using System;
using System.Diagnostics;
using System.Text;
using tRowcnt = System.UInt32; // 32-bit is the default

namespace Core.Command
{
    public partial class Insert
    {
        static void OpenTable(Parse p, int cur, int db, Table table, OP opcode)
        {
            Debug.Assert(!E.IsVirtual(table));
            Vdbe v = p.GetVdbe();
            Debug.Assert(opcode == OP.OpenWrite || opcode == OP.OpenRead);
            sqlite3TableLock(p, db, table.Id, (byte)(opcode == OP.OpenWrite ? 1 : 0), table.Name);
            v.AddOp3(opcode, cur, table.Id, db);
            v.ChangeP4(-1, table.Cols.length, Vdbe.P4T.INT32);
            v.Comment("%s", table.Name);
        }

        public static string IndexAffinityStr(Vdbe v, Index index)
        {
            if (index.ColAff == null || index.ColAff[0] == '\0')
            {
                // The first time a column affinity string for a particular index is required, it is allocated and populated here. It is then stored as
                // a member of the Index structure for subsequent use.
                //
                // The column affinity string will eventually be deleted by sqliteDeleteIndex() when the Index structure itself is cleaned up.
                Table table = index.Table;
                Context ctx = v.Ctx;
                StringBuilder b = new StringBuilder(index.Columns.length + 2); //: _tagalloc(nullptr, index->Columns.length+2);
                if (b == null)
                {
                    ctx.MallocFailed = true;
                    return null;
                }
                for (int n = 0; n < index.Columns.length; n++)
                    b.Append(table.Cols[index.Columns[n]].Affinity);
                b.Append(AFF.NONE);
                b.Append('\0');
                index.ColAff = b.ToString();
            }
            return index.ColAff;
        }

        public static void TableAffinityStr(Vdbe v, Table table)
        {
		    // The first time a column affinity string for a particular table is required, it is allocated and populated here. It is then 
		    // stored as a member of the Table structure for subsequent use.
		    //
		    // The column affinity string will eventually be deleted by sqlite3DeleteTable() when the Table structure itself is cleaned up.
            if (table.ColAff == null)
            {
                Context ctx = v.Ctx;
                StringBuilder b = new StringBuilder(table.Cols.length + 1);// (char)sqlite3DbMallocRaw(0, pTab->nCol+1);
                if (b == null)
                {
                    ctx.MallocFailed = true;
                    return;
                }
                for (int i = 0; i < table.Cols.; i++)
                    b.Append(table.Cols[i].Affinity);
                //b.Append('\0');
                table.ColAff = b.ToString();
            }
            v.ChangeP4(-1, table.ColAff, Vdbe.P4T.TRANSIENT);
        }

        static bool ReadsTable(Parse p, int startAddr, int db, Table table)
        {
            Vdbe v = p.GetVdbe();
            int end = v.CurrentAddr();
#if !OMIT_VIRTUALTABLE
            VTable vtable = (E.IsVirtual(table) ? VTable.GetVTable(p.Ctx, table) : null);
#endif
            for (int i = startAddr; i < end; i++)
            {
                Vdbe.VdbeOp op = v.GetOp(i);
                Debug.Assert(op != null);
                if (op.Opcode == OP.OpenRead && op.P3 == db)
                {
                    int id = op.P2;
                    if (id == table.Id)
                        return true;
                    for (Index index = table.Index; index != null; index = index.Next)
                        if (id == index.Id)
                            return true;
                }
#if !OMIT_VIRTUALTABLE
                if (op.Opcode == OP.VOpen && op.P4.VTable == vtable)
                {
                    Debug.Assert(op.P4.VTable != null);
                    Debug.Assert(op.P4Type == Vdbe.P4T.VTAB);
                    return true;
                }
#endif
            }
            return false;
        }

#if !OMIT_AUTOINCREMENT
        static int AutoIncBegin(Parse parse, int db, Table table)
        {
            int memId = 0;      // Register holding maximum rowid
            if ((table.TabFlags & TF.Autoincrement) != 0)
            {
                Parse toplevel = Parse.Toplevel(parse);
                AutoincInfo info = toplevel.Ainc;
                while (info != null && info.Table != table) info = info.Next;
                if (info == null)
                {
                    info = new AutoincInfo();
                    if (info == null) return 0;
                    info.Next = toplevel.Ainc;
                    toplevel.Ainc = info;
                    info.Table = table;
                    info.DB = db;
                    toplevel.Mems++;                // Register to hold name of table
                    info.RegCtr = ++toplevel.Mems;  // Max rowid register
                    toplevel.Mems++;                // Rowid in sqlite_sequence
                }
                memId = info.RegCtr;
            }
            return memId;
        }

        public static void AutoincrementBegin(Parse parse)
        {
            Context ctx = parse.Ctx; // The database connection
            Vdbe v = parse.V; // VDBE under construction

            // This routine is never called during trigger-generation.  It is only called from the top-level
            Debug.Assert(parse.TriggerTab == null);
            Debug.Assert(parse == Parse.Toplevel(parse));

            Debug.Assert(v != null); // We failed long ago if this is not so
            for (AutoincInfo p = parse.Ainc; p != null; p = p.Next) // Information about an AUTOINCREMENT
            {
                Context.DB dbAsObj = ctx.DBs[p.DB]; // Database only autoinc table
                int memId = p.RegCtr; // Register holding max rowid
                Debug.Assert(Btree.SchemaMutexHeld(ctx, 0, dbAsObj.Schema));
                OpenTable(parse, 0, p.DB, dbAsObj.Schema.SeqTable, OP.OpenRead);
                int addr = v.CurrentAddr(); // A VDBE address
                v.AddOp4(OP.String8, 0, memId - 1, 0, p.Table.Name, 0);
                v.AddOp2(OP.Rewind, 0, addr + 9);
                v.AddOp3(OP.Column, 0, 0, memId);
                v.AddOp3(OP.Ne, memId - 1, addr + 7, memId);
                v.ChangeP5(SQLITE_JUMPIFNULL);
                v.AddOp2(OP.Rowid, 0, memId + 1);
                v.AddOp3(OP.Column, 0, 1, memId);
                v.AddOp2(OP.Goto, 0, addr + 9);
                v.AddOp2(OP.Next, 0, addr + 2);
                v.AddOp2(OP.Integer, 0, memId);
                v.AddOp0(OP.Close);
            }
        }

        static void AutoIncStep(Parse parse, int memId, int regRowid)
        {
            if (memId > 0)
                parse.V.AddOp2(OP.MemMax, memId, regRowid);
        }

        public static void AutoincrementEnd(Parse parse)
        {
            Vdbe v = parse.V;
            Context ctx = parse.Ctx;

            Debug.Assert(v != null);
            for (AutoincInfo p = parse.Ainc; p != null; p = p.Next)
            {
                Context.DB dbAsObj = ctx.DBs[p.DB];
                int memId = p.RegCtr;

                int rec = Expr.GetTempReg(parse);
                Debug.Assert(Btree.SchemaMutexHeld(ctx, 0, dbAsObj.Schema));
                OpenTable(parse, 0, p.DB, dbAsObj.Schema.SeqTable, OP.OpenWrite);
                int j1 = v.AddOp1(OP.NotNull, memId + 1);
                int j2 = v.AddOp0(OP.Rewind);
                int j3 = v.AddOp3(OP.Column, 0, 0, rec);
                int j4 = v.AddOp3(OP.Eq, memId - 1, 0, rec);
                v.AddOp2(OP.Next, 0, j3);
                v.JumpHere(j2);
                v.AddOp2(OP.NewRowid, 0, memId + 1);
                int j5 = v.AddOp0(OP.Goto);
                v.JumpHere(j4);
                v.AddOp2(OP.Rowid, 0, memId + 1);
                v.JumpHere(j1);
                v.JumpHere(j5);
                v.AddOp3(OP.MakeRecord, memId - 1, 2, rec);
                v.AddOp3(OP.Insert, 0, rec, memId + 1);
                v.ChangeP5(Vdbe.OPFLAG.APPEND);
                v.AddOp0(OP.Close);
                Expr.ReleaseTempReg(parse, rec);
            }
        }
        //#else
        //#define AutoIncBegin(A,B,C) (0)
        //#define AutoIncStep(A,B,C)
#endif

#if false

        // OVERLOADS, so I don't need to rewrite parse.c
        static void Insert_(Parse parse, SrcList tabList, int dummy1, int dummy2, IdList column, int onError) { Insert_(parse, tabList, null, null, column, onError); }
        static void Insert_(Parse parse, SrcList tabList, int dummy1, Select select, IdList column, int onError) { Insert_(parse, tabList, null, select, column, onError); }
        static void Insert_(Parse parse, SrcList tabList, ExprList list, int dummy1, IdList column, int onError) { Insert_(parse, tabList, list, null, column, onError); }
        static void Insert_(Parse parse, SrcList tabList, ExprList list, Select select, IdList column, int onError)
        {
            
            int i = 0;
            int j = 0;
            int idx = 0;            // Loop counters
            Index pIdx;           // For looping over indices of the table
            int nColumn;          // Number of columns in the data
            int nHidden = 0;      // Number of hidden columns if TABLE is virtual
            int baseCur = 0;      // VDBE VdbeCursor number for pTab
            int keyColumn = -1;   // Column that is the INTEGER PRIMARY KEY
            int endOfLoop = 0;      /* Label for the end of the insertion loop */
            bool useTempTable = false; /* Store SELECT results in intermediate table */
            int srcTab = 0;       /* Data comes from this temporary cursor if >=0 */
            int addrInsTop = 0;   /* Jump to label "D" */
            int addrCont = 0;     /* Top of insert loop. Label "C" in templates 3 and 4 */
            int addrSelect = 0;   /* Address of coroutine that implements the SELECT */
            
            int iDb;              /* Index of database holding TABLE */
            Db pDb;               /* The database containing table being inserted into */
            bool appendFlag = false;   /* True if the insert is likely to be an append */

            // Register allocations
            int regFromSelect = 0;  // Base register for data coming from SELECT
            int regAutoinc = 0;   // Register holding the AUTOINCREMENT counter
            int regRowCount = 0;  // Memory cell used for the row counter
            int regIns;           // Block of regs holding rowid+data being inserted
            int regRowid;         // registers holding insert rowid
            int regData;          // register holding first column to insert
            int regEof = 0;       // Register recording end of SELECT data
            int[] aRegIdx = null; // One register allocated to each index

#if !OMIT_TRIGGER
            bool isView = false;        // True if attempting to insert into a view
            Trigger pTrigger;           // List of triggers on pTab, if required
            int tmask = 0;              // Mask of trigger times
#endif

            Context ctx = parse.Ctx; // The main database structure
            SelectDest dest = new SelectDest(); // Destination for SELECT on rhs of INSERT //: _memset(&dest, 0, sizeof(dest));
            if (parse.Errs != 0 || ctx.MallocFailed)
                goto insert_cleanup;

            // Locate the table into which we will be inserting new information.
            Debug.Assert(tabList.Srcs == 1);
            string tableName = tabList.Ids[0].Name; // Name of the table into which we are inserting
            if (C._NEVER(tableName == null))
                goto insert_cleanup;
            Table table = sqlite3SrcListLookup(parse, tabList); // The table to insert into.  aka TABLE
            if (table == null)
                goto insert_cleanup;
            iDb = sqlite3SchemaToIndex(ctx, table.pSchema);
            Debug.Assert(iDb < ctx.nDb);
            pDb = ctx.aDb[iDb];
#if !OMIT_AUTHORIZATION
            string zDb;           /* Name of the database holding this table */
            zDb = pDb.zName;
            if (sqlite3AuthCheck(parse, SQLITE_INSERT, table.zName, 0, zDb))
            {
                goto insert_cleanup;
            }
#endif
            // Figure out if we have any triggers and if the table being inserted into is a view
#if !OMIT_TRIGGER
            pTrigger = sqlite3TriggersExist(parse, table, TK_INSERT, null, out tmask);
            isView = table.pSelect != null;
#else
      Trigger pTrigger = null;  //# define pTrigger 0
      int tmask = 0;            //# define tmask 0
      bool isView = false;
#endif
#if OMIT_VIEW
//# undef isView
isView = false;
#endif
#if !OMIT_TRIGGER
            Debug.Assert((pTrigger != null && tmask != 0) || (pTrigger == null && tmask == 0));
#endif

#if !OMIT_VIEW
            // If pTab is really a view, make sure it has been initialized. ViewGetColumnNames() is a no-op if pTab is not a view (or virtual module table).
            if (sqlite3ViewGetColumnNames(parse, table) != -0)
                goto insert_cleanup;
#endif

            // Ensure that: (a) the table is not read-only,  (b) that if it is a view then ON INSERT triggers exist
            if (sqlite3IsReadOnly(parse, table, tmask))
                goto insert_cleanup;

            // Allocate a VDBE
            Vdbe v = parse.GetVdbe(); // Generate code into this virtual machine
            if (v == null)
                goto insert_cleanup;
            if (parse.nested == 0)
                sqlite3VdbeCountChanges(v);
            sqlite3BeginWriteOperation(parse, (select != null || pTrigger != null) ? 1 : 0, iDb);

#if !OMIT_XFER_OPT
            /* If the statement is of the form
**
**       INSERT INTO <table1> SELECT * FROM <table2>;
**
** Then special optimizations can be applied that make the transfer
** very fast and which reduce fragmentation of indices.
**
** This is the 2nd template.
*/
            if (column == null && xferOptimization(parse, table, select, onError, iDb) != 0)
            {
                Debug.Assert(null == pTrigger);
                Debug.Assert(list == null);
                goto insert_end;
            }
#endif

            /* If this is an AUTOINCREMENT table, look up the sequence number in the
** sqlite_sequence table and store it in memory cell regAutoinc.
*/
            regAutoinc = AutoIncBegin(parse, iDb, table);

            /* Figure out how many columns of data are supplied.  If the data
            ** is coming from a SELECT statement, then generate a co-routine that
            ** produces a single row of the SELECT on each invocation.  The
            ** co-routine is the common header to the 3rd and 4th templates.
            */
            if (select != null)
            {
                /* Data is coming from a SELECT.  Generate code to implement that SELECT
                ** as a co-routine.  The code is common to both the 3rd and 4th
                ** templates:
                **
                **         EOF <- 0
                **         X <- A
                **         goto B
                **      A: setup for the SELECT
                **         loop over the tables in the SELECT
                **           load value into register R..R+n
                **           yield X
                **         end loop
                **         cleanup after the SELECT
                **         EOF <- 1
                **         yield X
                **         halt-error
                **
                ** On each invocation of the co-routine, it puts a single row of the
                ** SELECT result into registers dest.iMem...dest.iMem+dest.nMem-1.
                ** (These output registers are allocated by sqlite3Select().)  When
                ** the SELECT completes, it sets the EOF flag stored in regEof.
                */
                int rc = 0, j1;

                regEof = ++parse.nMem;
                sqlite3VdbeAddOp2(v, OP_Integer, 0, regEof);      /* EOF <- 0 */
                VdbeComment(v, "SELECT eof flag");
                sqlite3SelectDestInit(dest, SRT_Coroutine, ++parse.nMem);
                addrSelect = sqlite3VdbeCurrentAddr(v) + 2;
                sqlite3VdbeAddOp2(v, OP_Integer, addrSelect - 1, dest.iParm);
                j1 = sqlite3VdbeAddOp2(v, OP_Goto, 0, 0);
                VdbeComment(v, "Jump over SELECT coroutine");
                /* Resolve the expressions in the SELECT statement and execute it. */
                rc = sqlite3Select(parse, select, ref dest);
                Debug.Assert(parse.nErr == 0 || rc != 0);
                if (rc != 0 || NEVER(parse.nErr != 0) /*|| db.mallocFailed != 0 */ )
                {
                    goto insert_cleanup;
                }
                sqlite3VdbeAddOp2(v, OP_Integer, 1, regEof);         /* EOF <- 1 */
                sqlite3VdbeAddOp1(v, OP_Yield, dest.iParm);   /* yield X */
                sqlite3VdbeAddOp2(v, OP_Halt, SQLITE_INTERNAL, OE_Abort);
                VdbeComment(v, "End of SELECT coroutine");
                sqlite3VdbeJumpHere(v, j1);                          /* label B: */

                regFromSelect = dest.iMem;
                Debug.Assert(select.pEList != null);
                nColumn = select.pEList.nExpr;
                Debug.Assert(dest.nMem == nColumn);

                /* Set useTempTable to TRUE if the result of the SELECT statement
                ** should be written into a temporary table (template 4).  Set to
                ** FALSE if each* row of the SELECT can be written directly into
                ** the destination table (template 3).
                **
                ** A temp table must be used if the table being updated is also one
                ** of the tables being read by the SELECT statement.  Also use a
                ** temp table in the case of row triggers.
                */
                if (pTrigger != null || ReadsTable(parse, addrSelect, iDb, table))
                {
                    useTempTable = true;
                }

                if (useTempTable)
                {
                    /* Invoke the coroutine to extract information from the SELECT
                    ** and add it to a transient table srcTab.  The code generated
                    ** here is from the 4th template:
                    **
                    **      B: open temp table
                    **      L: yield X
                    **         if EOF goto M
                    **         insert row from R..R+n into temp table
                    **         goto L
                    **      M: ...
                    */
                    int regRec;      /* Register to hold packed record */
                    int regTempRowid;    /* Register to hold temp table ROWID */
                    int addrTop;     /* Label "L" */
                    int addrIf;      /* Address of jump to M */

                    srcTab = parse.nTab++;
                    regRec = sqlite3GetTempReg(parse);
                    regTempRowid = sqlite3GetTempReg(parse);
                    sqlite3VdbeAddOp2(v, OP_OpenEphemeral, srcTab, nColumn);
                    addrTop = sqlite3VdbeAddOp1(v, OP_Yield, dest.iParm);
                    addrIf = sqlite3VdbeAddOp1(v, OP_If, regEof);
                    sqlite3VdbeAddOp3(v, OP_MakeRecord, regFromSelect, nColumn, regRec);
                    sqlite3VdbeAddOp2(v, OP_NewRowid, srcTab, regTempRowid);
                    sqlite3VdbeAddOp3(v, OP_Insert, srcTab, regRec, regTempRowid);
                    sqlite3VdbeAddOp2(v, OP_Goto, 0, addrTop);
                    sqlite3VdbeJumpHere(v, addrIf);
                    sqlite3ReleaseTempReg(parse, regRec);
                    sqlite3ReleaseTempReg(parse, regTempRowid);
                }
            }
            else
            {
                /* This is the case if the data for the INSERT is coming from a VALUES
                ** clause
                */
                NameContext sNC;
                sNC = new NameContext();// memset( &sNC, 0, sNC ).Length;
                sNC.pParse = parse;
                srcTab = -1;
                Debug.Assert(!useTempTable);
                nColumn = list != null ? list.nExpr : 0;
                for (i = 0; i < nColumn; i++)
                {
                    if (sqlite3ResolveExprNames(sNC, ref list.a[i].pExpr) != 0)
                    {
                        goto insert_cleanup;
                    }
                }
            }

            /* Make sure the number of columns in the source data matches the number
            ** of columns to be inserted into the table.
            */
            if (IsVirtual(table))
            {
                for (i = 0; i < table.nCol; i++)
                {
                    nHidden += (IsHiddenColumn(table.aCol[i]) ? 1 : 0);
                }
            }
            if (column == null && nColumn != 0 && nColumn != (table.nCol - nHidden))
            {
                sqlite3ErrorMsg(parse,
                "table %S has %d columns but %d values were supplied",
                tabList, 0, table.nCol - nHidden, nColumn);
                goto insert_cleanup;
            }
            if (column != null && nColumn != column.nId)
            {
                sqlite3ErrorMsg(parse, "%d values for %d columns", nColumn, column.nId);
                goto insert_cleanup;
            }

            /* If the INSERT statement included an IDLIST term, then make sure
            ** all elements of the IDLIST really are columns of the table and
            ** remember the column indices.
            **
            ** If the table has an INTEGER PRIMARY KEY column and that column
            ** is named in the IDLIST, then record in the keyColumn variable
            ** the index into IDLIST of the primary key column.  keyColumn is
            ** the index of the primary key as it appears in IDLIST, not as
            ** is appears in the original table.  (The index of the primary
            ** key in the original table is pTab.iPKey.)
            */
            if (column != null)
            {
                for (i = 0; i < column.nId; i++)
                {
                    column.a[i].idx = -1;
                }
                for (i = 0; i < column.nId; i++)
                {
                    for (j = 0; j < table.nCol; j++)
                    {
                        if (column.a[i].zName.Equals(table.aCol[j].zName, StringComparison.OrdinalIgnoreCase))
                        {
                            column.a[i].idx = j;
                            if (j == table.iPKey)
                            {
                                keyColumn = i;
                            }
                            break;
                        }
                    }
                    if (j >= table.nCol)
                    {
                        if (sqlite3IsRowid(column.a[i].zName))
                        {
                            keyColumn = i;
                        }
                        else
                        {
                            sqlite3ErrorMsg(parse, "table %S has no column named %s",
                            tabList, 0, column.a[i].zName);
                            parse.checkSchema = 1;
                            goto insert_cleanup;
                        }
                    }
                }
            }

            /* If there is no IDLIST term but the table has an integer primary
            ** key, the set the keyColumn variable to the primary key column index
            ** in the original table definition.
            */
            if (column == null && nColumn > 0)
            {
                keyColumn = table.iPKey;
            }

            /* Initialize the count of rows to be inserted
            */
            if ((ctx.flags & SQLITE_CountRows) != 0)
            {
                regRowCount = ++parse.nMem;
                sqlite3VdbeAddOp2(v, OP_Integer, 0, regRowCount);
            }

            /* If this is not a view, open the table and and all indices */
            if (!isView)
            {
                int nIdx;

                baseCur = parse.nTab;
                nIdx = sqlite3OpenTableAndIndices(parse, table, baseCur, OP_OpenWrite);
                aRegIdx = new int[nIdx + 1];// sqlite3DbMallocRaw( db, sizeof( int ) * ( nIdx + 1 ) );
                if (aRegIdx == null)
                {
                    goto insert_cleanup;
                }
                for (i = 0; i < nIdx; i++)
                {
                    aRegIdx[i] = ++parse.nMem;
                }
            }

            /* This is the top of the main insertion loop */
            if (useTempTable)
            {
                /* This block codes the top of loop only.  The complete loop is the
                ** following pseudocode (template 4):
                **
                **         rewind temp table
                **      C: loop over rows of intermediate table
                **           transfer values form intermediate table into <table>
                **         end loop
                **      D: ...
                */
                addrInsTop = sqlite3VdbeAddOp1(v, OP_Rewind, srcTab);
                addrCont = sqlite3VdbeCurrentAddr(v);
            }
            else if (select != null)
            {
                /* This block codes the top of loop only.  The complete loop is the
                ** following pseudocode (template 3):
                **
                **      C: yield X
                **         if EOF goto D
                **         insert the select result into <table> from R..R+n
                **         goto C
                **      D: ...
                */
                addrCont = sqlite3VdbeAddOp1(v, OP_Yield, dest.iParm);
                addrInsTop = sqlite3VdbeAddOp1(v, OP_If, regEof);
            }

            /* Allocate registers for holding the rowid of the new row,
            ** the content of the new row, and the assemblied row record.
            */
            regRowid = regIns = parse.nMem + 1;
            parse.nMem += table.nCol + 1;
            if (IsVirtual(table))
            {
                regRowid++;
                parse.nMem++;
            }
            regData = regRowid + 1;

            /* Run the BEFORE and INSTEAD OF triggers, if there are any
            */
            endOfLoop = sqlite3VdbeMakeLabel(v);
#if !OMIT_TRIGGER
            if ((tmask & TRIGGER_BEFORE) != 0)
            {
                int regCols = sqlite3GetTempRange(parse, table.nCol + 1);

                /* build the NEW.* reference row.  Note that if there is an INTEGER
                ** PRIMARY KEY into which a NULL is being inserted, that NULL will be
                ** translated into a unique ID for the row.  But on a BEFORE trigger,
                ** we do not know what the unique ID will be (because the insert has
                ** not happened yet) so we substitute a rowid of -1
                */
                if (keyColumn < 0)
                {
                    sqlite3VdbeAddOp2(v, OP_Integer, -1, regCols);
                }
                else
                {
                    int j1;
                    if (useTempTable)
                    {
                        sqlite3VdbeAddOp3(v, OP_Column, srcTab, keyColumn, regCols);
                    }
                    else
                    {
                        Debug.Assert(select == null);  /* Otherwise useTempTable is true */
                        sqlite3ExprCode(parse, list.a[keyColumn].pExpr, regCols);
                    }
                    j1 = sqlite3VdbeAddOp1(v, OP_NotNull, regCols);
                    sqlite3VdbeAddOp2(v, OP_Integer, -1, regCols);
                    sqlite3VdbeJumpHere(v, j1);
                    sqlite3VdbeAddOp1(v, OP_MustBeInt, regCols);
                }
                /* Cannot have triggers on a virtual table. If it were possible,
                ** this block would have to account for hidden column.
                */
                Debug.Assert(!IsVirtual(table));
                /* Create the new column data
                */
                for (i = 0; i < table.nCol; i++)
                {
                    if (column == null)
                    {
                        j = i;
                    }
                    else
                    {
                        for (j = 0; j < column.nId; j++)
                        {
                            if (column.a[j].idx == i)
                                break;
                        }
                    }
                    if ((!useTempTable && null == list) || (column != null && j >= column.nId))
                    {
                        sqlite3ExprCode(parse, table.aCol[i].pDflt, regCols + i + 1);
                    }
                    else if (useTempTable)
                    {
                        sqlite3VdbeAddOp3(v, OP_Column, srcTab, j, regCols + i + 1);
                    }
                    else
                    {
                        Debug.Assert(select == null); /* Otherwise useTempTable is true */
                        sqlite3ExprCodeAndCache(parse, list.a[j].pExpr, regCols + i + 1);
                    }
                }

                /* If this is an INSERT on a view with an INSTEAD OF INSERT trigger,
                ** do not attempt any conversions before assembling the record.
                ** If this is a real table, attempt conversions as required by the
                ** table column affinities.
                */
                if (!isView)
                {
                    sqlite3VdbeAddOp2(v, OP_Affinity, regCols + 1, table.nCol);
                    TableAffinityStr(v, table);
                }

                /* Fire BEFORE or INSTEAD OF triggers */
                sqlite3CodeRowTrigger(parse, pTrigger, TK_INSERT, null, TRIGGER_BEFORE,
                    table, regCols - table.nCol - 1, onError, endOfLoop);

                sqlite3ReleaseTempRange(parse, regCols, table.nCol + 1);
            }
#endif

            /* Push the record number for the new entry onto the stack.  The
** record number is a randomly generate integer created by NewRowid
** except when the table has an INTEGER PRIMARY KEY column, in which
** case the record number is the same as that column.
*/
            if (!isView)
            {
                if (IsVirtual(table))
                {
                    /* The row that the VUpdate opcode will delete: none */
                    sqlite3VdbeAddOp2(v, OP_Null, 0, regIns);
                }
                if (keyColumn >= 0)
                {
                    if (useTempTable)
                    {
                        sqlite3VdbeAddOp3(v, OP_Column, srcTab, keyColumn, regRowid);
                    }
                    else if (select != null)
                    {
                        sqlite3VdbeAddOp2(v, OP_SCopy, regFromSelect + keyColumn, regRowid);
                    }
                    else
                    {
                        VdbeOp pOp;
                        sqlite3ExprCode(parse, list.a[keyColumn].pExpr, regRowid);
                        pOp = sqlite3VdbeGetOp(v, -1);
                        if (ALWAYS(pOp != null) && pOp.opcode == OP_Null && !IsVirtual(table))
                        {
                            appendFlag = true;
                            pOp.opcode = OP_NewRowid;
                            pOp.p1 = baseCur;
                            pOp.p2 = regRowid;
                            pOp.p3 = regAutoinc;
                        }
                    }
                    /* If the PRIMARY KEY expression is NULL, then use OP_NewRowid
                    ** to generate a unique primary key value.
                    */
                    if (!appendFlag)
                    {
                        int j1;
                        if (!IsVirtual(table))
                        {
                            j1 = sqlite3VdbeAddOp1(v, OP_NotNull, regRowid);
                            sqlite3VdbeAddOp3(v, OP_NewRowid, baseCur, regRowid, regAutoinc);
                            sqlite3VdbeJumpHere(v, j1);
                        }
                        else
                        {
                            j1 = sqlite3VdbeCurrentAddr(v);
                            sqlite3VdbeAddOp2(v, OP_IsNull, regRowid, j1 + 2);
                        }
                        sqlite3VdbeAddOp1(v, OP_MustBeInt, regRowid);
                    }
                }
                else if (IsVirtual(table))
                {
                    sqlite3VdbeAddOp2(v, OP_Null, 0, regRowid);
                }
                else
                {
                    sqlite3VdbeAddOp3(v, OP_NewRowid, baseCur, regRowid, regAutoinc);
                    appendFlag = true;
                }
                AutoIncStep(parse, regAutoinc, regRowid);

                /* Push onto the stack, data for all columns of the new entry, beginning
                ** with the first column.
                */
                nHidden = 0;
                for (i = 0; i < table.nCol; i++)
                {
                    int iRegStore = regRowid + 1 + i;
                    if (i == table.iPKey)
                    {
                        /* The value of the INTEGER PRIMARY KEY column is always a NULL.
                        ** Whenever this column is read, the record number will be substituted
                        ** in its place.  So will fill this column with a NULL to avoid
                        ** taking up data space with information that will never be used. */
                        sqlite3VdbeAddOp2(v, OP_Null, 0, iRegStore);
                        continue;
                    }
                    if (column == null)
                    {
                        if (IsHiddenColumn(table.aCol[i]))
                        {
                            Debug.Assert(IsVirtual(table));
                            j = -1;
                            nHidden++;
                        }
                        else
                        {
                            j = i - nHidden;
                        }
                    }
                    else
                    {
                        for (j = 0; j < column.nId; j++)
                        {
                            if (column.a[j].idx == i)
                                break;
                        }
                    }
                    if (j < 0 || nColumn == 0 || (column != null && j >= column.nId))
                    {
                        sqlite3ExprCode(parse, table.aCol[i].pDflt, iRegStore);
                    }
                    else if (useTempTable)
                    {
                        sqlite3VdbeAddOp3(v, OP_Column, srcTab, j, iRegStore);
                    }
                    else if (select != null)
                    {
                        sqlite3VdbeAddOp2(v, OP_SCopy, regFromSelect + j, iRegStore);
                    }
                    else
                    {
                        sqlite3ExprCode(parse, list.a[j].pExpr, iRegStore);
                    }
                }

                /* Generate code to check constraints and generate index keys and
                ** do the insertion.
                */
#if !OMIT_VIRTUALTABLE
                if (IsVirtual(table))
                {
                    VTable pVTab = sqlite3GetVTable(ctx, table);
                    sqlite3VtabMakeWritable(parse, table);
                    sqlite3VdbeAddOp4(v, OP_VUpdate, 1, table.nCol + 2, regIns, pVTab, P4_VTAB);
                    sqlite3VdbeChangeP5(v, (byte)(onError == OE_Default ? OE_Abort : onError));
                    sqlite3MayAbort(parse);
                }
                else
#endif
                {
                    int isReplace = 0;    /* Set to true if constraints may cause a replace */
                    sqlite3GenerateConstraintChecks(parse, table, baseCur, regIns, aRegIdx,
                      keyColumn >= 0 ? 1 : 0, false, onError, endOfLoop, out isReplace
                    );
                    sqlite3FkCheck(parse, table, 0, regIns);
                    sqlite3CompleteInsertion(
                   parse, table, baseCur, regIns, aRegIdx, false, appendFlag, isReplace == 0
                    );
                }
            }

            /* Update the count of rows that are inserted
            */
            if ((ctx.flags & SQLITE_CountRows) != 0)
            {
                sqlite3VdbeAddOp2(v, OP_AddImm, regRowCount, 1);
            }

#if !OMIT_TRIGGER
            if (pTrigger != null)
            {
                /* Code AFTER triggers */
                sqlite3CodeRowTrigger(parse, pTrigger, TK_INSERT, null, TRIGGER_AFTER,
                    table, regData - 2 - table.nCol, onError, endOfLoop);
            }
#endif

            /* The bottom of the main insertion loop, if the data source
** is a SELECT statement.
*/
            sqlite3VdbeResolveLabel(v, endOfLoop);
            if (useTempTable)
            {
                sqlite3VdbeAddOp2(v, OP_Next, srcTab, addrCont);
                sqlite3VdbeJumpHere(v, addrInsTop);
                sqlite3VdbeAddOp1(v, OP_Close, srcTab);
            }
            else if (select != null)
            {
                sqlite3VdbeAddOp2(v, OP_Goto, 0, addrCont);
                sqlite3VdbeJumpHere(v, addrInsTop);
            }

            if (!IsVirtual(table) && !isView)
            {
                /* Close all tables opened */
                sqlite3VdbeAddOp1(v, OP_Close, baseCur);
                for (idx = 1, pIdx = table.pIndex; pIdx != null; pIdx = pIdx.pNext, idx++)
                {
                    sqlite3VdbeAddOp1(v, OP_Close, idx + baseCur);
                }
            }

        insert_end:
            /* Update the sqlite_sequence table by storing the content of the
            ** maximum rowid counter values recorded while inserting into
            ** autoincrement tables.
            */
            if (parse.nested == 0 && parse.pTriggerTab == null)
            {
                sqlite3AutoincrementEnd(parse);
            }

            /*
            ** Return the number of rows inserted. If this routine is
            ** generating code because of a call to sqlite3NestedParse(), do not
            ** invoke the callback function.
            */
            if ((ctx.flags & SQLITE_CountRows) != 0 && 0 == parse.nested && null == parse.pTriggerTab)
            {
                sqlite3VdbeAddOp2(v, OP_ResultRow, regRowCount, 1);
                sqlite3VdbeSetNumCols(v, 1);
                sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "rows inserted", SQLITE_STATIC);
            }

        insert_cleanup:
            sqlite3SrcListDelete(ctx, ref tabList);
            sqlite3ExprListDelete(ctx, ref list);
            sqlite3SelectDelete(ctx, ref select);
            sqlite3IdListDelete(ctx, ref column);
            sqlite3DbFree(ctx, ref aRegIdx);
        }


        static void sqlite3GenerateConstraintChecks(Parse pParse,Table pTab,int baseCur, int regRowid,       int[] aRegIdx,     int rowidChng,bool isUpdate,int overrideError,int ignoreDest,   out int pbMayReplace)
        {

            int i;               /* loop counter */
            Vdbe v;              /* VDBE under constrution */
            int nCol;            /* Number of columns */
            int onError;         /* Conflict resolution strategy */
            int j1;              /* Addresss of jump instruction */
            int j2 = 0, j3;      /* Addresses of jump instructions */
            int regData;         /* Register containing first data column */
            int iCur;            /* Table cursor number */
            Index pIdx;         /* Pointer to one of the indices */
            bool seenReplace = false; /* True if REPLACE is used to resolve INT PK conflict */
            int regOldRowid = (rowidChng != 0 && isUpdate) ? rowidChng : regRowid;

            v = sqlite3GetVdbe(pParse);
            Debug.Assert(v != null);
            Debug.Assert(pTab.pSelect == null);  /* This table is not a VIEW */
            nCol = pTab.nCol;
            regData = regRowid + 1;


            /* Test all NOT NULL constraints.
            */
            for (i = 0; i < nCol; i++)
            {
                if (i == pTab.iPKey)
                {
                    continue;
                }
                onError = pTab.aCol[i].notNull;
                if (onError == OE_None)
                    continue;
                if (overrideError != OE_Default)
                {
                    onError = overrideError;
                }
                else if (onError == OE_Default)
                {
                    onError = OE_Abort;
                }
                if (onError == OE_Replace && pTab.aCol[i].pDflt == null)
                {
                    onError = OE_Abort;
                }
                Debug.Assert(onError == OE_Rollback || onError == OE_Abort || onError == OE_Fail
                || onError == OE_Ignore || onError == OE_Replace);
                switch (onError)
                {
                    case OE_Abort:
                        {
                            sqlite3MayAbort(pParse);
                            goto case OE_Fail;
                        }
                    case OE_Rollback:
                    case OE_Fail:
                        {
                            string zMsg;
                            sqlite3VdbeAddOp3(v, OP_HaltIfNull,
                                        SQLITE_CONSTRAINT, onError, regData + i);
                            zMsg = sqlite3MPrintf(pParse.db, "%s.%s may not be NULL",
                            pTab.zName, pTab.aCol[i].zName);
                            sqlite3VdbeChangeP4(v, -1, zMsg, P4_DYNAMIC);
                            break;
                        }
                    case OE_Ignore:
                        {
                            sqlite3VdbeAddOp2(v, OP_IsNull, regData + i, ignoreDest);
                            break;
                        }
                    default:
                        {
                            Debug.Assert(onError == OE_Replace);
                            j1 = sqlite3VdbeAddOp1(v, OP_NotNull, regData + i);
                            sqlite3ExprCode(pParse, pTab.aCol[i].pDflt, regData + i);
                            sqlite3VdbeJumpHere(v, j1);
                            break;
                        }
                }
            }

            /* Test all CHECK constraints
            */
#if !OMIT_CHECK
            if (pTab.pCheck != null && (pParse.db.flags & SQLITE_IgnoreChecks) == 0)
            {
                int allOk = sqlite3VdbeMakeLabel(v);
                pParse.ckBase = regData;
                sqlite3ExprIfTrue(pParse, pTab.pCheck, allOk, SQLITE_JUMPIFNULL);
                onError = overrideError != OE_Default ? overrideError : OE_Abort;
                if (onError == OE_Ignore)
                {
                    sqlite3VdbeAddOp2(v, OP_Goto, 0, ignoreDest);
                }
                else
                {
                    if (onError == OE_Replace)
                        onError = OE_Abort; /* IMP: R-15569-63625 */
                    sqlite3HaltConstraint(pParse, onError, (string)null, 0);
                }
                sqlite3VdbeResolveLabel(v, allOk);
            }
#endif

            /* If we have an INTEGER PRIMARY KEY, make sure the primary key
** of the new record does not previously exist.  Except, if this
** is an UPDATE and the primary key is not changing, that is OK.
*/
            if (rowidChng != 0)
            {
                onError = pTab.keyConf;
                if (overrideError != OE_Default)
                {
                    onError = overrideError;
                }
                else if (onError == OE_Default)
                {
                    onError = OE_Abort;
                }

                if (isUpdate)
                {
                    j2 = sqlite3VdbeAddOp3(v, OP_Eq, regRowid, 0, rowidChng);
                }
                j3 = sqlite3VdbeAddOp3(v, OP_NotExists, baseCur, 0, regRowid);
                switch (onError)
                {
                    default:
                        {
                            onError = OE_Abort;
                            /* Fall thru into the next case */
                        }
                        goto case OE_Rollback;
                    case OE_Rollback:
                    case OE_Abort:
                    case OE_Fail:
                        {
                            sqlite3HaltConstraint(
                              pParse, onError, "PRIMARY KEY must be unique", P4_STATIC);
                            break;
                        }
                    case OE_Replace:
                        {
                            /* If there are DELETE triggers on this table and the
                            ** recursive-triggers flag is set, call GenerateRowDelete() to
                            ** remove the conflicting row from the the table. This will fire
                            ** the triggers and remove both the table and index b-tree entries.
                            **
                            ** Otherwise, if there are no triggers or the recursive-triggers
                            ** flag is not set, but the table has one or more indexes, call 
                            ** GenerateRowIndexDelete(). This removes the index b-tree entries 
                            ** only. The table b-tree entry will be replaced by the new entry 
                            ** when it is inserted.  
                            **
                            ** If either GenerateRowDelete() or GenerateRowIndexDelete() is called,
                            ** also invoke MultiWrite() to indicate that this VDBE may require
                            ** statement rollback (if the statement is aborted after the delete
                            ** takes place). Earlier versions called sqlite3MultiWrite() regardless,
                            ** but being more selective here allows statements like:
                            **
                            **   REPLACE INTO t(rowid) VALUES($newrowid)
                            **
                            ** to run without a statement journal if there are no indexes on the
                            ** table.
                            */
                            Trigger pTrigger = null;
                            if ((pParse.db.flags & SQLITE_RecTriggers) != 0)
                            {
                                int iDummy;
                                pTrigger = sqlite3TriggersExist(pParse, pTab, TK_DELETE, null, out iDummy);
                            }
                            if (pTrigger != null || sqlite3FkRequired(pParse, pTab, null, 0) != 0)
                            {
                                sqlite3MultiWrite(pParse);
                                sqlite3GenerateRowDelete(
                                    pParse, pTab, baseCur, regRowid, 0, pTrigger, OE_Replace
                                );
                            }
                            else
                                if (pTab.pIndex != null)
                                {
                                    sqlite3MultiWrite(pParse);
                                    sqlite3GenerateRowIndexDelete(pParse, pTab, baseCur, 0);
                                }
                            seenReplace = true;
                            break;
                        }
                    case OE_Ignore:
                        {
                            Debug.Assert(!seenReplace);
                            sqlite3VdbeAddOp2(v, OP_Goto, 0, ignoreDest);
                            break;
                        }
                }
                sqlite3VdbeJumpHere(v, j3);
                if (isUpdate)
                {
                    sqlite3VdbeJumpHere(v, j2);
                }
            }

            /* Test all UNIQUE constraints by creating entries for each UNIQUE
            ** index and making sure that duplicate entries do not already exist.
            ** Add the new records to the indices as we go.
            */
            for (iCur = 0, pIdx = pTab.pIndex; pIdx != null; pIdx = pIdx.pNext, iCur++)
            {
                int regIdx;
                int regR;

                if (aRegIdx[iCur] == 0)
                    continue;  /* Skip unused indices */

                /* Create a key for accessing the index entry */
                regIdx = sqlite3GetTempRange(pParse, pIdx.nColumn + 1);
                for (i = 0; i < pIdx.nColumn; i++)
                {
                    int idx = pIdx.aiColumn[i];
                    if (idx == pTab.iPKey)
                    {
                        sqlite3VdbeAddOp2(v, OP_SCopy, regRowid, regIdx + i);
                    }
                    else
                    {
                        sqlite3VdbeAddOp2(v, OP_SCopy, regData + idx, regIdx + i);
                    }
                }
                sqlite3VdbeAddOp2(v, OP_SCopy, regRowid, regIdx + i);
                sqlite3VdbeAddOp3(v, OP_MakeRecord, regIdx, pIdx.nColumn + 1, aRegIdx[iCur]);
                sqlite3VdbeChangeP4(v, -1, IndexAffinityStr(v, pIdx), P4_TRANSIENT);
                sqlite3ExprCacheAffinityChange(pParse, regIdx, pIdx.nColumn + 1);

                /* Find out what action to take in case there is an indexing conflict */
                onError = pIdx.onError;
                if (onError == OE_None)
                {
                    sqlite3ReleaseTempRange(pParse, regIdx, pIdx.nColumn + 1);
                    continue;  /* pIdx is not a UNIQUE index */
                }

                if (overrideError != OE_Default)
                {
                    onError = overrideError;
                }
                else if (onError == OE_Default)
                {
                    onError = OE_Abort;
                }
                if (seenReplace)
                {
                    if (onError == OE_Ignore)
                        onError = OE_Replace;
                    else if (onError == OE_Fail)
                        onError = OE_Abort;
                }


                /* Check to see if the new index entry will be unique */
                regR = sqlite3GetTempReg(pParse);
                sqlite3VdbeAddOp2(v, OP_SCopy, regOldRowid, regR);
                j3 = sqlite3VdbeAddOp4(v, OP_IsUnique, baseCur + iCur + 1, 0,
                regR, regIdx,//regR, SQLITE_INT_TO_PTR(regIdx),
                P4_INT32);
                sqlite3ReleaseTempRange(pParse, regIdx, pIdx.nColumn + 1);

                /* Generate code that executes if the new index entry is not unique */
                Debug.Assert(onError == OE_Rollback || onError == OE_Abort || onError == OE_Fail
                || onError == OE_Ignore || onError == OE_Replace);
                switch (onError)
                {
                    case OE_Rollback:
                    case OE_Abort:
                    case OE_Fail:
                        {
                            int j;
                            StrAccum errMsg = new StrAccum(200);
                            string zSep;
                            string zErr;

                            sqlite3StrAccumInit(errMsg, null, 0, 200);
                            errMsg.db = pParse.db;
                            zSep = pIdx.nColumn > 1 ? "columns " : "column ";
                            for (j = 0; j < pIdx.nColumn; j++)
                            {
                                string zCol = pTab.aCol[pIdx.aiColumn[j]].zName;
                                sqlite3StrAccumAppend(errMsg, zSep, -1);
                                zSep = ", ";
                                sqlite3StrAccumAppend(errMsg, zCol, -1);
                            }
                            sqlite3StrAccumAppend(errMsg,
                            pIdx.nColumn > 1 ? " are not unique" : " is not unique", -1);
                            zErr = sqlite3StrAccumFinish(errMsg);
                            sqlite3HaltConstraint(pParse, onError, zErr, 0);
                            sqlite3DbFree(errMsg.db, ref zErr);
                            break;
                        }
                    case OE_Ignore:
                        {
                            Debug.Assert(!seenReplace);
                            sqlite3VdbeAddOp2(v, OP_Goto, 0, ignoreDest);
                            break;
                        }
                    default:
                        {
                            Trigger pTrigger = null;
                            Debug.Assert(onError == OE_Replace);
                            sqlite3MultiWrite(pParse);
                            if ((pParse.db.flags & SQLITE_RecTriggers) != 0)
                            {
                                int iDummy;
                                pTrigger = sqlite3TriggersExist(pParse, pTab, TK_DELETE, null, out iDummy);
                            }
                            sqlite3GenerateRowDelete(
                                pParse, pTab, baseCur, regR, 0, pTrigger, OE_Replace
                            );
                            seenReplace = true;
                            break;
                        }
                }
                sqlite3VdbeJumpHere(v, j3);
                sqlite3ReleaseTempReg(pParse, regR);
            }
            //if ( pbMayReplace )
            {
                pbMayReplace = seenReplace ? 1 : 0;
            }
        }
#endif

        public static void CompleteInsertion(Parse parse, Table table, int baseCur, int regRowid, int[] regIdxs, bool isUpdate, bool appendBias, bool useSeekResult)
        {
            Vdbe v = parse.GetVdbe();
            Debug.Assert(v != null);
            Debug.Assert(table.Select == null); // This table is not a VIEW
            int indexsLength;
            Index index;
            for (indexsLength = 0, index = table.Index; index != null; index = index.Next, indexsLength++) { }
            for (int i = indexsLength - 1; i >= 0; i--)
            {
                if (regIdxs[i] == 0) continue;
                v.AddOp2(OP.IdxInsert, baseCur + i + 1, regIdxs[i]);
                if (useSeekResult)
                    v.ChangeP5(Vdbe.OPFLAG.USESEEKRESULT);
            }
            int regData = regRowid + 1;
            int regRec = Expr.GetTempReg(parse);
            v.AddOp3(OP.MakeRecord, regData, table.Cols.length, regRec);
            TableAffinityStr(v, table);
            Expr.CacheAffinityChange(parse, regData, table.Cols.length);
            Vdbe.OPFLAG pik_flags;
            if (parse.Nested != 0)
                pik_flags = 0;
            else
            {
                pik_flags = Vdbe.OPFLAG.NCHANGE;
                pik_flags |= (isUpdate ? Vdbe.OPFLAG.ISUPDATE : Vdbe.OPFLAG.LASTROWID);
            }
            if (appendBias) pik_flags |= Vdbe.OPFLAG.APPEND;
            if (useSeekResult) pik_flags |= Vdbe.OPFLAG.USESEEKRESULT;
            v.AddOp3(OP.Insert, baseCur, regRec, regRowid);
            if (parse.Nested == 0)
                v.ChangeP4(-1, table.Name, Vdbe.P4T.TRANSIENT);
            v.ChangeP5(pik_flags);
        }

        public static int OpenTableAndIndices(Parse parse, Table table, int baseCur, OP op)
        {
            if (E.IsVirtual(table)) return 0;
            int db = Prepare.SchemaToIndex(parse.Ctx, table.Schema);
            Vdbe v = parse.GetVdbe();
            Debug.Assert(v != null);
            OpenTable(parse, baseCur, db, table, op);
            int i;
            Index index;
            for (i = 1, index = table.Index; index != null; index = index.Next, i++)
            {
                KeyInfo pKey = parse.IndexKeyinfo(index);
                Debug.Assert(index.Schema == table.Schema);
                v.AddOp4(op, i + baseCur, index.Id, db, key, Vdbe.P4T.KEYINFO_HANDOFF);
                v.Comment("%s", index.Name);
            }
            if (parse.Tabs < baseCur + i)
                parse.Tabs = baseCur + i;
            return i - 1;
        }

#if TEST
        static int _xferopt_count = 0;
#endif
#if !OMIT_XFER_OPT
        static bool XferCompatibleCollation(string z1, string z2)
        {
            if (z1 == null) return (z2 == null);
            if (z2 == null) return false;
            return string.Equals(z1, z2, StringComparison.OrdinalIgnoreCase);
        }

        static bool XferCompatibleIndex(Index dest, Index src)
        {
            Debug.Assert(dest != null && src != null);
            Debug.Assert(dest.Table != src.Table);
            if (dest.Columns.length != src.Columns.length) return false;    // Different number of columns
            if (dest.OnError != src.OnError) return false;                  // Different conflict resolution strategies
            for (int i = 0; i < src.Columns.length; i++)
            {
                if (src.Columns[i] != dest.Columns[i]) return false;      // Different columns indexed
                if (src.SortOrders[i] != dest.SortOrders[i]) return false;  // Different sort orders
                if (!XferCompatibleCollation(src.CollNames[i], dest.CollNames[i])) return false; // Different collating sequences
            }
            // If no test above fails then the indices must be compatible
            return true;
        }

        static bool XferOptimization(Parse parse, Table dest, Select select, OE onError, int dbDestId)
        {
            if (select == null) return false;                                   // Must be of the form  INSERT INTO ... SELECT ...
            if (sqlite3TriggerList(parse, dest) != null) return false;          // tab1 must not have triggers
#if !OMIT_VIRTUALTABLE
            if ((dest.TabFlags & TF.Virtual) != 0) return false;				// tab1 must not be a virtual table
#endif
            if (onError == OE.Default)
            {
                if (dest.PKey >= 0) onError = dest.KeyConf;
                if (onError == OE.Default) onError = OE.Abort;
            }
            Debug.Assert(select.Src != null);                                   // allocated even if there is no FROM clause
            if (select.Src.Srcs != 1) return false;                             // FROM clause must have exactly one term
            if (select.Src.Ids[0].Select != null) return false;                 // FROM clause cannot contain a subquery
            if (select.Where != null) return false;                             // SELECT may not have a WHERE clause
            if (select.OrderBy != null) return false;                           // SELECT may not have an ORDER BY clause
            // Do not need to test for a HAVING clause.  If HAVING is present but there is no ORDER BY, we will get an error.
            if (select.GroupBy != null) return false;                           // SELECT may not have a GROUP BY clause
            if (select.Limit != null) return false;                             // SELECT may not have a LIMIT clause
            Debug.Assert(select.Offset == null);                                // Must be so if pLimit==0
            if (select.Prior != null) return false;                             // SELECT may not be a compound query
            if ((select.SelFlags & SF.Distinct) != 0) return false;             // SELECT may not be DISTINCT
            ExprList list = select.EList;                                      // The result set of the SELECT
            Debug.Assert(list != null);
            if (list.Exprs != 1) return false;                                 // The result set must have exactly one column
            Debug.Assert(list.Ids[0].Expr != null);
            if (list.Ids[0].Expr.OP != TK.ALL) return false;                   // The result set must be the special operator "*"

            // At this point we have established that the statement is of the correct syntactic form to participate in this optimization.  Now
            // we have to check the semantics.
            SrcList.SrcListItem item = select.Src.Ids[0];                       // An element of pSelect.pSrc
            Table src = parse.LocateTableItem(false, item);                     // The table in the FROM clause of SELECT
            if (src == null) return false;                                      // FROM clause does not contain a real table
            if (src == dest) return false;                                      // tab1 and tab2 may not be the same table
#if !OMIT_VIRTUALTABLE
            if ((src.TabFlags & TF.Virtual) != 0) return false;                 // tab2 must not be a virtual table
#endif
            if (src.Select != null) return false;                               // tab2 may not be a view
            if (dest.Cols.length != src.Cols.length) return false;              // Number of columns must be the same in tab1 and tab2
            if (dest.PKey != src.PKey) return false;                            // Both tables must have the same INTEGER PRIMARY KEY
            for (int i = 0; i < dest.Cols.length; i++)
            {
                if (dest.Cols[i].Affinity != src.Cols[i].Affinity) return false; // Affinity must be the same on all columns
                if (!XferCompatibleCollation(dest.Cols[i].Coll, src.Cols[i].Coll)) return false; // Collating sequence must be the same on all columns
                if (dest.Cols[i].NotNull != 0 && src.Cols[i].NotNull == 0) return false; // tab2 must be NOT NULL if tab1 is
            }
            bool destHasUniqueIdx = false;   // True if pDest has a UNIQUE index
            Index srcIdx, destIdx; // Source and destination indices
            for (destIdx = dest.Index; destIdx != null; destIdx = destIdx.Next)
            {
                if (destIdx.OnError != OE.None) destHasUniqueIdx = true;
                for (srcIdx = src.Index; srcIdx != null; srcIdx = srcIdx.Next)
                    if (XferCompatibleIndex(destIdx, srcIdx)) break;
                if (srcIdx == null) return false;                               // pDestIdx has no corresponding index in pSrc
            }
#if !OMIT_CHECK
            if (dest.Check != null && Expr.ListCompare(src.Check, dest.Check) != 0) return false; // Tables have different CHECK constraints.  Ticket #2252
#endif
#if !OMIT_FOREIGN_KEY
            // Disallow the transfer optimization if the destination table constains any foreign key constraints.  This is more restrictive than necessary.
            // But the main beneficiary of the transfer optimization is the VACUUM command, and the VACUUM command disables foreign key constraints.  So
            // the extra complication to make this rule less restrictive is probably not worth the effort.  Ticket [6284df89debdfa61db8073e062908af0c9b6118e]
            if ((parse.Ctx.Flags & Context.FLAG.ForeignKeys) != 0 && dest.FKeys != null) return false;
#endif
            if ((parse.Ctx.Flags & Context.FLAG.CountRows) != 0) return false;  // xfer opt does not play well with PRAGMA count_changes

            // If we get this far, it means that the xfer optimization is at least a possibility, though it might only work if the destination
            // table (tab1) is initially empty.
#if TEST
            _xferopt_count++;
#endif
            int dbSrcId = Prepare.SchemaToIndex(parse.Ctx, src.Schema); // The database of pSrc
            Vdbe v = parse.GetVdbe(); // The VDBE we are building
            parse.CodeVerifySchema(dbSrcId);
            int srcId = parse.Tabs++; // Cursors from source and destination 
            int destId = parse.Tabs++; // Cursors from source and destination 
            int regAutoinc = AutoIncBegin(parse, dbDestId, dest); // Memory register used by AUTOINC
            OpenTable(parse, destId, dbDestId, dest, OP.OpenWrite);
            int addr1; // Loop addresses
            int emptyDestTest; // Address of test for empty pDest
            if ((dest.PKey < 0 && dest.Index != null) ||		        // (1)
                destHasUniqueIdx ||									// (2)
                (onError != OE.Abort && onError != OE.Rollback))	    // (3)
            {
                // In some circumstances, we are able to run the xfer optimization only if the destination table is initially empty.  This code makes
                // that determination.  Conditions under which the destination must be empty:
                //
                // (1) There is no INTEGER PRIMARY KEY but there are indices. (If the destination is not initially empty, the rowid fields
                //     of index entries might need to change.)
                //
                // (2) The destination has a unique index.  (The xfer optimization is unable to test uniqueness.)
                //
                // (3) onError is something other than OE_Abort and OE_Rollback.
                addr1 = v.AddOp2(OP.Rewind, destId, 0);
                emptyDestTest = v.AddOp2(OP.Goto, 0, 0);
                v.JumpHere(addr1);
            }
            else
                emptyDestTest = 0;
            OpenTable(parse, srcId, dbSrcId, src, OP.OpenRead);
            int emptySrcTest = v.AddOp2(OP.Rewind, srcId, 0); // Address of test for empty pSrc
            int regData = Expr.GetTempReg(parse); // Registers holding data and rowid
            int regRowid = Expr.GetTempReg(parse);// Registers holding data and rowid
            if (dest.PKey >= 0)
            {
                addr1 = v.AddOp2(OP.Rowid, srcId, regRowid);
                int addr2 = v.AddOp3(OP.NotExists, destId, 0, regRowid); // Loop addresses
                sqlite3HaltConstraint(parse, onError, "PRIMARY KEY must be unique", Vdbe.P4T.STATIC);
                v.JumpHere(addr2);
                AutoIncStep(parse, regAutoinc, regRowid);
            }
            else if (dest.Index == null)
                addr1 = v.AddOp2(OP.NewRowid, destId, regRowid);
            else
            {
                addr1 = v.AddOp2(OP.Rowid, srcId, regRowid);
                Debug.Assert((dest.TabFlags & TF.Autoincrement) == 0);
            }
            v.AddOp2(OP.RowData, srcId, regData);
            v.AddOp3(OP.Insert, destId, regData, regRowid);
            v.ChangeP5(Vdbe.OPFLAG.NCHANGE | Vdbe.OPFLAG.LASTROWID | Vdbe.OPFLAG.APPEND);
            v.ChangeP4(-1, dest.Name, 0);
            v.AddOp2(OP.Next, srcId, addr1);
            KeyInfo key; // Key information for an index
            for (destIdx = dest.Index; destIdx != null; destIdx = destIdx.Next)
            {
                for (srcIdx = src.Index; srcIdx != null; srcIdx = srcIdx.Next)
                    if (XferCompatibleIndex(destIdx, srcIdx))
                        break;
                Debug.Assert(srcIdx != null);
                v.AddOp2(OP.Close, srcId, 0);
                v.AddOp2(OP.Close, destId, 0);
                key = parse.IndexKeyinfo(srcIdx);
                v.AddOp4(OP.OpenRead, srcId, srcIdx.Id, dbSrcId, key, Vdbe.P4T.KEYINFO_HANDOFF);
                v.Comment("%s", srcIdx.Name);
                key = parse.IndexKeyinfo(destIdx);
                v.AddOp4(OP.OpenWrite, destId, destIdx.Id, dbDestId, key, Vdbe.P4T.KEYINFO_HANDOFF);
                v.Comment("%s", destIdx.Name);
                addr1 = v.AddOp2(OP.Rewind, srcId, 0);
                v.AddOp2(OP.RowKey, srcId, regData);
                v.AddOp3(OP.IdxInsert, destId, regData, 1);
                v.AddOp2(OP.Next, srcId, addr1 + 1);
                v.JumpHere(addr1);
            }
            v.JumpHere(emptySrcTest);
            Expr.ReleaseTempReg(parse, regRowid);
            Expr.ReleaseTempReg(parse, regData);
            v.AddOp2(OP.Close, srcId, 0);
            v.AddOp2(OP.Close, destId, 0);
            if (emptyDestTest != 0)
            {
                v.AddOp2(OP.Halt, RC.OK, 0);
                v.JumpHere(emptyDestTest);
                v.AddOp2(OP.Close, destId, 0);
                return false;
            }
            return true;
        }

#endif
    }
}

