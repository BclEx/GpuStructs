using System;
using System.Diagnostics;
namespace Core.Command
{
    public partial class Update
    {
        public static void sqlite3ColumnDefault(Vdbe v, Table table, int i, int regId)
        {
            Debug.Assert(table != null);
            if (table.Select == null)
            {
                TEXTENCODE encode = Context.CTXENCODE(v.Ctx);
                Column col = table.Cols[i];
                v.VdbeComment("%s.%s", table.Name, col.Name);
                Debug.Assert(i < table.Cols.length);
                Mem value = new Mem();
                sqlite3ValueFromExpr(v.Ctx, col.Dflt, encode, col.Affinity, ref value);
                if (value != null)
                    v.ChangeP4(-1, value, Vdbe.P4T.MEM);
#if !OMIT_FLOATING_POINT
                if (regId >= 0 && table.Cols[i].Affinity == AFF.REAL)
                    v.AddOp1(OP.RealAffinity, regId);
#endif
            }
        }

        public static void sqlite3Update(Parse parse, SrcList tabList, ExprList changes, Expr where_, OE onError)
        {
            int i, j;                   // Loop counters
            Table pTab;                 // The table to be updated
            int addr = 0;               // VDBE instruction address of the start of the loop
            WhereInfo pWInfo;           // Information about the WHERE clause
            Vdbe v;                     // The virtual database engine
            Index pIdx;                 // For looping over indices
            int nIdx;                   // Number of indices that need updating
            int iCur;                   // VDBE Cursor number of pTab
            Context ctx;                // The database structure
            int[] aRegIdx = null;       // One register assigned to each index to be updated
            int[] aXRef = null;         // aXRef[i] is the index in pChanges.a[] of the
                                        // an expression for the i-th column of the table. aXRef[i]==-1 if the i-th column is not changed.
            bool chngRowid;             // True if the record number is being changed
            Expr pRowidExpr = null;     // Expression defining the new record number
            bool openAll = false;       // True if all indices need to be opened
            AuthContext sContext;       // The authorization context
            NameContext sNC;            // The name-context to resolve expressions in
            int iDb;                    // Database containing the table being updated
            bool okOnePass;             // True for one-pass algorithm without the FIFO
            bool hasFK;                 // True if foreign key processing is required

#if !SQLITE_OMIT_TRIGGER
            bool isView;            // True when updating a view (INSTEAD OF trigger)
            Trigger pTrigger;      // List of triggers on pTab, if required
            int tmask = 0;         // Mask of TRIGGER_BEFORE|TRIGGER_AFTER
#endif
            int newmask;           // Mask of NEW.* columns accessed by BEFORE triggers

            /* Register Allocations */
            int regRowCount = 0;         // A count of rows changed
            int regOldRowid;             // The old rowid
            int regNewRowid;             // The new rowid
            int regNew;
            int regOld = 0;
            int regRowSet = 0;           // Rowset of rows to be updated

            sContext = new AuthContext(); //memset( &sContext, 0, sizeof( sContext ) );
            ctx = parse.Ctx;
            if (parse.Errs != 0 || ctx.MallocFailed) goto update_cleanup;
            Debug.Assert(tabList.Srcs == 1);

            // Locate the table which we want to update.
            pTab = sqlite3SrcListLookup(parse, tabList);
            if (pTab == null)
                goto update_cleanup;
            iDb = sqlite3SchemaToIndex(parse.db, pTab.pSchema);

            // Figure out if we have any triggers and if the table being updated is a view.
#if !OMIT_TRIGGER
            pTrigger = sqlite3TriggersExist(parse, pTab, TK_UPDATE, changes, out tmask);
            isView = pTab.Select != null;
            Debug.Assert(pTrigger != null || tmask == 0);
#else
      const Trigger pTrigger = null;//# define pTrigger 0
      const int tmask = 0;          //# define tmask 0
#endif
#if SQLITE_OMIT_TRIGGER || SQLITE_OMIT_VIEW
//    # undef isView
      const bool isView = false;    //# define isView 0
#endif

            if (sqlite3ViewGetColumnNames(parse, pTab) != 0)
            {
                goto update_cleanup;
            }
            if (sqlite3IsReadOnly(parse, pTab, tmask))
            {
                goto update_cleanup;
            }
            aXRef = new int[pTab.nCol];// sqlite3DbMallocRaw(db, sizeof(int) * pTab.nCol);
            //if ( aXRef == null ) goto update_cleanup;
            for (i = 0; i < pTab.nCol; i++)
                aXRef[i] = -1;

            /* Allocate a cursors for the main database table and for all indices.
            ** The index cursors might not be used, but if they are used they
            ** need to occur right after the database cursor.  So go ahead and
            ** allocate enough space, just in case.
            */
            tabList.a[0].iCursor = iCur = parse.nTab++;
            for (pIdx = pTab.pIndex; pIdx != null; pIdx = pIdx.pNext)
            {
                parse.nTab++;
            }

            /* Initialize the name-context */
            sNC = new NameContext();// memset(&sNC, 0, sNC).Length;
            sNC.pParse = parse;
            sNC.pSrcList = tabList;

            /* Resolve the column names in all the expressions of the
            ** of the UPDATE statement.  Also find the column index
            ** for each column to be updated in the pChanges array.  For each
            ** column to be updated, make sure we have authorization to change
            ** that column.
            */
            chngRowid = false;
            for (i = 0; i < changes.nExpr; i++)
            {
                if (sqlite3ResolveExprNames(sNC, ref changes.a[i].pExpr) != 0)
                {
                    goto update_cleanup;
                }
                for (j = 0; j < pTab.nCol; j++)
                {
                    if (pTab.aCol[j].zName.Equals(changes.a[i].zName, StringComparison.OrdinalIgnoreCase))
                    {
                        if (j == pTab.iPKey)
                        {
                            chngRowid = true;
                            pRowidExpr = changes.a[i].pExpr;
                        }
                        aXRef[j] = i;
                        break;
                    }
                }
                if (j >= pTab.nCol)
                {
                    if (sqlite3IsRowid(changes.a[i].zName))
                    {
                        chngRowid = true;
                        pRowidExpr = changes.a[i].pExpr;
                    }
                    else
                    {
                        sqlite3ErrorMsg(parse, "no such column: %s", changes.a[i].zName);
                        parse.checkSchema = 1;
                        goto update_cleanup;
                    }
                }
#if !SQLITE_OMIT_AUTHORIZATION
                {
                    int rc;
                    rc = sqlite3AuthCheck(parse, SQLITE_UPDATE, pTab.zName,
                    pTab.aCol[j].zName, ctx.aDb[iDb].zName);
                    if (rc == SQLITE_DENY)
                    {
                        goto update_cleanup;
                    }
                    else if (rc == SQLITE_IGNORE)
                    {
                        aXRef[j] = -1;
                    }
                }
#endif
            }

            hasFK = sqlite3FkRequired(parse, pTab, aXRef, chngRowid ? 1 : 0) != 0;

            /* Allocate memory for the array aRegIdx[].  There is one entry in the
            ** array for each index associated with table being updated.  Fill in
            ** the value with a register number for indices that are to be used
            ** and with zero for unused indices.
            */
            for (nIdx = 0, pIdx = pTab.pIndex; pIdx != null; pIdx = pIdx.pNext, nIdx++)
            {
            }
            if (nIdx > 0)
            {
                aRegIdx = new int[nIdx]; // sqlite3DbMallocRaw(db, Index*.Length * nIdx);
                if (aRegIdx == null)
                    goto update_cleanup;
            }
            for (j = 0, pIdx = pTab.pIndex; pIdx != null; pIdx = pIdx.pNext, j++)
            {
                int reg;
                if (hasFK || chngRowid)
                {
                    reg = ++parse.nMem;
                }
                else
                {
                    reg = 0;
                    for (i = 0; i < pIdx.nColumn; i++)
                    {
                        if (aXRef[pIdx.aiColumn[i]] >= 0)
                        {
                            reg = ++parse.nMem;
                            break;
                        }
                    }
                }
                aRegIdx[j] = reg;
            }

            /* Begin generating code. */
            v = sqlite3GetVdbe(parse);
            if (v == null)
                goto update_cleanup;
            if (parse.nested == 0)
                sqlite3VdbeCountChanges(v);
            sqlite3BeginWriteOperation(parse, 1, iDb);

#if !SQLITE_OMIT_VIRTUALTABLE
            /* Virtual tables must be handled separately */
            if (IsVirtual(pTab))
            {
                updateVirtualTable(parse, tabList, pTab, changes, pRowidExpr, aXRef,
                                   where_, onError);
                where_ = null;
                tabList = null;
                goto update_cleanup;
            }
#endif

            /* Allocate required registers. */
            regOldRowid = regNewRowid = ++parse.nMem;
            if (pTrigger != null || hasFK)
            {
                regOld = parse.nMem + 1;
                parse.nMem += pTab.nCol;
            }
            if (chngRowid || pTrigger != null || hasFK)
            {
                regNewRowid = ++parse.nMem;
            }
            regNew = parse.nMem + 1;
            parse.nMem += pTab.nCol;

            /* Start the view context. */
            if (isView)
            {
                sqlite3AuthContextPush(parse, sContext, pTab.zName);
            }

            /* If we are trying to update a view, realize that view into
            ** a ephemeral table.
            */
#if !(SQLITE_OMIT_VIEW) && !(SQLITE_OMIT_TRIGGER)
            if (isView)
            {
                sqlite3MaterializeView(parse, pTab, where_, iCur);
            }
#endif

            /* Resolve the column names in all the expressions in the
** WHERE clause.
*/
            if (sqlite3ResolveExprNames(sNC, ref where_) != 0)
            {
                goto update_cleanup;
            }

            /* Begin the database scan
            */
            sqlite3VdbeAddOp2(v, OP_Null, 0, regOldRowid);
            ExprList NullOrderby = null;
            pWInfo = sqlite3WhereBegin(parse, tabList, where_, ref NullOrderby, WHERE_ONEPASS_DESIRED);
            if (pWInfo == null)
                goto update_cleanup;
            okOnePass = pWInfo.okOnePass != 0;

            /* Remember the rowid of every item to be updated.
            */
            sqlite3VdbeAddOp2(v, OP_Rowid, iCur, regOldRowid);
            if (!okOnePass)
            {
                regRowSet = ++parse.nMem;
                sqlite3VdbeAddOp2(v, OP_RowSetAdd, regRowSet, regOldRowid);
            }

            /* End the database scan loop.
            */
            sqlite3WhereEnd(pWInfo);

            /* Initialize the count of updated rows
            */
            if ((ctx.flags & SQLITE_CountRows) != 0 && null == parse.pTriggerTab)
            {
                regRowCount = ++parse.nMem;
                sqlite3VdbeAddOp2(v, OP_Integer, 0, regRowCount);
            }

            if (!isView)
            {
                /*
                ** Open every index that needs updating.  Note that if any
                ** index could potentially invoke a REPLACE conflict resolution
                ** action, then we need to open all indices because we might need
                ** to be deleting some records.
                */
                if (!okOnePass)
                    sqlite3OpenTable(parse, iCur, iDb, pTab, OP_OpenWrite);
                if (onError == OE_Replace)
                {
                    openAll = true;
                }
                else
                {
                    openAll = false;
                    for (pIdx = pTab.pIndex; pIdx != null; pIdx = pIdx.pNext)
                    {
                        if (pIdx.onError == OE_Replace)
                        {
                            openAll = true;
                            break;
                        }
                    }
                }
                for (i = 0, pIdx = pTab.pIndex; pIdx != null; pIdx = pIdx.pNext, i++)
                {
                    if (openAll || aRegIdx[i] > 0)
                    {
                        KeyInfo pKey = sqlite3IndexKeyinfo(parse, pIdx);
                        sqlite3VdbeAddOp4(v, OP_OpenWrite, iCur + i + 1, pIdx.tnum, iDb,
                        pKey, P4_KEYINFO_HANDOFF);
                        Debug.Assert(parse.nTab > iCur + i + 1);
                    }
                }
            }

            /* Top of the update loop */
            if (okOnePass)
            {
                int a1 = sqlite3VdbeAddOp1(v, OP_NotNull, regOldRowid);
                addr = sqlite3VdbeAddOp0(v, OP_Goto);
                sqlite3VdbeJumpHere(v, a1);
            }
            else
            {
                addr = sqlite3VdbeAddOp3(v, OP_RowSetRead, regRowSet, 0, regOldRowid);
            }

            /* Make cursor iCur point to the record that is being updated. If
            ** this record does not exist for some reason (deleted by a trigger,
            ** for example, then jump to the next iteration of the RowSet loop.  */
            sqlite3VdbeAddOp3(v, OP_NotExists, iCur, addr, regOldRowid);

            /* If the record number will change, set register regNewRowid to
            ** contain the new value. If the record number is not being modified,
            ** then regNewRowid is the same register as regOldRowid, which is
            ** already populated.  */
            Debug.Assert(chngRowid || pTrigger != null || hasFK || regOldRowid == regNewRowid);
            if (chngRowid)
            {
                sqlite3ExprCode(parse, pRowidExpr, regNewRowid);
                sqlite3VdbeAddOp1(v, OP_MustBeInt, regNewRowid);
            }

            /* If there are triggers on this table, populate an array of registers 
            ** with the required old.* column data.  */
            if (hasFK || pTrigger != null)
            {
                u32 oldmask = (hasFK ? sqlite3FkOldmask(parse, pTab) : 0);
                oldmask |= sqlite3TriggerColmask(parse,
                    pTrigger, changes, 0, TRIGGER_BEFORE | TRIGGER_AFTER, pTab, onError
                );
                for (i = 0; i < pTab.nCol; i++)
                {
                    if (aXRef[i] < 0 || oldmask == 0xffffffff || (i < 32 && 0 != (oldmask & (1 << i))))
                    {
                        sqlite3ExprCodeGetColumnOfTable(v, pTab, iCur, i, regOld + i);
                    }
                    else
                    {
                        sqlite3VdbeAddOp2(v, OP_Null, 0, regOld + i);
                    }
                }
                if (chngRowid == false)
                {
                    sqlite3VdbeAddOp2(v, OP_Copy, regOldRowid, regNewRowid);
                }
            }

            /* Populate the array of registers beginning at regNew with the new
            ** row data. This array is used to check constaints, create the new
            ** table and index records, and as the values for any new.* references
            ** made by triggers.
            **
            ** If there are one or more BEFORE triggers, then do not populate the
            ** registers associated with columns that are (a) not modified by
            ** this UPDATE statement and (b) not accessed by new.* references. The
            ** values for registers not modified by the UPDATE must be reloaded from 
            ** the database after the BEFORE triggers are fired anyway (as the trigger 
            ** may have modified them). So not loading those that are not going to
            ** be used eliminates some redundant opcodes.
            */
            newmask = (int)sqlite3TriggerColmask(
                parse, pTrigger, changes, 1, TRIGGER_BEFORE, pTab, onError
            );
            for (i = 0; i < pTab.nCol; i++)
            {
                if (i == pTab.iPKey)
                {
                    sqlite3VdbeAddOp2(v, OP_Null, 0, regNew + i);
                }
                else
                {
                    j = aXRef[i];
                    if (j >= 0)
                    {
                        sqlite3ExprCode(parse, changes.a[j].pExpr, regNew + i);
                    }
                    else if (0 == (tmask & TRIGGER_BEFORE) || i > 31 || (newmask & (1 << i)) != 0)
                    {
                        /* This branch loads the value of a column that will not be changed 
                        ** into a register. This is done if there are no BEFORE triggers, or
                        ** if there are one or more BEFORE triggers that use this value via
                        ** a new.* reference in a trigger program.
                        */
                        testcase(i == 31);
                        testcase(i == 32);
                        sqlite3VdbeAddOp3(v, OP_Column, iCur, i, regNew + i);
                        sqlite3ColumnDefault(v, pTab, i, regNew + i);
                    }
                }
            }

            /* Fire any BEFORE UPDATE triggers. This happens before constraints are
            ** verified. One could argue that this is wrong.
            */
            if ((tmask & TRIGGER_BEFORE) != 0)
            {
                sqlite3VdbeAddOp2(v, OP_Affinity, regNew, pTab.nCol);
                sqlite3TableAffinityStr(v, pTab);
                sqlite3CodeRowTrigger(parse, pTrigger, TK_UPDATE, changes,
                    TRIGGER_BEFORE, pTab, regOldRowid, onError, addr);

                /* The row-trigger may have deleted the row being updated. In this
                ** case, jump to the next row. No updates or AFTER triggers are 
                ** required. This behaviour - what happens when the row being updated
                ** is deleted or renamed by a BEFORE trigger - is left undefined in the
                ** documentation.
                */
                sqlite3VdbeAddOp3(v, OP_NotExists, iCur, addr, regOldRowid);

                /* If it did not delete it, the row-trigger may still have modified 
                ** some of the columns of the row being updated. Load the values for 
                ** all columns not modified by the update statement into their 
                ** registers in case this has happened.
                */
                for (i = 0; i < pTab.nCol; i++)
                {
                    if (aXRef[i] < 0 && i != pTab.iPKey)
                    {
                        sqlite3VdbeAddOp3(v, OP_Column, iCur, i, regNew + i);
                        sqlite3ColumnDefault(v, pTab, i, regNew + i);
                    }
                }
            }

            if (!isView)
            {
                int j1;                       /* Address of jump instruction */

                /* Do constraint checks. */
                int iDummy;
                sqlite3GenerateConstraintChecks(parse, pTab, iCur, regNewRowid,
                      aRegIdx, (chngRowid ? regOldRowid : 0), true, onError, addr, out iDummy);

                /* Do FK constraint checks. */
                if (hasFK)
                {
                    sqlite3FkCheck(parse, pTab, regOldRowid, 0);
                }

                /* Delete the index entries associated with the current record.  */
                j1 = sqlite3VdbeAddOp3(v, OP_NotExists, iCur, 0, regOldRowid);
                sqlite3GenerateRowIndexDelete(parse, pTab, iCur, aRegIdx);

                /* If changing the record number, delete the old record.  */
                if (hasFK || chngRowid)
                {
                    sqlite3VdbeAddOp2(v, OP_Delete, iCur, 0);
                }
                sqlite3VdbeJumpHere(v, j1);

                if (hasFK)
                {
                    sqlite3FkCheck(parse, pTab, 0, regNewRowid);
                }

                /* Insert the new index entries and the new record. */
                sqlite3CompleteInsertion(parse, pTab, iCur, regNewRowid, aRegIdx, true, false, false);

                /* Do any ON CASCADE, SET NULL or SET DEFAULT operations required to
                ** handle rows (possibly in other tables) that refer via a foreign key
                ** to the row just updated. */
                if (hasFK)
                {
                    sqlite3FkActions(parse, pTab, changes, regOldRowid);
                }
            }

            /* Increment the row counter 
            */
            if ((ctx.flags & SQLITE_CountRows) != 0 && null == parse.pTriggerTab)
            {
                sqlite3VdbeAddOp2(v, OP_AddImm, regRowCount, 1);
            }

            sqlite3CodeRowTrigger(parse, pTrigger, TK_UPDATE, changes,
                TRIGGER_AFTER, pTab, regOldRowid, onError, addr);

            /* Repeat the above with the next record to be updated, until
            ** all record selected by the WHERE clause have been updated.
            */
            sqlite3VdbeAddOp2(v, OP_Goto, 0, addr);
            sqlite3VdbeJumpHere(v, addr);

            /* Close all tables */
            for (i = 0, pIdx = pTab.pIndex; pIdx != null; pIdx = pIdx.pNext, i++)
            {
                if (openAll || aRegIdx[i] > 0)
                {
                    sqlite3VdbeAddOp2(v, OP_Close, iCur + i + 1, 0);
                }
            }
            sqlite3VdbeAddOp2(v, OP_Close, iCur, 0);

            /* Update the sqlite_sequence table by storing the content of the
            ** maximum rowid counter values recorded while inserting into
            ** autoincrement tables.
            */
            if (parse.nested == 0 && parse.pTriggerTab == null)
            {
                sqlite3AutoincrementEnd(parse);
            }

            /*
            ** Return the number of rows that were changed. If this routine is 
            ** generating code because of a call to sqlite3NestedParse(), do not
            ** invoke the callback function.
            */
            if ((ctx.flags & SQLITE_CountRows) != 0 && null == parse.pTriggerTab && 0 == parse.nested)
            {
                sqlite3VdbeAddOp2(v, OP_ResultRow, regRowCount, 1);
                sqlite3VdbeSetNumCols(v, 1);
                sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "rows updated", SQLITE_STATIC);
            }

        update_cleanup:
#if !SQLITE_OMIT_AUTHORIZATION
            sqlite3AuthContextPop(sContext);
#endif
            sqlite3DbFree(ctx, ref aRegIdx);
            sqlite3DbFree(ctx, ref aXRef);
            sqlite3SrcListDelete(ctx, ref tabList);
            sqlite3ExprListDelete(ctx, ref changes);
            sqlite3ExprDelete(ctx, ref where_);
            return;
        }
        /* Make sure "isView" and other macros defined above are undefined. Otherwise
        ** thely may interfere with compilation of other functions in this file
        ** (or in another file, if this file becomes part of the amalgamation).  */
        //#if isView
        // #undef isView
        //#endif
        //#if pTrigger
        // #undef pTrigger
        //#endif

#if !SQLITE_OMIT_VIRTUALTABLE
        /*
** Generate code for an UPDATE of a virtual table.
**
** The strategy is that we create an ephemerial table that contains
** for each row to be changed:
**
**   (A)  The original rowid of that row.
**   (B)  The revised rowid for the row. (note1)
**   (C)  The content of every column in the row.
**
** Then we loop over this ephemeral table and for each row in
** the ephermeral table call VUpdate.
**
** When finished, drop the ephemeral table.
**
** (note1) Actually, if we know in advance that (A) is always the same
** as (B) we only store (A), then duplicate (A) when pulling
** it out of the ephemeral table before calling VUpdate.
*/
        static void updateVirtualTable(
        Parse pParse,       /* The parsing context */
        SrcList pSrc,       /* The virtual table to be modified */
        Table pTab,         /* The virtual table */
        ExprList pChanges,  /* The columns to change in the UPDATE statement */
        Expr pRowid,        /* Expression used to recompute the rowid */
        int[] aXRef,        /* Mapping from columns of pTab to entries in pChanges */
        Expr pWhere,        /* WHERE clause of the UPDATE statement */
        int onError         /* ON CONFLICT strategy */
        )
        {
            Vdbe v = pParse.pVdbe;  /* Virtual machine under construction */
            ExprList pEList = null;     /* The result set of the SELECT statement */
            Select pSelect = null;      /* The SELECT statement */
            Expr pExpr;              /* Temporary expression */
            int ephemTab;             /* Table holding the result of the SELECT */
            int i;                    /* Loop counter */
            int addr;                 /* Address of top of loop */
            int iReg;                 /* First register in set passed to OP_VUpdate */
            sqlite3 db = pParse.db; /* Database connection */
            VTable pVTab = sqlite3GetVTable(db, pTab);
            SelectDest dest = new SelectDest();

            /* Construct the SELECT statement that will find the new values for
            ** all updated rows.
            */
            pEList = sqlite3ExprListAppend(pParse, 0, sqlite3Expr(db, TK_ID, "_rowid_"));
            if (pRowid != null)
            {
                pEList = sqlite3ExprListAppend(pParse, pEList,
                sqlite3ExprDup(db, pRowid, 0));
            }
            Debug.Assert(pTab.iPKey < 0);
            for (i = 0; i < pTab.nCol; i++)
            {
                if (aXRef[i] >= 0)
                {
                    pExpr = sqlite3ExprDup(db, pChanges.a[aXRef[i]].pExpr, 0);
                }
                else
                {
                    pExpr = sqlite3Expr(db, TK_ID, pTab.aCol[i].zName);
                }
                pEList = sqlite3ExprListAppend(pParse, pEList, pExpr);
            }
            pSelect = sqlite3SelectNew(pParse, pEList, pSrc, pWhere, null, null, null, 0, null, null);

            /* Create the ephemeral table into which the update results will
            ** be stored.
            */
            Debug.Assert(v != null);
            ephemTab = pParse.nTab++;
            sqlite3VdbeAddOp2(v, OP_OpenEphemeral, ephemTab, pTab.nCol + 1 + ((pRowid != null) ? 1 : 0));
            sqlite3VdbeChangeP5(v, BTREE_UNORDERED);

            /* fill the ephemeral table
            */
            sqlite3SelectDestInit(dest, SRT_Table, ephemTab);
            sqlite3Select(pParse, pSelect, ref dest);

            /* Generate code to scan the ephemeral table and call VUpdate. */
            iReg = ++pParse.nMem;
            pParse.nMem += pTab.nCol + 1;
            addr = sqlite3VdbeAddOp2(v, OP_Rewind, ephemTab, 0);
            sqlite3VdbeAddOp3(v, OP_Column, ephemTab, 0, iReg);
            sqlite3VdbeAddOp3(v, OP_Column, ephemTab, (pRowid != null ? 1 : 0), iReg + 1);
            for (i = 0; i < pTab.nCol; i++)
            {
                sqlite3VdbeAddOp3(v, OP_Column, ephemTab, i + 1 + ((pRowid != null) ? 1 : 0), iReg + 2 + i);
            }
            sqlite3VtabMakeWritable(pParse, pTab);
            sqlite3VdbeAddOp4(v, OP_VUpdate, 0, pTab.nCol + 2, iReg, pVTab, P4_VTAB);
            sqlite3VdbeChangeP5(v, (byte)(onError == OE_Default ? OE_Abort : onError));
            sqlite3MayAbort(pParse);
            sqlite3VdbeAddOp2(v, OP_Next, ephemTab, addr + 1);
            sqlite3VdbeJumpHere(v, addr);
            sqlite3VdbeAddOp2(v, OP_Close, ephemTab, 0);

            /* Cleanup */
            sqlite3SelectDelete(db, ref pSelect);
        }
#endif // * SQLITE_OMIT_VIRTUALTABLE */
    }
}
