#region OMIT_ALTERTABLE
#if !OMIT_ALTERTABLE
using System;
using System.Diagnostics;
using System.Text;

namespace Core.Command
{
    public partial class Alter
    {

        static void RenameTableFunc(FuncContext fctx, int notUsed, Mem[] argv)
        {
            Context ctx = sqlite3_context_db_handle(fctx);
            string sql = (Mem.Text(argv[0]);
            string tableName = Mem.Text(argv[1]);
            if (string.IsNullOrEmpty(sql))
                return;
            int length = 0;
            TK token = 0;
            Token tname = new Token();

            int z = 0, zLoc = 0;

            // The principle used to locate the table name in the CREATE TABLE statement is that the table name is the first non-space token that
            // is immediately followed by a TK_LP or TK_USING token.
            do
            {
                if (z == sql.Length)
                    return; // Ran out of input before finding an opening bracket. Return NULL.
                // Store the token that zCsr points to in tname.
                zLoc = z;
                tname.data = sql.Substring(z);
                tname.length = (uint)length;

                // Advance zCsr to the next token. Store that token type in 'token', and its length in 'len' (to be used next iteration of this loop).
                do
                {
                    z += length;
                    length = (z == sql.Length ? 1 : Parse.GetToken(sql, z, ref token));
                } while (token == TK.SPACE);
                Debug.Assert(length > 0);
            } while (token != TK.LP && token != TK.USING);

            string r = SysEx.Mprintf(ctx, "%.*s\"%w\"%s", zLoc, sql.Substring(0, zLoc), tableName, sql.Substring(zLoc + (int)tname.length));
            sqlite3_result_text(fctx, r, -1, E.DESTRUCTOR_DYNAMIC);
        }

#if !OMIT_FOREIGN_KEY
        static void RenameParentFunc(FuncContext fctx, int notUsed, Mem[] argv)
        {
            Context ctx = sqlite3_context_db_handle(fctx);
            string input = Mem.Text(argv[0]);
            string oldName = Mem.Text(argv[1]);
            string newName = Mem.Text(argv[2]);

            int zIdx;         // Pointer to token
            int zLeft = 0;    // Pointer to remainder of String

            TK token = 0;    // Type of token

            string output = string.Empty;
            int n; // Length of token z
            for (int z = 0; z < input.Length; z += n)
            {
                n = sqlite3GetToken(input, z, ref token);
                if (token == TK.REFERENCES)
                {
                    string parent;
                    do
                    {
                        z += n;
                        n = sqlite3GetToken(input, z, ref token);
                    } while (token == TK.SPACE);

                    parent = (z + n < input.Length ? input.Substring(z, n) : string.Empty);
                    if (string.IsNullOrEmpty(parent)) break;
                    Parse.Dequote(ref parent);
                    if (oldName.Equals(parent, StringComparison.OrdinalIgnoreCase))
                    {
                        string out_ = SysEx.Mprintf(ctx, "%s%.*s\"%w\"", output, z - zLeft, input.Substring(zLeft), newName);
                        SysEx.TagFree(ctx, ref output);
                        output = out_;
                        z += n;
                        zLeft = z;
                    }
                    SysEx.TagFree(ctx, ref parent);
                }
            }

            string r = SysEx.Mprintf(ctx, "%s%s", output, input.Substring(zLeft));
            sqlite3_result_text(fctx, r, -1, DESTRUCTOR.DYNAMIC);
            SysEx.TagFree(ctx, ref output);
        }
#endif

#if !OMIT_TRIGGER
        static void RenameTriggerFunc(FuncContext fctx, int notUsed, Mem[] argv)
        {
            Context ctx = sqlite3_context_db_handle(fctx);
            string sql = Mem.Text(argv[0]);
            string tableName = Mem.Text(argv[1]);

            int z = 0, zLoc = 0;
            int length = 1;
            TK token = 0;
            Token tname = new Token();
            int dist = 3;

            // The principle used to locate the table name in the CREATE TRIGGER statement is that the table name is the first token that is immediatedly
            // preceded by either TK_ON or TK_DOT and immediatedly followed by one of TK_WHEN, TK_BEGIN or TK_FOR.
            if (sql != null)
                return;
            do
            {
                if (z == sql.Length)
                    return; // Ran out of input before finding the table name. Return NULL.

                // Store the token that zCsr points to in tname.
                zLoc = z;
                tname.data = sql.Substring(z, length);
                tname.length = (uint)length;

                // Advance zCsr to the next token. Store that token type in 'token', and its length in 'len' (to be used next iteration of this loop).
                do
                {
                    z += length;
                    length = (z == sql.Length ? 1 : sqlite3GetToken(sql, z, ref token));
                } while (token == TK.SPACE);
                Debug.Assert(length > 0);

                // Variable 'dist' stores the number of tokens read since the most recent TK_DOT or TK_ON. This means that when a WHEN, FOR or BEGIN 
                // token is read and 'dist' equals 2, the condition stated above to be met.
                //
                // Note that ON cannot be a database, table or column name, so there is no need to worry about syntax like 
                // "CREATE TRIGGER ... ON ON.ON BEGIN ..." etc.

                dist++;
                if (token == TK.DOT || token == TK.ON)
                    dist = 0;
            } while (dist != 2 || (token != TK.WHEN && token != TK.FOR && token != TK.BEGIN));

            // Variable tname now contains the token that is the old table-name in the CREATE TRIGGER statement.
            string r = SysEx.Mprintf(ctx, "%.*s\"%w\"%s", zLoc, sql.Substring(0, zLoc), tableName, sql.Substring(zLoc + tname.length));
            sqlite3_result_text(fctx, r, -1, DESTRUCTOR.DYNAMIC);
        }
#endif

        static FuncDef[] _alterTableFuncs = new FuncDef[] {
            FUNCTION("sqlite_rename_table",   2, 0, 0, RenameTableFunc),
#if !OMIT_TRIGGER
FUNCTION("sqlite_rename_trigger", 2, 0, 0, RenameTriggerFunc),
#endif
#if !OMIT_FOREIGN_KEY
FUNCTION("sqlite_rename_parent",  3, 0, 0, RenameParentFunc),
#endif
  };
        public static void Functions()
        {
            FuncDefHash hash = Context.GlobalFunctions;
            FuncDef[] funcs = _alterTableFuncs;
            for (int i = 0; i < _alterTableFuncs.Length; i++)
                sqlite3FuncDefInsert(hash, funcs[i]);
        }

        static string WhereOrName(Context ctx, string where, string constant)
        {
            string newExpr;
            if (string.IsNullOrEmpty(where))
                newExpr = SysEx.Mprintf(ctx, "name=%Q", constant);
            else
            {
                newExpr = SysEx.Mprintf(ctx, "%s OR name=%Q", where, constant);
                SysEx.TagFree(ctx, ref where);
            }
            return newExpr;
        }

#if !OMIT_FOREIGN_KEY&& !OMIT_TRIGGER
        static string WhereForeignKeys(Parse parse, Table table)
        {
            string where = string.Empty;
            for (FKey p = sqlite3FkReferences(table); p != null; p = p.NextTo)
                where = WhereOrName(parse.Ctx, where, p.From.Name);
            return where;
        }
#endif

        static string WhereTempTriggers(Parse parse, Table table)
        {
            Context ctx = parse.Ctx;
            string where = string.Empty;
            Schema tempSchema = ctx.DBs[1].Schema; // Temp db schema
            // If the table is not located in the temp.db (in which case NULL is returned, loop through the tables list of triggers. For each trigger
            // that is not part of the temp.db schema, add a clause to the WHERE expression being built up in zWhere.
            if (table.Schema != tempSchema)
                for (Trigger trig = sqlite3TriggerList(parse, table); trig != null; trig = trig.Next)
                    if (trig.pSchema == tempSchema)
                        where = WhereOrName(ctx, where, trig.Name);
            if (!string.IsNullOrEmpty(where))
                where = SysEx.Mprintf(ctx, "type='trigger' AND (%s)", where);
            return where;
        }

        static void ReloadTableSchema(Parse parse, Table table, string name)
        {
            Context ctx = parse.Ctx;
            string where;
#if !SQLITE_OMIT_TRIGGER
            Trigger pTrig;
#endif

            Vdbe v = parse.V;
            if (SysEx.NEVER(v == null)) return;
            Debug.Assert(Btree.HoldsAllMutexes(ctxb));
            int db = sqlite3SchemaToIndex(ctx, table.Schema); // Index of database containing pTab
            Debug.Assert(db >= 0);

#if !SQLITE_OMIT_TRIGGER
            /* Drop any table triggers from the internal schema. */
            for (pTrig = sqlite3TriggerList(parse, table); pTrig != null; pTrig = pTrig.pNext)
            {
                int trigDb = sqlite3SchemaToIndex(parse.db, pTrig.pSchema);
                Debug.Assert(trigDb == db || trigDb == 1);
                v.AddOp4(OP.DropTrigger, trigDb, 0, 0, pTrig.zName, 0);
            }
#endif

            /* Drop the table and index from the internal schema. */
            sqlite3VdbeAddOp4(v, OP_DropTable, db, 0, 0, table.zName, 0);

            /* Reload the table, index and permanent trigger schemas. */
            where = sqlite3MPrintf(parse.db, "tbl_name=%Q", name);
            if (where == null)
                return;
            sqlite3VdbeAddParseSchemaOp(v, db, where);

#if !SQLITE_OMIT_TRIGGER
            /* Now, if the table is not stored in the temp database, reload any temp
** triggers. Don't use IN(...) in case SQLITE_OMIT_SUBQUERY is defined.
*/
            if ((where = whereTempTriggers(parse, table)) != "")
            {
                sqlite3VdbeAddParseSchemaOp(v, 1, where);
            }
#endif
        }

        /*
        ** Parameter zName is the name of a table that is about to be altered
        ** (either with ALTER TABLE ... RENAME TO or ALTER TABLE ... ADD COLUMN).
        ** If the table is a system table, this function leaves an error message
        ** in pParse->zErr (system tables may not be altered) and returns non-zero.
        **
        ** Or, if zName is not a system table, zero is returned.
        */
        static int isSystemTable(Parse pParse, string zName)
        {
            if (zName.StartsWith("sqlite_", System.StringComparison.OrdinalIgnoreCase))
            {
                sqlite3ErrorMsg(pParse, "table %s may not be altered", zName);
                return 1;
            }
            return 0;
        }

        /*
        ** Generate code to implement the "ALTER TABLE xxx RENAME TO yyy"
        ** command.
        */
        static void sqlite3AlterRenameTable(
        Parse pParse,             /* Parser context. */
        SrcList pSrc,             /* The table to rename. */
        Token pName               /* The new table name. */
        )
        {
            int iDb;                  /* Database that contains the table */
            string zDb;               /* Name of database iDb */
            Table pTab;               /* Table being renamed */
            string zName = null;      /* NULL-terminated version of pName */
            sqlite3 db = pParse.db;   /* Database connection */
            int nTabName;             /* Number of UTF-8 characters in zTabName */
            string zTabName;          /* Original name of the table */
            Vdbe v;
#if !SQLITE_OMIT_TRIGGER
            string zWhere = "";       /* Where clause to locate temp triggers */
#endif
            VTable pVTab = null;      /* Non-zero if this is a v-tab with an xRename() */
            int savedDbFlags;         /* Saved value of db->flags */

            savedDbFlags = db.flags;

            //if ( NEVER( db.mallocFailed != 0 ) ) goto exit_rename_table;
            Debug.Assert(pSrc.nSrc == 1);
            Debug.Assert(sqlite3BtreeHoldsAllMutexes(pParse.db));
            pTab = sqlite3LocateTable(pParse, 0, pSrc.a[0].zName, pSrc.a[0].zDatabase);
            if (pTab == null)
                goto exit_rename_table;
            iDb = sqlite3SchemaToIndex(pParse.db, pTab.pSchema);
            zDb = db.aDb[iDb].zName;
            db.flags |= SQLITE_PreferBuiltin;

            /* Get a NULL terminated version of the new table name. */
            zName = sqlite3NameFromToken(db, pName);
            if (zName == null)
                goto exit_rename_table;

            /* Check that a table or index named 'zName' does not already exist
            ** in database iDb. If so, this is an error.
            */
            if (sqlite3FindTable(db, zName, zDb) != null || sqlite3FindIndex(db, zName, zDb) != null)
            {
                sqlite3ErrorMsg(pParse,
                "there is already another table or index with this name: %s", zName);
                goto exit_rename_table;
            }

            /* Make sure it is not a system table being altered, or a reserved name
            ** that the table is being renamed to.
            */
            if (SQLITE_OK != isSystemTable(pParse, pTab.zName))
            {
                goto exit_rename_table;
            }
            if (SQLITE_OK != sqlite3CheckObjectName(pParse, zName))
            {
                goto exit_rename_table;
            }

#if !SQLITE_OMIT_VIEW
            if (pTab.pSelect != null)
            {
                sqlite3ErrorMsg(pParse, "view %s may not be altered", pTab.zName);
                goto exit_rename_table;
            }
#endif

#if !SQLITE_OMIT_AUTHORIZATION
            /* Invoke the authorization callback. */
            if (sqlite3AuthCheck(pParse, SQLITE_ALTER_TABLE, zDb, pTab.zName, 0))
            {
                goto exit_rename_table;
            }
#endif

            if (sqlite3ViewGetColumnNames(pParse, pTab) != 0)
            {
                goto exit_rename_table;
            }
#if !SQLITE_OMIT_VIRTUALTABLE
            if (IsVirtual(pTab))
            {
                pVTab = sqlite3GetVTable(db, pTab);
                if (pVTab.pVtab.pModule.xRename == null)
                {
                    pVTab = null;
                }
            }
#endif
            /* Begin a transaction and code the VerifyCookie for database iDb.
** Then modify the schema cookie (since the ALTER TABLE modifies the
** schema). Open a statement transaction if the table is a virtual
** table.
*/
            v = sqlite3GetVdbe(pParse);
            if (v == null)
            {
                goto exit_rename_table;
            }
            sqlite3BeginWriteOperation(pParse, pVTab != null ? 1 : 0, iDb);
            sqlite3ChangeCookie(pParse, iDb);

            /* If this is a virtual table, invoke the xRename() function if
            ** one is defined. The xRename() callback will modify the names
            ** of any resources used by the v-table implementation (including other
            ** SQLite tables) that are identified by the name of the virtual table.
            */
#if  !SQLITE_OMIT_VIRTUALTABLE
            if (pVTab != null)
            {
                int i = ++pParse.nMem;
                sqlite3VdbeAddOp4(v, OP_String8, 0, i, 0, zName, 0);
                sqlite3VdbeAddOp4(v, OP_VRename, i, 0, 0, pVTab, P4_VTAB);
                sqlite3MayAbort(pParse);
            }
#endif

            /* figure out how many UTF-8 characters are in zName */
            zTabName = pTab.zName;
            nTabName = sqlite3Utf8CharLen(zTabName, -1);

#if !(SQLITE_OMIT_FOREIGN_KEY) && !(SQLITE_OMIT_TRIGGER)
            if ((db.flags & SQLITE_ForeignKeys) != 0)
            {
                /* If foreign-key support is enabled, rewrite the CREATE TABLE 
                ** statements corresponding to all child tables of foreign key constraints
                ** for which the renamed table is the parent table.  */
                if ((zWhere = whereForeignKeys(pParse, pTab)) != null)
                {
                    sqlite3NestedParse(pParse,
                        "UPDATE \"%w\".%s SET " +
                            "sql = sqlite_rename_parent(sql, %Q, %Q) " +
                            "WHERE %s;", zDb, SCHEMA_TABLE(iDb), zTabName, zName, zWhere);
                    sqlite3DbFree(db, ref zWhere);
                }
            }
#endif

            /* Modify the sqlite_master table to use the new table name. */
            sqlite3NestedParse(pParse,
            "UPDATE %Q.%s SET " +
#if SQLITE_OMIT_TRIGGER
 "sql = sqlite_rename_table(sql, %Q), " +
#else
 "sql = CASE " +
            "WHEN type = 'trigger' THEN sqlite_rename_trigger(sql, %Q)" +
            "ELSE sqlite_rename_table(sql, %Q) END, " +
#endif
 "tbl_name = %Q, " +
            "name = CASE " +
            "WHEN type='table' THEN %Q " +
            "WHEN name LIKE 'sqlite_autoindex%%' AND type='index' THEN " +
            "'sqlite_autoindex_' || %Q || substr(name,%d+18) " +
            "ELSE name END " +
            "WHERE tbl_name=%Q AND " +
            "(type='table' OR type='index' OR type='trigger');",
            zDb, SCHEMA_TABLE(iDb), zName, zName, zName,
#if !SQLITE_OMIT_TRIGGER
 zName,
#endif
 zName, nTabName, zTabName
            );

#if !SQLITE_OMIT_AUTOINCREMENT
            /* If the sqlite_sequence table exists in this database, then update
** it with the new table name.
*/
            if (sqlite3FindTable(db, "sqlite_sequence", zDb) != null)
            {
                sqlite3NestedParse(pParse,
                "UPDATE \"%w\".sqlite_sequence set name = %Q WHERE name = %Q",
                zDb, zName, pTab.zName
                );
            }
#endif

#if !SQLITE_OMIT_TRIGGER
            /* If there are TEMP triggers on this table, modify the sqlite_temp_master
** table. Don't do this if the table being ALTERed is itself located in
** the temp database.
*/
            if ((zWhere = whereTempTriggers(pParse, pTab)) != "")
            {
                sqlite3NestedParse(pParse,
                "UPDATE sqlite_temp_master SET " +
                "sql = sqlite_rename_trigger(sql, %Q), " +
                "tbl_name = %Q " +
                "WHERE %s;", zName, zName, zWhere);
                sqlite3DbFree(db, ref zWhere);
            }
#endif

#if !(SQLITE_OMIT_FOREIGN_KEY) && !(SQLITE_OMIT_TRIGGER)
            if ((db.flags & SQLITE_ForeignKeys) != 0)
            {
                FKey p;
                for (p = sqlite3FkReferences(pTab); p != null; p = p.pNextTo)
                {
                    Table pFrom = p.pFrom;
                    if (pFrom != pTab)
                    {
                        reloadTableSchema(pParse, p.pFrom, pFrom.zName);
                    }
                }
            }
#endif

            /* Drop and reload the internal table schema. */
            reloadTableSchema(pParse, pTab, zName);

        exit_rename_table:
            sqlite3SrcListDelete(db, ref pSrc);
            sqlite3DbFree(db, ref zName);
            db.flags = savedDbFlags;
        }

        /*
        ** Generate code to make sure the file format number is at least minFormat.
        ** The generated code will increase the file format number if necessary.
        */
        static void sqlite3MinimumFileFormat(Parse pParse, int iDb, int minFormat)
        {
            Vdbe v;
            v = sqlite3GetVdbe(pParse);
            /* The VDBE should have been allocated before this routine is called.
            ** If that allocation failed, we would have quit before reaching this
            ** point */
            if (ALWAYS(v))
            {
                int r1 = sqlite3GetTempReg(pParse);
                int r2 = sqlite3GetTempReg(pParse);
                int j1;
                sqlite3VdbeAddOp3(v, OP_ReadCookie, iDb, r1, BTREE_FILE_FORMAT);
                sqlite3VdbeUsesBtree(v, iDb);
                sqlite3VdbeAddOp2(v, OP_Integer, minFormat, r2);
                j1 = sqlite3VdbeAddOp3(v, OP_Ge, r2, 0, r1);
                sqlite3VdbeAddOp3(v, OP_SetCookie, iDb, BTREE_FILE_FORMAT, r2);
                sqlite3VdbeJumpHere(v, j1);
                sqlite3ReleaseTempReg(pParse, r1);
                sqlite3ReleaseTempReg(pParse, r2);
            }
        }

        /*
        ** This function is called after an "ALTER TABLE ... ADD" statement
        ** has been parsed. Argument pColDef contains the text of the new
        ** column definition.
        **
        ** The Table structure pParse.pNewTable was extended to include
        ** the new column during parsing.
        */
        static void sqlite3AlterFinishAddColumn(Parse pParse, Token pColDef)
        {
            Table pNew;              /* Copy of pParse.pNewTable */
            Table pTab;              /* Table being altered */
            int iDb;                 /* Database number */
            string zDb;              /* Database name */
            string zTab;             /* Table name */
            string zCol;             /* Null-terminated column definition */
            Column pCol;             /* The new column */
            Expr pDflt;              /* Default value for the new column */
            sqlite3 db;              /* The database connection; */

            db = pParse.db;
            if (pParse.nErr != 0 /*|| db.mallocFailed != 0 */ )
                return;
            pNew = pParse.pNewTable;
            Debug.Assert(pNew != null);
            Debug.Assert(sqlite3BtreeHoldsAllMutexes(db));
            iDb = sqlite3SchemaToIndex(db, pNew.pSchema);
            zDb = db.aDb[iDb].zName;
            zTab = pNew.zName.Substring(16);// zTab = &pNew->zName[16]; /* Skip the "sqlite_altertab_" prefix on the name */
            pCol = pNew.aCol[pNew.nCol - 1];
            pDflt = pCol.pDflt;
            pTab = sqlite3FindTable(db, zTab, zDb);
            Debug.Assert(pTab != null);

#if !SQLITE_OMIT_AUTHORIZATION
            /* Invoke the authorization callback. */
            if (sqlite3AuthCheck(pParse, SQLITE_ALTER_TABLE, zDb, pTab.zName, 0))
            {
                return;
            }
#endif

            /* If the default value for the new column was specified with a
** literal NULL, then set pDflt to 0. This simplifies checking
** for an SQL NULL default below.
*/
            if (pDflt != null && pDflt.op == TK_NULL)
            {
                pDflt = null;
            }

            /* Check that the new column is not specified as PRIMARY KEY or UNIQUE.
            ** If there is a NOT NULL constraint, then the default value for the
            ** column must not be NULL.
            */
            if (pCol.isPrimKey != 0)
            {
                sqlite3ErrorMsg(pParse, "Cannot add a PRIMARY KEY column");
                return;
            }
            if (pNew.pIndex != null)
            {
                sqlite3ErrorMsg(pParse, "Cannot add a UNIQUE column");
                return;
            }
            if ((db.flags & SQLITE_ForeignKeys) != 0 && pNew.pFKey != null && pDflt != null)
            {
                sqlite3ErrorMsg(pParse,
                    "Cannot add a REFERENCES column with non-NULL default value");
                return;
            }
            if (pCol.notNull != 0 && pDflt == null)
            {
                sqlite3ErrorMsg(pParse,
                "Cannot add a NOT NULL column with default value NULL");
                return;
            }

            /* Ensure the default expression is something that sqlite3ValueFromExpr()
            ** can handle (i.e. not CURRENT_TIME etc.)
            */
            if (pDflt != null)
            {
                sqlite3_value pVal = null;
                if (sqlite3ValueFromExpr(db, pDflt, SQLITE_UTF8, SQLITE_AFF_NONE, ref pVal) != 0)
                {
                    //        db.mallocFailed = 1;
                    return;
                }
                if (pVal == null)
                {
                    sqlite3ErrorMsg(pParse, "Cannot add a column with non-constant default");
                    return;
                }
                sqlite3ValueFree(ref pVal);
            }

            /* Modify the CREATE TABLE statement. */
            zCol = pColDef.z.Substring(0, pColDef.n).Replace(";", " ").Trim();//sqlite3DbStrNDup(db, (char*)pColDef.z, pColDef.n);
            if (zCol != null)
            {
                //  char zEnd = zCol[pColDef.n-1];
                int savedDbFlags = db.flags;
                //      while( zEnd>zCol && (*zEnd==';' || sqlite3Isspace(*zEnd)) ){
                //    zEnd-- = '\0';
                //  }
                db.flags |= SQLITE_PreferBuiltin;
                sqlite3NestedParse(pParse,
                "UPDATE \"%w\".%s SET " +
                "sql = substr(sql,1,%d) || ', ' || %Q || substr(sql,%d) " +
                "WHERE type = 'table' AND name = %Q",
                zDb, SCHEMA_TABLE(iDb), pNew.addColOffset, zCol, pNew.addColOffset + 1,
                zTab
                );
                sqlite3DbFree(db, ref zCol);
                db.flags = savedDbFlags;
            }

            /* If the default value of the new column is NULL, then set the file
            ** format to 2. If the default value of the new column is not NULL,
            ** the file format becomes 3.
            */
            sqlite3MinimumFileFormat(pParse, iDb, pDflt != null ? 3 : 2);

            /* Reload the schema of the modified table. */
            reloadTableSchema(pParse, pTab, pTab.zName);
        }

        /*
        ** This function is called by the parser after the table-name in
        ** an "ALTER TABLE <table-name> ADD" statement is parsed. Argument
        ** pSrc is the full-name of the table being altered.
        **
        ** This routine makes a (partial) copy of the Table structure
        ** for the table being altered and sets Parse.pNewTable to point
        ** to it. Routines called by the parser as the column definition
        ** is parsed (i.e. sqlite3AddColumn()) add the new Column data to
        ** the copy. The copy of the Table structure is deleted by tokenize.c
        ** after parsing is finished.
        **
        ** Routine sqlite3AlterFinishAddColumn() will be called to complete
        ** coding the "ALTER TABLE ... ADD" statement.
        */
        static void sqlite3AlterBeginAddColumn(Parse pParse, SrcList pSrc)
        {
            Table pNew;
            Table pTab;
            Vdbe v;
            int iDb;
            int i;
            int nAlloc;
            sqlite3 db = pParse.db;

            /* Look up the table being altered. */
            Debug.Assert(pParse.pNewTable == null);
            Debug.Assert(sqlite3BtreeHoldsAllMutexes(db));
            //      if ( db.mallocFailed != 0 ) goto exit_begin_add_column;
            pTab = sqlite3LocateTable(pParse, 0, pSrc.a[0].zName, pSrc.a[0].zDatabase);
            if (pTab == null)
                goto exit_begin_add_column;

            if (IsVirtual(pTab))
            {
                sqlite3ErrorMsg(pParse, "virtual tables may not be altered");
                goto exit_begin_add_column;
            }

            /* Make sure this is not an attempt to ALTER a view. */
            if (pTab.pSelect != null)
            {
                sqlite3ErrorMsg(pParse, "Cannot add a column to a view");
                goto exit_begin_add_column;
            }
            if (SQLITE_OK != isSystemTable(pParse, pTab.zName))
            {
                goto exit_begin_add_column;
            }

            Debug.Assert(pTab.addColOffset > 0);
            iDb = sqlite3SchemaToIndex(db, pTab.pSchema);

            /* Put a copy of the Table struct in Parse.pNewTable for the
            ** sqlite3AddColumn() function and friends to modify.  But modify
            ** the name by adding an "sqlite_altertab_" prefix.  By adding this
            ** prefix, we insure that the name will not collide with an existing
            ** table because user table are not allowed to have the "sqlite_"
            ** prefix on their name.
            */
            pNew = new Table();// (Table*)sqlite3DbMallocZero( db, sizeof(Table))
            if (pNew == null)
                goto exit_begin_add_column;
            pParse.pNewTable = pNew;
            pNew.nRef = 1;
            pNew.nCol = pTab.nCol;
            Debug.Assert(pNew.nCol > 0);
            nAlloc = (((pNew.nCol - 1) / 8) * 8) + 8;
            Debug.Assert(nAlloc >= pNew.nCol && nAlloc % 8 == 0 && nAlloc - pNew.nCol < 8);
            pNew.aCol = new Column[nAlloc];// (Column*)sqlite3DbMallocZero( db, sizeof(Column) * nAlloc );
            pNew.zName = sqlite3MPrintf(db, "sqlite_altertab_%s", pTab.zName);
            if (pNew.aCol == null || pNew.zName == null)
            {
                //        db.mallocFailed = 1;
                goto exit_begin_add_column;
            }
            // memcpy( pNew.aCol, pTab.aCol, sizeof(Column) * pNew.nCol );
            for (i = 0; i < pNew.nCol; i++)
            {
                Column pCol = pTab.aCol[i].Copy();
                // sqlite3DbStrDup( db, pCol.zName );
                pCol.zColl = null;
                pCol.zType = null;
                pCol.pDflt = null;
                pCol.zDflt = null;
                pNew.aCol[i] = pCol;
            }
            pNew.pSchema = db.aDb[iDb].pSchema;
            pNew.addColOffset = pTab.addColOffset;
            pNew.nRef = 1;

            /* Begin a transaction and increment the schema cookie.  */
            sqlite3BeginWriteOperation(pParse, 0, iDb);
            v = sqlite3GetVdbe(pParse);
            if (v == null)
                goto exit_begin_add_column;
            sqlite3ChangeCookie(pParse, iDb);

        exit_begin_add_column:
            sqlite3SrcListDelete(db, ref pSrc);
            return;
        }

    }
}

#endif
#endregion