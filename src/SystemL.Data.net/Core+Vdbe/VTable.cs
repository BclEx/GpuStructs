#region OMIT_VIRTUALTABLE
#if !OMIT_VIRTUALTABLE
using System;
using System.Diagnostics;
using System.Text;

namespace Core
{
    public class VTableContext
    {
        public Table pTab;
        public VTable pVTable;
    }

    public partial class VTable
    {
        public static RC CreateModule(Context ctx, string name, ITableModule module, object aux, Action<object> destroy)
        {
            int rc, nName;
            Module pMod;

            sqlite3_mutex_enter(ctx.mutex);
            nName = sqlite3Strlen30(name);
            pMod = new Module();//  (Module)sqlite3DbMallocRaw( db, sizeof( Module ) + nName + 1 );
            if (pMod != null)
            {
                Module pDel;
                string zCopy;// = (char )(&pMod[1]);
                zCopy = name;//memcpy(zCopy, zName, nName+1);
                pMod.zName = zCopy;
                pMod.pModule = module;
                pMod.pAux = aux;
                pMod.xDestroy = destroy;
                pDel = (Module)sqlite3HashInsert(ref ctx.aModule, zCopy, nName, pMod);
                if (pDel != null && pDel.xDestroy != null)
                {
                    sqlite3ResetInternalSchema(ctx, -1);
                    pDel.xDestroy(ref pDel.pAux);
                }
                sqlite3DbFree(ctx, ref pDel);
                //if( pDel==pMod ){
                //  db.mallocFailed = 1;
                //}
            }
            else if (destroy != null)
            {
                destroy(ref aux);
            }
            rc = sqlite3ApiExit(ctx, SQLITE_OK);
            sqlite3_mutex_leave(ctx.mutex);
            return rc;
        }

        public void Lock()
        {
            Refs++;
        }

        static VTable sqlite3GetVTable(Context ctx, Table table)
        {
            Debug.Assert(IsVirtual(table));
            VTable vtable;
            for (vtable = table.VTables; vtable != null && vtable.Ctx != ctx; vtable = vtable.Next) ;
            return vtable;
        }

        public void Unlock()
        {
            Context ctx = Ctx;
            Debug.Assert(ctx != null);
            Debug.Assert(Refs > 0);
            Debug.Assert(ctx.Magic == MAGIC.OPEN || ctx.Magic == MAGIC.ZOMBIE);
            Refs--;
            if (Refs == 0)
            {
                if (IVTable)
                    IVTable.Module.Disconnect(ref IVTable);
                C._tagfree(ctx, ref this);
            }
        }

        static VTable VTableDisconnectAll(Context ctx, Table table)
        {
            // Assert that the mutex (if any) associated with the BtShared database that contains table p is held by the caller. See header comments 
            // above function sqlite3VtabUnlockList() for an explanation of why this makes it safe to access the sqlite3.pDisconnect list of any
            // database connection that may have an entry in the p->pVTable list.
            Debug.Assert(ctx == null || Btree.SchemaMutexHeld(ctx, 0, table.Schema));
            VTable r = null;
            VTable vtable = table.VTable;
            table.VTable = null;
            while (vtable != null)
            {
                VTable next = vtable.Next;
                Context ctx2 = vtable.Ctx;
                Debug.Assert(ctx2 != null);
                if (ctx2 == ctx)
                {
                    r = vtable;
                    table.VTable = r;
                    r.Next = null;
                }
                else
                {
                    vtable.Next = ctx2.Disconnect;
                    ctx2.Disconnect = vtable;
                }
                vtable = next;
            }
            Debug.Assert(ctx == null || r != null);
            return r;
        }

        public static void Disconnect(Context ctx, Table table)
        {
            Debug.Assert(IsVirtual(table));
            Debug.Aassert(Btree.HoldsAllMutexes(ctx));
            Debug.Aassert(MutexEx.Held(ctx.Mutex));
            for (VTable pvtable = table.VTables; pvtable; pvtable = pvtable.Next)
                if (pvtable.Ctx == ctx)
                {
                    VTable vtable = pvtable;
                    vtable = vtable.Next;
                    vtable.Unlock();
                    break;
                }
        }


        public static void UnlockList(Context ctx)
        {
            Debug.Assert(Btree.HoldsAllMutexes(ctx));
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            VTable vtable = ctx.Disconnect;
            ctx.Disconnect = null;
            if (vtable != null)
            {
                Vdbe.ExpirePreparedStatements(ctx);
                do
                {
                    VTable next = vtable.Next;
                    vtable.Unlock();
                    vtable = next;
                } while (vtable != null);
            }
        }

        public static void Clear(Context ctx, Table table)
        {
            if (ctx == null || ctx.BytesFreed == 0)
                VtableDisconnectAll(null, table);
            if (table.ModuleArgs != null)
            {
                for (int i = 0; i < table.ModuleArgs.Length; i++)
                    C._tagfree(ctx, ref table.ModuleArgs[i]);
                C._tagfree(ctx, ref table.ModuleArgs);
            }
        }

        static void AddModuleArgument(Context ctx, Table table, string arg)
        {
            int i = table.ModuleArgs.length++;
            //int nBytes = sizeof(char )*(1+pTable.nModuleArg);
            //string[] azModuleArg;
            //sqlite3DbRealloc( db, pTable.azModuleArg, nBytes );
            if (table.azModuleArg == null || table.azModuleArg.Length < table.nModuleArg)
                Array.Resize(ref table.azModuleArg, 3 + table.nModuleArg);
            //if ( azModuleArg == null )
            //{
            //  int j;
            //  for ( j = 0; j < i; j++ )
            //  {
            //    sqlite3DbFree( db, ref pTable.azModuleArg[j] );
            //  }
            //  sqlite3DbFree( db, ref zArg );
            //  sqlite3DbFree( db, ref pTable.azModuleArg );
            //  pTable.nModuleArg = 0;
            //}
            //else
            {
                table.azModuleArg[i] = arg;
                //pTable.azModuleArg[i + 1] = null;
                //azModuleArg[i+1] = 0;
            }
            //pTable.azModuleArg = azModuleArg;
        }

        static void sqlite3VtabBeginParse(
          Parse pParse,        /* Parsing context */
          Token pName1,        /* Name of new table, or database name */
          Token pName2,        /* Name of new table or NULL */
          Token pModuleName    /* Name of the module for the virtual table */
        )
        {
            int iDb;              /* The database the table is being created in */
            Table pTable;        /* The new virtual table */
            sqlite3 db;          /* Database connection */

            sqlite3StartTable(pParse, pName1, pName2, 0, 0, 1, 0);
            pTable = pParse.pNewTable;
            if (pTable == null)
                return;
            Debug.Assert(null == pTable.pIndex);

            db = pParse.db;
            iDb = sqlite3SchemaToIndex(db, pTable.pSchema);
            Debug.Assert(iDb >= 0);

            pTable.tabFlags |= TF_Virtual;
            pTable.nModuleArg = 0;
            addModuleArgument(db, pTable, sqlite3NameFromToken(db, pModuleName));
            addModuleArgument(db, pTable, db.aDb[iDb].zName);//sqlite3DbStrDup( db, db.aDb[iDb].zName ) );
            addModuleArgument(db, pTable, pTable.zName);//sqlite3DbStrDup( db, pTable.zName ) );
            pParse.sNameToken.n = pParse.sNameToken.z.Length;//      (int)[pModuleName.n] - pName1.z );

#if !OMIT_AUTHORIZATION
            /* Creating a virtual table invokes the authorization callback twice.
  ** The first invocation, to obtain permission to INSERT a row into the
  ** sqlite_master table, has already been made by sqlite3StartTable().
  ** The second call, to obtain permission to create the table, is made now.
  */
            if (pTable->azModuleArg)
            {
                sqlite3AuthCheck(pParse, SQLITE_CREATE_VTABLE, pTable->zName,
                        pTable->azModuleArg[0], pParse->db->aDb[iDb].zName);
            }
#endif
        }

        static void addArgumentToVtab(Parse pParse)
        {
            if (pParse.sArg.z != null && ALWAYS(pParse.pNewTable))
            {
                string z = pParse.sArg.z.Substring(0, pParse.sArg.n);
                int n = pParse.sArg.n;
                sqlite3 db = pParse.db;
                addModuleArgument(db, pParse.pNewTable, z);///sqlite3DbStrNDup( db, z, n ) );
            }
        }

        static void sqlite3VtabFinishParse(Parse pParse, Token pEnd)
        {
            Table pTab = pParse.pNewTable;  /* The table being constructed */
            sqlite3 db = pParse.db;         /* The database connection */

            if (pTab == null)
                return;
            addArgumentToVtab(pParse);
            pParse.sArg.z = "";
            if (pTab.nModuleArg < 1)
                return;

            /* If the CREATE VIRTUAL TABLE statement is being entered for the
            ** first time (in other words if the virtual table is actually being
            ** created now instead of just being read out of sqlite_master) then
            ** do additional initialization work and store the statement text
            ** in the sqlite_master table.
            */
            if (0 == db.init.busy)
            {
                string zStmt;
                string zWhere;
                int iDb;
                Vdbe v;

                /* Compute the complete text of the CREATE VIRTUAL TABLE statement */
                if (pEnd != null)
                {
                    pParse.sNameToken.n = pParse.sNameToken.z.Length;//(int)( pEnd.z - pParse.sNameToken.z ) + pEnd.n;
                }
                zStmt = sqlite3MPrintf(db, "CREATE VIRTUAL TABLE %T", pParse.sNameToken.z.Substring(0, pParse.sNameToken.n));

                /* A slot for the record has already been allocated in the 
                ** SQLITE_MASTER table.  We just need to update that slot with all
                ** the information we've collected.  
                **
                ** The VM register number pParse.regRowid holds the rowid of an
                ** entry in the sqlite_master table tht was created for this vtab
                ** by sqlite3StartTable().
                */
                iDb = sqlite3SchemaToIndex(db, pTab.pSchema);
                sqlite3NestedParse(pParse,
                  "UPDATE %Q.%s " +
                     "SET type='table', name=%Q, tbl_name=%Q, rootpage=0, sql=%Q " +
                   "WHERE rowid=#%d",
                  db.aDb[iDb].zName, SCHEMA_TABLE(iDb),
                  pTab.zName,
                  pTab.zName,
                  zStmt,
                  pParse.regRowid
                );
                sqlite3DbFree(db, ref zStmt);
                v = sqlite3GetVdbe(pParse);
                sqlite3ChangeCookie(pParse, iDb);

                sqlite3VdbeAddOp2(v, OP_Expire, 0, 0);
                zWhere = sqlite3MPrintf(db, "name='%q' AND type='table'", pTab.zName);
                sqlite3VdbeAddParseSchemaOp(v, iDb, zWhere);
                sqlite3VdbeAddOp4(v, OP_VCreate, iDb, 0, 0,
                                     pTab.zName, sqlite3Strlen30(pTab.zName) + 1);
            }

            /* If we are rereading the sqlite_master table create the in-memory
            ** record of the table. The xConnect() method is not called until
            ** the first time the virtual table is used in an SQL statement. This
            ** allows a schema that contains virtual tables to be loaded before
            ** the required virtual table implementations are registered.  */
            else
            {
                Table pOld;
                Schema pSchema = pTab.pSchema;
                string zName = pTab.zName;
                int nName = sqlite3Strlen30(zName);
                Debug.Assert(sqlite3SchemaMutexHeld(db, 0, pSchema));
                pOld = sqlite3HashInsert(ref pSchema.tblHash, zName, nName, pTab);
                if (pOld != null)
                {
                    //db.mallocFailed = 1;
                    Debug.Assert(pTab == pOld);  /* Malloc must have failed inside HashInsert() */
                    return;
                }
                pParse.pNewTable = null;
            }
        }

        static void sqlite3VtabArgInit(Parse pParse)
        {
            addArgumentToVtab(pParse);
            pParse.sArg.z = null;
            pParse.sArg.n = 0;
        }

        static void sqlite3VtabArgExtend(Parse pParse, Token p)
        {
            Token pArg = pParse.sArg;
            if (pArg.z == null)
            {
                pArg.z = p.z;
                pArg.n = p.n;
            }
            else
            {
                //Debug.Assert( pArg.z< p.z );
                pArg.n += p.n + 1;//(int)( p.z[p.n] - pArg.z );
            }
        }

        static int vtabCallConstructor(sqlite3 db, Table pTab, Module pMod, smdxCreateConnect xConstruct, ref string pzErr)
        {
            VtabCtx sCtx = new VtabCtx();
            VTable pVTable;
            int rc;
            string[] azArg = pTab.azModuleArg;
            int nArg = pTab.nModuleArg;
            string zErr = null;
            string zModuleName = sqlite3MPrintf(db, "%s", pTab.zName);

            //if ( String.IsNullOrEmpty( zModuleName ) )
            //{
            //  return SQLITE_NOMEM;
            //}

            pVTable = new VTable();//sqlite3DbMallocZero( db, sizeof( VTable ) );
            //if ( null == pVTable )
            //{
            //  sqlite3DbFree( db, ref zModuleName );
            //  return SQLITE_NOMEM;
            //}
            pVTable.db = db;
            pVTable.pMod = pMod;

            /* Invoke the virtual table constructor */
            //assert( &db->pVtabCtx );
            Debug.Assert(xConstruct != null);
            sCtx.pTab = pTab;
            sCtx.pVTable = pVTable;
            db.pVtabCtx = sCtx;
            rc = xConstruct(db, pMod.pAux, nArg, azArg, out pVTable.pVtab, out zErr);
            db.pVtabCtx = null;
            //if ( rc == SQLITE_NOMEM )
            //  db.mallocFailed = 1;

            if (SQLITE_OK != rc)
            {
                if (zErr == "")
                {
                    pzErr = sqlite3MPrintf(db, "vtable constructor failed: %s", zModuleName);
                }
                else
                {
                    pzErr = sqlite3MPrintf(db, "%s", zErr);
                    zErr = null;//sqlite3_free( zErr );
                }
                sqlite3DbFree(db, ref pVTable);
            }
            else if (ALWAYS(pVTable.pVtab))
            {
                /* Justification of ALWAYS():  A correct vtab constructor must allocate
                ** the sqlite3_vtab object if successful.  */
                pVTable.pVtab.pModule = pMod.pModule;
                pVTable.nRef = 1;
                if (sCtx.pTab != null)
                {
                    string zFormat = "vtable constructor did not declare schema: %s";
                    pzErr = sqlite3MPrintf(db, zFormat, pTab.zName);
                    sqlite3VtabUnlock(pVTable);
                    rc = SQLITE_ERROR;
                }
                else
                {
                    int iCol;
                    /* If everything went according to plan, link the new VTable structure
                    ** into the linked list headed by pTab->pVTable. Then loop through the 
                    ** columns of the table to see if any of them contain the token "hidden".
                    ** If so, set the Column.isHidden flag and remove the token from
                    ** the type string.  */
                    pVTable.pNext = pTab.pVTable;
                    pTab.pVTable = pVTable;

                    for (iCol = 0; iCol < pTab.nCol; iCol++)
                    {
                        if (String.IsNullOrEmpty(pTab.aCol[iCol].zType))
                            continue;
                        StringBuilder zType = new StringBuilder(pTab.aCol[iCol].zType);
                        int nType;
                        int i = 0;
                        //if ( zType )
                        //  continue;
                        nType = sqlite3Strlen30(zType);
                        if (sqlite3StrNICmp("hidden", 0, zType.ToString(), 6) != 0 || (zType.Length > 6 && zType[6] != ' '))
                        {
                            for (i = 0; i < nType; i++)
                            {
                                if ((0 == sqlite3StrNICmp(" hidden", zType.ToString().Substring(i), 7))
                                 && (i + 7 == zType.Length || (zType[i + 7] == '\0' || zType[i + 7] == ' '))
                                )
                                {
                                    i++;
                                    break;
                                }
                            }
                        }
                        if (i < nType)
                        {
                            int j;
                            int nDel = 6 + (zType.Length > i + 6 ? 1 : 0);
                            for (j = i; (j + nDel) < nType; j++)
                            {
                                zType[j] = zType[j + nDel];
                            }
                            if (zType[i] == '\0' && i > 0)
                            {
                                Debug.Assert(zType[i - 1] == ' ');
                                zType.Length = i;//[i - 1] = '\0';
                            }
                            pTab.aCol[iCol].isHidden = 1;
                            pTab.aCol[iCol].zType = zType.ToString().Substring(0, j);
                        }
                    }
                }
            }

            sqlite3DbFree(db, ref zModuleName);
            return rc;
        }

        static int sqlite3VtabCallConnect(Parse pParse, Table pTab)
        {
            sqlite3 db = pParse.db;
            string zMod;
            Module pMod;
            int rc;

            Debug.Assert(pTab != null);
            if ((pTab.tabFlags & TF_Virtual) == 0 || sqlite3GetVTable(db, pTab) != null)
            {
                return SQLITE_OK;
            }

            /* Locate the required virtual table module */
            zMod = pTab.azModuleArg[0];
            pMod = (Module)sqlite3HashFind(db.aModule, zMod, sqlite3Strlen30(zMod), (Module)null);

            if (null == pMod)
            {
                string zModule = pTab.azModuleArg[0];
                sqlite3ErrorMsg(pParse, "no such module: %s", zModule);
                rc = SQLITE_ERROR;
            }
            else
            {
                string zErr = null;
                rc = vtabCallConstructor(db, pTab, pMod, pMod.pModule.xConnect, ref zErr);
                if (rc != SQLITE_OK)
                {
                    sqlite3ErrorMsg(pParse, "%s", zErr);
                }
                zErr = null;//sqlite3DbFree( db, zErr );
            }

            return rc;
        }

        static int growVTrans(sqlite3 db)
        {
            const int ARRAY_INCR = 5;

            /* Grow the sqlite3.aVTrans array if required */
            if ((db.nVTrans % ARRAY_INCR) == 0)
            {
                //VTable** aVTrans;
                //int nBytes = sizeof( sqlite3_vtab* ) * ( db.nVTrans + ARRAY_INCR );
                //aVTrans = sqlite3DbRealloc( db, (void)db.aVTrans, nBytes );
                //if ( !aVTrans )
                //{
                //  return SQLITE_NOMEM;
                //}
                //memset( &aVTrans[db.nVTrans], 0, sizeof( sqlite3_vtab* ) * ARRAY_INCR );
                Array.Resize(ref db.aVTrans, db.nVTrans + ARRAY_INCR);
            }

            return SQLITE_OK;
        }

        static void addToVTrans(sqlite3 db, VTable pVTab)
        {
            /* Add pVtab to the end of sqlite3.aVTrans */
            db.aVTrans[db.nVTrans++] = pVTab;
            sqlite3VtabLock(pVTab);
        }

        static int sqlite3VtabCallCreate(sqlite3 db, int iDb, string zTab, ref string pzErr)
        {
            int rc = SQLITE_OK;
            Table pTab;
            Module pMod;
            string zMod;

            pTab = sqlite3FindTable(db, zTab, db.aDb[iDb].zName);
            Debug.Assert(pTab != null && (pTab.tabFlags & TF_Virtual) != 0 && null == pTab.pVTable);

            /* Locate the required virtual table module */
            zMod = pTab.azModuleArg[0];
            pMod = (Module)sqlite3HashFind(db.aModule, zMod, sqlite3Strlen30(zMod), (Module)null);

            /* If the module has been registered and includes a Create method, 
            ** invoke it now. If the module has not been registered, return an 
            ** error. Otherwise, do nothing.
            */
            if (null == pMod)
            {
                pzErr = sqlite3MPrintf(db, "no such module: %s", zMod);
                rc = SQLITE_ERROR;
            }
            else
            {
                rc = vtabCallConstructor(db, pTab, pMod, pMod.pModule.xCreate, ref pzErr);
            }

            /* Justification of ALWAYS():  The xConstructor method is required to
            ** create a valid sqlite3_vtab if it returns SQLITE_OK. */
            if (rc == SQLITE_OK && ALWAYS(sqlite3GetVTable(db, pTab)))
            {
                rc = growVTrans(db);
                if (rc == SQLITE_OK)
                {
                    addToVTrans(db, sqlite3GetVTable(db, pTab));
                }
            }

            return rc;
        }

        static int sqlite3_declare_vtab(sqlite3 db, string zCreateTable)
        {
            Parse pParse;

            int rc = SQLITE_OK;
            Table pTab;
            string zErr = "";

            sqlite3_mutex_enter(db.mutex);
            if (null == db.pVtabCtx || null == (pTab = db.pVtabCtx.pTab))
            {
                sqlite3Error(db, SQLITE_MISUSE, 0);
                sqlite3_mutex_leave(db.mutex);
                return SQLITE_MISUSE_BKPT();
            }
            Debug.Assert((pTab.tabFlags & TF_Virtual) != 0);

            pParse = new Parse();//sqlite3StackAllocZero(db, sizeof(*pParse));
            //if ( pParse == null )
            //{
            //  rc = SQLITE_NOMEM;
            //}
            //else
            {
                pParse.declareVtab = 1;
                pParse.db = db;
                pParse.nQueryLoop = 1;

                if (SQLITE_OK == sqlite3RunParser(pParse, zCreateTable, ref zErr)
                 && pParse.pNewTable != null
                    //&& !db.mallocFailed
                 && null == pParse.pNewTable.pSelect
                 && (pParse.pNewTable.tabFlags & TF_Virtual) == 0
                )
                {
                    if (null == pTab.aCol)
                    {
                        pTab.aCol = pParse.pNewTable.aCol;
                        pTab.nCol = pParse.pNewTable.nCol;
                        pParse.pNewTable.nCol = 0;
                        pParse.pNewTable.aCol = null;
                    }
                    db.pVtabCtx.pTab = null;
                }
                else
                {
                    sqlite3Error(db, SQLITE_ERROR, (zErr != null ? "%s" : null), zErr);
                    zErr = null;//sqlite3DbFree( db, zErr );
                    rc = SQLITE_ERROR;
                }
                pParse.declareVtab = 0;

                if (pParse.pVdbe != null)
                {
                    sqlite3VdbeFinalize(ref pParse.pVdbe);
                }
                sqlite3DeleteTable(db, ref pParse.pNewTable);
                //sqlite3StackFree( db, pParse );
            }

            Debug.Assert((rc & 0xff) == rc);
            rc = sqlite3ApiExit(db, rc);
            sqlite3_mutex_leave(db.mutex);
            return rc;
        }

        static int sqlite3VtabCallDestroy(sqlite3 db, int iDb, string zTab)
        {
            int rc = SQLITE_OK;
            Table pTab;

            pTab = sqlite3FindTable(db, zTab, db.aDb[iDb].zName);
            if (ALWAYS(pTab != null && pTab.pVTable != null))
            {
                VTable p = vtabDisconnectAll(db, pTab);

                Debug.Assert(rc == SQLITE_OK);
                object obj = p.pVtab;
                rc = p.pMod.pModule.xDestroy(ref obj);
                p.pVtab = null;

                /* Remove the sqlite3_vtab* from the aVTrans[] array, if applicable */
                if (rc == SQLITE_OK)
                {
                    Debug.Assert(pTab.pVTable == p && p.pNext == null);
                    p.pVtab = null;
                    pTab.pVTable = null;
                    sqlite3VtabUnlock(p);
                }
            }

            return rc;
        }

        static void callFinaliser(sqlite3 db, int offset)
        {
            int i;
            if (db.aVTrans != null)
            {
                for (i = 0; i < db.nVTrans; i++)
                {
                    VTable pVTab = db.aVTrans[i];
                    sqlite3_vtab p = pVTab.pVtab;
                    if (p != null)
                    {
                        //int (*x)(sqlite3_vtab );
                        //x = *(int (*)(sqlite3_vtab ))((char )p.pModule + offset);
                        //if( x ) x(p);
                        if (offset == 0)
                        {
                            if (p.pModule.xCommit != null)
                                p.pModule.xCommit(p);
                        }
                        else
                        {
                            if (p.pModule.xRollback != null)
                                p.pModule.xRollback(p);
                        }
                    }
                    pVTab.iSavepoint = 0;
                    sqlite3VtabUnlock(pVTab);
                }
                sqlite3DbFree(db, ref db.aVTrans);
                db.nVTrans = 0;
                db.aVTrans = null;
            }
        }

        static int sqlite3VtabSync(sqlite3 db, ref string pzErrmsg)
        {
            int i;
            int rc = SQLITE_OK;
            VTable[] aVTrans = db.aVTrans;

            db.aVTrans = null;
            for (i = 0; rc == SQLITE_OK && i < db.nVTrans; i++)
            {
                smdxFunction x;//int (*x)(sqlite3_vtab );
                sqlite3_vtab pVtab = aVTrans[i].pVtab;
                if (pVtab != null && (x = pVtab.pModule.xSync) != null)
                {
                    rc = x(pVtab);
                    //sqlite3DbFree(db, ref pzErrmsg);
                    pzErrmsg = pVtab.zErrMsg;// sqlite3DbStrDup( db, pVtab.zErrMsg );
                    pVtab.zErrMsg = null;//sqlite3_free( ref pVtab.zErrMsg );
                }
            }
            db.aVTrans = aVTrans;
            return rc;
        }

        static int sqlite3VtabRollback(sqlite3 db)
        {
            callFinaliser(db, 1);//offsetof( sqlite3_module, xRollback ) );
            return SQLITE_OK;
        }

        static int sqlite3VtabCommit(sqlite3 db)
        {
            callFinaliser(db, 0);//offsetof( sqlite3_module, xCommit ) );
            return SQLITE_OK;
        }

        static int sqlite3VtabBegin(sqlite3 db, VTable pVTab)
        {
            int rc = SQLITE_OK;
            sqlite3_module pModule;

            /* Special case: If db.aVTrans is NULL and db.nVTrans is greater
            ** than zero, then this function is being called from within a
            ** virtual module xSync() callback. It is illegal to write to 
            ** virtual module tables in this case, so return SQLITE_LOCKED.
            */
            if (sqlite3VtabInSync(db))
            {
                return SQLITE_LOCKED;
            }
            if (null == pVTab)
            {
                return SQLITE_OK;
            }
            pModule = pVTab.pVtab.pModule;

            if (pModule.xBegin != null)
            {
                int i;

                /* If pVtab is already in the aVTrans array, return early */
                for (i = 0; i < db.nVTrans; i++)
                {
                    if (db.aVTrans[i] == pVTab)
                    {
                        return SQLITE_OK;
                    }
                }

                /* Invoke the xBegin method. If successful, add the vtab to the 
                ** sqlite3.aVTrans[] array. */
                rc = growVTrans(db);
                if (rc == SQLITE_OK)
                {
                    rc = pModule.xBegin(pVTab.pVtab);
                    if (rc == SQLITE_OK)
                    {
                        addToVTrans(db, pVTab);
                    }
                }
            }
            return rc;
        }

        static int sqlite3VtabSavepoint(sqlite3 db, int op, int iSavepoint)
        {
            int rc = SQLITE_OK;

            Debug.Assert(op == SAVEPOINT_RELEASE || op == SAVEPOINT_ROLLBACK || op == SAVEPOINT_BEGIN);
            Debug.Assert(iSavepoint >= 0);
            if (db.aVTrans != null)
            {
                int i;
                for (i = 0; rc == SQLITE_OK && i < db.nVTrans; i++)
                {
                    VTable pVTab = db.aVTrans[i];
                    sqlite3_module pMod = pVTab.pMod.pModule;
                    if (pMod.iVersion >= 2)
                    {
                        smdxFunctionArg xMethod = null; //int (*xMethod)(sqlite3_vtab *, int);
                        switch (op)
                        {
                            case SAVEPOINT_BEGIN:
                                xMethod = pMod.xSavepoint;
                                pVTab.iSavepoint = iSavepoint + 1;
                                break;
                            case SAVEPOINT_ROLLBACK:
                                xMethod = pMod.xRollbackTo;
                                break;
                            default:
                                xMethod = pMod.xRelease;
                                break;
                        }
                        if (xMethod != null && pVTab.iSavepoint > iSavepoint)
                        {
                            rc = xMethod(db.aVTrans[i].pVtab, iSavepoint);
                        }
                    }
                }
            }
            return rc;
        }

        static FuncDef sqlite3VtabOverloadFunction(sqlite3 db, FuncDef pDef, int nArg, Expr pExpr)
        {
            Table pTab;
            sqlite3_vtab pVtab;
            sqlite3_module pMod;
            dxFunc xFunc = null;//void (*xFunc)(sqlite3_context*,int,sqlite3_value*) = 0;
            object pArg = null;
            FuncDef pNew;
            int rc = 0;
            string zLowerName;
            string z;

            /* Check to see the left operand is a column in a virtual table */
            if (NEVER(pExpr == null))
                return pDef;
            if (pExpr.op != TK_COLUMN)
                return pDef;
            pTab = pExpr.pTab;
            if (NEVER(pTab == null))
                return pDef;
            if ((pTab.tabFlags & TF_Virtual) == 0)
                return pDef;
            pVtab = sqlite3GetVTable(db, pTab).pVtab;
            Debug.Assert(pVtab != null);
            Debug.Assert(pVtab.pModule != null);
            pMod = (sqlite3_module)pVtab.pModule;
            if (pMod.xFindFunction == null)
                return pDef;

            /* Call the xFindFunction method on the virtual table implementation
            ** to see if the implementation wants to overload this function 
            */
            zLowerName = pDef.zName;//sqlite3DbStrDup(db, pDef.zName);
            if (zLowerName != null)
            {
                //for(z=(unsigned char)zLowerName; *z; z++){
                //  *z = sqlite3UpperToLower[*z];
                //}
                rc = pMod.xFindFunction(pVtab, nArg, zLowerName.ToLowerInvariant(), ref xFunc, ref pArg);
                sqlite3DbFree(db, ref zLowerName);
            }
            if (rc == 0)
            {
                return pDef;
            }

            /* Create a new ephemeral function definition for the overloaded
            ** function */
            //sqlite3DbMallocZero(db, sizeof(*pNew)
            //      + sqlite3Strlen30(pDef.zName) + 1);
            //if ( pNew == null )
            //{
            //  return pDef;
            //}
            pNew = pDef.Copy();
            pNew.zName = pDef.zName;
            //pNew.zName = (char )&pNew[1];
            //memcpy(pNew.zName, pDef.zName, sqlite3Strlen30(pDef.zName)+1);
            pNew.xFunc = xFunc;
            pNew.pUserData = pArg;
            pNew.flags |= SQLITE_FUNC_EPHEM;
            return pNew;
        }

        static void sqlite3VtabMakeWritable(Parse pParse, Table pTab)
        {
            Parse pToplevel = sqlite3ParseToplevel(pParse);
            int i, n;
            //Table[] apVtabLock = null;

            Debug.Assert(IsVirtual(pTab));
            for (i = 0; i < pToplevel.nVtabLock; i++)
            {
                if (pTab == pToplevel.apVtabLock[i])
                    return;
            }
            n = pToplevel.apVtabLock == null ? 1 : pToplevel.apVtabLock.Length + 1;//(pToplevel.nVtabLock+1)*sizeof(pToplevel.apVtabLock[0]);
            //sqlite3_realloc( pToplevel.apVtabLock, n );
            //if ( apVtabLock != null )
            {
                Array.Resize(ref pToplevel.apVtabLock, n);// pToplevel.apVtabLock= apVtabLock;
                pToplevel.apVtabLock[pToplevel.nVtabLock++] = pTab;
            }
            //else
            //{
            //  pToplevel.db.mallocFailed = 1;
            //}
        }

        static int[] aMap = new int[] { SQLITE_ROLLBACK, SQLITE_ABORT, SQLITE_FAIL, SQLITE_IGNORE, SQLITE_REPLACE };
        static int sqlite3_vtab_on_conflict(sqlite3 db)
        {
            Debug.Assert(OE_Rollback == 1 && OE_Abort == 2 && OE_Fail == 3);
            Debug.Assert(OE_Ignore == 4 && OE_Replace == 5);
            Debug.Assert(db.vtabOnConflict >= 1 && db.vtabOnConflict <= 5);
            return (int)aMap[db.vtabOnConflict - 1];
        }

        static int sqlite3_vtab_config(sqlite3 db, int op, params object[] ap)
        { // TODO ...){
            //va_list ap;
            int rc = SQLITE_OK;

            sqlite3_mutex_enter(db.mutex);

            va_start(ap, "op");
            switch (op)
            {
                case SQLITE_VTAB_CONSTRAINT_SUPPORT:
                    {
                        VtabCtx p = db.pVtabCtx;
                        if (null == p)
                        {
                            rc = SQLITE_MISUSE_BKPT();
                        }
                        else
                        {
                            Debug.Assert(p.pTab == null || (p.pTab.tabFlags & TF_Virtual) != 0);
                            p.pVTable.bConstraint = (Byte)va_arg(ap, (Int32)0);
                        }
                        break;
                    }
                default:
                    rc = SQLITE_MISUSE_BKPT();
                    break;
            }
            va_end(ref ap);

            if (rc != SQLITE_OK) sqlite3Error(db, rc, 0);
            sqlite3_mutex_leave(db.mutex);
            return rc;
        }
    }
}
#endif
#endregion

