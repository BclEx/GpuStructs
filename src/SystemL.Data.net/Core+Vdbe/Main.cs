using System;
using System.Diagnostics;
using System.Text;
using Pid = System.UInt32;

namespace Core
{
    #region From: Util_c

    public partial class Parse
    {
        public void ErrorMsg(string fmt, params object[] args)
        {
            Context ctx = Ctx;
            string msg = C._vmtagprintf(ctx, fmt, args);
            if (ctx.SuppressErr != 0)
                C._tagfree(ctx, ref msg);
            else
            {
                Errs++;
                C._tagfree(ctx, ref ErrMsg);
                ErrMsg = msg;
                RC = RC.ERROR;
            }
        }
    }

    public partial class Main
    {
        public static RC ApiExit(Context ctx, RC rc)
        {
            // If the ctx handle is not NULL, then we must hold the connection handle mutex here. Otherwise the read (and possible write) of db->mallocFailed is unsafe, as is the call to sqlite3Error().
            Debug.Assert(ctx == null || MutexEx.Held(ctx.Mutex));
            if (ctx != null && (ctx.MallocFailed || rc == RC.IOERR_NOMEM))
            {
                Error(ctx, RC.NOMEM, null);
                ctx.MallocFailed = false;
                rc = RC.NOMEM;
            }
            return (RC)((int)rc & (ctx != null ? ctx.ErrMask : 0xff));
        }

        //public static void Error(Context ctx, RC rc, int noString) { Error(ctx, rc, (rc == 0 ? null : string.Empty)); }
        public static void Error(Context ctx, RC rc, string fmt, params object[] args)
        {
            if (ctx != null && (ctx.Err != null || (ctx.Err = Vdbe.ValueNew(ctx)) != null))
            {
                ctx.ErrCode = rc;
                if (fmt != null)
                {
                    string z = C._vmtagprintf(ctx, fmt, args);
                    Vdbe.ValueSetStr(ctx.Err, -1, z, TEXTENCODE.UTF8, C.DESTRUCTOR_DYNAMIC);
                }
                else
                    Vdbe.ValueSetStr(ctx.Err, 0, null, TEXTENCODE.UTF8, C.DESTRUCTOR_STATIC);
            }
        }

        static void LogBadConnection(string type)
        {
            SysEx.LOG(RC.MISUSE, "API call with %s database connection pointer", type);
        }

        public static bool SafetyCheckOk(Context ctx)
        {
            if (ctx == null)
            {
                LogBadConnection("NULL");
                return false;
            }
            MAGIC magic = ctx.Magic;
            if (magic != MAGIC.OPEN)
            {
                if (SafetyCheckSickOrOk(ctx))
                {
                    C.ASSERTCOVERAGE(SysEx._GlobalStatics.Log != null);
                    LogBadConnection("unopened");
                }
                return false;
            }
            return true;
        }

        public static bool SafetyCheckSickOrOk(Context ctx)
        {
            MAGIC magic = ctx.Magic;
            if (magic != MAGIC.SICK && magic != MAGIC.OPEN && magic != MAGIC.BUSY)
            {
                C.ASSERTCOVERAGE(SysEx._GlobalStatics.Log != null);
                LogBadConnection("invalid");
                return false;
            }
            return true;
        }
    }

    #endregion

    public partial class Main
    {
        #region Initialize/Shutdown/Config

        // If the following global variable points to a string which is the name of a directory, then that directory will be used to store temporary files.
        // See also the "PRAGMA temp_store_directory" SQL command.
        static string g_temp_directory = null;
        // If the following global variable points to a string which is the name of a directory, then that directory will be used to store
        // all database files specified with a relative pathname.
        // See also the "PRAGMA data_store_directory" SQL command.
        static string g_data_directory = null;

        public class GlobalStatics
        {
            public bool UseCis;						    // Use covering indices for full-scans
            //public sqlite3_pcache_methods pcache;     // Low-level page-cache interface
            public Btree.MemPage Page;                  // Page cache memory
            public int PageSize;                        // Size of each page in pPage[]
            public int Pages;                           // Number of pages in pPage[]
            public int MaxParserStack;                  // maximum depth of the parser stack
            // The above might be initialized to non-zero.  The following need to always initially be zero, however.
            public bool IsPCacheInit;                    // True after malloc is initialized

            public GlobalStatics(
                bool useCis,
                //sqlite3_pcache_methods pcache,
                Btree.MemPage page,
                int pageSize,
                int pages,
                int maxParserStack,
                bool isPCacheInit)
            {
                UseCis = useCis;
                //pcache = pcache;
                Page = page;
                PageSize = pageSize;
                Pages = pages;
                MaxParserStack = maxParserStack;
                IsPCacheInit = isPCacheInit;
            }
        }

        enum CONFIG
        {
            PAGECACHE = 7,				// void*, int sz, int N
            PCACHE = 14,					// no-op
            GETPCACHE = 15,				// no-op
            PCACHE2 = 18,				// sqlite3_pcache_methods2*
            GETPCACHE2 = 19,				// sqlite3_pcache_methods2*
            COVERING_INDEX_SCAN = 20,	// int
        }

        const bool ALLOW_COVERING_INDEX_SCAN = true;

        // The following singleton contains the global configuration for the SQLite library.
        public static readonly GlobalStatics _GlobalStatics = new GlobalStatics(
            ALLOW_COVERING_INDEX_SCAN,		// UseCis
            //{0,0,0,0,0,0,0,0,0,0,0,0,0},	// pcache2
            null,				            // Page
            0,								// PageSize
            0,								// Pages
            0,								// MaxParserStack
            // All the rest should always be initialized to zero
            false							// IsPCacheInit
        );
        public static FuncDefHash _GlobalFunctions;

        public static RC Initialize()
        {
            if (SysEx._GlobalStatics.IsInit) return RC.OK;

            MutexEx masterMutex;
            RC rc = SysEx.PreInitialize(out masterMutex);
            if (rc != RC.OK) return rc;

            //: FuncDefHash hash = _GlobalFunctions;
            _GlobalFunctions = new FuncDefHash(); //: _memset(hash, 0, sizeof(_GlobalFunctions));
            Command.Func.RegisterGlobalFunctions();
            if (!_GlobalStatics.IsPCacheInit)
                rc = PCache.Initialize();
            if (rc == RC.OK)
            {
                _GlobalStatics.IsPCacheInit = true;
                PCache.PageBufferSetup(_GlobalStatics.Page, _GlobalStatics.PageSize, _GlobalStatics.Pages);
                SysEx._GlobalStatics.IsInit = true;
            }
            SysEx._GlobalStatics.InProgress = false;

            // Do extra initialization steps requested by the EXTRA_INIT compile-time option.
#if EXTRA_INIT
            if (rc == RC.OK && SysEx._GlobalStatics.IsInit)
                rc = EXTRA_INIT(null);
#endif

            SysEx.PostInitialize(masterMutex);
            return rc;
        }

        public static RC Shutdown()
        {
            if (!SysEx._GlobalStatics.IsInit) return RC.OK;

            // Do extra shutdown steps requested by the EXTRA_SHUTDOWN compile-time option.
#if EXTRA_SHUTDOWN
		EXTRA_SHUTDOWN();
#endif

            if (_GlobalStatics.IsPCacheInit)
            {
                PCache.Shutdown();
                _GlobalStatics.IsPCacheInit = false;
            }
#if !OMIT_SHUTDOWN_DIRECTORIES
            // The heap subsystem has now been shutdown and these values are supposed to be NULL or point to memory that was obtained from sqlite3_malloc(),
            // which would rely on that heap subsystem; therefore, make sure these values cannot refer to heap memory that was just invalidated when the
            // heap subsystem was shutdown.  This is only done if the current call to this function resulted in the heap subsystem actually being shutdown.
            g_data_directory = null;
            g_temp_directory = null;
#endif

            SysEx.Shutdown();
            return RC.OK;
        }

        public static RC Main.Config(CONFIG op, params object[] args)
        {
            if ((int)op < (int)CONFIG.PAGECACHE) return SysEx.Config((SysEx.CONFIG)op, args);
            RC rc = RC.OK;
            switch (op)
            {
                case CONFIG.PAGECACHE:
                    { // Designate a buffer for page cache memory space
                        _GlobalStatics.Page = (Btree.MemPage)args[0];
                        _GlobalStatics.PageSize = (int)args[1];
                        _GlobalStatics.Pages = (int)args[2];
                        break;
                    }
                case CONFIG.PCACHE:
                    { // no-op
                        break;
                    }
                case CONFIG.GETPCACHE:
                    { // now an error
                        rc = RC.ERROR;
                        break;
                    }
                case CONFIG.PCACHE2:
                    { // Specify an alternative page cache implementation
                        //_GlobalStatics.PCache2 = (sqlite3_pcache_methods2)args[0];
                        break;
                    }
                case CONFIG.GETPCACHE2:
                    {
                        //if (_GlobalStatics.Pcache2.Init == 0)
                        //	PCacheSetDefault();
                        //args[0] = _GlobalStatics.pcache2;
                        break;
                    }
                case CONFIG.COVERING_INDEX_SCAN:
                    {
                        _GlobalStatics.UseCis = (bool)args[0];
                        break;
                    }
                default:
                    {
                        rc = RC.ERROR;
                        break;
                    }
            }
            return rc;

        }

        public enum CTXCONFIG
        {
            LOOKASIDE = 1001,  // void* int int
            ENABLE_FKEY = 1002,  // int int*
            ENABLE_TRIGGER = 1003,  // int int*
        }

        public class FlagOp
        {
            public CTXCONFIG OP;      // The opcode
            public Context.FLAG Mask;       // Mask of the bit in sqlite3.flags to set/clear

            public FlagOp(CTXCONFIG op, Context.FLAG mask)
            {
                OP = op;
                Mask = mask;
            }
        }

        static readonly FlagOp[] _flagOps = new[]
        {
            new FlagOp(CTXCONFIG.ENABLE_FKEY,    Context.FLAG.ForeignKeys),
            new FlagOp(CTXCONFIG.ENABLE_TRIGGER, Context.FLAG.EnableTrigger),
        };

        public static RC CtxConfig(Context ctx, CTXCONFIG op, params object[] args)
        {
            RC rc;
            switch (op)
            {
                case CTXCONFIG.LOOKASIDE:
                    {
                        byte[] buf = (byte[])args[0]; // IMP: R-26835-10964
                        int size = (int)args[1];       // IMP: R-47871-25994
                        int count = (int)args[2];      // IMP: R-04460-53386
                        rc = SysEx.SetupLookaside(ctx, buf, size, count);
                        break;
                    }
                default:
                    {
                        rc = RC.ERROR; // IMP: R-42790-23372
                        for (int i = 0; i < _flagOps.Length; i++)
                        {
                            if (_flagOps[i].OP == op)
                            {
                                bool set = (bool)args[0];
                                Action<bool> r = (Action<bool>)args[1];
                                Context.FLAG oldFlags = ctx.Flags;
                                if (set)
                                    ctx.Flags |= _flagOps[i].Mask;
                                else
                                    ctx.Flags &= ~_flagOps[i].Mask;
                                if (oldFlags != ctx.Flags)
                                    Vdbe.ExpirePreparedStatements(ctx);
                                if (r != null)
                                    r((ctx.Flags & _flagOps[i].Mask) != 0);
                                rc = RC.OK;
                                break;
                            }
                        }
                        break;
                    }
            }
            return rc;
        }

        #endregion

        //public static MutexEx CtxMutex(Context ctx) { return ctx.Mutex; }

        public static RC CtxReleaseMemory(Context ctx)
        {
            MutexEx.Enter(ctx.Mutex);
            Btree.EnterAll(ctx);
            for (int i = 0; i < ctx.DBs.length; i++)
            {
                Btree bt = ctx.DBs[i].Bt;
                if (bt != null)
                {
                    Pager pager = bt.get_Pager();
                    pager.Shrink();
                }
            }
            Btree.LeaveAll(ctx);
            MutexEx.Leave(ctx.Mutex);
            return RC.OK;
        }

        static bool AllSpaces(string z, int iStart, int n)
        {
            while (n > 0 && z[iStart + n - 1] == ' ') { n--; }
            return (n == 0);
        }

        static int BinCollFunc(object padFlag, int key1Length, string key1, int key2Length, string key2)
        {
            int n = (key1Length < key2Length ? key1Length : key2Length);
            int rc = C._memcmp(key1, key2, n);
            if (rc == 0)
            {
                if ((int)padFlag != 0 && AllSpaces(key1, n, key1Length - n) && AllSpaces(key2, n, key2Length - n)) { } // Leave rc unchanged at 0
                else rc = key1Length - key2Length;
            }
            return rc;
        }

        static int NocaseCollatingFunc(object dummy1, int key1Length, string key1, int key2Length, string key2)
        {
            int r = string.Compare(key1, 0, key2, 0, (key1Length < key2Length ? key1Length : key2Length), StringComparison.OrdinalIgnoreCase);
            if (r == 0)
                r = key1Length - key2Length;
            return r;
        }

        //public static long CtxLastInsertRowid(Context ctx) { return ctx.LastRowID; }
        //public static int CtxChanges(Context ctx) { return ctx.Changes; }
        //public static int CtxTotalChanges(Context ctx) { return ctx.TotalChanges; }

        public static void CloseSavepoints(Context ctx)
        {
            while (ctx.Savepoints != null)
            {
                Savepoint t = ctx.Savepoints;
                ctx.Savepoints = t.Next;
                C._tagfree(ctx, ref t);
            }
            ctx.SavepointsLength = 0;
            ctx.Statements = 0;
            ctx.IsTransactionSavepoint = 0;
        }

        static void FunctionDestroy(Context ctx, FuncDef p)
        {
            FuncDestructor destructor = p.Destructor;
            if (destructor != null)
            {
                destructor.Refs--;
                if (destructor.Refs == 0)
                {
                    destructor.Destroy(destructor.UserData);
                    C._tagfree(ctx, ref destructor);
                }
            }
        }

        static void DisconnectAllVtab(Context ctx)
        {
#if !OMIT_VIRTUALTABLE
            Btree.EnterAll(ctx);
            for (int i = 0; i < ctx.DBs.length; i++)
            {
                Schema schema = ctx.DBs[i].Schema;
                if (ctx.DBs[i].Schema != null)
                    for (HashElem p = schema.TableHash.First; p != null; p = p.Next)
                    {
                        Table table = (Table)p.Data;
                        if (E.IsVirtual(table)) VTable.Disconnect(ctx, table);
                    }
            }
            Btree.LeaveAll(ctx);
#endif
        }

        static bool ConnectionIsBusy(Context ctx)
        {
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            if (ctx.Vdbes != null) return true;
            for (int j = 0; j < ctx.DBs.length; j++)
            {
                Btree bt = ctx.DBs[j].Bt;
                if (bt != null && bt.IsInBackup()) return true;
            }
            return false;
        }

        #region Close/Rollback

        public static RC Close(Context ctx) { return Close(ctx, false); }
        public static RC Close_v2(Context ctx) { return Close(ctx, true); }
        public static RC Close(Context ctx, bool forceZombie)
        {
            HashElem i;
            int j;

            if (ctx == null)
                return RC.OK;
            if (!SafetyCheckSickOrOk(ctx))
                return SysEx.MISUSE_BKPT();
            MutexEx.Enter(ctx.Mutex);

            // Force xDestroy calls on all virtual tables
            DisconnectAllVtab(ctx);

            // If a transaction is open, the disconnectAllVtab() call above will not have called the xDisconnect() method on any virtual
            // tables in the ctx->aVTrans[] array. The following sqlite3VtabRollback() call will do so. We need to do this before the check for active
            // SQL statements below, as the v-table implementation may be storing some prepared statements internally.
            VTable.Rollback(ctx);

            // Legacy behavior (sqlite3_close() behavior) is to return SQLITE_BUSY if the connection can not be closed immediately.
            if (!forceZombie && ConnectionIsBusy(ctx))
            {
                Error(ctx, RC.BUSY, "unable to close due to unfinalized statements or unfinished backups");
                MutexEx.Leave(ctx.Mutex);
                return RC.BUSY;
            }

#if ENABLE_SQLLOG
		if (SysEx._GlobalStatics.Sqllog != null)
			SysEx._GlobalStatics.Sqllog(SysEx._GlobalStatics.SqllogArg, ctx, 0, 2); // Closing the handle. Fourth parameter is passed the value 2.
#endif

            // Convert the connection into a zombie and then close it.
            ctx.Magic = MAGIC.ZOMBIE;
            LeaveMutexAndCloseZombie(ctx);
            return RC.OK;
        }

        public static void LeaveMutexAndCloseZombie(Context ctx)
        {
            // If there are outstanding sqlite3_stmt or sqlite3_backup objects or if the connection has not yet been closed by sqlite3_close_v2(),
            // then just leave the mutex and return.
            if (ctx.Magic != MAGIC.ZOMBIE || ConnectionIsBusy(ctx))
            {
                MutexEx.Leave(ctx.Mutex);
                return;
            }

            // If we reach this point, it means that the database connection has closed all sqlite3_stmt and sqlite3_backup objects and has been
            // passed to sqlite3_close (meaning that it is a zombie).  Therefore, go ahead and free all resources.

            // Free any outstanding Savepoint structures.
            CloseSavepoints(ctx);

            // Close all database connections
            int j;
            for (j = 0; j < ctx.DBs.length; j++)
            {
                Context.DB dbAsObj = ctx.DBs[j];
                if (dbAsObj.Bt != null)
                {
                    dbAsObj.Bt.Close();
                    dbAsObj.Bt = null;
                    if (j != 1)
                        dbAsObj.Schema = null;
                }
            }

            // Clear the TEMP schema separately and last
            if (ctx.DBs[1].Schema != null)
                Callback.SchemaClear(ctx.DBs[1].Schema);
            VTable.UnlockList(ctx);

            // Free up the array of auxiliary databases
            Parse.CollapseDatabaseArray(ctx);
            Debug.Assert(ctx.DBs.length <= 2);
            Debug.Assert(ctx.DBs.data == ctx.DBStatics);

            // Tell the code in notify.c that the connection no longer holds any locks and does not require any further unlock-notify callbacks.
            //Notify::ConnectionClosed(ctx);

            for (j = 0; j < ctx.Funcs.data.Length; j++)
            {
                FuncDef next, hash;
                for (FuncDef p = ctx.Funcs.data[j]; p != null; p = hash)
                {
                    hash = p.Hash;
                    while (p != null)
                    {
                        FunctionDestroy(ctx, p);
                        next = p.Next;
                        C._tagfree(ctx, ref p);
                        p = next;
                    }
                }
            }
            HashElem i;
            for (i = ctx.CollSeqs.First; i != null; i = i.Next) // Hash table iterator
            {
                CollSeq[] coll = (CollSeq[])i.Data;
                // Invoke any destructors registered for collation sequence user data.
                for (j = 0; j < 3; j++)
                    if (coll[j].Del != null)
                        coll[j].Del(ref coll[j].User);
                C._tagfree(ctx, ref coll);
            }
            ctx.CollSeqs.Clear();
#if !OMIT_VIRTUALTABLE
            for (i = ctx.Modules.First; i != null; i = i.Next)
            {
                TableModule mod = (TableModule)i.Data;
                if (mod.Destroy != null)
                    mod.Destroy(mod.Aux);
                C._tagfree(ctx, ref mod);
            }
            ctx.Modules.Clear();
#endif

            Error(ctx, RC.OK, null); // Deallocates any cached error strings.
            if (ctx.Err != null)
                Vdbe.ValueFree(ref ctx.Err);
            //LoadExt.CloseExtensions(ctx);

            ctx.Magic = MAGIC.ERROR;

            // The temp-database schema is allocated differently from the other schema objects (using sqliteMalloc() directly, instead of sqlite3BtreeSchema()).
            // So it needs to be freed here. Todo: Why not roll the temp schema into the same sqliteMalloc() as the one that allocates the database structure?
            C._tagfree(ctx, ref ctx.DBs[1].Schema);
            MutexEx.Leave(ctx.Mutex);
            ctx.Magic = MAGIC.CLOSED;
            MutexEx.Free(ctx.Mutex);
            Debug.Assert(ctx.Lookaside.Outs == 0); // Fails on a lookaside memory leak
            if (ctx.Lookaside.Malloced)
                C._free(ref ctx.Lookaside.Start);
            C._free(ref ctx);
        }

        public static void RollbackAll(Context ctx, RC tripCode)
        {
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            C._benignalloc_begin();
            bool inTrans = false;
            for (int i = 0; i < ctx.DBs.length; i++)
            {
                Btree p = ctx.DBs[i].Bt;
                if (p != null)
                {
                    if (p.IsInTrans())
                        inTrans = true;
                    p.Rollback(tripCode);
                    ctx.DBs[i].InTrans = 0;
                }
            }
            VTable.Rollback(ctx);
            C._benignalloc_end();

            if ((ctx.Flags & Context.FLAG.InternChanges) != 0)
            {
                Vdbe.ExpirePreparedStatements(ctx);
                Parse.ResetAllSchemasOfConnection(ctx);
            }

            // Any deferred constraint violations have now been resolved.
            ctx.DeferredCons = 0;

            // If one has been configured, invoke the rollback-hook callback
            if (ctx.RollbackCallback != null && (inTrans || ctx.AutoCommit == 0))
                ctx.RollbackCallback(ctx.RollbackArg);
        }

        #endregion

        static string[] _msgs = new string[]
            {
                /* SQLITE_OK          */ "not an error",
                /* SQLITE_ERROR       */ "SQL logic error or missing database",
                /* SQLITE_INTERNAL    */ "",
                /* SQLITE_PERM        */ "access permission denied",
                /* SQLITE_ABORT       */ "callback requested query abort",
                /* SQLITE_BUSY        */ "database is locked",
                /* SQLITE_LOCKED      */ "database table is locked",
                /* SQLITE_NOMEM       */ "out of memory",
                /* SQLITE_READONLY    */ "attempt to write a readonly database",
                /* SQLITE_INTERRUPT   */ "interrupted",
                /* SQLITE_IOERR       */ "disk I/O error",
                /* SQLITE_CORRUPT     */ "database disk image is malformed",
                /* SQLITE_NOTFOUND    */ "unknown operation",
                /* SQLITE_FULL        */ "database or disk is full",
                /* SQLITE_CANTOPEN    */ "unable to open database file",
                /* SQLITE_PROTOCOL    */ "locking protocol",
                /* SQLITE_EMPTY       */ "table contains no data",
                /* SQLITE_SCHEMA      */ "database schema has changed",
                /* SQLITE_TOOBIG      */ "string or blob too big",
                /* SQLITE_CONSTRAINT  */ "constraint failed",
                /* SQLITE_MISMATCH    */ "datatype mismatch",
                /* SQLITE_MISUSE      */ "library routine called out of sequence",
                /* SQLITE_NOLFS       */ "large file support is disabled",
                /* SQLITE_AUTH        */ "authorization denied",
                /* SQLITE_FORMAT      */ "auxiliary database format error",
                /* SQLITE_RANGE       */ "bind or column index out of range",
                /* SQLITE_NOTADB      */ "file is encrypted or is not a database",
            };
        public static string ErrStr(RC rc)
        {
            string err = "unknown error";
            switch (rc)
            {
                case RC.ABORT_ROLLBACK:
                    {
                        err = "abort due to ROLLBACK";
                        break;
                    }
                default:
                    {
                        rc = (RC)((int)rc & 0xff);
                        if (C._ALWAYS(rc >= 0) && (int)rc < _msgs.Length && _msgs[(int)rc] != null)
                            err = _msgs[(int)rc];
                        break;
                    }
            }
            return err;
        }

        #region Busy Handler

#if OS_WIN || HAVE_USLEEP
        static readonly byte[] _delays = new byte[] { 1, 2, 5, 10, 15, 20, 25, 25, 25, 50, 50, 100 };
        static readonly byte[] _totals = new byte[] { 0, 1, 3, 8, 18, 33, 53, 78, 103, 128, 178, 228 };
        static readonly int NDELAY = _delays.Length;
#endif

        public static int DefaultBusyCallback(object ptr, int count)
        {
            Context ctx = (Context)ptr;
            int timeout = ctx.BusyTimeout;
            Debug.Assert(count >= 0);
#if OS_WIN || HAVE_USLEEP
            int delay, prior;
            if (count < NDELAY)
            {
                delay = _delays[count];
                prior = _totals[count];
            }
            else
            {
                delay = _delays[NDELAY - 1];
                prior = _totals[NDELAY - 1] + delay * (count - (NDELAY - 1));
            }
            if (prior + delay > timeout)
            {
                delay = timeout - prior;
                if (delay <= 0) return 0;
            }
            ctx.Vfs.Sleep(delay * 1000);
            return 1;
#else
            if ((count + 1) * 1000 > timeout)
                return 0;
            ctx.Vfs.Sleep(1000000);
            return 1;
#endif
        }

        public static int InvokeBusyHandler(Context.BusyHandlerType p)
        {
            if (C._NEVER(p == null) || p.Func == null || p.Busys < 0) return 0;
            int rc = p.Func(p.Arg, p.Busys);
            if (rc == 0)
                p.Busys = -1;
            else
                p.Busys++;
            return rc;
        }

        public static RC BusyHandler(Context ctx, Func<object, int, int> busy, object arg)
        {
            MutexEx.Enter(ctx.Mutex);
            ctx.BusyHandler.Func = busy;
            ctx.BusyHandler.Arg = arg;
            ctx.BusyHandler.Busys = 0;
            MutexEx.Leave(ctx.Mutex);
            return RC.OK;
        }

#if !OMIT_PROGRESS_CALLBACK
        public static void ProgressHandler(Context ctx, int ops, Func<object, int> progress, object arg)
        {
            MutexEx.Enter(ctx.Mutex);
            if (ops > 0)
            {
                ctx.Progress = progress;
                ctx.ProgressOps = ops;
                ctx.ProgressArg = arg;
            }
            else
            {
                ctx.Progress = null;
                ctx.ProgressOps = 0;
                ctx.ProgressArg = null;
            }
            MutexEx.Leave(ctx.Mutex);
        }
#endif

        public static RC BusyTimeout(Context ctx, int ms)
        {
            if (ms > 0)
            {
                ctx.BusyTimeout = ms;
                BusyHandler(ctx, DefaultBusyCallback, ctx);
            }
            else
                BusyHandler(ctx, null, null);
            return RC.OK;
        }

        public static void Interrupt(Context ctx)
        {
            ctx.u1.IsInterrupted = true;
        }

        #endregion

        #region Function

        public static RC CreateFunc(Context ctx, string funcName, int args, TEXTENCODE encode, object userData, Action<FuncContext, int, Mem[]> func, Action<FuncContext, int, Mem[]> step, Action<FuncContext> final_, FuncDestructor destructor)
        {
            FuncDef p;
            int funcNameLength;
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            if (funcName == null ||
            (func != null && (final_ != null || step != null)) ||
            (func == null && (final_ != null && step == null)) ||
            (func == null && (final_ == null && step != null)) ||
            (args < -1 || args > MAX_FUNCTION_ARG) ||
            (255 < (funcNameLength = funcName.Length)))
                return SysEx.MISUSE_BKPT();

#if !OMIT_UTF16
            // If SQLITE_UTF16 is specified as the encoding type, transform this to one of SQLITE_UTF16LE or SQLITE_UTF16BE using the
            // SQLITE_UTF16NATIVE macro. SQLITE_UTF16 is not used internally.
            //
            // If SQLITE_ANY is specified, add three versions of the function to the hash table.
            if (encode == TEXTENCODE.UTF16)
                encode = TEXTENCODE.UTF16NATIVE;
            else if (encode == TEXTENCODE.ANY)
            {
                RC rc = CreateFunc(ctx, funcName, args, TEXTENCODE.UTF8, userData, func, step, final_, destructor);
                if (rc == RC.OK)
                    rc = CreateFunc(ctx, funcName, args, TEXTENCODE.UTF16LE, userData, func, step, final_, destructor);
                if (rc != RC.OK)
                    return rc;
                encode = TEXTENCODE.UTF16BE;
            }
#else
            encode = TEXTENCODE.UTF8;
#endif

            // Check if an existing function is being overridden or deleted. If so, and there are active VMs, then return SQLITE_BUSY. If a function
            // is being overridden/deleted but there are no active VMs, allow the operation to continue but invalidate all precompiled statements.
            FuncDef p = Callback.FindFunction(ctx, funcName, funcNameLength, args, encode, false);
            if (p != null && p.PrefEncode == encode && p.Args == args)
            {
                if (ctx.ActiveVdbeCnt != 0)
                {
                    Error(ctx, RC.BUSY, "unable to delete/modify user-function due to active statements");
                    Debug.Assert(!ctx.MallocFailed);
                    return RC.BUSY;
                }
                else
                    Vdbe.ExpirePreparedStatements(ctx);
            }

            p = Callback.FindFunction(ctx, funcName, funcNameLength, args, encode, true);
            Debug.Assert(p != null || ctx.MallocFailed);
            if (p == null)
                return RC.NOMEM;

            // If an older version of the function with a configured destructor is being replaced invoke the destructor function here.
            FunctionDestroy(ctx, p);

            if (destructor != null)
                destructor.Refs++;
            p.Destructor = destructor;
            p.Flags = 0;
            p.Func = func;
            p.Step = step;
            p.Finalize = final_;
            p.UserData = userData;
            p.Args = (short)args;
            return RC.OK;
        }

        public static RC CreateFunction(Context ctx, string funcName, int args, TEXTENCODE encode, object p, Action<FuncContext, int, Mem[]> func, Action<FuncContext, int, Mem[]> step, Action<FuncContext> final_) { return CreateFunction_v2(ctx, funcName, args, encode, p, func, step, final_, null); }
        public static RC CreateFunction_v2(Context ctx, string funcName, int args, TEXTENCODE encode, object p, Action<FuncContext, int, Mem[]> func, Action<FuncContext, int, Mem[]> step, Action<FuncContext> final_, Action<object> destroy)
        {
            RC rc = RC.ERROR;
            FuncDestructor arg = null;
            MutexEx.Enter(ctx.Mutex);
            if (destroy != null)
            {
                arg = new FuncDestructor();
                if (arg == null)
                {
                    destroy(p);
                    goto _out;
                }
                arg.Destroy = destroy;
                arg.UserData = p;
            }
            rc = CreateFunc(ctx, funcName, args, encode, p, func, step, final_, arg);
            if (arg != null && arg.Refs == 0)
            {
                Debug.Assert(rc != RC.OK);
                destroy(p);
                C._tagfree(ctx, ref arg);
            }
        _out:
            rc = ApiExit(ctx, rc);
            MutexEx.Leave(ctx.Mutex);
            return rc;
        }

#if !OMIT_UTF16
        public static RC CreateFunction16(Context ctx, string funcName, int args, TEXTENCODE encode, object p, Action<FuncContext, int, Mem[]> func, Action<FuncContext, int, Mem[]> step, Action<FuncContext> final_)
        {
            MutexEx.Enter(ctx.Mutex);
            Debug.Assert(!ctx.MallocFailed);
            string funcName8 = Vdbe.Utf16to8(ctx, funcName, -1, TEXTENCODE.UTF16NATIVE);
            RC rc = CreateFunc(ctx, funcName8, args, encode, p, func, step, final_, null);
            C._tagfree(ctx, ref funcName8);
            rc = ApiExit(ctx, rc);
            MutexEx.Leave(ctx.Mutex);
            return rc;
        }
#endif

        public static RC OverloadFunction(Context ctx, string funcName, int args)
        {
            int funcNameLength = funcName.Length;
            RC rc;
            MutexEx.Enter(ctx.Mutex);
            if (Callback.FindFunction(ctx, funcName, funcNameLength, args, TEXTENCODE.UTF8, false) == null)
                rc = CreateFunc(ctx, funcName, args, TEXTENCODE.UTF8, 0, Vdbe.InvalidFunction, null, null, null);
            rc = ApiExit(ctx, RC.OK);
            MutexEx.Leave(ctx.Mutex);
            return rc;
        }

        #endregion

        #region Callback

#if !OMIT_TRACE
        public static object Trace(Context ctx, Action<object, string> trace, object arg)
        {
            MutexEx.Enter(ctx.Mutex);
            object oldArg = ctx.TraceArg;
            ctx.Trace = trace;
            ctx.TraceArg = arg;
            MutexEx.Leave(ctx.Mutex);
            return oldArg;
        }

        public static object Profile(Context ctx, Action<object, string, ulong> profile, object arg)
        {
            MutexEx.Enter(ctx.Mutex);
            object oldArg = ctx.ProfileArg;
            ctx.Profile = profile;
            ctx.ProfileArg = arg;
            MutexEx.Leave(ctx.Mutex);
            return oldArg;
        }
#endif

        public static object CommitHook(Context ctx, Func<object, RC> callback, object arg)
        {
            MutexEx.Enter(ctx.Mutex);
            object oldArg = ctx.CommitArg;
            ctx.CommitCallback = callback;
            ctx.CommitArg = arg;
            MutexEx.Leave(ctx.Mutex);
            return oldArg;
        }

        public static object UpdateHook(Context ctx, Action<object, int, string, string, long> callback, object arg)
        {
            MutexEx.Enter(ctx.Mutex);
            object oldArg = ctx.UpdateArg;
            ctx.UpdateCallback = callback;
            ctx.UpdateArg = arg;
            MutexEx.Leave(ctx.Mutex);
            return oldArg;
        }

        public static object RollbackHook(Context ctx, Action<object> callback, object arg)
        {
            MutexEx.Enter(ctx.Mutex);
            object oldArg = ctx.RollbackArg;
            ctx.RollbackCallback = callback;
            ctx.RollbackArg = arg;
            MutexEx.Leave(ctx.Mutex);
            return oldArg;
        }

#if !OMIT_WAL
        public RC WalDefaultHook(object clientData, Context ctx, string dbName, int frames)
        {
            if (frames >= (int)clientData)
            {
                C._benignalloc_begin();
                WalCheckpoint(ctx, dbName);
                C._benignalloc_end();
            }
            return RC.OK;
        }
#endif

        public static RC WalAutocheckpoint(Context ctx, int frames)
        {
#if OMIT_WAL
            return RC.OK;
#else
            if (frames > 0)
                WalHook(ctx, WalDefaultHook, frames);
            else
                WalHook(ctx, null, 0);
            return RC.OK;
#endif
        }

        public static object WalHook(Context ctx, Func<object, Context, string, int, RC> callback, object arg)
        {
#if OMIT_WAL
            return null;
#else
            MutexEx.Enter(ctx.Mutex);
            object oldArg = ctx.WalArg;
            ctx.WalCallback = callback;
            ctx.WalArg = arg;
            MutexEx.Leave(ctx.Mutex);
            return oldArg;
#endif
        }

        #endregion

        #region Checkpoint

        public static RC WalCheckpoint(Context ctx, string dbName) { int dummy1; return WalCheckpoint_v2(ctx, dbName, IPager.CHECKPOINT.PASSIVE, out dummy1, out dummy1); }
        public static RC WalCheckpoint_v2(Context ctx, string dbName, IPager.CHECKPOINT mode, out int logsOut, out int ckptsOut)
        {
#if OMIT_WAL
            logsOut = 0;
            ckptsOut = 0;
            return RC.OK;
#else
            int db = MAX_ATTACHED;  // sqlite3.aDb[] index of db to checkpoint

            // Initialize the output variables to -1 in case an error occurs.
            logsOut = -1;
            ckptsOut = -1;

            Debug.Assert(IPager.CHECKPOINT.FULL > IPager.CHECKPOINT.PASSIVE);
            Debug.Assert(IPager.CHECKPOINT.FULL < IPager.CHECKPOINT.RESTART);
            Debug.Assert(IPager.CHECKPOINT.PASSIVE + 2 == IPager.CHECKPOINT.RESTART);
            if (mode < IPager.CHECKPOINT.PASSIVE || mode > IPager.CHECKPOINT.RESTART)
                return RC.MISUSE;

            MutexEx.Enter(ctx.Mutex);
            if (dbName != null && dbName.Length > 0)
                db = Parse.FindDbName(ctx, dbName);
            RC rc;
            if (db < 0)
            {
                rc = RC.ERROR;
                Error(ctx, RC.ERROR, "unknown database: %s", dbName);
            }
            else
            {
                rc = Checkpoint(ctx, db, mode, out logsOut, out ckptsOut);
                Error(ctx, rc, null);
            }
            rc = ApiExit(ctx, rc);
            MutexEx.Leave(ctx.Mutex);
            return rc;
#endif
        }

#if !OMIT_WAL
        int Checkpoint(Context ctx, int db, IPager.CHECKPOINT mode, out int logsOut, out int ckptsOut)
        {
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            Debug.Assert(logsOut == -1);
            Debug.Assert(ckptsOut == -1);

            RC rc = RC.OK;
            bool busy = false; // True if SQLITE_BUSY has been encountered
            for (int i = 0; i < ctx.DBs.length && rc == RC.OK; i++)
            {
                if (i == db || db == MAX_ATTACHED)
                {
                    int dummy1;
                    if (!busy)
                        rc = Btree.Checkpoint(ctx.DBs[i].Bt, mode, logsOut, ckptsOut);
                    else
                        rc = Btree.Checkpoint(ctx.DBs[i].Bt, mode, dummy1, dummy1);
                    if (rc == RC.BUSY)
                    {
                        busy = true;
                        rc = RC.OK;
                    }
                }
            }
            return (rc == RC.OK && busy ? RC.BUSY : rc);
        }
#endif

        #endregion

        public static bool TempInMemory(Context ctx)
        {
            if (TEMP_STORE == 1)
                return (ctx.TempStore == 2);
            if (TEMP_STORE == 2)
                return (ctx.TempStore != 1);
            if (TEMP_STORE == 3)
                return true;
            return false;
        }

        public static string ErrMsg(Context ctx)
        {
            if (ctx == null)
                return ErrStr(RC.NOMEM);
            if (!SafetyCheckSickOrOk(ctx))
                return ErrStr(SysEx.MISUSE_BKPT());
            MutexEx.Enter(ctx.Mutex);
            string z;
            if (ctx.MallocFailed)
                z = ErrStr(RC.NOMEM);
            else
            {
                z = Vdbe.Value_Text(ctx.Err);
                Debug.Assert(!ctx.MallocFailed);
                if (z == null)
                    z = ErrStr(ctx.ErrCode);
            }
            MutexEx.Leave(ctx.Mutex);
            return z;
        }

#if !OMIT_UTF16
        static readonly string _outOfMem = "out of memory";
        static readonly string _misuse = "library routine called out of sequence";

        public static string ErrMsg16(Context ctx)
        {
            if (ctx == null)
                return _outOfMem;
            if (SafetyCheckSickOrOk(ctx))
                return _misuse;
            MutexEx.Enter(ctx.Mutex);
            string z;
            if (ctx.MallocFailed)
                z = _outOfMem;
            else
            {
                z = Vdbe.Value_Text16(ctx.Err);
                if (z == null)
                {
                    Vdbe.ValueSetStr(ctx.Err, -1, ErrStr(ctx.ErrCode), TEXTENCODE.UTF8, C.DESTRUCTOR_STATIC);
                    z = Vdbe.Value_Text16(ctx.Err);
                }
                // A malloc() may have failed within the call to sqlite3_value_text16() above. If this is the case, then the ctx->mallocFailed flag needs to
                // be cleared before returning. Do this directly, instead of via sqlite3ApiExit(), to avoid setting the database handle error message.
                ctx.MallocFailed = false;
            }
            MutexEx.Leave(ctx.Mutex);
            return z;
        }
#endif

        public static RC ErrCode(Context ctx)
        {
            if (ctx != null && !SafetyCheckSickOrOk(ctx))
                return SysEx.MISUSE_BKPT();
            if (ctx == null || ctx.MallocFailed)
                return RC.NOMEM;
            return (RC)((int)ctx.ErrCode & ctx.ErrMask);
        }

        public static RC ExtendedErrCode(Context ctx)
        {
            if (ctx != null && !SafetyCheckSickOrOk(ctx))
                return SysEx.MISUSE_BKPT();
            if (ctx == null || ctx.MallocFailed)
                return RC.NOMEM;
            return ctx.ErrCode;
        }

        //public static string ErrStr(RC rc) { return ErrStr(rc); }

        static RC CreateCollation(Context ctx, string name, TEXTENCODE encode, object ctx2, Func<object, int, object, int, object, int> compare, RefAction<object> del)
        {
            int nameLength = name.Length;
            Debug.Assert(MutexEx.Held(ctx.Mutex));

            // If SQLITE_UTF16 is specified as the encoding type, transform this to one of SQLITE_UTF16LE or SQLITE_UTF16BE using the
            // SQLITE_UTF16NATIVE macro. SQLITE_UTF16 is not used internally.
            TEXTENCODE encode2 = encode;
            C.ASSERTCOVERAGE(encode2 == TEXTENCODE.UTF16);
            C.ASSERTCOVERAGE(encode2 == TEXTENCODE.UTF16_ALIGNED);
            if (encode2 == TEXTENCODE.UTF16 || encode2 == TEXTENCODE.UTF16_ALIGNED)
                encode2 = TEXTENCODE.UTF16NATIVE;
            if (encode2 < TEXTENCODE.UTF8 || encode2 > TEXTENCODE.UTF16BE)
                return SysEx.MISUSE_BKPT();

            // Check if this call is removing or replacing an existing collation sequence. If so, and there are active VMs, return busy. If there
            // are no active VMs, invalidate any pre-compiled statements.
            CollSeq coll = Callback.FindCollSeq(ctx, encode2, name, false);
            if (coll != null && coll.Cmp != null)
            {
                if (ctx.ActiveVdbeCnt != 0)
                {
                    Error(ctx, RC.BUSY, "unable to delete/modify collation sequence due to active statements");
                    return RC.BUSY;
                }
                Vdbe.ExpirePreparedStatements(ctx);

                // If collation sequence pColl was created directly by a call to sqlite3_create_collation, and not generated by synthCollSeq(),
                // then any copies made by synthCollSeq() need to be invalidated. Also, collation destructor - CollSeq.xDel() - function may need
                // to be called.
                if ((coll.Encode & ~TEXTENCODE.UTF16_ALIGNED) == encode2)
                {
                    CollSeq[] colls = ctx.CollSeqs.Find(name, nameLength, (CollSeq[])null);
                    for (int j = 0; j < 3; j++)
                    {
                        CollSeq p = colls[j];
                        if (p.Encode == coll.Encode)
                        {
                            if (p.Del != null)
                                p.Del(ref p.User);
                            p.Cmp = null;
                        }
                    }
                }
            }

            coll = Callback.FindCollSeq(ctx, encode2, name, true);
            if (coll == null)
                return RC.NOMEM;
            coll.Cmp = compare;
            coll.User = ctx2;
            coll.Del = del;
            coll.Encode = (encode2 | (encode & TEXTENCODE.UTF16_ALIGNED));
            Error(ctx, RC.OK, null);
            return RC.OK;
        }

        #region Limit

        // Make sure the hard limits are set to reasonable values
        //#if MAX_LENGTH < 100
        //#error MAX_LENGTH must be at least 100
        //#endif
        //#if MAX_SQL_LENGTH < 100
        //#error MAX_SQL_LENGTH must be at least 100
        //#endif
        //#if MAX_SQL_LENGTH > MAX_LENGTH
        //#error MAX_SQL_LENGTH must not be greater than MAX_LENGTH
        //#endif
        //#if MAX_COMPOUND_SELECT < 2
        //#error MAX_COMPOUND_SELECT must be at least 2
        //#endif
        //#if MAX_VDBE_OP < 40
        //#error MAX_VDBE_OP must be at least 40
        //#endif
        //#if MAX_FUNCTION_ARG < 0 || MAX_FUNCTION_ARG > 1000
        //#error MAX_FUNCTION_ARG must be between 0 and 1000
        //#endif
        //#if MAX_ATTACHED<0 || MAX_ATTACHED > 62
        //#error MAX_ATTACHED must be between 0 and 62
        //#endif
        //#if MAX_LIKE_PATTERN_LENGTH < 1
        //#error MAX_LIKE_PATTERN_LENGTH must be at least 1
        //#endif
        //#if MAX_COLUMN > 32767
        //#error MAX_COLUMN must not exceed 32767
        //#endif
        //#if MAX_TRIGGER_DEPTH < 1
        //#error MAX_TRIGGER_DEPTH must be at least 1
        //#endif

        static readonly int[] _hardLimits = new int[]  {  // kept in sync with the LIMIT_*
            MAX_LENGTH,
            MAX_SQL_LENGTH,
            MAX_COLUMN,
            MAX_EXPR_DEPTH,
            MAX_COMPOUND_SELECT,
            MAX_VDBE_OP,
            MAX_FUNCTION_ARG,
            MAX_ATTACHED,
            MAX_LIKE_PATTERN_LENGTH,
            MAX_VARIABLE_NUMBER,
            MAX_TRIGGER_DEPTH,
        };

        public static int Limit(Context ctx, LIMIT limit, int newLimit)
        {
            // EVIDENCE-OF: R-30189-54097 For each limit category SQLITE_LIMIT_NAME there is a hard upper bound set at compile-time by a C preprocessor
            // macro called SQLITE_MAX_NAME. (The "_LIMIT_" in the name is changed to "_MAX_".)
            Debug.Assert(_hardLimits[(int)LIMIT.LENGTH] == MAX_LENGTH);
            Debug.Assert(_hardLimits[(int)LIMIT.SQL_LENGTH] == MAX_SQL_LENGTH);
            Debug.Assert(_hardLimits[(int)LIMIT.COLUMN] == MAX_COLUMN);
            Debug.Assert(_hardLimits[(int)LIMIT.EXPR_DEPTH] == MAX_EXPR_DEPTH);
            Debug.Assert(_hardLimits[(int)LIMIT.COMPOUND_SELECT] == MAX_COMPOUND_SELECT);
            Debug.Assert(_hardLimits[(int)LIMIT.VDBE_OP] == MAX_VDBE_OP);
            Debug.Assert(_hardLimits[(int)LIMIT.FUNCTION_ARG] == MAX_FUNCTION_ARG);
            Debug.Assert(_hardLimits[(int)LIMIT.ATTACHED] == MAX_ATTACHED);
            Debug.Assert(_hardLimits[(int)LIMIT.LIKE_PATTERN_LENGTH] == MAX_LIKE_PATTERN_LENGTH);
            Debug.Assert(_hardLimits[(int)LIMIT.VARIABLE_NUMBER] == MAX_VARIABLE_NUMBER);
            Debug.Assert(_hardLimits[(int)LIMIT.TRIGGER_DEPTH] == MAX_TRIGGER_DEPTH);
            Debug.Assert((int)LIMIT.TRIGGER_DEPTH == ((int)LIMIT.MAX_ - 1));

            if (limit < 0 || limit >= LIMIT.MAX_)
                return -1;
            int oldLimit = ctx.Limits[(int)limit];
            if (newLimit >= 0) // IMP: R-52476-28732
            {
                if (newLimit > _hardLimits[(int)limit])
                    newLimit = _hardLimits[(int)limit]; // IMP: R-51463-25634
                ctx.Limits[(int)limit] = newLimit;
            }
            return oldLimit; // IMP: R-53341-35419
        }

        #endregion

#if false
    static int openDatabase(
    string zFilename,   /* Database filename UTF-8 encoded */
    out sqlite3 ppDb,   /* OUT: Returned database handle */
    int flags,         /* Operational flags */
    string zVfs         /* Name of the VFS to use */
    )
    {
      sqlite3 db;                     /* Store allocated handle here */
      int rc;                         /* Return code */
      int isThreadsafe;               /* True for threadsafe connections */
      string zOpen = "";              /* Filename argument to pass to BtreeOpen() */
      string zErrMsg = "";            /* Error message from sqlite3ParseUri() */

      ppDb = null;
#if !OMIT_AUTOINIT
      rc = sqlite3_initialize();
      if ( rc != 0 )
        return rc;
#endif

      /* Only allow sensible combinations of bits in the flags argument.
** Throw an error if any non-sense combination is used.  If we
** do not block illegal combinations here, it could trigger
** Debug.Assert() statements in deeper layers.  Sensible combinations
** are:
**
**  1:  SQLITE_OPEN_READONLY
**  2:  SQLITE_OPEN_READWRITE
**  6:  SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE
*/
      Debug.Assert( SQLITE_OPEN_READONLY == 0x01 );
      Debug.Assert( SQLITE_OPEN_READWRITE == 0x02 );
      Debug.Assert( SQLITE_OPEN_CREATE == 0x04 );
      testcase( ( 1 << ( flags & 7 ) ) == 0x02 ); /* READONLY */
      testcase( ( 1 << ( flags & 7 ) ) == 0x04 ); /* READWRITE */
      testcase( ( 1 << ( flags & 7 ) ) == 0x40 ); /* READWRITE | CREATE */
      if ( ( ( 1 << ( flags & 7 ) ) & 0x46 ) == 0 ) return SQLITE_MISUSE_BKPT();

      if ( SysEx_GlobalStatics.bCoreMutex == false )
      {
        isThreadsafe = 0;
      }
      else if ( ( flags & SQLITE_OPEN_NOMUTEX ) != 0 )
      {
        isThreadsafe = 0;
      }
      else if ( ( flags & SQLITE_OPEN_FULLMUTEX ) != 0 )
      {
        isThreadsafe = 1;
      }
      else
      {
        isThreadsafe = SysEx_GlobalStatics.bFullMutex ? 1 : 0;
      }
      if ( ( flags & SQLITE_OPEN_PRIVATECACHE ) != 0 )
      {
        flags &= ~SQLITE_OPEN_SHAREDCACHE;
      }
      else if ( SysEx_GlobalStatics.sharedCacheEnabled )
      {
        flags |= SQLITE_OPEN_SHAREDCACHE;
      }

      /* Remove harmful bits from the flags parameter
      **
      ** The SQLITE_OPEN_NOMUTEX and SQLITE_OPEN_FULLMUTEX flags were
      ** dealt with in the previous code block.  Besides these, the only
      ** valid input flags for sqlite3_open_v2() are SQLITE_OPEN_READONLY,
      ** SQLITE_OPEN_READWRITE, SQLITE_OPEN_CREATE, SQLITE_OPEN_SHAREDCACHE,
      ** SQLITE_OPEN_PRIVATECACHE, and some reserved bits.  Silently mask
      ** off all other flags.
      */
      flags &= ~( SQLITE_OPEN_DELETEONCLOSE |
      SQLITE_OPEN_EXCLUSIVE |
      SQLITE_OPEN_MAIN_DB |
      SQLITE_OPEN_TEMP_DB |
      SQLITE_OPEN_TRANSIENT_DB |
      SQLITE_OPEN_MAIN_JOURNAL |
      SQLITE_OPEN_TEMP_JOURNAL |
      SQLITE_OPEN_SUBJOURNAL |
      SQLITE_OPEN_MASTER_JOURNAL |
      SQLITE_OPEN_NOMUTEX |
      SQLITE_OPEN_FULLMUTEX |
      SQLITE_OPEN_WAL
      );


      /* Allocate the sqlite data structure */
      db = new sqlite3();//sqlite3MallocZero( sqlite3.Length );
      if ( db == null )
        goto opendb_out;
      if ( SysEx_GlobalStatics.bFullMutex && isThreadsafe != 0 )
      {
        db.mutex = sqlite3MutexAlloc( SQLITE_MUTEX_RECURSIVE );
        if ( db.mutex == null )
        {
          //sqlite3_free( ref db );
          goto opendb_out;
        }
      }
      sqlite3_mutex_enter( db.mutex );
      db.errMask = 0xff;
      db.nDb = 2;
      db.magic = SQLITE_MAGIC_BUSY;
      Array.Copy( db.aDbStatic, db.aDb, db.aDbStatic.Length );// db.aDb = db.aDbStatic;
      Debug.Assert( db.aLimit.Length == aHardLimit.Length );
      Buffer.BlockCopy( aHardLimit, 0, db.aLimit, 0, aHardLimit.Length * sizeof( int ) );//memcpy(db.aLimit, aHardLimit, sizeof(db.aLimit));
      db.autoCommit = 1;
      db.nextAutovac = -1;
      db.nextPagesize = 0;
      db.flags |= SQLITE_ShortColNames | SQLITE_AutoIndex | SQLITE_EnableTrigger;
      if ( SQLITE_DEFAULT_FILE_FORMAT < 4 )
        db.flags |= SQLITE_LegacyFileFmt
#if  ENABLE_LOAD_EXTENSION
| SQLITE_LoadExtension
#endif
#if DEFAULT_RECURSIVE_TRIGGERS
   | SQLITE_RecTriggers
#endif
#if (DEFAULT_FOREIGN_KEYS) //&& SQLITE_DEFAULT_FOREIGN_KEYS
   | SQLITE_ForeignKeys
#endif
;
      sqlite3HashInit( db.aCollSeq );
#if !OMIT_VIRTUALTABLE
      db.aModule = new Hash();
      sqlite3HashInit( db.aModule );
#endif

      /* Add the default collation sequence BINARY. BINARY works for both UTF-8
      ** and UTF-16, so add a version for each to avoid any unnecessary
      ** conversions. The only error that can occur here is a malloc() failure.
      */
      createCollation( db, "BINARY", SQLITE_UTF8, SQLITE_COLL_BINARY, 0,
               binCollFunc, null );
      createCollation( db, "BINARY", SQLITE_UTF16BE, SQLITE_COLL_BINARY, 0,
              binCollFunc, null );
      createCollation( db, "BINARY", SQLITE_UTF16LE, SQLITE_COLL_BINARY, 0,
              binCollFunc, null );
      createCollation( db, "RTRIM", SQLITE_UTF8, SQLITE_COLL_USER, 1,
              binCollFunc, null );
      //if ( db.mallocFailed != 0 )
      //{
      //  goto opendb_out;
      //}
      db.pDfltColl = sqlite3FindCollSeq( db, SQLITE_UTF8, "BINARY", 0 );
      Debug.Assert( db.pDfltColl != null );

      /* Also add a UTF-8 case-insensitive collation sequence. */
      createCollation( db, "NOCASE", SQLITE_UTF8, SQLITE_COLL_NOCASE, 0,
               nocaseCollatingFunc, null );

  /* Parse the filename/URI argument. */
  db.openFlags = flags;
  rc = sqlite3ParseUri( zVfs, zFilename, ref flags, ref db.pVfs, ref zOpen, ref zErrMsg );
  if( rc!=SQLITE_OK ){
    //if( rc==SQLITE_NOMEM ) db.mallocFailed = 1;
    sqlite3Error(db, rc, zErrMsg.Length>0 ? "%s" : "", zErrMsg);
    //sqlite3_free(zErrMsg);
    goto opendb_out;
  }

  /* Open the backend database driver */
  rc = sqlite3BtreeOpen(db.pVfs, zOpen, db, ref db.aDb[0].pBt, 0,
                        flags | SQLITE_OPEN_MAIN_DB);
if ( rc != SQLITE_OK )
      {
        if ( rc == SQLITE_IOERR_NOMEM )
        {
          rc = SQLITE_NOMEM;
        }
        sqlite3Error( db, rc, 0 );
        goto opendb_out;
      }
      db.aDb[0].pSchema = sqlite3SchemaGet( db, db.aDb[0].pBt );
      db.aDb[1].pSchema = sqlite3SchemaGet( db, null );


      /* The default safety_level for the main database is 'full'; for the temp
      ** database it is 'NONE'. This matches the pager layer defaults.
      */
      db.aDb[0].zName = "main";
      db.aDb[0].safety_level = 3;
      db.aDb[1].zName = "temp";
      db.aDb[1].safety_level = 1;

      db.magic = SQLITE_MAGIC_OPEN;
      //if ( db.mallocFailed != 0 )
      //{
      //  goto opendb_out;
      //}

      /* Register all built-in functions, but do not attempt to read the
      ** database schema yet. This is delayed until the first time the database
      ** is accessed.
      */
      sqlite3Error( db, SQLITE_OK, 0 );
      sqlite3RegisterBuiltinFunctions( db );

      /* Load automatic extensions - extensions that have been registered
      ** using the sqlite3_automatic_extension() API.
      */
      sqlite3AutoLoadExtensions( db );
      rc = sqlite3_errcode( db );
      if ( rc != SQLITE_OK )
      {
        goto opendb_out;
      }


#if ENABLE_FTS1
if( 0==db.mallocFailed ){
extern int sqlite3Fts1Init(sqlite3);
rc = sqlite3Fts1Init(db);
}
#endif

#if ENABLE_FTS2
if( 0==db.mallocFailed && rc==SQLITE_OK ){
extern int sqlite3Fts2Init(sqlite3);
rc = sqlite3Fts2Init(db);
}
#endif

#if ENABLE_FTS3
if( 0==db.mallocFailed && rc==SQLITE_OK ){
rc = sqlite3Fts3Init(db);
}
#endif

#if ENABLE_ICU
if( 0==db.mallocFailed && rc==SQLITE_OK ){
extern int sqlite3IcuInit(sqlite3);
rc = sqlite3IcuInit(db);
}
#endif

#if ENABLE_RTREE
if( 0==db.mallocFailed && rc==SQLITE_OK){
rc = sqlite3RtreeInit(db);
}
#endif

      sqlite3Error( db, rc, 0 );

      /* -DSQLITE_DEFAULT_LOCKING_MODE=1 makes EXCLUSIVE the default locking
      ** mode.  -DSQLITE_DEFAULT_LOCKING_MODE=0 make NORMAL the default locking
      ** mode.  Doing nothing at all also makes NORMAL the default.
      */
#if DEFAULT_LOCKING_MODE
db.dfltLockMode = SQLITE_DEFAULT_LOCKING_MODE;
sqlite3PagerLockingMode(sqlite3BtreePager(db.aDb[0].pBt),
SQLITE_DEFAULT_LOCKING_MODE);
#endif

      /* Enable the lookaside-malloc subsystem */
      setupLookaside( db, null, SysEx_GlobalStatics.szLookaside,
      SysEx_GlobalStatics.nLookaside );

      sqlite3_wal_autocheckpoint( db, SQLITE_DEFAULT_WAL_AUTOCHECKPOINT );

opendb_out:
      //sqlite3_free(zOpen);
      if ( db != null )
      {
        Debug.Assert( db.mutex != null || isThreadsafe == 0 || !SysEx_GlobalStatics.bFullMutex );
        sqlite3_mutex_leave( db.mutex );
      }
      rc = sqlite3_errcode( db );
      if ( rc == SQLITE_NOMEM )
      {
        sqlite3_close( db );
        db = null;
      }
      else if ( rc != SQLITE_OK )
      {
        db.magic = SQLITE_MAGIC_SICK;
      }
      ppDb = db;
      return sqlite3ApiExit( 0, rc );
    }

    /*
    ** Open a new database handle.
    */
    static public int sqlite3_open(
    string zFilename,
    out sqlite3 ppDb
    )
    {
      return openDatabase( zFilename, out ppDb,
      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, null );
    }

    static public int sqlite3_open_v2(
    string filename,   /* Database filename (UTF-8) */
    out sqlite3 ppDb,  /* OUT: SQLite db handle */
    int flags,         /* Flags */
    string zVfs        /* Name of VFS module to use */
    )
    {
      return openDatabase( filename, out ppDb, flags, zVfs );
    }

#if !OMIT_UTF16
int sqlite3_open16(
string zFilename,
sqlite3 **ppDb
){
char const *zFilename8;   /* zFilename encoded in UTF-8 instead of UTF-16 */
sqlite3_value pVal;
int rc;

Debug.Assert(zFilename );
Debug.Assert(ppDb );
*ppDb = 0;
#if !OMIT_AUTOINIT
rc = sqlite3_initialize();
if( rc !=0) return rc;
#endif
pVal = sqlite3ValueNew(0);
sqlite3ValueSetStr(pVal, -1, zFilename, SQLITE_UTF16NATIVE, SQLITE_STATIC);
zFilename8 = sqlite3ValueText(pVal, SQLITE_UTF8);
if( zFilename8 ){
rc = openDatabase(zFilename8, ppDb,
SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, 0);
Debug.Assert(*ppDb || rc==SQLITE_NOMEM );
if( rc==SQLITE_OK && !DbHasProperty(*ppDb, 0, DB_SchemaLoaded) ){
ENC(*ppDb) = SQLITE_UTF16NATIVE;
}
}else{
rc = SQLITE_NOMEM;
}
sqlite3ValueFree(pVal);
return sqlite3ApiExit(0, rc);
}
#endif 

    static int sqlite3_create_collation(
    sqlite3 db,
    string zName,
    int enc,
    object pCtx,
    dxCompare xCompare
    )
    {
      int rc;
      sqlite3_mutex_enter( db.mutex );
      //Debug.Assert( 0 == db.mallocFailed );
      rc = createCollation( db, zName, (u8)enc, SQLITE_COLL_USER, pCtx, xCompare, null );
      rc = sqlite3ApiExit( db, rc );
      sqlite3_mutex_leave( db.mutex );
      return rc;
    }

    /*
    ** Register a new collation sequence with the database handle db.
    */
    static int sqlite3_create_collation_v2(
    sqlite3 db,
    string zName,
    int enc,
    object pCtx,
    dxCompare xCompare, //int(*xCompare)(void*,int,const void*,int,const void),
    dxDelCollSeq xDel  //void(*xDel)(void)
    )
    {
      int rc;
      sqlite3_mutex_enter( db.mutex );
      //Debug.Assert( 0 == db.mallocFailed );
      rc = createCollation( db, zName, (u8)enc, SQLITE_COLL_USER, pCtx, xCompare, xDel );
      rc = sqlite3ApiExit( db, rc );
      sqlite3_mutex_leave( db.mutex );
      return rc;
    }

#if !SQLITE_OMIT_UTF16

//int sqlite3_create_collation16(
//  sqlite3* db,
//  string zName,
//  int enc,
//  void* pCtx,
//  int(*xCompare)(void*,int,const void*,int,const void)
//){
//  int rc = SQLITE_OK;
//  string zName8;
//  sqlite3_mutex_enter(db.mutex);
//  Debug.Assert( 0==db.mallocFailed );
//  zName8 = sqlite3Utf16to8(db, zName, -1, SQLITE_UTF16NATIVE);
//  if( zName8 ){
//    rc = createCollation(db, zName8, (u8)enc, SQLITE_COLL_USER, pCtx, xCompare, 0);
//    sqlite3DbFree(db,ref zName8);
//  }
//  rc = sqlite3ApiExit(db, rc);
//  sqlite3_mutex_leave(db.mutex);
//  return rc;
//}
#endif


    static int sqlite3_collation_needed(
    sqlite3 db,
    object pCollNeededArg,
    dxCollNeeded xCollNeeded
    )
    {
      sqlite3_mutex_enter( db.mutex );
      db.xCollNeeded = xCollNeeded;
      db.xCollNeeded16 = null;
      db.pCollNeededArg = pCollNeededArg;
      sqlite3_mutex_leave( db.mutex );
      return SQLITE_OK;
    }

#if !SQLITE_OMIT_UTF16
//int sqlite3_collation_needed16(
//  sqlite3 db,
//  void pCollNeededArg,
//  void(*xCollNeeded16)(void*,sqlite3*,int eTextRep,const void)
//){
//  sqlite3_mutex_enter(db.mutex);
//  db.xCollNeeded = 0;
//  db.xCollNeeded16 = xCollNeeded16;
//  db.pCollNeededArg = pCollNeededArg;
//  sqlite3_mutex_leave(db.mutex);
//  return SQLITE_OK;
//}
#endif



    /*
** Test to see whether or not the database connection is in autocommit
** mode.  Return TRUE if it is and FALSE if not.  Autocommit mode is on
** by default.  Autocommit is disabled by a BEGIN statement and reenabled
** by the next COMMIT or ROLLBACK.
**
******* THIS IS AN EXPERIMENTAL API AND IS SUBJECT TO CHANGE ******
*/
    static u8 sqlite3_get_autocommit( sqlite3 db )
    {
      return db.autoCommit;
    }

    /*
    ** The following routines are subtitutes for constants SQLITE_CORRUPT,
    ** SQLITE_MISUSE, SQLITE_CANTOPEN, SQLITE_IOERR and possibly other error
    ** constants.  They server two purposes:
    **
    **   1.  Serve as a convenient place to set a breakpoint in a debugger
    **       to detect when version error conditions occurs.
    **
    **   2.  Invoke sqlite3_log() to provide the source code location where
    **       a low-level error is first detected.
    */
    static int sqlite3CorruptError( int lineno )
    {
      testcase( SysEx_GlobalStatics.xLog != null );
      sqlite3_log( SQLITE_CORRUPT,
      "database corruption at line %d of [%.10s]",
      lineno, 20 + sqlite3_sourceid() );
      return SQLITE_CORRUPT;
    }
    static int sqlite3MisuseError( int lineno )
    {
      testcase( SysEx_GlobalStatics.xLog != null );
      sqlite3_log( SQLITE_MISUSE,
      "misuse at line %d of [%.10s]",
      lineno, 20 + sqlite3_sourceid() );
      return SQLITE_MISUSE;
    }
    static int sqlite3CantopenError( int lineno )
    {
      testcase( SysEx_GlobalStatics.xLog != null );
      sqlite3_log( SQLITE_CANTOPEN,
      "cannot open file at line %d of [%.10s]",
      lineno, 20 + sqlite3_sourceid() );
      return SQLITE_CANTOPEN;
    }


    /*
** Return meta information about a specific column of a database table.
** See comment in sqlite3.h (sqlite.h.in) for details.
*/
#if SQLITE_ENABLE_COLUMN_METADATA

    static int sqlite3_table_column_metadata(
    sqlite3 db,            /* Connection handle */
    string zDbName,        /* Database name or NULL */
    string zTableName,     /* Table name */
    string zColumnName,    /* Column name */
    ref string pzDataType, /* OUTPUT: Declared data type */
    ref string pzCollSeq,  /* OUTPUT: Collation sequence name */
    ref int pNotNull,      /* OUTPUT: True if NOT NULL constraint exists */
    ref int pPrimaryKey,   /* OUTPUT: True if column part of PK */
    ref int pAutoinc       /* OUTPUT: True if column is auto-increment */
    )
    {
      int rc;
      string zErrMsg = "";
      Table pTab = null;
      Column pCol = null;
      int iCol;

      string zDataType = null;
      string zCollSeq = null;
      int notnull = 0;
      int primarykey = 0;
      int autoinc = 0;

      /* Ensure the database schema has been loaded */
      sqlite3_mutex_enter( db.mutex );
      sqlite3BtreeEnterAll( db );
      rc = sqlite3Init( db, ref zErrMsg );
      if ( SQLITE_OK != rc )
      {
        goto error_out;
      }

      /* Locate the table in question */
      pTab = sqlite3FindTable( db, zTableName, zDbName );
      if ( null == pTab || pTab.pSelect != null )
      {
        pTab = null;
        goto error_out;
      }

      /* Find the column for which info is requested */
      if ( sqlite3IsRowid( zColumnName ) )
      {
        iCol = pTab.iPKey;
        if ( iCol >= 0 )
        {
          pCol = pTab.aCol[iCol];
        }
      }
      else
      {
        for ( iCol = 0; iCol < pTab.nCol; iCol++ )
        {
          pCol = pTab.aCol[iCol];
          if ( pCol.zName.Equals( zColumnName, StringComparison.InvariantCultureIgnoreCase ) )
          {
            break;
          }
        }
        if ( iCol == pTab.nCol )
        {
          pTab = null;
          goto error_out;
        }
      }

      /* The following block stores the meta information that will be returned
      ** to the caller in local variables zDataType, zCollSeq, notnull, primarykey
      ** and autoinc. At this point there are two possibilities:
      **
      **     1. The specified column name was rowid", "oid" or "_rowid_"
      **        and there is no explicitly declared IPK column.
      **
      **     2. The table is not a view and the column name identified an
      **        explicitly declared column. Copy meta information from pCol.
      */
      if ( pCol != null )
      {
        zDataType = pCol.zType;
        zCollSeq = pCol.zColl;
        notnull = pCol.notNull != 0 ? 1 : 0;
        primarykey = pCol.isPrimKey != 0 ? 1 : 0;
        autoinc = ( pTab.iPKey == iCol && ( pTab.tabFlags & TF_Autoincrement ) != 0 ) ? 1 : 0;
      }
      else
      {
        zDataType = "INTEGER";
        primarykey = 1;
      }
      if ( String.IsNullOrEmpty( zCollSeq ) )
      {
        zCollSeq = "BINARY";
      }

error_out:
      sqlite3BtreeLeaveAll( db );

      /* Whether the function call succeeded or failed, set the output parameters
      ** to whatever their local counterparts contain. If an error did occur,
      ** this has the effect of zeroing all output parameters.
      */
      //if ( pzDataType )
      pzDataType = zDataType;
      //if ( pzCollSeq )
      pzCollSeq = zCollSeq;
      //if ( pNotNull )
      pNotNull = notnull;
      //if ( pPrimaryKey )
      pPrimaryKey = primarykey;
      //if ( pAutoinc )
      pAutoinc = autoinc;

      if ( SQLITE_OK == rc && null == pTab )
      {
        sqlite3DbFree( db, ref zErrMsg );
        zErrMsg = sqlite3MPrintf( db, "no such table column: %s.%s", zTableName,
        zColumnName );
        rc = SQLITE_ERROR;
      }
      sqlite3Error( db, rc, ( !String.IsNullOrEmpty( zErrMsg ) ? "%s" : null ), zErrMsg );
      sqlite3DbFree( db, ref zErrMsg );
      rc = sqlite3ApiExit( db, rc );
      sqlite3_mutex_leave( db.mutex );
      return rc;
    }
#endif

    /*
** Sleep for a little while.  Return the amount of time slept.
*/
    static public int sqlite3_sleep( int ms )
    {
      sqlite3_vfs pVfs;
      int rc;
      pVfs = sqlite3_vfs_find( null );
      if ( pVfs == null )
        return 0;

      /* This function works in milliseconds, but the underlying OsSleep()
      ** API uses microseconds. Hence the 1000's.
      */
      rc = ( sqlite3OsSleep( pVfs, 1000 * ms ) / 1000 );
      return rc;
    }

    /*
    ** Enable or disable the extended result codes.
    */
    static int sqlite3_extended_result_codes( sqlite3 db, bool onoff )
    {
      sqlite3_mutex_enter( db.mutex );
      db.errMask = (int)( onoff ? 0xffffffff : 0xff );
      sqlite3_mutex_leave( db.mutex );
      return SQLITE_OK;
    }

    /*
    ** Invoke the xFileControl method on a particular database.
    */
    static int sqlite3_file_control( sqlite3 db, string zDbName, int op, ref sqlite3_int64 pArg )
    {
      int rc = SQLITE_ERROR;
      int iDb;
      sqlite3_mutex_enter( db.mutex );
      if ( zDbName == null )
      {
        iDb = 0;
      }
      else
      {
        for ( iDb = 0; iDb < db.nDb; iDb++ )
        {
          if ( db.aDb[iDb].zName == zDbName )
            break;
        }
      }
      if ( iDb < db.nDb )
      {
        Btree pBtree = db.aDb[iDb].pBt;
        if ( pBtree != null )
        {
          Pager pPager;
          sqlite3_file fd;
          sqlite3BtreeEnter( pBtree );
          pPager = sqlite3BtreePager( pBtree );
          Debug.Assert( pPager != null );
          fd = sqlite3PagerFile( pPager );
          Debug.Assert( fd != null );
          if ( op == SQLITE_FCNTL_FILE_POINTER )
          {
#if (SQLITE_SILVERLIGHT || WINDOWS_MOBILE)
              pArg = (long)-1; // not supported
#else
            pArg = (long)fd.fs.Handle;
#endif
            rc = SQLITE_OK;
          }
          else if ( fd.pMethods != null )
          {
            rc = sqlite3OsFileControl( fd, (u32)op, ref pArg );
          }
          else
          {
            rc = SQLITE_NOTFOUND;
          }
          sqlite3BtreeLeave( pBtree );
        }
      }
      sqlite3_mutex_leave( db.mutex );
      return rc;
    }

    /*
    ** Interface to the testing logic.
    */
    static int sqlite3_test_control( int op, params object[] ap )
    {
      int rc = 0;
#if !SQLITE_OMIT_BUILTIN_TEST
      //  va_list ap;
      lock ( lock_va_list )
      {
        va_start( ap, "op" );
        switch ( op )
        {

          /*
          ** Save the current state of the PRNG.
          */
          case SQLITE_TESTCTRL_PRNG_SAVE:
            {
              sqlite3PrngSaveState();
              break;
            }

          /*
          ** Restore the state of the PRNG to the last state saved using
          ** PRNG_SAVE.  If PRNG_SAVE has never before been called, then
          ** this verb acts like PRNG_RESET.
          */
          case SQLITE_TESTCTRL_PRNG_RESTORE:
            {
              sqlite3PrngRestoreState();
              break;
            }

          /*
          ** Reset the PRNG back to its uninitialized state.  The next call
          ** to sqlite3_randomness() will reseed the PRNG using a single call
          ** to the xRandomness method of the default VFS.
          */
          case SQLITE_TESTCTRL_PRNG_RESET:
            {
              sqlite3PrngResetState();
              break;
            }

          /*
          **  sqlite3_test_control(BITVEC_TEST, size, program)
          **
          ** Run a test against a Bitvec object of size.  The program argument
          ** is an array of integers that defines the test.  Return -1 on a
          ** memory allocation error, 0 on success, or non-zero for an error.
          ** See the sqlite3BitvecBuiltinTest() for additional information.
          */
          case SQLITE_TESTCTRL_BITVEC_TEST:
            {
              int sz = va_arg( ap, ( Int32 ) 0 );
              int[] aProg = va_arg( ap, (Int32[])null );
              rc = sqlite3BitvecBuiltinTest( (u32)sz, aProg );
              break;
            }

          /*
          **  sqlite3_test_control(BENIGN_MALLOC_HOOKS, xBegin, xEnd)
          **
          ** Register hooks to call to indicate which malloc() failures
          ** are benign.
          */
          case SQLITE_TESTCTRL_BENIGN_MALLOC_HOOKS:
            {
              //typedef void (*void_function)(void);
              void_function xBenignBegin;
              void_function xBenignEnd;
              xBenignBegin = va_arg( ap, (void_function)null );
              xBenignEnd = va_arg( ap, (void_function)null );
              sqlite3BenignMallocHooks( xBenignBegin, xBenignEnd );
              break;
            }
          /*
          **  sqlite3_test_control(SQLITE_TESTCTRL_PENDING_BYTE, unsigned int X)
          **
          ** Set the PENDING byte to the value in the argument, if X>0.
          ** Make no changes if X==0.  Return the value of the pending byte
          ** as it existing before this routine was called.
          **
          ** IMPORTANT:  Changing the PENDING byte from 0x40000000 results in
          ** an incompatible database file format.  Changing the PENDING byte
          ** while any database connection is open results in undefined and
          ** dileterious behavior.
          */
          case SQLITE_TESTCTRL_PENDING_BYTE:
            {
              rc = PENDING_BYTE;
#if !SQLITE_OMIT_WSD
              {
                u32 newVal = va_arg( ap, (UInt32)0 );
                if ( newVal != 0 )
                {
                  if ( sqlite3PendingByte != newVal )
                    sqlite3PendingByte = (int)newVal;
#if DEBUG && TCLSH
                  TCLsqlite3PendingByte.iValue = sqlite3PendingByte;
#endif
                  PENDING_BYTE = sqlite3PendingByte;
                }
              }
#endif
              break;
            }

          /*
          **  sqlite3_test_control(SQLITE_TESTCTRL_ASSERT, int X)
          **
          ** This action provides a run-time test to see whether or not
          ** Debug.Assert() was enabled at compile-time.  If X is true and Debug.Assert()
          ** is enabled, then the return value is true.  If X is true and
          ** Debug.Assert() is disabled, then the return value is zero.  If X is
          ** false and Debug.Assert() is enabled, then the assertion fires and the
          ** process aborts.  If X is false and Debug.Assert() is disabled, then the
          ** return value is zero.
          */
          case SQLITE_TESTCTRL_ASSERT:
            {
              int x = 0;
              Debug.Assert( ( x = va_arg( ap, ( Int32 ) 0 ) ) != 0 );
              rc = x;
              break;
            }


          /*
          **  sqlite3_test_control(SQLITE_TESTCTRL_ALWAYS, int X)
          **
          ** This action provides a run-time test to see how the ALWAYS and
          ** NEVER macros were defined at compile-time.
          **
          ** The return value is ALWAYS(X).
          **
          ** The recommended test is X==2.  If the return value is 2, that means
          ** ALWAYS() and NEVER() are both no-op pass-through macros, which is the
          ** default setting.  If the return value is 1, then ALWAYS() is either
          ** hard-coded to true or else it asserts if its argument is false.
          ** The first behavior (hard-coded to true) is the case if
          ** SQLITE_TESTCTRL_ASSERT shows that Debug.Assert() is disabled and the second
          ** behavior (assert if the argument to ALWAYS() is false) is the case if
          ** SQLITE_TESTCTRL_ASSERT shows that Debug.Assert() is enabled.
          **
          ** The run-time test procedure might look something like this:
          **
          **    if( sqlite3_test_control(SQLITE_TESTCTRL_ALWAYS, 2)==2 ){
          **      // ALWAYS() and NEVER() are no-op pass-through macros
          **    }else if( sqlite3_test_control(SQLITE_TESTCTRL_ASSERT, 1) ){
          **      // ALWAYS(x) asserts that x is true. NEVER(x) asserts x is false.
          **    }else{
          **      // ALWAYS(x) is a constant 1.  NEVER(x) is a constant 0.
          **    }
          */
          case SQLITE_TESTCTRL_ALWAYS:
            {
              int x = va_arg( ap, ( Int32 ) 0 );
              rc = ALWAYS( x );
              break;
            }

          /*   sqlite3_test_control(SQLITE_TESTCTRL_RESERVE, sqlite3 db, int N)
          **
          ** Set the nReserve size to N for the main database on the database
          ** connection db.
          */
          case SQLITE_TESTCTRL_RESERVE:
            {
              sqlite3 db = va_arg( ap, (sqlite3)null );
              int x = va_arg( ap, ( Int32 ) 0 );
              sqlite3_mutex_enter( db.mutex );
              sqlite3BtreeSetPageSize( db.aDb[0].pBt, 0, x, 0 );
              sqlite3_mutex_leave( db.mutex );
              break;
            }

          /*  sqlite3_test_control(SQLITE_TESTCTRL_OPTIMIZATIONS, sqlite3 db, int N)
          **
          ** Enable or disable various optimizations for testing purposes.  The 
          ** argument N is a bitmask of optimizations to be disabled.  For normal
          ** operation N should be 0.  The idea is that a test program (like the
          ** SQL Logic Test or SLT test module) can run the same SQL multiple times
          ** with various optimizations disabled to verify that the same answer
          ** is obtained in every case.
          */
          case SQLITE_TESTCTRL_OPTIMIZATIONS:
            {
              sqlite3 db = va_arg( ap, (sqlite3)null );//sqlite3 db = va_arg(ap, sqlite3);
              int x = va_arg( ap, (Int32)0 );//int x = va_arg(ap,int);
              db.flags = ( x & SQLITE_OptMask ) | ( db.flags & ~SQLITE_OptMask );
              break;
            }

          //#if SQLITE_N_KEYWORD
          /* sqlite3_test_control(SQLITE_TESTCTRL_ISKEYWORD, const string zWord)
          **
          ** If zWord is a keyword recognized by the parser, then return the
          ** number of keywords.  Or if zWord is not a keyword, return 0.
          ** 
          ** This test feature is only available in the amalgamation since
          ** the SQLITE_N_KEYWORD macro is not defined in this file if SQLite
          ** is built using separate source files.
          */
          case SQLITE_TESTCTRL_ISKEYWORD:
            {
              string zWord = (string)va_arg( ap, "char*" );
              int n = sqlite3Strlen30( zWord );
              rc = ( sqlite3KeywordCode( zWord, n ) != TK_ID ) ? SQLITE_N_KEYWORD : 0;
              break;
            }
          //#endif 
          /* sqlite3_test_control(SQLITE_TESTCTRL_PGHDRSZ)
          **
          ** Return the size of a pcache header in bytes.
          */
          case SQLITE_TESTCTRL_PGHDRSZ:
            {
              rc = -1;// sizeof(PgHdr);
              break;
            }

          /* sqlite3_test_control(SQLITE_TESTCTRL_SCRATCHMALLOC, sz, &pNew, pFree);
          **
          ** Pass pFree into sqlite3ScratchFree(). 
          ** If sz>0 then allocate a scratch buffer into pNew.  
          */
          case SQLITE_TESTCTRL_SCRATCHMALLOC:
            {
              //void pFree, *ppNew;
              //int sz;
              //sz = va_arg(ap, int);
              //ppNew = va_arg(ap, void*);
              //pFree = va_arg(ap, void);
              //if( sz ) *ppNew = sqlite3ScratchMalloc(sz);
              //sqlite3ScratchFree(pFree);
              break;
            }
    /*   sqlite3_test_control(SQLITE_TESTCTRL_LOCALTIME_FAULT, int onoff);
    **
    ** If parameter onoff is non-zero, configure the wrappers so that all
    ** subsequent calls to localtime() and variants fail. If onoff is zero,
    ** undo this setting.
    */
    case SQLITE_TESTCTRL_LOCALTIME_FAULT: {
      SysEx_GlobalStatics.bLocaltimeFault = va_arg( ap, (Boolean)true );
      break;
    }

        }
        va_end( ref ap );
      }
#endif
      return rc;
    }



static string sqlite3_uri_parameter(string zFilename, string zParam){
  Debugger.Break();
  //zFilename += sqlite3Strlen30(zFilename) + 1;
  //while( zFilename[0] ){
  //  int x = strcmp(zFilename, zParam);
  //  zFilename += sqlite3Strlen30(zFilename) + 1;
  //  if( x==0 ) return zFilename;
  //  zFilename += sqlite3Strlen30(zFilename) + 1;
  //}
  return null;
}

}

#endif
    }
}
