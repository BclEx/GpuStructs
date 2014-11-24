using System;
using System.Diagnostics;
using System.Text;
using System.Globalization;

namespace Core.Command
{
    public partial class Pragma
    {
        // 123456789 123456789
        static readonly string _safetyLevelText = "onoffalseyestruefull";
        static readonly int[] _safetyLevelOffset = new int[] { 0, 1, 2, 4, 9, 12, 16 };
        static readonly int[] _safetyLevelLength = new int[] { 2, 2, 3, 5, 3, 4, 4 };
        static readonly byte[] _safetyLevelValue = new byte[] { 1, 0, 0, 0, 1, 1, 2 };
        static byte GetSafetyLevel(string z, int omitFull, byte dflt)
        {
            if (char.IsDigit(z[0]))
                return (byte)ConvertEx.Atoi(z);
            int n = z.Length;
            for (int i = 0; i < _safetyLevelLength.Length - omitFull; i++)
                if (_safetyLevelLength[i] == n && string.CompareOrdinal(_safetyLevelText.Substring(_safetyLevelOffset[i]), 0, z, 0, n) == 0)
                    return _safetyLevelValue[i];
            return dflt;
        }

        public static byte GetBoolean(string z, byte dflt)
        {
            return (GetSafetyLevel(z, 1, dflt) != 0);
        }

#if !(OMIT_PRAGMA)

        static int GetLockingMode(string z)
        {
            if (z != null)
            {
                if (string.Equals(z, "exclusive", StringComparison.OrdinalIgnoreCase)) return PAGER_LOCKINGMODE_EXCLUSIVE;
                if (string.Equals(z, "normal", StringComparison.OrdinalIgnoreCase)) return PAGER_LOCKINGMODE_NORMAL;
            }
            return PAGER_LOCKINGMODE_QUERY;
        }

#if !OMIT_AUTOVACUUM
        static byte GetAutoVacuum(string z)
        {
            if (string.Equals(z, "none", StringComparison.OrdinalIgnoreCase)) return BTREE_AUTOVACUUM_NONE;
            if (string.Equals(z, "full", StringComparison.OrdinalIgnoreCase)) return BTREE_AUTOVACUUM_FULL;
            if (string.Equals(z, "incremental", StringComparison.OrdinalIgnoreCase)) return BTREE_AUTOVACUUM_INCR;
            int i = ConvertEx.Atoi(z);
            return (byte)(i >= 0 && i <= 2 ? i : 0);
        }
#endif

#if !OMIT_PAGER_PRAGMAS
        static int GetTempStore(string z)
        {
            if (z[0] >= '0' && z[0] <= '2') return z[0] - '0';
            else if (string.Equals(z, "file", StringComparison.OrdinalIgnoreCase)) return 1;
            else if (string.Equals(z, "memory", StringComparison.OrdinalIgnoreCase)) return 2;
            else return 0;
        }
#endif

#if !OMIT_PAGER_PRAGMAS
        static RC InvalidateTempStorage(Parse parse)
        {
            Context ctx = parse.Ctx;
            if (ctx.DBs[1].Bt != null)
            {
                if (!ctx.AutoCommit || ctx.DBs[1].Bt.IsInReadTrans())
                {
                    parse.ErrorMsg("temporary storage cannot be changed from within a transaction");
                    return RC.ERROR;
                }
                ctx.DBs[1].Bt.Close();
                ctx.DBs[1].Bt = null;
                sqlite3ResetInternalSchema(ctx, -1);
            }
            return RC.OK;
        }
#endif

#if !OMIT_PAGER_PRAGMAS
        static RC ChangeTempStorage(Parse parse, string storageType)
        {
            int ts = GetTempStore(storageType);
            Context ctx = parse.Ctx;
            if (ctx.TempStore == ts) return RC.OK;
            if (InvalidateTempStorage(parse) != RC.OK)
                return SQLITE_ERROR;
            ctx.TempStore = (byte)ts;
            return RC.OK;
        }
#endif

        static void ReturnSingleInt(Parse parse, string label, long value)
        {
            Vdbe v = parse.GetVdbe();
            int mem = ++parse.Mems;
            v.AddOp4(OP.Int64, 0, mem, 0, value, Vdbe.P4T.INT64);
            v.SetNumCols(1);
            v.SetColName(0, COLNAME_NAME, label, C.DESTRUCTOR_STATIC);
            v.AddOp2(OP.ResultRow, mem, 1);
        }

#if !OMIT_FLAG_PRAGMAS
        struct sPragmaType
        {
            public string Name; // Name of the pragma
            public int Mask; // Mask for the db.flags value
            public sPragmaType(string name, int mask)
            {
                Name = name;
                Mask = mask;
            }
        }
        static readonly sPragmaType[] _pragmas = new sPragmaType[]
        {
            new sPragmaType("full_column_names",        SQLITE_FullColNames),
            new sPragmaType("short_column_names",       SQLITE_ShortColNames),
            new sPragmaType("count_changes",            SQLITE_CountRows),
            new sPragmaType("empty_result_callbacks",   SQLITE_NullCallback),
            new sPragmaType("legacy_file_format",       SQLITE_LegacyFileFmt),
            new sPragmaType("fullfsync",                SQLITE_FullFSync),
            new sPragmaType("checkpoint_fullfsync",     SQLITE_CkptFullFSync),
            new sPragmaType("reverse_unordered_selects", SQLITE_ReverseOrder),
            #if !OMIT_AUTOMATIC_INDEX
            new sPragmaType("automatic_index",          SQLITE_AutoIndex),
            #endif
            #if DEBUG
            new sPragmaType("sql_trace",                SQLITE_SqlTrace),
            new sPragmaType("vdbe_listing",             SQLITE_VdbeListing),
            new sPragmaType("vdbe_trace",               SQLITE_VdbeTrace),
		    new sPragmaType("vdbe_addoptrace",          SQLITE_VdbeAddopTrace),
		    new sPragmaType("vdbe_debug",               SQLITE_SqlTrace|SQLITE_VdbeListing|SQLITE_VdbeTrace),
            #endif
            #if !OMIT_CHECK
            new sPragmaType("ignore_check_constraints", SQLITE_IgnoreChecks),
            #endif
            // The following is VERY experimental
            new sPragmaType("writable_schema",          SQLITE_WriteSchema|SQLITE_RecoveryMode),

            // TODO: Maybe it shouldn't be possible to change the ReadUncommitted flag if there are any active statements.
            new sPragmaType( "read_uncommitted",         SQLITE_ReadUncommitted),
            new sPragmaType( "recursive_triggers",       SQLITE_RecTriggers),

            // This flag may only be set if both foreign-key and trigger support are present in the build.
            #if !OMIT_FOREIGN_KEY && !OMIT_TRIGGER
            new sPragmaType("foreign_keys",             SQLITE_ForeignKeys),
            #endif
        };

        static bool FlagPragma(Parse parse, string left, string right)
        {
            int i;
            sPragmaType p;
            for (i = 0; i < _pragmas.Length; i++)
            {
                p = _pragmas[i];
                if (string.Equals(left, p.Name, StringComparison.OrdinalIgnoreCase))
                {
                    Context ctx = parse.Ctx;
                    Vdbe v = parse.GetVdbe();
                    Debug.Assert(v != null); // Already allocated by sqlite3Pragma()
                    if (C._ALWAYS(v))
                    {
                        if (right == null)
                            ReturnSingleInt(parse, p.Name, ((ctx.Flags & p.Mask) != 0) ? 1 : 0);
                        else
                        {
                            int mask = p.Mask; // Mask of bits to set or clear.
                            if (ctx.AutoCommit == 0)
                                mask &= ~(SQLITE_ForeignKeys); // Foreign key support may not be enabled or disabled while not in auto-commit mode.
                            if (GetBoolean(right, 0))
                                ctx.Flags |= mask;
                            else
                                ctx.Flags &= ~mask;
                            // Many of the flag-pragmas modify the code generated by the SQL compiler (eg. count_changes). So add an opcode to expire all
                            // compiled SQL statements after modifying a pragma value.
                            v.AddOp2(OP.Expire, 0, 0);
                        }
                    }
                    return true;
                }
            }
            return false;
        }
#endif

#if !OMIT_FOREIGN_KEY
        static string ActionName(OE action)
        {
            switch (action)
            {
                case OE.SetNull: return "SET NULL";
                case OE.SetDflt: return "SET DEFAULT";
                case OE.Cascade: return "CASCADE";
                case OE.Restrict: return "RESTRICT";
                default: Debug.Assert(action == OE.None); return "NO ACTION";
            }
        }
#endif

        static readonly string[] _modeNames = {
"delete", "persist", "off", "truncate", "memory"
#if !OMIT_WAL
, "wal"
#endif
};
        public static string JournalModename(IPager.JOURNALMODE mode)
        {
            Debug.Assert((int)IPager.JOURNALMODE.DELETE == 0);
            Debug.Assert((int)IPager.JOURNALMODE.PERSIST == 1);
            Debug.Assert((int)IPager.JOURNALMODE.OFF == 2);
            Debug.Assert((int)IPager.JOURNALMODE.TRUNCATE == 3);
            Debug.Assert((int)IPager.JOURNALMODE.JMEMORY == 4);
            Debug.Assert((int)IPager.JOURNALMODE.WAL == 5);
            Debug.Assert(mode >= 0 && (int)mode <= _modeNames.Length);
            if ((int)mode == _modeNames.Length) return null;
            return _modeNames[(int)mode];
        }

        class EncName
        {
            public string Name;
            public TEXTENCODE Encode;
            public EncName(string name, TEXTENCODE encode) { Name = name; Encode = encode; }
        };
        static EncName[] _encnames = new EncName[]
        {
            new EncName("UTF8",     TEXTENCODE.UTF8),
            new EncName("UTF-8",    TEXTENCODE.UTF8), // Must be element [1]
            new EncName("UTF-16le (not supported)", TEXTENCODE.UTF16LE), // Must be element [2]
            new EncName("UTF-16be (not supported)", TEXTENCODE.UTF16BE), // Must be element [3]
            new EncName("UTF16le (not supported)",  TEXTENCODE.UTF16LE),
            new EncName("UTF16be (not supported)",  TEXTENCODE.UTF16BE),
            new EncName("UTF-16 (not supported)",   0), // SQLITE_UTF16NATIVE
            new EncName("UTF16",    0), // SQLITE_UTF16NATIVE
            new EncName(null, 0 )
        };

        static readonly Vdbe.VdbeOpList[] _getCacheSize = new Vdbe.VdbeOpList[]
        {
            new Vdbe.VdbeOpList(OP.Transaction, 0, 0,        0),                         // 0
            new Vdbe.VdbeOpList(OP.ReadCookie,  0, 1,        BTREE_DEFAULT_CACHE_SIZE),  // 1
            new Vdbe.VdbeOpList(OP.IfPos,       1, 7,        0),
            new Vdbe.VdbeOpList(OP.Integer,     0, 2,        0),
            new Vdbe.VdbeOpList(OP.Subtract,    1, 2,        1),
            new Vdbe.VdbeOpList(OP.IfPos,       1, 7,        0),
            new Vdbe.VdbeOpList(OP.Integer,     0, 1,        0),  // 6
            new Vdbe.VdbeOpList(OP.ResultRow,   1, 1,        0),
        };

        // OVERLOADS, so I don't need to rewrite parse.c
        public static void Pragma_(Parse parse, Token id1, Token id2, int null4, bool minusFlag) { Pragma_(parse, id1, id2, null, minusFlag); }
        public static void Pragma_(Parse parse, Token id1, Token id2, Token value, bool minusFlag)
        {
            Context ctx = parse.Ctx;
            Vdbe v = parse.V = Vdbe.Create(ctx);
            if (v == null) return;
            v.RunOnlyOnce();
            parse.Mems = 2;

            // Interpret the [database.] part of the pragma statement. iDb is the index of the database this pragma is being applied to in db.aDb[].
            Token id = new Token(); // Pointer to <id> token
            int db = parse.TwoPartName(id1, id2, ref id); // Database index for <database>
            if (db < 0) return;
            Context.DB dbAsObj = ctx.DBs[db]; // The specific database being pragmaed

            // If the temp database has been explicitly named as part of the pragma, make sure it is open.
            if (db == 1 && parse.OpenTempDatabase() != 0)
                return;

            string left = Parse.NameFromToken(ctx, id); // Nul-terminated UTF-8 string <id>
            if (string.IsNullOrEmpty(left)) return;
            string right = (minusFlag ? C._mtagprintf(ctx, "-%T", value) : Parse.NameFromToken(ctx, value)); // Nul-terminated UTF-8 string <value>, or NULL

            Debug.Assert(id2 != null);
            string dbName = (dbName = id2.length > 0 ? dbAsObj.Name : null); // The database name
#if !OMIT_AUTHORIZATION
            if (Auth.Check(parse, AUTH.PRAGMA, left, right, dbName) != ARC.OK)
                goto pragma_out;
#endif
#if !OMIT_PAGER_PRAGMAS
            //  PRAGMA [database.]default_cache_size
            //  PRAGMA [database.]default_cache_size=N
            //
            // The first form reports the current persistent setting for the page cache size.  The value returned is the maximum number of
            // pages in the page cache.  The second form sets both the current page cache size value and the persistent page cache size value
            // stored in the database file.
            //
            // Older versions of SQLite would set the default cache size to a negative number to indicate synchronous=OFF.  These days, synchronous
            // is always on by default regardless of the sign of the default cache size.  But continue to take the absolute value of the default cache
            // size of historical compatibility.
            if (string.Equals(left, "default_cache_size", StringComparison.OrdinalIgnoreCase))
            {
                if (sqlite3ReadSchema(parse) != 0) goto pragma_out;
                v.UsesBtree(db);
                if (right == null)
                {
                    v.SetNumCols(1);
                    v.SetColName(0, COLNAME_NAME, "cache_size", C.DESTRUCTOR_STATIC);
                    parse.Mems += 2;
                    int addr = v.AddOpList(_getCacheSize.Length, _getCacheSize);
                    v.ChangeP1(addr, db);
                    v.ChangeP1(addr + 1, db);
                    v.ChangeP1(addr + 6, SQLITE_DEFAULT_CACHE_SIZE);
                }
                else
                {
                    int size = ConvertEx.AbsInt32(ConvertEx.Atoi(right));
                    parse.BeginWriteOperation(parse, 0, db);
                    v.AddOp2(OP.Integer, size, 1);
                    v.AddOp3(OP.SetCookie, db, BTREE_DEFAULT_CACHE_SIZE, 1);
                    Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
                    dbAsObj.Schema.CacheSize = size;
                    dbAsObj.Bt.SetCacheSize(dbAsObj.Schema.CacheSize);
                }
            }
            else
                //  PRAGMA [database.]page_size
                //  PRAGMA [database.]page_size=N
                //
                // The first form reports the current setting for the database page size in bytes.  The second form sets the
                // database page size value.  The value can only be set if the database has not yet been created.
                if (string.Equals(left, "page_size", StringComparison.OrdinalIgnoreCase))
                {
                    Btree bt = dbAsObj.Bt;
                    Debug.Assert(bt != null);
                    if (right == null)
                    {
                        int size = (C._ALWAYS(bt) ? bt.GetPageSize() : 0);
                        ReturnSingleInt(parse, "page_size", size);
                    }
                    else
                    {
                        // Malloc may fail when setting the page-size, as there is an internal buffer that the pager module resizes using sqlite3_realloc().
                        ctx.NextPagesize = ConvertEx.Atoi(right);
                        if (bt.SetPageSize(ctx.NextPagesize, -1, false) == RC.NOMEM)
                            ctx.MallocFailed = true;
                    }
                }
                else

                    //  PRAGMA [database.]secure_delete
                    //  PRAGMA [database.]secure_delete=ON/OFF
                    //
                    // The first form reports the current setting for the secure_delete flag.  The second form changes the secure_delete
                    // flag setting and reports thenew value.
                    if (string.Equals(left, "secure_delete", StringComparison.OrdinalIgnoreCase))
                    {
                        Btree bt = dbAsObj.Bt;
                        Debug.Assert(bt != null);
                        int b = (right != null ? GetBoolean(right, 0) : -1);
                        if (id2.length == 0 && b >= 0)
                            for (int ii = 0; ii < ctx.DBs.length; ii++)
                                ctx.DBs[ii].Bt.SecureDelete(b);
                        b = bt.SecureDelete(b);
                        ReturnSingleInt(parse, "secure_delete", b);
                    }
                    else
                        /*
                        **  PRAGMA [database.]max_page_count
                        **  PRAGMA [database.]max_page_count=N
                        **
                        ** The first form reports the current setting for the
                        ** maximum number of pages in the database file.  The 
                        ** second form attempts to change this setting.  Both
                        ** forms return the current setting.
                        **
                        **  PRAGMA [database.]page_count
                        **
                        ** Return the number of pages in the specified database.
                        */
                        if (left.Equals("page_count", StringComparison.OrdinalIgnoreCase)
                        || left.Equals("max_page_count", StringComparison.OrdinalIgnoreCase)
                        )
                        {
                            int iReg;
                            if (sqlite3ReadSchema(parse) != 0)
                                goto pragma_out;
                            sqlite3CodeVerifySchema(parse, db);
                            iReg = ++parse.nMem;
                            if (left[0] == 'p')
                            {
                                sqlite3VdbeAddOp2(v, OP_Pagecount, db, iReg);
                            }
                            else
                            {
                                sqlite3VdbeAddOp3(v, OP_MaxPgcnt, db, iReg, sqlite3Atoi(right));
                            }
                            sqlite3VdbeAddOp2(v, OP_ResultRow, iReg, 1);
                            sqlite3VdbeSetNumCols(v, 1);
                            sqlite3VdbeSetColName(v, 0, COLNAME_NAME, left, SQLITE_TRANSIENT);
                        }
                        else

                            /*
                            **  PRAGMA [database.]page_count
                            **
                            ** Return the number of pages in the specified database.
                            */
                            if (left.Equals("page_count", StringComparison.OrdinalIgnoreCase))
                            {
                                Vdbe _v;
                                int iReg;
                                _v = sqlite3GetVdbe(parse);
                                if (_v == null || sqlite3ReadSchema(parse) != 0)
                                    goto pragma_out;
                                sqlite3CodeVerifySchema(parse, db);
                                iReg = ++parse.nMem;
                                sqlite3VdbeAddOp2(_v, OP_Pagecount, db, iReg);
                                sqlite3VdbeAddOp2(_v, OP_ResultRow, iReg, 1);
                                sqlite3VdbeSetNumCols(_v, 1);
                                sqlite3VdbeSetColName(_v, 0, COLNAME_NAME, "page_count", SQLITE_STATIC);
                            }
                            else

                                /*
                                **  PRAGMA [database.]locking_mode
                                **  PRAGMA [database.]locking_mode = (normal|exclusive)
                                */
                                if (left.Equals("locking_mode", StringComparison.OrdinalIgnoreCase))
                                {
                                    string zRet = "normal";
                                    int eMode = GetLockingMode(right);

                                    if (id2.n == 0 && eMode == PAGER_LOCKINGMODE_QUERY)
                                    {
                                        /* Simple "PRAGMA locking_mode;" statement. This is a query for
                                        ** the current default locking mode (which may be different to
                                        ** the locking-mode of the main database).
                                        */
                                        eMode = ctx.dfltLockMode;
                                    }
                                    else
                                    {
                                        Pager pPager;
                                        if (id2.n == 0)
                                        {
                                            /* This indicates that no database name was specified as part
                                            ** of the PRAGMA command. In this case the locking-mode must be
                                            ** set on all attached databases, as well as the main db file.
                                            **
                                            ** Also, the sqlite3.dfltLockMode variable is set so that
                                            ** any subsequently attached databases also use the specified
                                            ** locking mode.
                                            */
                                            int ii;
                                            Debug.Assert(dbAsObj == ctx.aDb[0]);
                                            for (ii = 2; ii < ctx.nDb; ii++)
                                            {
                                                pPager = sqlite3BtreePager(ctx.aDb[ii].pBt);
                                                sqlite3PagerLockingMode(pPager, eMode);
                                            }
                                            ctx.dfltLockMode = (u8)eMode;
                                        }
                                        pPager = sqlite3BtreePager(dbAsObj.pBt);
                                        eMode = sqlite3PagerLockingMode(pPager, eMode) ? 1 : 0;
                                    }

                                    Debug.Assert(eMode == PAGER_LOCKINGMODE_NORMAL || eMode == PAGER_LOCKINGMODE_EXCLUSIVE);
                                    if (eMode == PAGER_LOCKINGMODE_EXCLUSIVE)
                                    {
                                        zRet = "exclusive";
                                    }
                                    sqlite3VdbeSetNumCols(v, 1);
                                    sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "locking_mode", SQLITE_STATIC);
                                    sqlite3VdbeAddOp4(v, OP_String8, 0, 1, 0, zRet, 0);
                                    sqlite3VdbeAddOp2(v, OP_ResultRow, 1, 1);
                                }
                                else
                                    /*
                                    **  PRAGMA [database.]journal_mode
                                    **  PRAGMA [database.]journal_mode =
                                    **                      (delete|persist|off|truncate|memory|wal|off)
                                    */
                                    if (left.Equals("journal_mode", StringComparison.OrdinalIgnoreCase))
                                    {
                                        int eMode;        /* One of the PAGER_JOURNALMODE_XXX symbols */
                                        int ii;           /* Loop counter */

                                        /* Force the schema to be loaded on all databases.  This cases all
                                        ** database files to be opened and the journal_modes set. */
                                        if (sqlite3ReadSchema(parse) != 0)
                                        {
                                            goto pragma_out;
                                        }

                                        sqlite3VdbeSetNumCols(v, 1);
                                        sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "journal_mode", SQLITE_STATIC);
                                        if (null == right)
                                        {
                                            /* If there is no "=MODE" part of the pragma, do a query for the
                                            ** current mode */
                                            eMode = PAGER_JOURNALMODE_QUERY;
                                        }
                                        else
                                        {
                                            string zMode;
                                            int n = sqlite3Strlen30(right);
                                            for (eMode = 0; (zMode = JournalModename(eMode)) != null; eMode++)
                                            {
                                                if (sqlite3StrNICmp(right, zMode, n) == 0)
                                                    break;
                                            }
                                            if (null == zMode)
                                            {
                                                /* If the "=MODE" part does not match any known journal mode,
                                                ** then do a query */
                                                eMode = PAGER_JOURNALMODE_QUERY;
                                            }
                                        }
                                        if (eMode == PAGER_JOURNALMODE_QUERY && id2.n == 0)
                                        {
                                            /* Convert "PRAGMA journal_mode" into "PRAGMA main.journal_mode" */
                                            db = 0;
                                            id2.n = 1;
                                        }
                                        for (ii = ctx.nDb - 1; ii >= 0; ii--)
                                        {
                                            if (ctx.aDb[ii].pBt != null && (ii == db || id2.n == 0))
                                            {
                                                sqlite3VdbeUsesBtree(v, ii);
                                                sqlite3VdbeAddOp3(v, OP_JournalMode, ii, 1, eMode);
                                            }
                                        }
                                        sqlite3VdbeAddOp2(v, OP_ResultRow, 1, 1);
                                    }
                                    else

                                        /*
                                        **  PRAGMA [database.]journal_size_limit
                                        **  PRAGMA [database.]journal_size_limit=N
                                        **
                                        ** Get or set the size limit on rollback journal files.
                                        */
                                        if (left.Equals("journal_size_limit", StringComparison.OrdinalIgnoreCase))
                                        {
                                            Pager pPager = sqlite3BtreePager(dbAsObj.pBt);
                                            i64 iLimit = -2;
                                            if (!String.IsNullOrEmpty(right))
                                            {
                                                sqlite3Atoi64(right, ref iLimit, 1000000, SQLITE_UTF8);
                                                if (iLimit < -1)
                                                    iLimit = -1;
                                            }
                                            iLimit = sqlite3PagerJournalSizeLimit(pPager, iLimit);
                                            ReturnSingleInt(parse, "journal_size_limit", iLimit);
                                        }
                                        else

#endif

                                            /*
**  PRAGMA [database.]auto_vacuum
**  PRAGMA [database.]auto_vacuum=N
**
** Get or set the value of the database 'auto-vacuum' parameter.
** The value is one of:  0 NONE 1 FULL 2 INCREMENTAL
*/
#if !OMIT_AUTOVACUUM
                                            if (left.Equals("auto_vacuum", StringComparison.OrdinalIgnoreCase))
                                            {
                                                Btree pBt = dbAsObj.pBt;
                                                Debug.Assert(pBt != null);
                                                if (sqlite3ReadSchema(parse) != 0)
                                                {
                                                    goto pragma_out;
                                                }
                                                if (null == right)
                                                {
                                                    int auto_vacuum;
                                                    if (ALWAYS(pBt))
                                                    {
                                                        auto_vacuum = sqlite3BtreeGetAutoVacuum(pBt);
                                                    }
                                                    else
                                                    {
                                                        auto_vacuum = SQLITE_DEFAULT_AUTOVACUUM;
                                                    }
                                                    ReturnSingleInt(parse, "auto_vacuum", auto_vacuum);
                                                }
                                                else
                                                {
                                                    int eAuto = GetAutoVacuum(right);
                                                    Debug.Assert(eAuto >= 0 && eAuto <= 2);
                                                    ctx.nextAutovac = (u8)eAuto;
                                                    if (ALWAYS(eAuto >= 0))
                                                    {
                                                        /* Call SetAutoVacuum() to set initialize the internal auto and
                                                        ** incr-vacuum flags. This is required in case this connection
                                                        ** creates the database file. It is important that it is created
                                                        ** as an auto-vacuum capable db.
                                                        */
                                                        int rc = sqlite3BtreeSetAutoVacuum(pBt, eAuto);
                                                        if (rc == SQLITE_OK && (eAuto == 1 || eAuto == 2))
                                                        {
                                                            /* When setting the auto_vacuum mode to either "full" or
                                                            ** "incremental", write the value of meta[6] in the database
                                                            ** file. Before writing to meta[6], check that meta[3] indicates
                                                            ** that this really is an auto-vacuum capable database.
                                                            */
                                                            VdbeOpList[] setMeta6 = new VdbeOpList[] {
new VdbeOpList( OP_Transaction,    0,               1,        0),    /* 0 */
new VdbeOpList( OP_ReadCookie,     0,               1,        BTREE_LARGEST_ROOT_PAGE),    /* 1 */
new VdbeOpList( OP_If,             1,               0,        0),    /* 2 */
new VdbeOpList( OP_Halt,           SQLITE_OK,       OE_Abort, 0),    /* 3 */
new VdbeOpList( OP_Integer,        0,               1,        0),    /* 4 */
new VdbeOpList( OP_SetCookie,      0,               BTREE_INCR_VACUUM, 1),    /* 5 */
};
                                                            int iAddr;
                                                            iAddr = sqlite3VdbeAddOpList(v, ArraySize(setMeta6), setMeta6);
                                                            sqlite3VdbeChangeP1(v, iAddr, db);
                                                            sqlite3VdbeChangeP1(v, iAddr + 1, db);
                                                            sqlite3VdbeChangeP2(v, iAddr + 2, iAddr + 4);
                                                            sqlite3VdbeChangeP1(v, iAddr + 4, eAuto - 1);
                                                            sqlite3VdbeChangeP1(v, iAddr + 5, db);
                                                            sqlite3VdbeUsesBtree(v, db);
                                                        }
                                                    }
                                                }
                                            }
                                            else
#endif

                                                /*
**  PRAGMA [database.]incremental_vacuum(N)
**
** Do N steps of incremental vacuuming on a database.
*/
#if !OMIT_AUTOVACUUM
                                                if (left.Equals("incremental_vacuum", StringComparison.OrdinalIgnoreCase))
                                                {
                                                    int iLimit = 0, addr;
                                                    if (sqlite3ReadSchema(parse) != 0)
                                                    {
                                                        goto pragma_out;
                                                    }
                                                    if (right == null || !sqlite3GetInt32(right, ref iLimit) || iLimit <= 0)
                                                    {
                                                        iLimit = 0x7fffffff;
                                                    }
                                                    sqlite3BeginWriteOperation(parse, 0, db);
                                                    sqlite3VdbeAddOp2(v, OP_Integer, iLimit, 1);
                                                    addr = sqlite3VdbeAddOp1(v, OP_IncrVacuum, db);
                                                    sqlite3VdbeAddOp1(v, OP_ResultRow, 1);
                                                    sqlite3VdbeAddOp2(v, OP_AddImm, 1, -1);
                                                    sqlite3VdbeAddOp2(v, OP_IfPos, 1, addr);
                                                    sqlite3VdbeJumpHere(v, addr);
                                                }
                                                else
#endif

#if !OMIT_PAGER_PRAGMAS
                                                    /*
**  PRAGMA [database.]cache_size
**  PRAGMA [database.]cache_size=N
**
** The first form reports the current local setting for the
** page cache size.  The local setting can be different from
** the persistent cache size value that is stored in the database
** file itself.  The value returned is the maximum number of
** pages in the page cache.  The second form sets the local
** page cache size value.  It does not change the persistent
** cache size stored on the disk so the cache size will revert
** to its default value when the database is closed and reopened.
** N should be a positive integer.
*/
                                                    if (left.Equals("cache_size", StringComparison.OrdinalIgnoreCase))
                                                    {
                                                        if (sqlite3ReadSchema(parse) != 0)
                                                            goto pragma_out;
                                                        Debug.Assert(sqlite3SchemaMutexHeld(ctx, db, null));
                                                        if (null == right)
                                                        {
                                                            ReturnSingleInt(parse, "cache_size", dbAsObj.pSchema.cache_size);
                                                        }
                                                        else
                                                        {
                                                            int size = sqlite3AbsInt32(sqlite3Atoi(right));
                                                            dbAsObj.pSchema.cache_size = size;
                                                            sqlite3BtreeSetCacheSize(dbAsObj.pBt, dbAsObj.pSchema.cache_size);
                                                        }
                                                    }
                                                    else

                                                        /*
                                                        **   PRAGMA temp_store
                                                        **   PRAGMA temp_store = "default"|"memory"|"file"
                                                        **
                                                        ** Return or set the local value of the temp_store flag.  Changing
                                                        ** the local value does not make changes to the disk file and the default
                                                        ** value will be restored the next time the database is opened.
                                                        **
                                                        ** Note that it is possible for the library compile-time options to
                                                        ** override this setting
                                                        */
                                                        if (left.Equals("temp_store", StringComparison.OrdinalIgnoreCase))
                                                        {
                                                            if (right == null)
                                                            {
                                                                ReturnSingleInt(parse, "temp_store", ctx.temp_store);
                                                            }
                                                            else
                                                            {
                                                                ChangeTempStorage(parse, right);
                                                            }
                                                        }
                                                        else

                                                            /*
                                                            **   PRAGMA temp_store_directory
                                                            **   PRAGMA temp_store_directory = ""|"directory_name"
                                                            **
                                                            ** Return or set the local value of the temp_store_directory flag.  Changing
                                                            ** the value sets a specific directory to be used for temporary files.
                                                            ** Setting to a null string reverts to the default temporary directory search.
                                                            ** If temporary directory is changed, then invalidateTempStorage.
                                                            **
                                                            */
                                                            if (left.Equals("temp_store_directory", StringComparison.OrdinalIgnoreCase))
                                                            {
                                                                if (null == right)
                                                                {
                                                                    if (sqlite3_temp_directory != "")
                                                                    {
                                                                        sqlite3VdbeSetNumCols(v, 1);
                                                                        sqlite3VdbeSetColName(v, 0, COLNAME_NAME,
                                                                            "temp_store_directory", SQLITE_STATIC);
                                                                        sqlite3VdbeAddOp4(v, OP_String8, 0, 1, 0, sqlite3_temp_directory, 0);
                                                                        sqlite3VdbeAddOp2(v, OP_ResultRow, 1, 1);
                                                                    }
                                                                }
                                                                else
                                                                {
#if !OMIT_WSD
                                                                    if (right.Length > 0)
                                                                    {
                                                                        int rc;
                                                                        int res = 0;
                                                                        rc = sqlite3OsAccess(ctx.pVfs, right, SQLITE_ACCESS_READWRITE, ref res);
                                                                        if (rc != SQLITE_OK || res == 0)
                                                                        {
                                                                            sqlite3ErrorMsg(parse, "not a writable directory");
                                                                            goto pragma_out;
                                                                        }
                                                                    }
                                                                    if (SQLITE_TEMP_STORE == 0
                                                                     || (SQLITE_TEMP_STORE == 1 && ctx.temp_store <= 1)
                                                                     || (SQLITE_TEMP_STORE == 2 && ctx.temp_store == 1)
                                                                    )
                                                                    {
                                                                        InvalidateTempStorage(parse);
                                                                    }
                                                                    //sqlite3_free( ref sqlite3_temp_directory );
                                                                    if (right.Length > 0)
                                                                    {
                                                                        sqlite3_temp_directory = right;//sqlite3_mprintf("%s", zRight);
                                                                    }
                                                                    else
                                                                    {
                                                                        sqlite3_temp_directory = "";
                                                                    }
#endif
                                                                }
                                                            }
                                                            else

#if !(ENABLE_LOCKING_STYLE)
#  if (__APPLE__)
//#define ENABLE_LOCKING_STYLE 1
#  else                                                                //#define ENABLE_LOCKING_STYLE 0
#  endif
#endif
#if ENABLE_LOCKING_STYLE
/*
**   PRAGMA [database.]lock_proxy_file
**   PRAGMA [database.]lock_proxy_file = ":auto:"|"lock_file_path"
**
** Return or set the value of the lock_proxy_file flag.  Changing
** the value sets a specific file to be used for database access locks.
**
*/
if ( zLeft.Equals( "lock_proxy_file", StringComparison.OrdinalIgnoreCase )  )
{
if ( zRight !="")
{
Pager pPager = sqlite3BtreePager( pDb.pBt );
int proxy_file_path = 0;
sqlite3_file pFile = sqlite3PagerFile( pPager );
sqlite3OsFileControl( pFile, SQLITE_GET_LOCKPROXYFILE,
ref proxy_file_path );

if ( proxy_file_path!=0 )
{
sqlite3VdbeSetNumCols( v, 1 );
sqlite3VdbeSetColName( v, 0, COLNAME_NAME,
"lock_proxy_file", SQLITE_STATIC );
sqlite3VdbeAddOp4( v, OP_String8, 0, 1, 0, proxy_file_path, 0 );
sqlite3VdbeAddOp2( v, OP_ResultRow, 1, 1 );
}
}
else
{
Pager pPager = sqlite3BtreePager( pDb.pBt );
sqlite3_file pFile = sqlite3PagerFile( pPager );
int res;
int iDummy = 0;
if ( zRight[0]!=0 )
{
iDummy = zRight[0];
res = sqlite3OsFileControl( pFile, SQLITE_SET_LOCKPROXYFILE,
ref iDummy );
}
else
{
res = sqlite3OsFileControl( pFile, SQLITE_SET_LOCKPROXYFILE,
ref iDummy );
}
if ( res != SQLITE_OK )
{
sqlite3ErrorMsg( pParse, "failed to set lock proxy file" );
goto pragma_out;
}
}
}
else
#endif

                                                                /*
**   PRAGMA [database.]synchronous
**   PRAGMA [database.]synchronous=OFF|ON|NORMAL|FULL
**
** Return or set the local value of the synchronous flag.  Changing
** the local value does not make changes to the disk file and the
** default value will be restored the next time the database is
** opened.
*/
                                                                if (left.Equals("synchronous", StringComparison.OrdinalIgnoreCase))
                                                                {
                                                                    if (sqlite3ReadSchema(parse) != 0)
                                                                        goto pragma_out;
                                                                    if (null == right)
                                                                    {
                                                                        ReturnSingleInt(parse, "synchronous", dbAsObj.safety_level - 1);
                                                                    }
                                                                    else
                                                                    {
                                                                        if (0 == ctx.autoCommit)
                                                                        {
                                                                            sqlite3ErrorMsg(parse,
                                                                              "Safety level may not be changed inside a transaction");
                                                                        }
                                                                        else
                                                                        {
                                                                            dbAsObj.safety_level = (byte)(GetSafetyLevel(right) + 1);
                                                                        }
                                                                    }
                                                                }
                                                                else
#endif

#if !OMIT_FLAG_PRAGMAS
                                                                    if (FlagPragma(parse, left, right) != 0)
                                                                    {
                                                                        /* The flagPragma() subroutine also generates any necessary code
                                                                        ** there is nothing more to do here */
                                                                    }
                                                                    else
#endif

#if !OMIT_SCHEMA_PRAGMAS
                                                                        /*
**   PRAGMA table_info(<table>)
**
** Return a single row for each column of the named table. The columns of
** the returned data set are:
**
** cid:        Column id (numbered from left to right, starting at 0)
** name:       Column name
** type:       Column declaration type.
** notnull:    True if 'NOT NULL' is part of column declaration
** dflt_value: The default value for the column, if any.
*/
                                                                        if (left.Equals("table_info", StringComparison.OrdinalIgnoreCase) && right != null)
                                                                        {
                                                                            Table pTab;
                                                                            if (sqlite3ReadSchema(parse) != 0)
                                                                                goto pragma_out;
                                                                            pTab = sqlite3FindTable(ctx, right, dbName);
                                                                            if (pTab != null)
                                                                            {
                                                                                int i;
                                                                                int nHidden = 0;
                                                                                Column pCol;
                                                                                sqlite3VdbeSetNumCols(v, 6);
                                                                                parse.nMem = 6;
                                                                                sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "cid", SQLITE_STATIC);
                                                                                sqlite3VdbeSetColName(v, 1, COLNAME_NAME, "name", SQLITE_STATIC);
                                                                                sqlite3VdbeSetColName(v, 2, COLNAME_NAME, "type", SQLITE_STATIC);
                                                                                sqlite3VdbeSetColName(v, 3, COLNAME_NAME, "notnull", SQLITE_STATIC);
                                                                                sqlite3VdbeSetColName(v, 4, COLNAME_NAME, "dflt_value", SQLITE_STATIC);
                                                                                sqlite3VdbeSetColName(v, 5, COLNAME_NAME, "pk", SQLITE_STATIC);
                                                                                sqlite3ViewGetColumnNames(parse, pTab);
                                                                                for (i = 0; i < pTab.nCol; i++)//, pCol++)
                                                                                {
                                                                                    pCol = pTab.aCol[i];
                                                                                    if (IsHiddenColumn(pCol))
                                                                                    {
                                                                                        nHidden++;
                                                                                        continue;
                                                                                    }
                                                                                    sqlite3VdbeAddOp2(v, OP_Integer, i - nHidden, 1);
                                                                                    sqlite3VdbeAddOp4(v, OP_String8, 0, 2, 0, pCol.zName, 0);
                                                                                    sqlite3VdbeAddOp4(v, OP_String8, 0, 3, 0,
                                                                                       pCol.zType != null ? pCol.zType : "", 0);
                                                                                    sqlite3VdbeAddOp2(v, OP_Integer, (pCol.notNull != 0 ? 1 : 0), 4);
                                                                                    if (pCol.zDflt != null)
                                                                                    {
                                                                                        sqlite3VdbeAddOp4(v, OP_String8, 0, 5, 0, pCol.zDflt, 0);
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        sqlite3VdbeAddOp2(v, OP_Null, 0, 5);
                                                                                    }
                                                                                    sqlite3VdbeAddOp2(v, OP_Integer, pCol.isPrimKey != 0 ? 1 : 0, 6);
                                                                                    sqlite3VdbeAddOp2(v, OP_ResultRow, 1, 6);
                                                                                }
                                                                            }
                                                                        }
                                                                        else

                                                                            if (left.Equals("index_info", StringComparison.OrdinalIgnoreCase) && right != null)
                                                                            {
                                                                                Index pIdx;
                                                                                Table pTab;
                                                                                if (sqlite3ReadSchema(parse) != 0)
                                                                                    goto pragma_out;
                                                                                pIdx = sqlite3FindIndex(ctx, right, dbName);
                                                                                if (pIdx != null)
                                                                                {
                                                                                    int i;
                                                                                    pTab = pIdx.pTable;
                                                                                    sqlite3VdbeSetNumCols(v, 3);
                                                                                    parse.nMem = 3;
                                                                                    sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "seqno", SQLITE_STATIC);
                                                                                    sqlite3VdbeSetColName(v, 1, COLNAME_NAME, "cid", SQLITE_STATIC);
                                                                                    sqlite3VdbeSetColName(v, 2, COLNAME_NAME, "name", SQLITE_STATIC);
                                                                                    for (i = 0; i < pIdx.nColumn; i++)
                                                                                    {
                                                                                        int cnum = pIdx.aiColumn[i];
                                                                                        sqlite3VdbeAddOp2(v, OP_Integer, i, 1);
                                                                                        sqlite3VdbeAddOp2(v, OP_Integer, cnum, 2);
                                                                                        Debug.Assert(pTab.nCol > cnum);
                                                                                        sqlite3VdbeAddOp4(v, OP_String8, 0, 3, 0, pTab.aCol[cnum].zName, 0);
                                                                                        sqlite3VdbeAddOp2(v, OP_ResultRow, 1, 3);
                                                                                    }
                                                                                }
                                                                            }
                                                                            else

                                                                                if (left.Equals("index_list", StringComparison.OrdinalIgnoreCase) && right != null)
                                                                                {
                                                                                    Index pIdx;
                                                                                    Table pTab;
                                                                                    if (sqlite3ReadSchema(parse) != 0)
                                                                                        goto pragma_out;
                                                                                    pTab = sqlite3FindTable(ctx, right, dbName);
                                                                                    if (pTab != null)
                                                                                    {
                                                                                        v = sqlite3GetVdbe(parse);
                                                                                        pIdx = pTab.pIndex;
                                                                                        if (pIdx != null)
                                                                                        {
                                                                                            int i = 0;
                                                                                            sqlite3VdbeSetNumCols(v, 3);
                                                                                            parse.nMem = 3;
                                                                                            sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "seq", SQLITE_STATIC);
                                                                                            sqlite3VdbeSetColName(v, 1, COLNAME_NAME, "name", SQLITE_STATIC);
                                                                                            sqlite3VdbeSetColName(v, 2, COLNAME_NAME, "unique", SQLITE_STATIC);
                                                                                            while (pIdx != null)
                                                                                            {
                                                                                                sqlite3VdbeAddOp2(v, OP_Integer, i, 1);
                                                                                                sqlite3VdbeAddOp4(v, OP_String8, 0, 2, 0, pIdx.zName, 0);
                                                                                                sqlite3VdbeAddOp2(v, OP_Integer, (pIdx.onError != OE_None) ? 1 : 0, 3);
                                                                                                sqlite3VdbeAddOp2(v, OP_ResultRow, 1, 3);
                                                                                                ++i;
                                                                                                pIdx = pIdx.pNext;
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                                else

                                                                                    if (left.Equals("database_list", StringComparison.OrdinalIgnoreCase))
                                                                                    {
                                                                                        int i;
                                                                                        if (sqlite3ReadSchema(parse) != 0)
                                                                                            goto pragma_out;
                                                                                        sqlite3VdbeSetNumCols(v, 3);
                                                                                        parse.nMem = 3;
                                                                                        sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "seq", SQLITE_STATIC);
                                                                                        sqlite3VdbeSetColName(v, 1, COLNAME_NAME, "name", SQLITE_STATIC);
                                                                                        sqlite3VdbeSetColName(v, 2, COLNAME_NAME, "file", SQLITE_STATIC);
                                                                                        for (i = 0; i < ctx.nDb; i++)
                                                                                        {
                                                                                            if (ctx.aDb[i].pBt == null)
                                                                                                continue;
                                                                                            Debug.Assert(ctx.aDb[i].zName != null);
                                                                                            sqlite3VdbeAddOp2(v, OP_Integer, i, 1);
                                                                                            sqlite3VdbeAddOp4(v, OP_String8, 0, 2, 0, ctx.aDb[i].zName, 0);
                                                                                            sqlite3VdbeAddOp4(v, OP_String8, 0, 3, 0,
                                                                                                 sqlite3BtreeGetFilename(ctx.aDb[i].pBt), 0);
                                                                                            sqlite3VdbeAddOp2(v, OP_ResultRow, 1, 3);
                                                                                        }
                                                                                    }
                                                                                    else

                                                                                        if (left.Equals("collation_list", StringComparison.OrdinalIgnoreCase))
                                                                                        {
                                                                                            int i = 0;
                                                                                            HashElem p;
                                                                                            sqlite3VdbeSetNumCols(v, 2);
                                                                                            parse.nMem = 2;
                                                                                            sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "seq", SQLITE_STATIC);
                                                                                            sqlite3VdbeSetColName(v, 1, COLNAME_NAME, "name", SQLITE_STATIC);
                                                                                            for (p = ctx.aCollSeq.first; p != null; p = p.next)//( p = sqliteHashFirst( db.aCollSeq ) ; p; p = sqliteHashNext( p ) )
                                                                                            {
                                                                                                CollSeq pColl = ((CollSeq[])p.data)[0];// sqliteHashData( p );
                                                                                                sqlite3VdbeAddOp2(v, OP_Integer, i++, 1);
                                                                                                sqlite3VdbeAddOp4(v, OP_String8, 0, 2, 0, pColl.zName, 0);
                                                                                                sqlite3VdbeAddOp2(v, OP_ResultRow, 1, 2);
                                                                                            }
                                                                                        }
                                                                                        else
#endif

#if !OMIT_FOREIGN_KEY
                                                                                            if (left.Equals("foreign_key_list", StringComparison.OrdinalIgnoreCase) && right != null)
                                                                                            {
                                                                                                FKey pFK;
                                                                                                Table pTab;
                                                                                                if (sqlite3ReadSchema(parse) != 0)
                                                                                                    goto pragma_out;
                                                                                                pTab = sqlite3FindTable(ctx, right, dbName);
                                                                                                if (pTab != null)
                                                                                                {
                                                                                                    v = sqlite3GetVdbe(parse);
                                                                                                    pFK = pTab.pFKey;
                                                                                                    if (pFK != null)
                                                                                                    {
                                                                                                        int i = 0;
                                                                                                        sqlite3VdbeSetNumCols(v, 8);
                                                                                                        parse.nMem = 8;
                                                                                                        sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "id", SQLITE_STATIC);
                                                                                                        sqlite3VdbeSetColName(v, 1, COLNAME_NAME, "seq", SQLITE_STATIC);
                                                                                                        sqlite3VdbeSetColName(v, 2, COLNAME_NAME, "table", SQLITE_STATIC);
                                                                                                        sqlite3VdbeSetColName(v, 3, COLNAME_NAME, "from", SQLITE_STATIC);
                                                                                                        sqlite3VdbeSetColName(v, 4, COLNAME_NAME, "to", SQLITE_STATIC);
                                                                                                        sqlite3VdbeSetColName(v, 5, COLNAME_NAME, "on_update", SQLITE_STATIC);
                                                                                                        sqlite3VdbeSetColName(v, 6, COLNAME_NAME, "on_delete", SQLITE_STATIC);
                                                                                                        sqlite3VdbeSetColName(v, 7, COLNAME_NAME, "match", SQLITE_STATIC);
                                                                                                        while (pFK != null)
                                                                                                        {
                                                                                                            int j;
                                                                                                            for (j = 0; j < pFK.nCol; j++)
                                                                                                            {
                                                                                                                string zCol = pFK.aCol[j].zCol;
                                                                                                                string zOnDelete = ActionName(pFK.aAction[0]);
                                                                                                                string zOnUpdate = ActionName(pFK.aAction[1]);
                                                                                                                sqlite3VdbeAddOp2(v, OP_Integer, i, 1);
                                                                                                                sqlite3VdbeAddOp2(v, OP_Integer, j, 2);
                                                                                                                sqlite3VdbeAddOp4(v, OP_String8, 0, 3, 0, pFK.zTo, 0);
                                                                                                                sqlite3VdbeAddOp4(v, OP_String8, 0, 4, 0,
                                                                                                                                  pTab.aCol[pFK.aCol[j].iFrom].zName, 0);
                                                                                                                sqlite3VdbeAddOp4(v, zCol != null ? OP_String8 : OP_Null, 0, 5, 0, zCol, 0);
                                                                                                                sqlite3VdbeAddOp4(v, OP_String8, 0, 6, 0, zOnUpdate, 0);
                                                                                                                sqlite3VdbeAddOp4(v, OP_String8, 0, 7, 0, zOnDelete, 0);
                                                                                                                sqlite3VdbeAddOp4(v, OP_String8, 0, 8, 0, "NONE", 0);
                                                                                                                sqlite3VdbeAddOp2(v, OP_ResultRow, 1, 8);
                                                                                                            }
                                                                                                            ++i;
                                                                                                            pFK = pFK.pNextFrom;
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                            else
#endif

#if !NDEBUG
                                                                                                if (left.Equals("parser_trace", StringComparison.OrdinalIgnoreCase))
                                                                                                {
                                                                                                    if (right != null)
                                                                                                    {
                                                                                                        if (GetBoolean(right) != 0)
                                                                                                        {
                                                                                                            sqlite3ParserTrace(Console.Out, "parser: ");
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            sqlite3ParserTrace(null, "");
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                                else
#endif

                                                                                                    /* Reinstall the LIKE and GLOB functions.  The variant of LIKE
** used will be case sensitive or not depending on the RHS.
*/
                                                                                                    if (left.Equals("case_sensitive_like", StringComparison.OrdinalIgnoreCase))
                                                                                                    {
                                                                                                        if (right != null)
                                                                                                        {
                                                                                                            sqlite3RegisterLikeFunctions(ctx, GetBoolean(right));
                                                                                                        }
                                                                                                    }
                                                                                                    else

#if !INTEGRITY_CHECK_ERROR_MAX
                                                                                                        //const int SQLITE_INTEGRITY_CHECK_ERROR_MAX = 100;
#endif

#if !OMIT_INTEGRITY_CHECK
                                                                                                        /* Pragma "quick_check" is an experimental reduced version of
** integrity_check designed to detect most database corruption
** without most of the overhead of a full integrity-check.
*/
                                                                                                        if (left.Equals("integrity_check", StringComparison.OrdinalIgnoreCase)
                                                                                                         || left.Equals("quick_check", StringComparison.OrdinalIgnoreCase)
                                                                                                        )
                                                                                                        {
                                                                                                            const int SQLITE_INTEGRITY_CHECK_ERROR_MAX = 100;
                                                                                                            int i, j, addr, mxErr;

                                                                                                            /* Code that appears at the end of the integrity check.  If no error
                                                                                                            ** messages have been generated, refput OK.  Otherwise output the
                                                                                                            ** error message
                                                                                                            */
                                                                                                            VdbeOpList[] endCode = new VdbeOpList[]  {
new VdbeOpList( OP_AddImm,      1, 0,        0),    /* 0 */
new                    VdbeOpList( OP_IfNeg,       1, 0,        0),    /* 1 */
new    VdbeOpList( OP_String8,     0, 3,        0),    /* 2 */
new  VdbeOpList( OP_ResultRow,   3, 1,        0),
};

                                                                                                            bool isQuick = (left[0] == 'q');

                                                                                                            /* Initialize the VDBE program */
                                                                                                            if (sqlite3ReadSchema(parse) != 0)
                                                                                                                goto pragma_out;
                                                                                                            parse.nMem = 6;
                                                                                                            sqlite3VdbeSetNumCols(v, 1);
                                                                                                            sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "integrity_check", SQLITE_STATIC);

                                                                                                            /* Set the maximum error count */
                                                                                                            mxErr = SQLITE_INTEGRITY_CHECK_ERROR_MAX;
                                                                                                            if (right != null)
                                                                                                            {
                                                                                                                sqlite3GetInt32(right, ref mxErr);
                                                                                                                if (mxErr <= 0)
                                                                                                                {
                                                                                                                    mxErr = SQLITE_INTEGRITY_CHECK_ERROR_MAX;
                                                                                                                }
                                                                                                            }
                                                                                                            sqlite3VdbeAddOp2(v, OP_Integer, mxErr, 1);  /* reg[1] holds errors left */

                                                                                                            /* Do an integrity check on each database file */
                                                                                                            for (i = 0; i < ctx.nDb; i++)
                                                                                                            {
                                                                                                                HashElem x;
                                                                                                                Hash pTbls;
                                                                                                                int cnt = 0;

                                                                                                                if (OMIT_TEMPDB != 0 && i == 1)
                                                                                                                    continue;

                                                                                                                sqlite3CodeVerifySchema(parse, i);
                                                                                                                addr = sqlite3VdbeAddOp1(v, OP_IfPos, 1); /* Halt if out of errors */
                                                                                                                sqlite3VdbeAddOp2(v, OP_Halt, 0, 0);
                                                                                                                sqlite3VdbeJumpHere(v, addr);

                                                                                                                /* Do an integrity check of the B-Tree
                                                                                                                **
                                                                                                                ** Begin by filling registers 2, 3, ... with the root pages numbers
                                                                                                                ** for all tables and indices in the database.
                                                                                                                */
                                                                                                                Debug.Assert(sqlite3SchemaMutexHeld(ctx, db, null));
                                                                                                                pTbls = ctx.aDb[i].pSchema.tblHash;
                                                                                                                for (x = pTbls.first; x != null; x = x.next)
                                                                                                                {//          for(x=sqliteHashFirst(pTbls); x; x=sqliteHashNext(x)){
                                                                                                                    Table pTab = (Table)x.data;// sqliteHashData( x );
                                                                                                                    Index pIdx;
                                                                                                                    sqlite3VdbeAddOp2(v, OP_Integer, pTab.tnum, 2 + cnt);
                                                                                                                    cnt++;
                                                                                                                    for (pIdx = pTab.pIndex; pIdx != null; pIdx = pIdx.pNext)
                                                                                                                    {
                                                                                                                        sqlite3VdbeAddOp2(v, OP_Integer, pIdx.tnum, 2 + cnt);
                                                                                                                        cnt++;
                                                                                                                    }
                                                                                                                }

                                                                                                                /* Make sure sufficient number of registers have been allocated */
                                                                                                                if (parse.nMem < cnt + 4)
                                                                                                                {
                                                                                                                    parse.nMem = cnt + 4;
                                                                                                                }

                                                                                                                /* Do the b-tree integrity checks */
                                                                                                                sqlite3VdbeAddOp3(v, OP_IntegrityCk, 2, cnt, 1);
                                                                                                                sqlite3VdbeChangeP5(v, (u8)i);
                                                                                                                addr = sqlite3VdbeAddOp1(v, OP_IsNull, 2);
                                                                                                                sqlite3VdbeAddOp4(v, OP_String8, 0, 3, 0,
                                                                                                                   sqlite3MPrintf(ctx, "*** in database %s ***\n", ctx.aDb[i].zName),
                                                                                                                   P4_DYNAMIC);
                                                                                                                sqlite3VdbeAddOp3(v, OP_Move, 2, 4, 1);
                                                                                                                sqlite3VdbeAddOp3(v, OP_Concat, 4, 3, 2);
                                                                                                                sqlite3VdbeAddOp2(v, OP_ResultRow, 2, 1);
                                                                                                                sqlite3VdbeJumpHere(v, addr);

                                                                                                                /* Make sure all the indices are constructed correctly.
                                                                                                                */
                                                                                                                for (x = pTbls.first; x != null && !isQuick; x = x.next)
                                                                                                                {
                                                                                                                    ;//          for(x=sqliteHashFirst(pTbls); x && !isQuick; x=sqliteHashNext(x)){
                                                                                                                    Table pTab = (Table)x.data;// sqliteHashData( x );
                                                                                                                    Index pIdx;
                                                                                                                    int loopTop;

                                                                                                                    if (pTab.pIndex == null)
                                                                                                                        continue;
                                                                                                                    addr = sqlite3VdbeAddOp1(v, OP_IfPos, 1);  /* Stop if out of errors */
                                                                                                                    sqlite3VdbeAddOp2(v, OP_Halt, 0, 0);
                                                                                                                    sqlite3VdbeJumpHere(v, addr);
                                                                                                                    sqlite3OpenTableAndIndices(parse, pTab, 1, OP_OpenRead);
                                                                                                                    sqlite3VdbeAddOp2(v, OP_Integer, 0, 2);  /* reg(2) will count entries */
                                                                                                                    loopTop = sqlite3VdbeAddOp2(v, OP_Rewind, 1, 0);
                                                                                                                    sqlite3VdbeAddOp2(v, OP_AddImm, 2, 1);   /* increment entry count */
                                                                                                                    for (j = 0, pIdx = pTab.pIndex; pIdx != null; pIdx = pIdx.pNext, j++)
                                                                                                                    {
                                                                                                                        int jmp2;
                                                                                                                        int r1;
                                                                                                                        VdbeOpList[] idxErr = new VdbeOpList[]  {
new VdbeOpList( OP_AddImm,      1, -1,  0),
new VdbeOpList( OP_String8,     0,  3,  0),    /* 1 */
new VdbeOpList( OP_Rowid,       1,  4,  0),
new VdbeOpList( OP_String8,     0,  5,  0),    /* 3 */
new VdbeOpList( OP_String8,     0,  6,  0),    /* 4 */
new VdbeOpList( OP_Concat,      4,  3,  3),
new VdbeOpList( OP_Concat,      5,  3,  3),
new VdbeOpList( OP_Concat,      6,  3,  3),
new VdbeOpList( OP_ResultRow,   3,  1,  0),
new VdbeOpList(  OP_IfPos,       1,  0,  0),    /* 9 */
new VdbeOpList(  OP_Halt,        0,  0,  0),
};
                                                                                                                        r1 = sqlite3GenerateIndexKey(parse, pIdx, 1, 3, false);
                                                                                                                        jmp2 = sqlite3VdbeAddOp4Int(v, OP_Found, j + 2, 0, r1, pIdx.nColumn + 1);
                                                                                                                        addr = sqlite3VdbeAddOpList(v, ArraySize(idxErr), idxErr);
                                                                                                                        sqlite3VdbeChangeP4(v, addr + 1, "rowid ", SQLITE_STATIC);
                                                                                                                        sqlite3VdbeChangeP4(v, addr + 3, " missing from index ", SQLITE_STATIC);
                                                                                                                        sqlite3VdbeChangeP4(v, addr + 4, pIdx.zName, P4_TRANSIENT);
                                                                                                                        sqlite3VdbeJumpHere(v, addr + 9);
                                                                                                                        sqlite3VdbeJumpHere(v, jmp2);
                                                                                                                    }
                                                                                                                    sqlite3VdbeAddOp2(v, OP_Next, 1, loopTop + 1);
                                                                                                                    sqlite3VdbeJumpHere(v, loopTop);
                                                                                                                    for (j = 0, pIdx = pTab.pIndex; pIdx != null; pIdx = pIdx.pNext, j++)
                                                                                                                    {
                                                                                                                        VdbeOpList[] cntIdx = new VdbeOpList[] {
new VdbeOpList( OP_Integer,      0,  3,  0),
new VdbeOpList( OP_Rewind,       0,  0,  0),  /* 1 */
new VdbeOpList( OP_AddImm,       3,  1,  0),
new VdbeOpList( OP_Next,         0,  0,  0),  /* 3 */
new VdbeOpList( OP_Eq,           2,  0,  3),  /* 4 */
new VdbeOpList( OP_AddImm,       1, -1,  0),
new VdbeOpList( OP_String8,      0,  2,  0),  /* 6 */
new VdbeOpList( OP_String8,      0,  3,  0),  /* 7 */
new VdbeOpList( OP_Concat,       3,  2,  2),
new VdbeOpList( OP_ResultRow,    2,  1,  0),
};
                                                                                                                        addr = sqlite3VdbeAddOp1(v, OP_IfPos, 1);
                                                                                                                        sqlite3VdbeAddOp2(v, OP_Halt, 0, 0);
                                                                                                                        sqlite3VdbeJumpHere(v, addr);
                                                                                                                        addr = sqlite3VdbeAddOpList(v, ArraySize(cntIdx), cntIdx);
                                                                                                                        sqlite3VdbeChangeP1(v, addr + 1, j + 2);
                                                                                                                        sqlite3VdbeChangeP2(v, addr + 1, addr + 4);
                                                                                                                        sqlite3VdbeChangeP1(v, addr + 3, j + 2);
                                                                                                                        sqlite3VdbeChangeP2(v, addr + 3, addr + 2);
                                                                                                                        sqlite3VdbeJumpHere(v, addr + 4);
                                                                                                                        sqlite3VdbeChangeP4(v, addr + 6,
                                                                                                                                   "wrong # of entries in index ", P4_STATIC);
                                                                                                                        sqlite3VdbeChangeP4(v, addr + 7, pIdx.zName, P4_TRANSIENT);
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                            addr = sqlite3VdbeAddOpList(v, ArraySize(endCode), endCode);
                                                                                                            sqlite3VdbeChangeP2(v, addr, -mxErr);
                                                                                                            sqlite3VdbeJumpHere(v, addr + 1);
                                                                                                            sqlite3VdbeChangeP4(v, addr + 2, "ok", P4_STATIC);
                                                                                                        }
                                                                                                        else
#endif

                                                                                                            /*
**   PRAGMA encoding
**   PRAGMA encoding = "utf-8"|"utf-16"|"utf-16le"|"utf-16be"
**
** In its first form, this pragma returns the encoding of the main
** database. If the database is not initialized, it is initialized now.
**
** The second form of this pragma is a no-op if the main database file
** has not already been initialized. In this case it sets the default
** encoding that will be used for the main database file if a new file
** is created. If an existing main database file is opened, then the
** default text encoding for the existing database is used.
**
** In all cases new databases created using the ATTACH command are
** created to use the same default text encoding as the main database. If
** the main database has not been initialized and/or created when ATTACH
** is executed, this is done before the ATTACH operation.
**
** In the second form this pragma sets the text encoding to be used in
** new database files created using this database handle. It is only
** useful if invoked immediately after the main database i
*/
                                                                                                            if (left.Equals("encoding", StringComparison.OrdinalIgnoreCase))
                                                                                                            {
                                                                                                                int iEnc;
                                                                                                                if (null == right)
                                                                                                                {    /* "PRAGMA encoding" */
                                                                                                                    if (sqlite3ReadSchema(parse) != 0)
                                                                                                                    {
                                                                                                                        parse.nErr = 0;
                                                                                                                        parse.zErrMsg = null;
                                                                                                                        parse.rc = 0;//  reset errors goto pragma_out;
                                                                                                                    }
                                                                                                                    sqlite3VdbeSetNumCols(v, 1);
                                                                                                                    sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "encoding", SQLITE_STATIC);
                                                                                                                    sqlite3VdbeAddOp2(v, OP_String8, 0, 1);
                                                                                                                    Debug.Assert(_encnames[SQLITE_UTF8].Encode == SQLITE_UTF8);
                                                                                                                    Debug.Assert(_encnames[SQLITE_UTF16LE].Encode == SQLITE_UTF16LE);
                                                                                                                    Debug.Assert(_encnames[SQLITE_UTF16BE].Encode == SQLITE_UTF16BE);
                                                                                                                    sqlite3VdbeChangeP4(v, -1, _encnames[ENC(parse.db)].Name, P4_STATIC);
                                                                                                                    sqlite3VdbeAddOp2(v, OP_ResultRow, 1, 1);
                                                                                                                }
#if !OMIT_UTF16
                                                                                                                else
                                                                                                                {                        /* "PRAGMA encoding = XXX" */
                                                                                                                    /* Only change the value of sqlite.enc if the database handle is not
                                                                                                                    ** initialized. If the main database exists, the new sqlite.enc value
                                                                                                                    ** will be overwritten when the schema is next loaded. If it does not
                                                                                                                    ** already exists, it will be created to use the new encoding value.
                                                                                                                    */
                                                                                                                    if (
                                                                                                                        //!(DbHasProperty(db, 0, DB_SchemaLoaded)) ||
                                                                                                                        //DbHasProperty(db, 0, DB_Empty)
                                                                                                                    (ctx.flags & DB_SchemaLoaded) != DB_SchemaLoaded || (ctx.flags & DB_Empty) == DB_Empty
                                                                                                                    )
                                                                                                                    {
                                                                                                                        for (iEnc = 0; _encnames[iEnc].Name != null; iEnc++)
                                                                                                                        {
                                                                                                                            if (right.Equals(_encnames[iEnc].Name, StringComparison.OrdinalIgnoreCase))
                                                                                                                            {
                                                                                                                                parse.db.aDbStatic[0].pSchema.enc = _encnames[iEnc].Encode != 0 ? _encnames[iEnc].Encode : SQLITE_UTF16NATIVE;
                                                                                                                                break;
                                                                                                                            }
                                                                                                                        }
                                                                                                                        if (_encnames[iEnc].Name == null)
                                                                                                                        {
                                                                                                                            sqlite3ErrorMsg(parse, "unsupported encoding: %s", right);
                                                                                                                        }
                                                                                                                    }
                                                                                                                }
#endif
                                                                                                            }
                                                                                                            else

#if !OMIT_SCHEMA_VERSION_PRAGMAS
                                                                                                                /*
**   PRAGMA [database.]schema_version
**   PRAGMA [database.]schema_version = <integer>
**
**   PRAGMA [database.]user_version
**   PRAGMA [database.]user_version = <integer>
**
** The pragma's schema_version and user_version are used to set or get
** the value of the schema-version and user-version, respectively. Both
** the schema-version and the user-version are 32-bit signed integers
** stored in the database header.
**
** The schema-cookie is usually only manipulated internally by SQLite. It
** is incremented by SQLite whenever the database schema is modified (by
** creating or dropping a table or index). The schema version is used by
** SQLite each time a query is executed to ensure that the internal cache
** of the schema used when compiling the SQL query matches the schema of
** the database against which the compiled query is actually executed.
** Subverting this mechanism by using "PRAGMA schema_version" to modify
** the schema-version is potentially dangerous and may lead to program
** crashes or database corruption. Use with caution!
**
** The user-version is not used internally by SQLite. It may be used by
** applications for any purpose.
*/
                                                                                                                if (left.Equals("schema_version", StringComparison.OrdinalIgnoreCase)
                                                                                                                 || left.Equals("user_version", StringComparison.OrdinalIgnoreCase)
                                                                                                                 || left.Equals("freelist_count", StringComparison.OrdinalIgnoreCase)
                                                                                                                )
                                                                                                                {
                                                                                                                    int iCookie;   /* Cookie index. 1 for schema-cookie, 6 for user-cookie. */
                                                                                                                    sqlite3VdbeUsesBtree(v, db);
                                                                                                                    switch (left[0])
                                                                                                                    {
                                                                                                                        case 'f':
                                                                                                                        case 'F':
                                                                                                                            iCookie = BTREE_FREE_PAGE_COUNT;
                                                                                                                            break;
                                                                                                                        case 's':
                                                                                                                        case 'S':
                                                                                                                            iCookie = BTREE_SCHEMA_VERSION;
                                                                                                                            break;
                                                                                                                        default:
                                                                                                                            iCookie = BTREE_USER_VERSION;
                                                                                                                            break;
                                                                                                                    }

                                                                                                                    if (right != null && iCookie != BTREE_FREE_PAGE_COUNT)
                                                                                                                    {
                                                                                                                        /* Write the specified cookie value */
                                                                                                                        VdbeOpList[] setCookie = new VdbeOpList[] {
new VdbeOpList( OP_Transaction,    0,  1,  0),    /* 0 */
new   VdbeOpList( OP_Integer,        0,  1,  0),    /* 1 */
new VdbeOpList( OP_SetCookie,      0,  0,  1),    /* 2 */
};
                                                                                                                        int addr = sqlite3VdbeAddOpList(v, ArraySize(setCookie), setCookie);
                                                                                                                        sqlite3VdbeChangeP1(v, addr, db);
                                                                                                                        sqlite3VdbeChangeP1(v, addr + 1, sqlite3Atoi(right));
                                                                                                                        sqlite3VdbeChangeP1(v, addr + 2, db);
                                                                                                                        sqlite3VdbeChangeP2(v, addr + 2, iCookie);
                                                                                                                    }
                                                                                                                    else
                                                                                                                    {
                                                                                                                        /* Read the specified cookie value */
                                                                                                                        VdbeOpList[] readCookie = new VdbeOpList[]  {
new VdbeOpList( OP_Transaction,     0,  0,  0),    /* 0 */
new VdbeOpList( OP_ReadCookie,      0,  1,  0),    /* 1 */
new VdbeOpList( OP_ResultRow,       1,  1,  0)
};
                                                                                                                        int addr = sqlite3VdbeAddOpList(v, readCookie.Length, readCookie);// ArraySize(readCookie), readCookie);
                                                                                                                        sqlite3VdbeChangeP1(v, addr, db);
                                                                                                                        sqlite3VdbeChangeP1(v, addr + 1, db);
                                                                                                                        sqlite3VdbeChangeP3(v, addr + 1, iCookie);
                                                                                                                        sqlite3VdbeSetNumCols(v, 1);
                                                                                                                        sqlite3VdbeSetColName(v, 0, COLNAME_NAME, left, SQLITE_TRANSIENT);
                                                                                                                    }
                                                                                                                }
                                                                                                                else if (left.Equals("reload_schema", StringComparison.OrdinalIgnoreCase))
                                                                                                                {
                                                                                                                    /* force schema reloading*/
                                                                                                                    sqlite3ResetInternalSchema(ctx, -1);
                                                                                                                }
                                                                                                                else if (left.Equals("file_format", StringComparison.OrdinalIgnoreCase))
                                                                                                                {
                                                                                                                    dbAsObj.pSchema.file_format = (u8)atoi(right);
                                                                                                                    sqlite3ResetInternalSchema(ctx, -1);
                                                                                                                }
                                                                                                                else
#endif

#if !OMIT_COMPILEOPTION_DIAGS
                                                                                                                    /*
**   PRAGMA compile_options
**
** Return the names of all compile-time options used in this build,
** one option per row.
*/
                                                                                                                    if (left.Equals("compile_options", StringComparison.OrdinalIgnoreCase))
                                                                                                                    {
                                                                                                                        int i = 0;
                                                                                                                        string zOpt;
                                                                                                                        sqlite3VdbeSetNumCols(v, 1);
                                                                                                                        parse.nMem = 1;
                                                                                                                        sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "compile_option", SQLITE_STATIC);
                                                                                                                        while ((zOpt = sqlite3_compileoption_get(i++)) != null)
                                                                                                                        {
                                                                                                                            sqlite3VdbeAddOp4(v, OP_String8, 0, 1, 0, zOpt, 0);
                                                                                                                            sqlite3VdbeAddOp2(v, OP_ResultRow, 1, 1);
                                                                                                                        }
                                                                                                                    }
                                                                                                                    else
#endif


#if !SQLITE_OMIT_WAL
                                                                                                                        /*
  **   PRAGMA [database.]wal_checkpoint = passive|full|restart
  **
  ** Checkpoint the database.
  */
                                                                                                                        if (sqlite3StrICmp(left, "wal_checkpoint") == 0)
                                                                                                                        {
                                                                                                                            int iBt = (id2->z ? db : SQLITE_MAX_ATTACHED);
                                                                                                                            int eMode = SQLITE_CHECKPOINT_PASSIVE;
                                                                                                                            if (right)
                                                                                                                            {
                                                                                                                                if (sqlite3StrICmp(right, "full") == 0)
                                                                                                                                {
                                                                                                                                    eMode = SQLITE_CHECKPOINT_FULL;
                                                                                                                                }
                                                                                                                                else if (sqlite3StrICmp(right, "restart") == 0)
                                                                                                                                {
                                                                                                                                    eMode = SQLITE_CHECKPOINT_RESTART;
                                                                                                                                }
                                                                                                                            }
                                                                                                                            if (sqlite3ReadSchema(parse)) goto pragma_out;
                                                                                                                            sqlite3VdbeSetNumCols(v, 3);
                                                                                                                            parse->nMem = 3;
                                                                                                                            sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "busy", SQLITE_STATIC);
                                                                                                                            sqlite3VdbeSetColName(v, 1, COLNAME_NAME, "log", SQLITE_STATIC);
                                                                                                                            sqlite3VdbeSetColName(v, 2, COLNAME_NAME, "checkpointed", SQLITE_STATIC);

                                                                                                                            sqlite3VdbeAddOp3(v, OP_Checkpoint, iBt, eMode, 1);
                                                                                                                            sqlite3VdbeAddOp2(v, OP_ResultRow, 1, 3);
                                                                                                                        }
                                                                                                                        else

                                                                                                                            /*
                                                                                                                            **   PRAGMA wal_autocheckpoint
                                                                                                                            **   PRAGMA wal_autocheckpoint = N
                                                                                                                            **
                                                                                                                            ** Configure a database connection to automatically checkpoint a database
                                                                                                                            ** after accumulating N frames in the log. Or query for the current value
                                                                                                                            ** of N.
                                                                                                                            */
                                                                                                                            if (sqlite3StrICmp(left, "wal_autocheckpoint") == 0)
                                                                                                                            {
                                                                                                                                if (right)
                                                                                                                                {
                                                                                                                                    sqlite3_wal_autocheckpoint(ctx, sqlite3Atoi(right));
                                                                                                                                }
                                                                                                                                ReturnSingleInt(parse, "wal_autocheckpoint",
                                                                                                                                   ctx->xWalCallback == sqlite3WalDefaultHook ?
                                                                                                                                       SQLITE_PTR_TO_INT(ctx->pWalArg) : 0);
                                                                                                                            }
                                                                                                                            else
#endif

#if DEBUG || TEST
                                                                                                                                /*
** Report the current state of file logs for all databases
*/
                                                                                                                                if (left.Equals("lock_status", StringComparison.OrdinalIgnoreCase))
                                                                                                                                {
                                                                                                                                    string[] azLockName = {
"unlocked", "shared", "reserved", "pending", "exclusive"
};
                                                                                                                                    int i;
                                                                                                                                    sqlite3VdbeSetNumCols(v, 2);
                                                                                                                                    parse.nMem = 2;
                                                                                                                                    sqlite3VdbeSetColName(v, 0, COLNAME_NAME, "database", SQLITE_STATIC);
                                                                                                                                    sqlite3VdbeSetColName(v, 1, COLNAME_NAME, "status", SQLITE_STATIC);
                                                                                                                                    for (i = 0; i < ctx.nDb; i++)
                                                                                                                                    {
                                                                                                                                        Btree pBt;
                                                                                                                                        Pager pPager;
                                                                                                                                        string zState = "unknown";
                                                                                                                                        sqlite3_int64 j = 0;
                                                                                                                                        if (ctx.aDb[i].zName == null)
                                                                                                                                            continue;
                                                                                                                                        sqlite3VdbeAddOp4(v, OP_String8, 0, 1, 0, ctx.aDb[i].zName, P4_STATIC);
                                                                                                                                        pBt = ctx.aDb[i].pBt;
                                                                                                                                        if (pBt == null || (pPager = sqlite3BtreePager(pBt)) == null)
                                                                                                                                        {
                                                                                                                                            zState = "closed";
                                                                                                                                        }
                                                                                                                                        else if (sqlite3_file_control(ctx, i != 0 ? ctx.aDb[i].zName : null,
                                                                                                                                 SQLITE_FCNTL_LOCKSTATE, ref j) == SQLITE_OK)
                                                                                                                                        {
                                                                                                                                            zState = azLockName[j];
                                                                                                                                        }
                                                                                                                                        sqlite3VdbeAddOp4(v, OP_String8, 0, 2, 0, zState, P4_STATIC);
                                                                                                                                        sqlite3VdbeAddOp2(v, OP_ResultRow, 1, 2);
                                                                                                                                    }
                                                                                                                                }
                                                                                                                                else
#endif

#if HAS_CODEC
                                                                                                                                    // needed to support key/rekey/hexrekey with pragma cmds
                                                                                                                                    if (left.Equals("key", StringComparison.OrdinalIgnoreCase) && !String.IsNullOrEmpty(right))
                                                                                                                                    {
                                                                                                                                        sqlite3_key(ctx, right, sqlite3Strlen30(right));
                                                                                                                                    }
                                                                                                                                    else
                                                                                                                                        if (left.Equals("rekey", StringComparison.OrdinalIgnoreCase) && !String.IsNullOrEmpty(right))
                                                                                                                                        {
                                                                                                                                            sqlite3_rekey(ctx, right, sqlite3Strlen30(right));
                                                                                                                                        }
                                                                                                                                        else
                                                                                                                                            if (!String.IsNullOrEmpty(right) && (left.Equals("hexkey", StringComparison.OrdinalIgnoreCase) ||
                                                                                                                                            left.Equals("hexrekey", StringComparison.OrdinalIgnoreCase)))
                                                                                                                                            {
                                                                                                                                                StringBuilder zKey = new StringBuilder(40);
                                                                                                                                                right.ToLower(new CultureInfo("en-us"));
                                                                                                                                                // expected '0x0102030405060708090a0b0c0d0e0f10'
                                                                                                                                                if (right.Length != 34)
                                                                                                                                                    return;

                                                                                                                                                for (int i = 2; i < right.Length; i += 2)
                                                                                                                                                {
                                                                                                                                                    int h1 = right[i];
                                                                                                                                                    int h2 = right[i + 1];
                                                                                                                                                    h1 += 9 * (1 & (h1 >> 6));
                                                                                                                                                    h2 += 9 * (1 & (h2 >> 6));
                                                                                                                                                    zKey.Append(Convert.ToChar((h2 & 0x0f) | ((h1 & 0xf) << 4)));
                                                                                                                                                }
                                                                                                                                                if ((left[3] & 0xf) == 0xb)
                                                                                                                                                {
                                                                                                                                                    sqlite3_key(ctx, zKey.ToString(), zKey.Length);
                                                                                                                                                }
                                                                                                                                                else
                                                                                                                                                {
                                                                                                                                                    sqlite3_rekey(ctx, zKey.ToString(), zKey.Length);
                                                                                                                                                }
                                                                                                                                            }
                                                                                                                                            else
#endif
#if HAS_CODEC || ENABLE_CEROD
                                                                                                                                                if (left.Equals("activate_extensions", StringComparison.OrdinalIgnoreCase))
                                                                                                                                                {
#if SQLITE_HAS_CODEC
                                                                      if ( !String.IsNullOrEmpty( zRight ) && zRight.Length > 4 && sqlite3StrNICmp( zRight, "see-", 4 ) == 0 )
                                                                      {
                                                                        sqlite3_activate_see( zRight.Substring( 4 ) );
                                                                      }
#endif
#if SQLITE_ENABLE_CEROD
if(  !String.IsNullOrEmpty( zRight ) &&  zRight.StartsWith("cerod-", StringComparison.OrdinalIgnoreCase))
{
sqlite3_activate_cerod( zRight.Substring( 6 ));
}
#endif
                                                                                                                                                }
                                                                                                                                                else
#endif
                                                                                                                                                { /* Empty ELSE clause */
                                                                                                                                                }

            /*
            ** Reset the safety level, in case the fullfsync flag or synchronous
            ** setting changed.
            */
#if !OMIT_PAGER_PRAGMAS
            if (ctx.autoCommit != 0)
            {
                sqlite3BtreeSetSafetyLevel(dbAsObj.pBt, dbAsObj.safety_level,
                ((ctx.flags & SQLITE_FullFSync) != 0) ? 1 : 0,
                ((ctx.flags & SQLITE_CkptFullFSync) != 0) ? 1 : 0);
            }
#endif
        pragma_out:
            sqlite3DbFree(ctx, ref left);
            sqlite3DbFree(ctx, ref right);
            ;
        }

#endif
    }
}
