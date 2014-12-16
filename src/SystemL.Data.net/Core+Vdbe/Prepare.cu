#include "Core+Vdbe.cu.h"

namespace Core
{
	__device__ static void CorruptSchema(InitData *data, const char *obj, const char *extra)
	{
		Context *ctx = data->Ctx;
		if (!ctx->MallocFailed && (ctx->Flags & Context::FLAG_RecoveryMode) == 0)
		{
			if (!obj) obj = "?";
			sqlite3SetString(data->ErrMsg, ctx, "malformed database schema (%s)", obj);
			if (extra)
				*data->ErrMsg = _mtagappendf(ctx, *data->ErrMsg, "%s - %s", *data->ErrMsg, extra);
		}
		data->RC = (ctx->MallocFailed ? RC_NOMEM : SysEx_CORRUPT_BKPT);
	}

	__device__ bool Prepare::InitCallback(void *init, int argc, char **argv, char **notUsed1)
	{
		InitData *data = (InitData *)init;
		Context *ctx = data->Ctx;
		int db = data->Db;

		_assert(argc == 3);
		_assert(MutexEx::Held(ctx->Mutex));
		DbClearProperty(ctx, db, DB_Empty);
		if (ctx->MallocFailed)
		{
			CorruptSchema(data, argv[0], nullptr);
			return true;
		}

		_assert(db >= 0 && db < ctx->DBs.length);
		if (argv == nullptr) return false; // Might happen if EMPTY_RESULT_CALLBACKS are on
		if (argv[1] == 0)
			CorruptSchema(data, argv[0], nullptr);
		else if (argv[2] && argv[2][0])
		{
			// Call the parser to process a CREATE TABLE, INDEX or VIEW. But because ctx->init.busy is set to 1, no VDBE code is generated
			// or executed.  All the parser does is build the internal data structures that describe the table, index, or view.
			_assert(ctx->Init.Busy);
			ctx->Init.DB = db;
			ctx->Init.NewTid = ConvertEx::Atoi(argv[1]);
			ctx->Init.OrphanTrigger = false;
			Vdbe *stmt;
#if DEBUG
			int rcp = Prepare(ctx, argv[2], -1, &stmt, 0);
#else
			Prepare(ctx, argv[2], -1, &stmt, 0);
#endif
			RC rc = ctx->ErrCode;
			_assert((rc&0xFF) == (rcp&0xFF));
			ctx->Init.DB = 0;
			if (rc != RC_OK)
			{
				if (ctx->Init.OrphanTrigger)
					_assert(db == 1);
				else
				{
					data->RC = rc;
					if (rc == RC_NOMEM)
						ctx->MallocFailed = true;
					else if (rc != RC_INTERRUPT && (rc&0xFF) != RC_LOCKED)
						CorruptSchema(data, argv[0], sqlite3_errmsg(ctx));
				}
			}
			stmt->Finalize();
		}
		else if (argv[0] == 0)
			CorruptSchema(data, nullptr, nullptr);
		else
		{
			// If the SQL column is blank it means this is an index that was created to be the PRIMARY KEY or to fulfill a UNIQUE
			// constraint for a CREATE TABLE.  The index should have already been created when we processed the CREATE TABLE.  All we have
			// to do here is record the root page number for that index.
			Index *index = Build::FindIndex(ctx, argv[0], ctx->DBs[db].Name);
			if (!index)
			{
				// This can occur if there exists an index on a TEMP table which has the same name as another index on a permanent index.  Since
				// the permanent table is hidden by the TEMP table, we can also safely ignore the index on the permanent table.
				// Do Nothing
			}
			else if (!ConvertEx::Atoi(argv[1], &index->Id))
				CorruptSchema(data, argv[0], "invalid rootpage");
		}
		return false;
	}

	// The master database table has a structure like this
	__constant__ static const char master_schema[] = 
		"CREATE TABLE sqlite_master(\n"
		"  type text,\n"
		"  name text,\n"
		"  tbl_name text,\n"
		"  rootpage integer,\n"
		"  sql text\n"
		")";
#ifndef OMIT_TEMPDB
	__constant__ static const char temp_master_schema[] = 
		"CREATE TEMP TABLE sqlite_temp_master(\n"
		"  type text,\n"
		"  name text,\n"
		"  tbl_name text,\n"
		"  rootpage integer,\n"
		"  sql text\n"
		")";
#else
#define temp_master_schema nullptr
#endif

	__device__ static RC Prepare::InitOne(Context *ctx, int iDb, char **errMsg)
	{
		_assert(db >= 0 && db < ctx->DBs.length);
		_assert(ctx->DBs[db].Schema);
		_assert(MutexEx::Held(ctx->Mutex));
		_assert(db == 1 || Btree::HoldsMutex(ctx->DBs[db].Bt));
		RC rc;
		int i;

		// zMasterSchema and zInitScript are set to point at the master schema and initialisation script appropriate for the database being
		// initialized. zMasterName is the name of the master table.
		char const *masterSchema = (!OMIT_TEMPDB && db == 1 ? temp_master_schema : master_schema);
		char const *masterName = SCHEMA_TABLE(iDb);

		// Construct the schema tables.
		char const *args[4];
		args[0] = masterName;
		args[1] = "1";
		args[2] = masterSchema;
		args[3] = 0;
		InitData initData;
		initData.Ctx = ctx;
		initData.Db = db;
		initData.RC = RC_OK;
		initData.ErrMsg = errMsg;
		InitCallback(&initData, 3, (char **)args, nullptr);
		if (initData.RC)
		{
			rc = initData.RC;
			goto error_out;
		}
		Table *table = Build::FindTable(ctx, masterName, ctx->DBs[db].Name);
		if (_ALWAYS(table))
			table->TabFlags |= TF_Readonly;

		// Create a cursor to hold the database open
		Context::DB *dbAsObj = &ctx->DBs[db];
		if (!dbAsObj->Bt)
		{
			if (!OMIT_TEMPDB && _ALWAYS(db == 1))
				DbSetProperty(ctx, 1, DB_SchemaLoaded);
			return RC_OK;
		}

		// If there is not already a read-only (or read-write) transaction opened on the b-tree database, open one now. If a transaction is opened, it 
		// will be closed before this function returns.  */
		bool openedTransaction = false;
		dbAsObj->Bt->Enter();
		if (!dbAsObj->Bt->IsInReadTrans())
		{
			rc = dbAsObj->Bt->BeginTrans(0);
			if (rc != RC_OK)
			{
				SysEx::TagStrSet(errMsg, ctx, "%s", SysEx::ErrStr(rc));
				goto initone_error_out;
			}
			openedTransaction = true;
		}

		// Get the database meta information.
		// Meta values are as follows:
		//    meta[0]   Schema cookie.  Changes with each schema change.
		//    meta[1]   File format of schema layer.
		//    meta[2]   Size of the page cache.
		//    meta[3]   Largest rootpage (auto/incr_vacuum mode)
		//    meta[4]   Db text encoding. 1:UTF-8 2:UTF-16LE 3:UTF-16BE
		//    meta[5]   User version
		//    meta[6]   Incremental vacuum mode
		//    meta[7]   unused
		//    meta[8]   unused
		//    meta[9]   unused
		// Note: The #defined SQLITE_UTF* symbols in sqliteInt.h correspond to the possible values of meta[4].
		int meta[5];
		for (i = 0; i < _lengthof(meta); i++)
			dbAsObj->Bt->GetMeta((Btree::META)i+1, (uint32 *)&meta[i]);
		dbAsObj->Schema->SchemaCookie = meta[Btree::META_SCHEMA_VERSION-1];

		int i;
		// If opening a non-empty database, check the text encoding. For the main database, set sqlite3.enc to the encoding of the main database.
		// For an attached ctx, it is an error if the encoding is not the same as sqlite3.enc.
		if (meta[Btree::META_TEXT_ENCODING-1]) // text encoding
		{  
			if (db == 0)
			{
#ifndef OMIT_UTF16
				uint8 encoding;
				// If opening the main database, set ENC(ctx).
				encoding = (uint8)meta[Btree::META_TEXT_ENCODING-1] & 3;
				if (encoding == 0) encoding = TEXTENCODE_UTF8;
				CTXENCODE(ctx) = encoding;
#else
				CTXENCODE(ctx) = TEXTENCODE_UTF8;
#endif
			}
			else
			{
				// If opening an attached database, the encoding much match CTXENCODE(ctx)
				if (meta[Btree::META_TEXT_ENCODING-1] != CTXENCODE(ctx))
				{
					SysEx::TagStrSet(errMsg, ctx, "attached databases must use the same text encoding as main database");
					rc = RC_ERROR;
					goto initone_error_out;
				}
			}
		}
		else
			DbSetProperty(ctx, db, DB_Empty);
		dbAsObj->Schema->Encode = CTXENCODE(ctx);

		if (dbAsObj->Schema->CacheSize == 0)
		{
			dbAsObj->Schema->CacheSize = DEFAULT_CACHE_SIZE;
			dbAsObj->Bt->SetCacheSize(dbAsObj->Schema->CacheSize);
		}

		// file_format==1    Version 3.0.0.
		// file_format==2    Version 3.1.3.  // ALTER TABLE ADD COLUMN
		// file_format==3    Version 3.1.4.  // ditto but with non-NULL defaults
		// file_format==4    Version 3.3.0.  // DESC indices.  Boolean constants
		dbAsObj->Schema->FileFormat = (uint8)meta[Btree::META_FILE_FORMAT-1];
		if (dbAsObj->Schema->FileFormat == 0)
			dbAsObj->Schema->FileFormat = 1;
		if (dbAsObj->Schema->FileFormat > MAX_FILE_FORMAT)
		{
			SysEx::TagStrSet(errMsg, ctx, "unsupported file format");
			rc = RC_ERROR;
			goto initone_error_out;
		}

		// Ticket #2804:  When we open a database in the newer file format, clear the legacy_file_format pragma flag so that a VACUUM will
		// not downgrade the database and thus invalidate any descending indices that the user might have created.
		if (db == 0 && meta[Btree::META_FILE_FORMAT-1] >= 4)
			ctx->Flags &= ~Context::FLAG_LegacyFileFmt;

		// Read the schema information out of the schema tables
		_assert(ctx->Init.Busy);
		{
			char *sql = _mtagprintf(ctx,  "SELECT name, rootpage, sql FROM '%q'.%s ORDER BY rowid", ctx->DBs[db].Name, masterName);
#ifndef OMIT_AUTHORIZATION
			{
				int (*auth)(void*,int,const char*,const char*,const char*,const char*);
				auth = ctx->Auth;
				ctx->Auth = nullptr;
#endif
				rc = sqlite3_exec(ctx, sql, InitCallback, &initData, 0);
#ifndef OMIT_AUTHORIZATION
				ctx->Auth = auth;
			}
#endif
			if (rc == RC_OK) rc = initData.RC;
			_tagfree(ctx, sql);
#ifndef OMIT_ANALYZE
			if (rc == RC_OK)
				sqlite3AnalysisLoad(ctx, db);
#endif
		}
		if (ctx->MallocFailed)
		{
			rc = RC_NOMEM;
			Main::ResetAllSchemasOfConnection(ctx);
		}
		if (rc == RC_OK || (ctx->Flags&Context::FLAG_RecoveryMode))
		{
			// Black magic: If the SQLITE_RecoveryMode flag is set, then consider the schema loaded, even if errors occurred. In this situation the 
			// current sqlite3_prepare() operation will fail, but the following one will attempt to compile the supplied statement against whatever subset
			// of the schema was loaded before the error occurred. The primary purpose of this is to allow access to the sqlite_master table
			// even when its contents have been corrupted.
			DbSetProperty(ctx, db, DB_SchemaLoaded);
			rc = RC_OK;
		}

		// Jump here for an error that occurs after successfully allocating curMain and calling sqlite3BtreeEnter(). For an error that occurs
		// before that point, jump to error_out.
initone_error_out:
		if (openedTransaction)
			dbAsObj->Bt->Commit();
		dbAsObj->Bt->Leave();

error_out:
		if (rc == RC_NOMEM || rc == RC_IOERR_NOMEM)
			ctx->MallocFailed = true;
		return rc;
	}

	__device__ RC Prepare::Init(Context *ctx, char **errMsg)
	{
		_assert(MutexEx::Held(ctx->Mutex));
		RC rc = RC_OK;
		ctx->Init.Busy = true;
		for (int i = 0; rc == RC_OK && i < ctx->DBs.length; i++)
		{
			if (DbHasProperty(ctx, i, DB_SchemaLoaded) || i == 1) continue;
			rc = InitOne(ctx, i, errMsg);
			if (rc)
				ResetOneSchema(ctx, i);
		}

		// Once all the other databases have been initialized, load the schema for the TEMP database. This is loaded last, as the TEMP database
		// schema may contain references to objects in other databases.
#ifndef OMIT_TEMPDB
		if (rc == RC_OK && _ALWAYS(ctx->DBs.length > 1) && !DbHasProperty(ctx, 1, DB_SchemaLoaded))
		{
			rc = InitOne(ctx, 1, errMsg);
			if (rc)
				ResetOneSchema(ctx, 1);
		}
#endif

		bool commitInternal = !(ctx->Flags&Context::FLAG_InternChanges);
		ctx->Init.Busy = false;
		if (rc == RC_OK && CommitInternal)
			CommitInternalChanges(ctx);
		return rc; 
	}

	__device__ int Prepare::ReadSchema(Parse *parse)
	{
		RC rc = RC_OK;
		Context *ctx = pParse->Ctx;
		_assert(MutexEx::Held(ctx->Mutex));
		if (!ctx->Init.Busy)
			rc = Init(ctx, &parse->ErrMsg);
		if (rc != RC_OK)
		{
			parse->RC = rc;
			parse->Errs++;
		}
		return rc;
	}

	__device__ static void SchemaIsValid(Parse *parse)
	{
		Context *ctx = parse->Ctx;
		_assert(parse->CheckSchema);
		_assert(MutexEx::Held(ctx->Mutex));
		for (int db = 0; db <ctx->DBs.length; db++)
		{
			bool openedTransaction = false; // True if a transaction is opened
			Btree *bt = ctx->DBs[db].Bt; // Btree database to read cookie from
			if (!bt) continue;

			// If there is not already a read-only (or read-write) transaction opened on the b-tree database, open one now. If a transaction is opened, it 
			// will be closed immediately after reading the meta-value.
			if (!bt->IsInReadTrans())
			{
				RC rc = bt->BeginTrans(0);
				if (rc == RC_NOMEM || rc == RC_IOERR_NOMEM)
					ctx->MallocFailed = true;
				if (rc != RC_OK) return;
				openedTransaction = true;
			}

			// Read the schema cookie from the database. If it does not match the value stored as part of the in-memory schema representation,
			// set Parse.rc to SQLITE_SCHEMA.
			int cookie;
			bt->GetMeta(Btree::META_SCHEMA_VERSION, (uint32 *)&cookie);
			_assert(Btree::SchemaMutexHeld(ctx, db, 0));
			if (cookie != ctx->DBs[db].Schema->SchemaCookie)
			{
				sqlite3ResetOneSchema(ctx, db);
				parse->RC = RC_SCHEMA;
			}

			// Close the transaction, if one was opened.
			if (openedTransaction)
				bt->Commit();
		}
	}

	__device__ int Prepare::SchemaToIndex(Context *ctx, Schema *schema)
	{
		// If pSchema is NULL, then return -1000000. This happens when code in expr.c is trying to resolve a reference to a transient table (i.e. one
		// created by a sub-select). In this case the return value of this function should never be used.
		//
		// We return -1000000 instead of the more usual -1 simply because using -1000000 as the incorrect index into ctx->aDb[] is much 
		// more likely to cause a segfault than -1 (of course there are assert() statements too, but it never hurts to play the odds).
		_assert(MutexEx::Held(ctx->Mutex));
		int i = -1000000;
		if (schema)
		{
			for (i = 0; _ALWAYS(i < ctx->DBs.length); i++)
				if (ctx->DBs[i].Schema == schema)
					break;
			_assert(i >= 0 && i < ctx->DBs.length);
		}
		return i;
	}

	/*
	** Compile the UTF-8 encoded SQL statement zSql into a statement handle.
	*/
	static int sqlite3Prepare(
		sqlite3 *ctx,              /* Database handle. */
		const char *zSql,         /* UTF-8 encoded SQL statement. */
		int nBytes,               /* Length of zSql in bytes. */
		int saveSqlFlag,          /* True to copy SQL text into the sqlite3_stmt */
		Vdbe *pReprepare,         /* VM being reprepared */
		sqlite3_stmt **ppStmt,    /* OUT: A pointer to the prepared statement */
		const char **pzTail       /* OUT: End of parsed string */
		){
			Parse *pParse;            /* Parsing context */
			char *zErrMsg = 0;        /* Error message */
			int rc = SQLITE_OK;       /* Result code */
			int i;                    /* Loop counter */

			/* Allocate the parsing context */
			pParse = sqlite3StackAllocZero(ctx, sizeof(*pParse));
			if( pParse==0 ){
				rc = SQLITE_NOMEM;
				goto end_prepare;
			}
			pParse->pReprepare = pReprepare;
			assert( ppStmt && *ppStmt==0 );
			assert( !ctx->mallocFailed );
			assert( sqlite3_mutex_held(ctx->mutex) );

			/* Check to verify that it is possible to get a read lock on all
			** database schemas.  The inability to get a read lock indicates that
			** some other database connection is holding a write-lock, which in
			** turn means that the other connection has made uncommitted changes
			** to the schema.
			**
			** Were we to proceed and prepare the statement against the uncommitted
			** schema changes and if those schema changes are subsequently rolled
			** back and different changes are made in their place, then when this
			** prepared statement goes to run the schema cookie would fail to detect
			** the schema change.  Disaster would follow.
			**
			** This thread is currently holding mutexes on all Btrees (because
			** of the sqlite3BtreeEnterAll() in sqlite3LockAndPrepare()) so it
			** is not possible for another thread to start a new schema change
			** while this routine is running.  Hence, we do not need to hold 
			** locks on the schema, we just need to make sure nobody else is 
			** holding them.
			**
			** Note that setting READ_UNCOMMITTED overrides most lock detection,
			** but it does *not* override schema lock detection, so this all still
			** works even if READ_UNCOMMITTED is set.
			*/
			for(i=0; i<ctx->nDb; i++) {
				Btree *pBt = ctx->aDb[i].pBt;
				if( pBt ){
					assert( sqlite3BtreeHoldsMutex(pBt) );
					rc = sqlite3BtreeSchemaLocked(pBt);
					if( rc ){
						const char *zDb = ctx->aDb[i].zName;
						sqlite3Error(ctx, rc, "database schema is locked: %s", zDb);
						testcase( ctx->flags & SQLITE_ReadUncommitted );
						goto end_prepare;
					}
				}
			}

			sqlite3VtabUnlockList(ctx);

			pParse->Ctx = ctx;
			pParse->nQueryLoop = (double)1;
			if( nBytes>=0 && (nBytes==0 || zSql[nBytes-1]!=0) ){
				char *zSqlCopy;
				int mxLen = ctx->aLimit[SQLITE_LIMIT_SQL_LENGTH];
				testcase( nBytes==mxLen );
				testcase( nBytes==mxLen+1 );
				if( nBytes>mxLen ){
					sqlite3Error(ctx, SQLITE_TOOBIG, "statement too long");
					rc = sqlite3ApiExit(ctx, SQLITE_TOOBIG);
					goto end_prepare;
				}
				zSqlCopy = sqlite3DbStrNDup(ctx, zSql, nBytes);
				if( zSqlCopy ){
					sqlite3RunParser(pParse, zSqlCopy, &zErrMsg);
					sqlite3DbFree(ctx, zSqlCopy);
					pParse->zTail = &zSql[pParse->zTail-zSqlCopy];
				}else{
					pParse->zTail = &zSql[nBytes];
				}
			}else{
				sqlite3RunParser(pParse, zSql, &zErrMsg);
			}
			assert( 1==(int)pParse->nQueryLoop );

			if( ctx->mallocFailed ){
				pParse->rc = SQLITE_NOMEM;
			}
			if( pParse->rc==SQLITE_DONE ) pParse->rc = SQLITE_OK;
			if( pParse->checkSchema ){
				schemaIsValid(pParse);
			}
			if( ctx->mallocFailed ){
				pParse->rc = SQLITE_NOMEM;
			}
			if( pzTail ){
				*pzTail = pParse->zTail;
			}
			rc = pParse->rc;

#ifndef SQLITE_OMIT_EXPLAIN
			if( rc==SQLITE_OK && pParse->pVdbe && pParse->explain ){
				static const char * const azColName[] = {
					"addr", "opcode", "p1", "p2", "p3", "p4", "p5", "comment",
					"selectid", "order", "from", "detail"
				};
				int iFirst, mx;
				if( pParse->explain==2 ){
					sqlite3VdbeSetNumCols(pParse->pVdbe, 4);
					iFirst = 8;
					mx = 12;
				}else{
					sqlite3VdbeSetNumCols(pParse->pVdbe, 8);
					iFirst = 0;
					mx = 8;
				}
				for(i=iFirst; i<mx; i++){
					sqlite3VdbeSetColName(pParse->pVdbe, i-iFirst, COLNAME_NAME,
						azColName[i], SQLITE_STATIC);
				}
			}
#endif

			assert( ctx->init.busy==0 || saveSqlFlag==0 );
			if( ctx->init.busy==0 ){
				Vdbe *pVdbe = pParse->pVdbe;
				sqlite3VdbeSetSql(pVdbe, zSql, (int)(pParse->zTail-zSql), saveSqlFlag);
			}
			if( pParse->pVdbe && (rc!=SQLITE_OK || ctx->mallocFailed) ){
				sqlite3VdbeFinalize(pParse->pVdbe);
				assert(!(*ppStmt));
			}else{
				*ppStmt = (sqlite3_stmt*)pParse->pVdbe;
			}

			if( zErrMsg ){
				sqlite3Error(ctx, rc, "%s", zErrMsg);
				sqlite3DbFree(ctx, zErrMsg);
			}else{
				sqlite3Error(ctx, rc, 0);
			}

			/* Delete any TriggerPrg structures allocated while parsing this statement. */
			while( pParse->pTriggerPrg ){
				TriggerPrg *pT = pParse->pTriggerPrg;
				pParse->pTriggerPrg = pT->pNext;
				sqlite3DbFree(ctx, pT);
			}

end_prepare:

			sqlite3StackFree(ctx, pParse);
			rc = sqlite3ApiExit(ctx, rc);
			assert( (rc&ctx->errMask)==rc );
			return rc;
	}
	static int sqlite3LockAndPrepare(
		sqlite3 *ctx,              /* Database handle. */
		const char *zSql,         /* UTF-8 encoded SQL statement. */
		int nBytes,               /* Length of zSql in bytes. */
		int saveSqlFlag,          /* True to copy SQL text into the sqlite3_stmt */
		Vdbe *pOld,               /* VM being reprepared */
		sqlite3_stmt **ppStmt,    /* OUT: A pointer to the prepared statement */
		const char **pzTail       /* OUT: End of parsed string */
		){
			int rc;
			assert( ppStmt!=0 );
			*ppStmt = 0;
			if( !sqlite3SafetyCheckOk(ctx) ){
				return SQLITE_MISUSE_BKPT;
			}
			sqlite3_mutex_enter(ctx->mutex);
			sqlite3BtreeEnterAll(ctx);
			rc = sqlite3Prepare(ctx, zSql, nBytes, saveSqlFlag, pOld, ppStmt, pzTail);
			if( rc==SQLITE_SCHEMA ){
				sqlite3_finalize(*ppStmt);
				rc = sqlite3Prepare(ctx, zSql, nBytes, saveSqlFlag, pOld, ppStmt, pzTail);
			}
			sqlite3BtreeLeaveAll(ctx);
			sqlite3_mutex_leave(ctx->mutex);
			assert( rc==SQLITE_OK || *ppStmt==0 );
			return rc;
	}

	/*
	** Rerun the compilation of a statement after a schema change.
	**
	** If the statement is successfully recompiled, return SQLITE_OK. Otherwise,
	** if the statement cannot be recompiled because another connection has
	** locked the sqlite3_master table, return SQLITE_LOCKED. If any other error
	** occurs, return SQLITE_SCHEMA.
	*/
	int sqlite3Reprepare(Vdbe *p){
		int rc;
		sqlite3_stmt *pNew;
		const char *zSql;
		sqlite3 *ctx;

		assert( sqlite3_mutex_held(sqlite3VdbeDb(p)->mutex) );
		zSql = sqlite3_sql((sqlite3_stmt *)p);
		assert( zSql!=0 );  /* Reprepare only called for prepare_v2() statements */
		ctx = sqlite3VdbeDb(p);
		assert( sqlite3_mutex_held(ctx->mutex) );
		rc = sqlite3LockAndPrepare(ctx, zSql, -1, 0, p, &pNew, 0);
		if( rc ){
			if( rc==SQLITE_NOMEM ){
				ctx->mallocFailed = 1;
			}
			assert( pNew==0 );
			return rc;
		}else{
			assert( pNew!=0 );
		}
		sqlite3VdbeSwap((Vdbe*)pNew, p);
		sqlite3TransferBindings(pNew, (sqlite3_stmt*)p);
		sqlite3VdbeResetStepResult((Vdbe*)pNew);
		sqlite3VdbeFinalize((Vdbe*)pNew);
		return SQLITE_OK;
	}


	/*
	** Two versions of the official API.  Legacy and new use.  In the legacy
	** version, the original SQL text is not saved in the prepared statement
	** and so if a schema change occurs, SQLITE_SCHEMA is returned by
	** sqlite3_step().  In the new version, the original SQL text is retained
	** and the statement is automatically recompiled if an schema change
	** occurs.
	*/
	int sqlite3_prepare(
		sqlite3 *ctx,              /* Database handle. */
		const char *zSql,         /* UTF-8 encoded SQL statement. */
		int nBytes,               /* Length of zSql in bytes. */
		sqlite3_stmt **ppStmt,    /* OUT: A pointer to the prepared statement */
		const char **pzTail       /* OUT: End of parsed string */
		){
			int rc;
			rc = sqlite3LockAndPrepare(ctx,zSql,nBytes,0,0,ppStmt,pzTail);
			assert( rc==SQLITE_OK || ppStmt==0 || *ppStmt==0 );  /* VERIFY: F13021 */
			return rc;
	}
	int sqlite3_prepare_v2(
		sqlite3 *ctx,              /* Database handle. */
		const char *zSql,         /* UTF-8 encoded SQL statement. */
		int nBytes,               /* Length of zSql in bytes. */
		sqlite3_stmt **ppStmt,    /* OUT: A pointer to the prepared statement */
		const char **pzTail       /* OUT: End of parsed string */
		){
			int rc;
			rc = sqlite3LockAndPrepare(ctx,zSql,nBytes,1,0,ppStmt,pzTail);
			assert( rc==SQLITE_OK || ppStmt==0 || *ppStmt==0 );  /* VERIFY: F13021 */
			return rc;
	}


#ifndef SQLITE_OMIT_UTF16
	/*
	** Compile the UTF-16 encoded SQL statement zSql into a statement handle.
	*/
	static int sqlite3Prepare16(
		sqlite3 *ctx,              /* Database handle. */ 
		const void *zSql,         /* UTF-16 encoded SQL statement. */
		int nBytes,               /* Length of zSql in bytes. */
		int saveSqlFlag,          /* True to save SQL text into the sqlite3_stmt */
		sqlite3_stmt **ppStmt,    /* OUT: A pointer to the prepared statement */
		const void **pzTail       /* OUT: End of parsed string */
		){
			/* This function currently works by first transforming the UTF-16
			** encoded string to UTF-8, then invoking sqlite3_prepare(). The
			** tricky bit is figuring out the pointer to return in *pzTail.
			*/
			char *zSql8;
			const char *zTail8 = 0;
			int rc = SQLITE_OK;

			assert( ppStmt );
			*ppStmt = 0;
			if( !sqlite3SafetyCheckOk(ctx) ){
				return SQLITE_MISUSE_BKPT;
			}
			sqlite3_mutex_enter(ctx->mutex);
			zSql8 = sqlite3Utf16to8(ctx, zSql, nBytes, SQLITE_UTF16NATIVE);
			if( zSql8 ){
				rc = sqlite3LockAndPrepare(ctx, zSql8, -1, saveSqlFlag, 0, ppStmt, &zTail8);
			}

			if( zTail8 && pzTail ){
				/* If sqlite3_prepare returns a tail pointer, we calculate the
				** equivalent pointer into the UTF-16 string by counting the unicode
				** characters between zSql8 and zTail8, and then returning a pointer
				** the same number of characters into the UTF-16 string.
				*/
				int chars_parsed = sqlite3Utf8CharLen(zSql8, (int)(zTail8-zSql8));
				*pzTail = (u8 *)zSql + sqlite3Utf16ByteLen(zSql, chars_parsed);
			}
			sqlite3DbFree(ctx, zSql8); 
			rc = sqlite3ApiExit(ctx, rc);
			sqlite3_mutex_leave(ctx->mutex);
			return rc;
	}

	/*
	** Two versions of the official API.  Legacy and new use.  In the legacy
	** version, the original SQL text is not saved in the prepared statement
	** and so if a schema change occurs, SQLITE_SCHEMA is returned by
	** sqlite3_step().  In the new version, the original SQL text is retained
	** and the statement is automatically recompiled if an schema change
	** occurs.
	*/
	int sqlite3_prepare16(
		sqlite3 *ctx,              /* Database handle. */ 
		const void *zSql,         /* UTF-16 encoded SQL statement. */
		int nBytes,               /* Length of zSql in bytes. */
		sqlite3_stmt **ppStmt,    /* OUT: A pointer to the prepared statement */
		const void **pzTail       /* OUT: End of parsed string */
		){
			int rc;
			rc = sqlite3Prepare16(ctx,zSql,nBytes,0,ppStmt,pzTail);
			assert( rc==SQLITE_OK || ppStmt==0 || *ppStmt==0 );  /* VERIFY: F13021 */
			return rc;
	}
	int sqlite3_prepare16_v2(
		sqlite3 *ctx,              /* Database handle. */ 
		const void *zSql,         /* UTF-16 encoded SQL statement. */
		int nBytes,               /* Length of zSql in bytes. */
		sqlite3_stmt **ppStmt,    /* OUT: A pointer to the prepared statement */
		const void **pzTail       /* OUT: End of parsed string */
		){
			int rc;
			rc = sqlite3Prepare16(ctx,zSql,nBytes,1,ppStmt,pzTail);
			assert( rc==SQLITE_OK || ppStmt==0 || *ppStmt==0 );  /* VERIFY: F13021 */
			return rc;
	}

#endif
}