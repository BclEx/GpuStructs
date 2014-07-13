// vdbeapi.cu
#include "VdbeInt.cu.h"

#pragma region Name1

__device__ static int vdbeSafety(Vdbe *p)
{
	if (p->Db == nullptr)
	{
		SysEx_LOG(RC_MISUSE, "API called with finalized prepared statement");
		return 1;
	}
	return 0;
}

__device__ static int vdbeSafetyNotNull(Vdbe *p)
{
	if (p == nullptr)
	{
		SysEx_LOG(RC_MISUSE, "API called with NULL prepared statement");
		return 1;
	}
	return vdbeSafety(p);
}

__device__ RC sqlite3_finalize(Vdbe *p)
{
	if (p == nullptr)
		return RC_OK; // IMPLEMENTATION-OF: R-57228-12904 Invoking sqlite3_finalize() on a NULL pointer is a harmless no-op.
	Context *db = p->Db;
	if (vdbeSafety(p)) return RC_MISUSE_BKPT;
	MutexEx::Enter(db->Mutex);
	RC rc = p->Finalize();
	rc = sqlite3ApiExit(db, rc);
	sqlite3LeaveMutexAndCloseZombie(db);
	return rc;
}

__device__ RC sqlite3_reset(Vdbe *p)
{
	if (p == nullptr)
		return RC_OK;
#if THREADSAFE
	MutexEx mutex = p->Db->Mutex;
#endif
	MutexEx::Enter(mutex);
	RC rc = p->Reset();
	p->Rewind();
	_assert((rc & p->Db->ErrMask) == rc);
	rc = sqlite3ApiExit(p->Db, rc);
	MutexEx::Leave(mutex);
	return rc;
}

__device__ RC sqlite3_clear_bindings(Vdbe *p)
{
	RC rc = RC_OK;
#if THREADSAFE
	MutexEx mutex = p->Db->Mutex;
#endif
	MutexEx::Enter(mutex);
	for (int i = 0; i < p->Vars.length; i++)
	{
		Vdbe::MemRelease(&p->Vars[i]);
		p->Vars[i].Flags = MEM_Null;
	}
	if (p->IsPrepareV2 && p->Expmask)
		p->Expired = true;
	MutexEx::Leave(mutex);
	return rc;
}

#pragma endregion

#pragma region Value

__device__ const void *sqlite3_value_blob(Mem *p)
{
	if (p->Flags & (MEM_Blob|MEM_Str))
	{
		Vdbe::MemExpandBlob(p);
		p->Flags &= ~MEM_Str;
		p->Flags |= MEM_Blob;
		return (p->N ? p->Z : nullptr);
	}
	return sqlite3_value_text(val);
}
__device__ int sqlite3_value_bytes(Mem *val) { return sqlite3ValueBytes(val, TEXTENCODE_UTF8); }
__device__ int sqlite3_value_bytes16(Mem *val) { return sqlite3ValueBytes(val, TEXTENCODE_UTF16NATIVE); }
__device__ double sqlite3_value_double(Mem *val) { return Vdbe::RealValue(val); }
__device__ int sqlite3_value_int(Mem *val) { return (int)Vdbe::IntValue(val); }
__device__ int64 sqlite3_value_int64(Mem *val) { return Vdbe::IntValue(val); }
__device__ const unsigned char *sqlite3_value_text(Mem *val) { return (const unsigned char *)sqlite3ValueText(val, TEXTENCODE_UTF8); }
#ifndef OMIT_UTF16
__device__ const void *sqlite3_value_text16(Mem *val) { return sqlite3ValueText(val, TEXTENCODE_UTF16NATIVE); }
__device__ const void *sqlite3_value_text16be(Mem *val) { return sqlite3ValueText(val, TEXTENCODE_UTF16BE); }
__device__ const void *sqlite3_value_text16le(Mem *val) { return sqlite3ValueText(val, TEXTENCODE_UTF16LE); }
#endif
__device__ int sqlite3_value_type(Mem *val) { return val->Type; }

#pragma endregion

#pragma region Result

__device__ inline static void setResultStrOrError(FuncContext *ctx, const char *z, int n, uint8 enc, void (*del)(void *))
{
	if (Vdbe::MemSetStr(&ctx->S, z, n, enc, del) == RC_TOOBIG)
		sqlite3_result_error_toobig(ctx);
}
__device__ void sqlite3_result_blob(FuncContext *ctx, const void *z, int n, void (*del)(void *))
{
	_assert(n >= 0);
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	setResultStrOrError(ctx, z, n, 0, del);
}
__device__ void sqlite3_result_double(FuncContext *ctx, double value)
{
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	Vdbe::MemSetDouble(&pCtx->s, value);
}
__device__ void sqlite3_result_error(FuncContext *ctx, const char *z, int n)
{
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	ctx->IsError = RC_ERROR;
	Vdbe::MemSetStr(&ctx->S, z, n, TEXTENCODE_UTF8, SQLITE_TRANSIENT);
}
#ifndef OMIT_UTF16
__device__ void sqlite3_result_error16(FuncContext *ctx, const void *z, int n)
{
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	pCtx->isError = RC_ERROR;
	Vdbe::MemSetStr(&ctx->S, z, n, TEXTENCODE_UTF16NATIVE, SQLITE_TRANSIENT);
}
#endif
__device__ void sqlite3_result_int(FuncContext *ctx, int value)
{
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	Vdbe::MemSetInt64(&ctx->s, (int64)value);
}
__device__ void sqlite3_result_int64(FuncContext *ctx, int64 value)
{
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	Vdbe::MemSetInt64(&ctx->s, value);
}
__device__ void sqlite3_result_null(sqlite3_context *ctx)
{
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	Vdbe::MemSetNull(&ctx->S);
}
__device__ void sqlite3_result_text(FuncContext *ctx, const char *z, int n, void (*del)(void *))
{
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	setResultStrOrError(ctx, z, n, TEXTENCODE_UTF8, del);
}
#ifndef OMIT_UTF16
__device__ void sqlite3_result_text16(FuncContext *ctx, const void *z, int n, void (*del)(void *))
{
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	setResultStrOrError(ctx, z, n, TEXTENCODE_UTF16NATIVE, del);
}
__device__ void sqlite3_result_text16be(FuncContext *ctx, const void *z, int n, void (*del)(void *))
{
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	setResultStrOrError(ctx, z, n, TEXTENCODE_UTF16BE, del);
}
__device__ void sqlite3_result_text16le(FuncContext *ctx, const void *z, int n, void (*del)(void *))
{
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	setResultStrOrError(ctx, z, n, TEXTENCODE_UTF16LE, del);
}
#endif

__device__ void sqlite3_result_value(FuncContext *ctx, Mem *value)
{
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	Vdbe::MemCopy(&ctx->S, value);
}
__device__ void sqlite3_result_zeroblob(FuncContext *ctx, int n)
{
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	Vdbe::MemSetZeroBlob(&ctx->S, n);
}

__device__ void sqlite3_result_error_code(FuncContext *ctx, int errCode)
{
	ctx->IsError = errCode;
	if (ctx->S.Flags & MEM_Null)
		Vdbe::MemSetStr(&ctx->S, sqlite3ErrStr(errCode), -1, TEXTENCODE_UTF8, SQLITE_STATIC);
}

__device__ void sqlite3_result_error_toobig(FuncContext *ctx)
{
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	ctx->IsError = RC_TOOBIG;
	Vdbe::MemSetStr(&ctx->S, "string or blob too big", -1, TEXTENCODE_UTF8, SQLITE_STATIC);
}

__device__ void sqlite3_result_error_nomem(FuncContext *ctx)
{
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	sqlite3VdbeMemSetNull(&ctx->S);
	ctx->IsError = RC_NOMEM;
	ctx->S.Db->MallocFailed = true;
}

#pragma endregion

#pragma region Step

__device__ static RC doWalCallbacks(Context *db)
{
	RC rc = RC_OK;
#ifndef OMIT_WAL
	for (int i = 0; i < db->Dbs.length; i++)
	{
		Btree *bt = db->Dbs[i].Bt;
		if (bt)
		{
			int entrys = sqlite3PagerWalCallback(sqlite3BtreePager(bt));
			if (db->WalCallback && entrys > 0 && rc == RC_OK)
				rc = db->WalCallback(db->WalArg, db, db->Dbs[i].Name, entrys);
		}
	}
#endif
	return rc;
}

__device__ static RC sqlite3Step(Vdbe *p)
{
	_assert(p);
	if (p->Magic != VDBE_MAGIC_RUN)
	{
		// We used to require that sqlite3_reset() be called before retrying sqlite3_step() after any error or after SQLITE_DONE.  But beginning
		// with version 3.7.0, we changed this so that sqlite3_reset() would be called automatically instead of throwing the SQLITE_MISUSE error.
		// This "automatic-reset" change is not technically an incompatibility, since any application that receives an SQLITE_MISUSE is broken by
		// definition.
		//
		// Nevertheless, some published applications that were originally written for version 3.6.23 or earlier do in fact depend on SQLITE_MISUSE 
		// returns, and those were broken by the automatic-reset change.  As a work-around, the SQLITE_OMIT_AUTORESET compile-time restores the
		// legacy behavior of returning SQLITE_MISUSE for cases where the previous sqlite3_step() returned something other than a SQLITE_LOCKED
		// or SQLITE_BUSY error.
#ifdef OMIT_AUTORESET
		if (p->RC == RC_BUSY || p->RC == RC_LOCKED)
			sqlite3_reset(p);
		else
			return RC_MISUSE_BKPT;
#else
		sqlite3_reset(p);
#endif
	}

	// Check that malloc() has not failed. If it has, return early.
	RC rc;
	Context *db = p->Db;
	if (db->MallocFailed)
	{
		p->RC = RC_NOMEM;
		return RC_NOMEM;
	}
	if (p->PC <= 0 && P->Expired)
	{
		p->RC = RC_SCHEMA;
		rc = RC_ERROR;
		goto end_of_step;
	}
	if (p->PC < 0)
	{
		// If there are no other statements currently running, then reset the interrupt flag.  This prevents a call to sqlite3_interrupt
		// from interrupting a statement that has not yet started.
		if (db->ActiveVdbeCnt == 0)
			db->U1.IsInterrupted = false;
		_assert(db->WriteVdbeCnt > 0 || !db->AutoCommit || !db->DeferredCons);
#ifndef OMIT_TRACE
		if (db->Profile && !db->Init.Busy)
			db->Vfs->CurrentTimeInt64(&p->StartTime);
#endif
		db->ActiveVdbeCnt++;
		if (!p->ReadOnly) db->WriteVdbeCnt++;
		p->PC = 0;
	}

#ifndef OMIT_EXPLAIN
	if (p->Explain)
		rc = p->List();
	else
#endif
	{
		db->VdbeExecCnt++;
		rc = p->Exec();
		db->VdbeExecCnt--;
	}

#ifndef OMIT_TRACE
	// Invoke the profile callback if there is one
	if (rc != RC_ROW && db->Profile && !db->Init.Busy && p->Sql)
	{
		int64 now;
		db->Vfs->CurrentTimeInt64(&iNow);
		db->Profile(db->ProfileArg, p->Sql, (now - p->StartTime)*1000000);
	}
#endif

	if (rc == RC_DONE)
	{
		_assert(p->RC == RC_OK);
		p->RC = doWalCallbacks(db);
		if (p->RC != RC_OK)
			rc = RC_ERROR;
	}

	db->ErrCode = rc;
	if (sqlite3ApiExit(p->Db, p->RC) == RC_NOMEM)
		p->RC = RC_NOMEM;

end_of_step:
	// At this point local variable rc holds the value that should be returned if this statement was compiled using the legacy 
	// sqlite3_prepare() interface. According to the docs, this can only be one of the values in the first assert() below. Variable p->rc 
	// contains the value that would be returned if sqlite3_finalize() were called on statement p.
	_assert(rc == RC_ROW || rc == RC_DONE || rc == RC_ERROR || rc == RC_BUSY || rc == RC_MISUSE);
	_assert(p->RC != RC_ROW && p->RC != RC_DONE);
	// If this statement was prepared using sqlite3_prepare_v2(), and an error has occurred, then return the error code in p->rc to the
	// caller. Set the error code in the database handle to the same value.
	if (p->IsPrepareV2 && rc != RC_ROW && rc != RC_DONE)
		rc = p->TransferError();
	return (rc & db->ErrMask);
}

#ifndef MAX_SCHEMA_RETRY
#define MAX_SCHEMA_RETRY 5
#endif

__device__ RC sqlite3_step(Vdbe *p)
{
	RC rc = RC_OK;      // Result from sqlite3Step()
	RC rc2 = RC_OK;     // Result from sqlite3Reprepare()
	int cnt = 0; // Counter to prevent infinite loop of reprepares
	Context *db; // The database connection

	if (vdbeSafetyNotNull(p))
		return RC_MISUSE_BKPT;
	db = p->db;
	MutexEx::Enter(db->Mutex);
	p->DoingRerun = false;
	while ((rc = sqlite3Step(p)) == RC_SCHEMA &&
		cnt++ < MAX_SCHEMA_RETRY &&
		(rc2 = rc = sqlite3Reprepare(p)) == RC_OK)
	{
		sqlite3_reset(p);
		p->DoingRerun = true;
		_assert(!p->Expired);
	}
	if (rc2 != RC_OK && ALWAYS(p->IsPrepareV2) && ALWAYS(db->Err))
	{
		// This case occurs after failing to recompile an sql statement. The error message from the SQL compiler has already been loaded 
		// into the database handle. This block copies the error message from the database handle into the statement and sets the statement
		// program counter to 0 to ensure that when the statement is finalized or reset the parser error message is available via
		// sqlite3_errmsg() and sqlite3_errcode().
		const char *err = (const char *)sqlite3_value_text(db->Err); 
		sqlite3DbFree(db, p->ErrMsg);
		if (!db->MallocFailed) { p->ErrMsg = SysEx::StrDup(db, err); p->rc = rc2; }
		else { p->ErrMsg = nullptr; p->RC = rc = RC_NOMEM; }
	}
	rc = sqlite3ApiExit(db, rc);
	MutexEx::Leave(db->Mutex);
	return rc;
}

#pragma endregion

#pragma region Name3

__device__ void *sqlite3_user_data(FuncContext *ctx)
{
	_assert(ctx && ctx->Func);
	return ctx->Func->UserData;
}

__device__ Context *sqlite3_context_db_handle(FuncContext *ctx)
{
	_assert(ctx && ctx->Func);
	return ctx->S.Db;
}

__device__ void sqlite3InvalidFunction(FuncContext *ctx, int notUsed, Mem **notUsed2)
{
	const char *name = ctx->Func->Name;
	char *err = _mprintf("unable to use function %s in the requested context", name);
	sqlite3_result_error(ctx, err, -1);
	SysEx::Free(err);
}

__device__ void *sqlite3_aggregate_context(FuncContext *ctx, int bytes)
{
	_assert(ctx && ctx->Func && ctx->Func->xStep);
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	Mem *mem = ctx->Mem;
	ASSERTCOVERAGE(bytes < 0);
	if ((mem->Flags & MEM_Agg) == 0)
	{
		if (bytes <= 0)
		{
			Vdbe::MemReleaseExternal(mem);
			mem->Flags = MEM_Null;
			mem->Z = nullptr;
		}
		else
		{
			Vdbe::MemGrow(mem, bytes, 0);
			mem->Flags = MEM_Agg;
			mem->U.Def = ctx->Func;
			if (mem->Z)
				memset(mem->Z, 0, bytes);
		}
	}
	return (void *)mem->Z;
}

__device__ void *sqlite3_get_auxdata(FuncContext *ctx, int arg)
{
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	VdbeFunc *vdbeFunc = ctx->pVdbeFunc;
	if (!vdbeFunc || arg >= vdbeFunc->AuxsLength || arg < 0)
		return nullptr;
	return vdbeFunc->Auxs[arg].Aux;
}

__device__ void sqlite3_set_auxdata(FuncContext *ctx, int args, void *aux, void (*delete_)(void*))
{
	if (args < 0) goto failed;
	_assert(MutexEx::Held(ctx->S.Db->Mutex));
	VdbeFunc *vdbeFunc = ctx->VdbeFunc;
	if (!vdbeFunc || vdbeFunc->nAux <= args)
	{
		int auxLength = (vdbeFunc ? vdbeFunc->AuxLength : 0);
		int newSize = sizeof(VdbeFunc) + sizeof(AuxData)*args;
		vdbeFunc = SysEx::Realloc(ctx->S.Db, vdbeFunc, newSize);
		if (!vdbeFunc)
			goto failed;
		ctx->VdbeFunc = vdbeFunc;
		memset(&vdbeFunc->Auxs[auxLength], 0, sizeof(AuxData)*(args+1-auxLength));
		vdbeFunc->AuxLength = args+1;
		vdbeFunc->Func = ctx->Func;
	}
	AuxData *auxData = &vdbeFunc->Auxs[args];
	if (auxData->Aux && auxData->Delete)
		auxData->Delete(auxData->Aux);
	auxData->Aux = aux;
	auxData->Delete = delete_;
	return;
failed:
	if (delete_)
		delete_(aux);
}

__device__ int sqlite3_column_count(Vdbe *p)
{
	return (p ? p->ResColumns : 0);
}

__device__ int sqlite3_data_count(Vdbe *p)
{
	if (!p || !p->ResultSet) return 0;
	return p->ResColumns;
}

// If the value passed as the second argument is out of range, return a pointer to the following static Mem object which contains the
// value SQL NULL. Even though the Mem structure contains an element of type i64, on certain architectures (x86) with certain compiler
// switches (-Os), gcc may align this Mem object on a 4-byte boundary instead of an 8-byte one. This all works fine, except that when
// running with SQLITE_DEBUG defined the SQLite code sometimes assert()s that a Mem structure is located on an 8-byte boundary. To prevent
// these assert()s from failing, when building with SQLITE_DEBUG defined using gcc, we force nullMem to be 8-byte aligned using the magical
// __attribute__((aligned(8))) macro.
__device__ static const Mem nullMem 
#if defined(_DEBUG) && defined(__GNUC__)
	__attribute__((aligned(8))) 
#endif
	= {0, "", (double)0, {0}, 0, MEM_Null, SQLITE_NULL, 0,
#ifdef _DEBUG
	0, 0,  // pScopyFrom, pFiller
#endif
	0, 0 };
__device__ static Mem *columnMem(Vdbe *p, int i)
{
	Mem *r;
	if (p && p->ResultSet != 0 && i < p->ResColumns && i >= 0)
	{
		MutexEx::Enter(p->Db->Mutex);
		r = &p->ResultSet[i];
	}
	else
	{
		if (p && SysEx_ALWAYS(p->Db))
		{
			MutexEx::Enter(p->Db->Mutex);
			sqlite3Error(p->Db, SQLITE_RANGE, 0);
		}
		r = (Mem *)&nullMem;
	}
	return r;
}

__device__ static void columnMallocFailure(Vdbe *p)
{
	// If malloc() failed during an encoding conversion within an sqlite3_column_XXX API, then set the return code of the statement to
	// SQLITE_NOMEM. The next call to _step() (if any) will return SQLITE_ERROR and _finalize() will return NOMEM.
	if (p)
	{
		p->RC = sqlite3ApiExit(p->Db, p->RC);
		MutexEx::Leave(p->Db->Mutex);
	}
}

#pragma endregion

#pragma region sqlite3_column_

const void *sqlite3_column_blob(sqlite3_stmt *pStmt, int i){
	const void *val;
	val = sqlite3_value_blob( columnMem(pStmt,i) );
	/* Even though there is no encoding conversion, value_blob() might
	** need to call malloc() to expand the result of a zeroblob() 
	** expression. 
	*/
	columnMallocFailure(pStmt);
	return val;
}
int sqlite3_column_bytes(sqlite3_stmt *pStmt, int i){
	int val = sqlite3_value_bytes( columnMem(pStmt,i) );
	columnMallocFailure(pStmt);
	return val;
}
int sqlite3_column_bytes16(sqlite3_stmt *pStmt, int i){
	int val = sqlite3_value_bytes16( columnMem(pStmt,i) );
	columnMallocFailure(pStmt);
	return val;
}
double sqlite3_column_double(sqlite3_stmt *pStmt, int i){
	double val = sqlite3_value_double( columnMem(pStmt,i) );
	columnMallocFailure(pStmt);
	return val;
}
int sqlite3_column_int(sqlite3_stmt *pStmt, int i){
	int val = sqlite3_value_int( columnMem(pStmt,i) );
	columnMallocFailure(pStmt);
	return val;
}
sqlite_int64 sqlite3_column_int64(sqlite3_stmt *pStmt, int i){
	sqlite_int64 val = sqlite3_value_int64( columnMem(pStmt,i) );
	columnMallocFailure(pStmt);
	return val;
}
const unsigned char *sqlite3_column_text(sqlite3_stmt *pStmt, int i){
	const unsigned char *val = sqlite3_value_text( columnMem(pStmt,i) );
	columnMallocFailure(pStmt);
	return val;
}
sqlite3_value *sqlite3_column_value(sqlite3_stmt *pStmt, int i){
	Mem *pOut = columnMem(pStmt, i);
	if( pOut->flags&MEM_Static ){
		pOut->flags &= ~MEM_Static;
		pOut->flags |= MEM_Ephem;
	}
	columnMallocFailure(pStmt);
	return (sqlite3_value *)pOut;
}
#ifndef SQLITE_OMIT_UTF16
const void *sqlite3_column_text16(sqlite3_stmt *pStmt, int i){
	const void *val = sqlite3_value_text16( columnMem(pStmt,i) );
	columnMallocFailure(pStmt);
	return val;
}
#endif /* SQLITE_OMIT_UTF16 */
int sqlite3_column_type(sqlite3_stmt *pStmt, int i){
	int iType = sqlite3_value_type( columnMem(pStmt,i) );
	columnMallocFailure(pStmt);
	return iType;
}

/* The following function is experimental and subject to change or
** removal */
/*int sqlite3_column_numeric_type(sqlite3_stmt *pStmt, int i){
**  return sqlite3_value_numeric_type( columnMem(pStmt,i) );
**}
*/

/*
** Convert the N-th element of pStmt->pColName[] into a string using
** xFunc() then return that string.  If N is out of range, return 0.
**
** There are up to 5 names for each column.  useType determines which
** name is returned.  Here are the names:
**
**    0      The column name as it should be displayed for output
**    1      The datatype name for the column
**    2      The name of the database that the column derives from
**    3      The name of the table that the column derives from
**    4      The name of the table column that the result column derives from
**
** If the result is not a simple column reference (if it is an expression
** or a constant) then useTypes 2, 3, and 4 return NULL.
*/
static const void *columnName(
	sqlite3_stmt *pStmt,
	int N,
	const void *(*xFunc)(Mem*),
	int useType
	){
		const void *ret = 0;
		Vdbe *p = (Vdbe *)pStmt;
		int n;
		sqlite3 *db = p->db;

		assert( db!=0 );
		n = sqlite3_column_count(pStmt);
		if( N<n && N>=0 ){
			N += useType*n;
			sqlite3_mutex_enter(db->mutex);
			assert( db->mallocFailed==0 );
			ret = xFunc(&p->aColName[N]);
			/* A malloc may have failed inside of the xFunc() call. If this
			** is the case, clear the mallocFailed flag and return NULL.
			*/
			if( db->mallocFailed ){
				db->mallocFailed = 0;
				ret = 0;
			}
			sqlite3_mutex_leave(db->mutex);
		}
		return ret;
}

/*
** Return the name of the Nth column of the result set returned by SQL
** statement pStmt.
*/
const char *sqlite3_column_name(sqlite3_stmt *pStmt, int N){
	return columnName(
		pStmt, N, (const void*(*)(Mem*))sqlite3_value_text, COLNAME_NAME);
}
#ifndef SQLITE_OMIT_UTF16
const void *sqlite3_column_name16(sqlite3_stmt *pStmt, int N){
	return columnName(
		pStmt, N, (const void*(*)(Mem*))sqlite3_value_text16, COLNAME_NAME);
}
#endif

/*
** Constraint:  If you have ENABLE_COLUMN_METADATA then you must
** not define OMIT_DECLTYPE.
*/
#if defined(SQLITE_OMIT_DECLTYPE) && defined(SQLITE_ENABLE_COLUMN_METADATA)
# error "Must not define both SQLITE_OMIT_DECLTYPE \
	and SQLITE_ENABLE_COLUMN_METADATA"
#endif

#ifndef SQLITE_OMIT_DECLTYPE
/*
** Return the column declaration type (if applicable) of the 'i'th column
** of the result set of SQL statement pStmt.
*/
const char *sqlite3_column_decltype(sqlite3_stmt *pStmt, int N){
	return columnName(
		pStmt, N, (const void*(*)(Mem*))sqlite3_value_text, COLNAME_DECLTYPE);
}
#ifndef SQLITE_OMIT_UTF16
const void *sqlite3_column_decltype16(sqlite3_stmt *pStmt, int N){
	return columnName(
		pStmt, N, (const void*(*)(Mem*))sqlite3_value_text16, COLNAME_DECLTYPE);
}
#endif /* SQLITE_OMIT_UTF16 */
#endif /* SQLITE_OMIT_DECLTYPE */

#ifdef SQLITE_ENABLE_COLUMN_METADATA
/*
** Return the name of the database from which a result column derives.
** NULL is returned if the result column is an expression or constant or
** anything else which is not an unabiguous reference to a database column.
*/
const char *sqlite3_column_database_name(sqlite3_stmt *pStmt, int N){
	return columnName(
		pStmt, N, (const void*(*)(Mem*))sqlite3_value_text, COLNAME_DATABASE);
}
#ifndef SQLITE_OMIT_UTF16
const void *sqlite3_column_database_name16(sqlite3_stmt *pStmt, int N){
	return columnName(
		pStmt, N, (const void*(*)(Mem*))sqlite3_value_text16, COLNAME_DATABASE);
}
#endif /* SQLITE_OMIT_UTF16 */

/*
** Return the name of the table from which a result column derives.
** NULL is returned if the result column is an expression or constant or
** anything else which is not an unabiguous reference to a database column.
*/
const char *sqlite3_column_table_name(sqlite3_stmt *pStmt, int N){
	return columnName(
		pStmt, N, (const void*(*)(Mem*))sqlite3_value_text, COLNAME_TABLE);
}
#ifndef SQLITE_OMIT_UTF16
const void *sqlite3_column_table_name16(sqlite3_stmt *pStmt, int N){
	return columnName(
		pStmt, N, (const void*(*)(Mem*))sqlite3_value_text16, COLNAME_TABLE);
}
#endif /* SQLITE_OMIT_UTF16 */

/*
** Return the name of the table column from which a result column derives.
** NULL is returned if the result column is an expression or constant or
** anything else which is not an unabiguous reference to a database column.
*/
const char *sqlite3_column_origin_name(sqlite3_stmt *pStmt, int N){
	return columnName(
		pStmt, N, (const void*(*)(Mem*))sqlite3_value_text, COLNAME_COLUMN);
}
#ifndef SQLITE_OMIT_UTF16
const void *sqlite3_column_origin_name16(sqlite3_stmt *pStmt, int N){
	return columnName(
		pStmt, N, (const void*(*)(Mem*))sqlite3_value_text16, COLNAME_COLUMN);
}
#endif /* SQLITE_OMIT_UTF16 */
#endif /* SQLITE_ENABLE_COLUMN_METADATA */

#pragma endregion

#pragma region sqlite3_bind_

static int vdbeUnbind(Vdbe *p, int i){
	Mem *pVar;
	if( vdbeSafetyNotNull(p) ){
		return SQLITE_MISUSE_BKPT;
	}
	sqlite3_mutex_enter(p->db->mutex);
	if( p->magic!=VDBE_MAGIC_RUN || p->pc>=0 ){
		sqlite3Error(p->db, SQLITE_MISUSE, 0);
		sqlite3_mutex_leave(p->db->mutex);
		sqlite3_log(SQLITE_MISUSE, 
			"bind on a busy prepared statement: [%s]", p->zSql);
		return SQLITE_MISUSE_BKPT;
	}
	if( i<1 || i>p->nVar ){
		sqlite3Error(p->db, SQLITE_RANGE, 0);
		sqlite3_mutex_leave(p->db->mutex);
		return SQLITE_RANGE;
	}
	i--;
	pVar = &p->aVar[i];
	sqlite3VdbeMemRelease(pVar);
	pVar->flags = MEM_Null;
	sqlite3Error(p->db, SQLITE_OK, 0);

	/* If the bit corresponding to this variable in Vdbe.expmask is set, then 
	** binding a new value to this variable invalidates the current query plan.
	**
	** IMPLEMENTATION-OF: R-48440-37595 If the specific value bound to host
	** parameter in the WHERE clause might influence the choice of query plan
	** for a statement, then the statement will be automatically recompiled,
	** as if there had been a schema change, on the first sqlite3_step() call
	** following any change to the bindings of that parameter.
	*/
	if( p->isPrepareV2 &&
		((i<32 && p->expmask & ((u32)1 << i)) || p->expmask==0xffffffff)
		){
			p->expired = 1;
	}
	return SQLITE_OK;
}

/*
** Bind a text or BLOB value.
*/
static int bindText(
	sqlite3_stmt *pStmt,   /* The statement to bind against */
	int i,                 /* Index of the parameter to bind */
	const void *zData,     /* Pointer to the data to be bound */
	int nData,             /* Number of bytes of data to be bound */
	void (*xDel)(void*),   /* Destructor for the data */
	u8 encoding            /* Encoding for the data */
	){
		Vdbe *p = (Vdbe *)pStmt;
		Mem *pVar;
		int rc;

		rc = vdbeUnbind(p, i);
		if( rc==SQLITE_OK ){
			if( zData!=0 ){
				pVar = &p->aVar[i-1];
				rc = sqlite3VdbeMemSetStr(pVar, zData, nData, encoding, xDel);
				if( rc==SQLITE_OK && encoding!=0 ){
					rc = sqlite3VdbeChangeEncoding(pVar, ENC(p->db));
				}
				sqlite3Error(p->db, rc, 0);
				rc = sqlite3ApiExit(p->db, rc);
			}
			sqlite3_mutex_leave(p->db->mutex);
		}else if( xDel!=SQLITE_STATIC && xDel!=SQLITE_TRANSIENT ){
			xDel((void*)zData);
		}
		return rc;
}


/*
** Bind a blob value to an SQL statement variable.
*/
int sqlite3_bind_blob(
	sqlite3_stmt *pStmt, 
	int i, 
	const void *zData, 
	int nData, 
	void (*xDel)(void*)
	){
		return bindText(pStmt, i, zData, nData, xDel, 0);
}
int sqlite3_bind_double(sqlite3_stmt *pStmt, int i, double rValue){
	int rc;
	Vdbe *p = (Vdbe *)pStmt;
	rc = vdbeUnbind(p, i);
	if( rc==SQLITE_OK ){
		sqlite3VdbeMemSetDouble(&p->aVar[i-1], rValue);
		sqlite3_mutex_leave(p->db->mutex);
	}
	return rc;
}
int sqlite3_bind_int(sqlite3_stmt *p, int i, int iValue){
	return sqlite3_bind_int64(p, i, (i64)iValue);
}
int sqlite3_bind_int64(sqlite3_stmt *pStmt, int i, sqlite_int64 iValue){
	int rc;
	Vdbe *p = (Vdbe *)pStmt;
	rc = vdbeUnbind(p, i);
	if( rc==SQLITE_OK ){
		sqlite3VdbeMemSetInt64(&p->aVar[i-1], iValue);
		sqlite3_mutex_leave(p->db->mutex);
	}
	return rc;
}
int sqlite3_bind_null(sqlite3_stmt *pStmt, int i){
	int rc;
	Vdbe *p = (Vdbe*)pStmt;
	rc = vdbeUnbind(p, i);
	if( rc==SQLITE_OK ){
		sqlite3_mutex_leave(p->db->mutex);
	}
	return rc;
}
int sqlite3_bind_text( 
	sqlite3_stmt *pStmt, 
	int i, 
	const char *zData, 
	int nData, 
	void (*xDel)(void*)
	){
		return bindText(pStmt, i, zData, nData, xDel, SQLITE_UTF8);
}
#ifndef SQLITE_OMIT_UTF16
int sqlite3_bind_text16(
	sqlite3_stmt *pStmt, 
	int i, 
	const void *zData, 
	int nData, 
	void (*xDel)(void*)
	){
		return bindText(pStmt, i, zData, nData, xDel, SQLITE_UTF16NATIVE);
}
#endif /* SQLITE_OMIT_UTF16 */
int sqlite3_bind_value(sqlite3_stmt *pStmt, int i, const sqlite3_value *pValue){
	int rc;
	switch( pValue->type ){
	case SQLITE_INTEGER: {
		rc = sqlite3_bind_int64(pStmt, i, pValue->u.i);
		break;
						 }
	case SQLITE_FLOAT: {
		rc = sqlite3_bind_double(pStmt, i, pValue->r);
		break;
					   }
	case SQLITE_BLOB: {
		if( pValue->flags & MEM_Zero ){
			rc = sqlite3_bind_zeroblob(pStmt, i, pValue->u.nZero);
		}else{
			rc = sqlite3_bind_blob(pStmt, i, pValue->z, pValue->n,SQLITE_TRANSIENT);
		}
		break;
					  }
	case SQLITE_TEXT: {
		rc = bindText(pStmt,i,  pValue->z, pValue->n, SQLITE_TRANSIENT,
			pValue->enc);
		break;
					  }
	default: {
		rc = sqlite3_bind_null(pStmt, i);
		break;
			 }
	}
	return rc;
}
int sqlite3_bind_zeroblob(sqlite3_stmt *pStmt, int i, int n){
	int rc;
	Vdbe *p = (Vdbe *)pStmt;
	rc = vdbeUnbind(p, i);
	if( rc==SQLITE_OK ){
		sqlite3VdbeMemSetZeroBlob(&p->aVar[i-1], n);
		sqlite3_mutex_leave(p->db->mutex);
	}
	return rc;
}

/*
** Return the number of wildcards that can be potentially bound to.
** This routine is added to support DBD::SQLite.  
*/
int sqlite3_bind_parameter_count(sqlite3_stmt *pStmt){
	Vdbe *p = (Vdbe*)pStmt;
	return p ? p->nVar : 0;
}

/*
** Return the name of a wildcard parameter.  Return NULL if the index
** is out of range or if the wildcard is unnamed.
**
** The result is always UTF-8.
*/
const char *sqlite3_bind_parameter_name(sqlite3_stmt *pStmt, int i){
	Vdbe *p = (Vdbe*)pStmt;
	if( p==0 || i<1 || i>p->nzVar ){
		return 0;
	}
	return p->azVar[i-1];
}

/*
** Given a wildcard parameter name, return the index of the variable
** with that name.  If there is no variable with the given name,
** return 0.
*/
int sqlite3VdbeParameterIndex(Vdbe *p, const char *zName, int nName){
	int i;
	if( p==0 ){
		return 0;
	}
	if( zName ){
		for(i=0; i<p->nzVar; i++){
			const char *z = p->azVar[i];
			if( z && strncmp(z,zName,nName)==0 && z[nName]==0 ){
				return i+1;
			}
		}
	}
	return 0;
}
int sqlite3_bind_parameter_index(sqlite3_stmt *pStmt, const char *zName){
	return sqlite3VdbeParameterIndex((Vdbe*)pStmt, zName, sqlite3Strlen30(zName));
}

/*
** Transfer all bindings from the first statement over to the second.
*/
int sqlite3TransferBindings(sqlite3_stmt *pFromStmt, sqlite3_stmt *pToStmt){
	Vdbe *pFrom = (Vdbe*)pFromStmt;
	Vdbe *pTo = (Vdbe*)pToStmt;
	int i;
	assert( pTo->db==pFrom->db );
	assert( pTo->nVar==pFrom->nVar );
	sqlite3_mutex_enter(pTo->db->mutex);
	for(i=0; i<pFrom->nVar; i++){
		sqlite3VdbeMemMove(&pTo->aVar[i], &pFrom->aVar[i]);
	}
	sqlite3_mutex_leave(pTo->db->mutex);
	return SQLITE_OK;
}

#pragma endregion

#pragma region Name4

/*
** Return the sqlite3* database handle to which the prepared statement given
** in the argument belongs.  This is the same database handle that was
** the first argument to the sqlite3_prepare() that was used to create
** the statement in the first place.
*/
sqlite3 *sqlite3_db_handle(sqlite3_stmt *pStmt)
{
	return pStmt ? ((Vdbe*)pStmt)->db : 0;
}

/*
** Return true if the prepared statement is guaranteed to not modify the
** database.
*/
int sqlite3_stmt_readonly(sqlite3_stmt *pStmt){
	return pStmt ? ((Vdbe*)pStmt)->readOnly : 1;
}

/*
** Return true if the prepared statement is in need of being reset.
*/
int sqlite3_stmt_busy(sqlite3_stmt *pStmt){
	Vdbe *v = (Vdbe*)pStmt;
	return v!=0 && v->pc>0 && v->magic==VDBE_MAGIC_RUN;
}

/*
** Return a pointer to the next prepared statement after pStmt associated
** with database connection pDb.  If pStmt is NULL, return the first
** prepared statement for the database connection.  Return NULL if there
** are no more.
*/
sqlite3_stmt *sqlite3_next_stmt(sqlite3 *pDb, sqlite3_stmt *pStmt){
	sqlite3_stmt *pNext;
	sqlite3_mutex_enter(pDb->mutex);
	if( pStmt==0 ){
		pNext = (sqlite3_stmt*)pDb->pVdbe;
	}else{
		pNext = (sqlite3_stmt*)((Vdbe*)pStmt)->pNext;
	}
	sqlite3_mutex_leave(pDb->mutex);
	return pNext;
}

/*
** Return the value of a status counter for a prepared statement
*/
int sqlite3_stmt_status(sqlite3_stmt *pStmt, int op, int resetFlag){
	Vdbe *pVdbe = (Vdbe*)pStmt;
	int v = pVdbe->aCounter[op-1];
	if( resetFlag ) pVdbe->aCounter[op-1] = 0;
	return v;
}

#pragma endregion
