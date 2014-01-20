namespace Core
{
	struct VTableContext;

	enum LIMIT
	{
		LIMIT_LENGTH = 0,
		LIMIT_SQL_LENGTH = 1,
		LIMIT_COLUMN = 2,
		LIMIT_EXPR_DEPTH = 3,
		LIMIT_COMPOUND_SELECT = 4,
		LIMIT_VDBE_OP = 5,
		LIMIT_FUNCTION_ARG = 6,
		LIMIT_ATTACHED = 7,
		LIMIT_LIKE_PATTERN_LENGTH = 8,
		LIMIT_VARIABLE_NUMBER = 9,
		LIMIT_TRIGGER_DEPTH = 10,
		LIMIT_MAX_ = 11,
	};

	class Context : public BContext
	{
	public:

		VSystem *Vfs;				// OS Interface
		//array_t<Vdbe> Vdbe;				// List of active virtual machines
		//CollSeq *DefaultColl;		// The default collating sequence (BINARY)
		int64 LastRowID;            // ROWID of most recent insert (see above)
		//unsigned int OpenFlags;	// Flags passed to sqlite3_vfs.xOpen()
		RC ErrCode;					// Most recent error code (RC_*)
		//int ErrMask;				// & result codes with this before returning
		//Mem *Err;          // Most recent error message
		//char *zErrMsg;                // Most recent error message (UTF-8 encoded)
		//char *zErrMsg16;              // Most recent error message (UTF-16 encoded)

		bool MallocFailed;			// True if we have seen a malloc failure

		uint8 IsTransactionSavepoint;    // True if the outermost savepoint is a TS
		int Limits[LIMIT_MAX_];	// Limits
#ifndef OMIT_VIRTUALTABLE
		Hash Modules;					// populated by sqlite3_create_module()
		VTableContext *VTableCtx;       // Context for active vtab connect/create
		array_t<VTable> VTrans;			// Virtual tables with open transactions / Allocated size of aVTrans
		VTable *Disconnect;				// Disconnect these in next sqlite3_prepare()
#endif

		//__device__ inline static RC ApiExit(Context *ctx, RC rc)
		//{
		//	// If the db handle is not NULL, then we must hold the connection handle mutex here. Otherwise the read (and possible write) of db->mallocFailed 
		//	// is unsafe, as is the call to sqlite3Error().
		//	_assert(!ctx || MutexEx::Held(ctx->Mutex));
		//	if (ctx && (ctx->MallocFailed || rc == RC_IOERR_NOMEM))
		//	{
		//		sqlite3Error(ctx, RC_NOMEM, 0);
		//		ctx->MallocFailed = false;
		//		rc = RC_NOMEM;
		//	}
		//	return (RC)(rc & (ctx ? ctx->ErrMask : 0xff));
		//}

	};
}