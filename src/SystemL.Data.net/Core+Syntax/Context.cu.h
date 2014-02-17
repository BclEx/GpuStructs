namespace Core
{
	struct VTable;
	struct VTableContext;

	enum LIMIT : uint8
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

	enum MAGIC : uint32
	{
		MAGIC_OPEN = 0xa029a697,  // Database is open
		MAGIC_CLOSED = 0x9f3c2d33,  // Database is closed
		MAGIC_SICK = 0x4b771290,  // Error and awaiting close
		MAGIC_BUSY = 0xf03b7906,  // Database currently in use
		MAGIC_ERROR = 0xb5357930,  // An SQLITE_MISUSE error occurred
		MAGIC_ZOMBIE = 0x64cffc7f,  // Close with last statement close
	};

	class Context : public BContext
	{
	public:
		struct sqlite3InitInfo
		{
			int NewTid;					// Rootpage of table being initialized
			uint8 DB;                   // Which db file is being initialized
			bool Busy;					// TRUE if currently initializing
			uint8 OrphanTrigger;        // Last statement is orphaned TEMP trigger
		};

		VSystem *Vfs;					// OS Interface
		//array_t<Vdbe> Vdbe;			// List of active virtual machines
		//CollSeq *DefaultColl;			// The default collating sequence (BINARY)
		int64 LastRowID;				// ROWID of most recent insert (see above)
		//unsigned int OpenFlags;		// Flags passed to sqlite3_vfs.xOpen()
		RC ErrCode;						// Most recent error code (RC_*)
		int ErrMask;					// & result codes with this before returning
		//Mem *Err;						// Most recent error message
		//char *zErrMsg;                // Most recent error message (UTF-8 encoded)
		//char *zErrMsg16;              // Most recent error message (UTF-16 encoded)

		bool MallocFailed;				// True if we have seen a malloc failure
		uint8 VTableOnConflict;         // Value to return for s3_vtab_on_conflict()
		uint8 IsTransactionSavepoint;   // True if the outermost savepoint is a TS
		MAGIC Magic;					// Magic number for detect library misuse
		int Limits[LIMIT_MAX_];			// Limits
		sqlite3InitInfo Init;			// Information used during initialization
#ifndef OMIT_VIRTUALTABLE
		Hash Modules;					// populated by sqlite3_create_module()
		VTableContext *VTableCtx;       // Context for active vtab connect/create
		array_t<VTable *> VTrans;		// Virtual tables with open transactions / Allocated size of aVTrans
		VTable *Disconnect;				// Disconnect these in next sqlite3_prepare()
#endif
		int *BytesFreed;				// If not NULL, increment this in DbFree()

#pragma region FromBuild_c



#pragma endregion

		__device__ inline static RC ApiExit(Context *ctx, RC rc)
		{
			// If the db handle is not NULL, then we must hold the connection handle mutex here. Otherwise the read (and possible write) of db->mallocFailed 
			// is unsafe, as is the call to sqlite3Error().
			_assert(!ctx || MutexEx::Held(ctx->Mutex));
			if (ctx && (ctx->MallocFailed || rc == RC_IOERR_NOMEM))
			{
				Error(ctx, RC_NOMEM, nullptr);
				ctx->MallocFailed = false;
				rc = RC_NOMEM;
			}
			return (RC)(rc & (ctx ? ctx->ErrMask : 0xff));
		}

		//////////////////////
		// ERROR
#pragma region ERROR

		inline __device__ static void Error(void *tag, RC errorCode, const char *fmt) { }
		template <typename T1> inline __device__ static void Error(void *tag, RC errorCode, const char *fmt, T1 arg1) { }
		template <typename T1, typename T2> inline __device__ static void Error(void *tag, RC errorCode, const char *fmt, T1 arg1, T2 arg2) { }
		template <typename T1, typename T2, typename T3> inline __device__ static void Error(void *tag, RC errorCode, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { }
		template <typename T1, typename T2, typename T3, typename T4> inline __device__ static void Error(void *tag, RC errorCode, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { }
		template <typename T1, typename T2, typename T3, typename T4, typename T5> inline __device__ static void Error(void *tag, RC errorCode, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { }

#pragma endregion
	};

}