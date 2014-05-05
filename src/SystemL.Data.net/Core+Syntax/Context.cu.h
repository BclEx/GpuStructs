namespace Core
{
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
		MAGIC_OPEN = 0xa029a697,	// Database is open
		MAGIC_CLOSED = 0x9f3c2d33,  // Database is closed
		MAGIC_SICK = 0x4b771290,	// Error and awaiting close
		MAGIC_BUSY = 0xf03b7906,	// Database currently in use
		MAGIC_ERROR = 0xb5357930,	// An SQLITE_MISUSE error occurred
		MAGIC_ZOMBIE = 0x64cffc7f,  // Close with last statement close
	};

	struct LookasideSlot
	{
		LookasideSlot *Next;    // Next buffer in the list of free buffers
	};
	struct Lookaside
	{
		uint16 Size;            // Size of each buffer in bytes
		bool Enabled;           // False to disable new lookaside allocations
		bool Malloced;          // True if pStart obtained from sqlite3_malloc()
		int Outs;               // Number of buffers currently checked out
		int MaxOuts;            // Highwater mark for nOut
		int Stats[3];			// 0: hits.  1: size misses.  2: full misses
		LookasideSlot *Free;	// List of available buffers
		void *Start;			// First byte of available memory space
		void *End;				// First byte past end of available space
	};

#ifndef OMIT_BUILTIN_TEST
#define CtxOptimizationDisabled(ctx, mask)  (((ctx)->OptFlags&(mask))!=0)
#define CtxOptimizationEnabled(ctx, mask)   (((ctx)->OptFlags&(mask))==0)
#else
#define CtxOptimizationDisabled(ctx, mask)  0
#define CtxOptimizationEnabled(ctx, mask)   1
#endif

	enum OPTFLAG : uint16
	{
		OPTFLAG_QueryFlattener = 0x0001,	// Query flattening
		OPTFLAG_ColumnCache = 0x0002,		// Column cache
		OPTFLAG_GroupByOrder = 0x0004,		// GROUPBY cover of ORDERBY
		OPTFLAG_FactorOutConst = 0x0008,	// Constant factoring
		OPTFLAG_IdxRealAsInt = 0x0010,		// Store REAL as INT in indices
		OPTFLAG_DistinctOpt = 0x0020,		// DISTINCT using indexes
		OPTFLAG_CoverIdxScan = 0x0040,		// Covering index scans
		OPTFLAG_OrderByIdxJoin = 0x0080,	// ORDER BY of joins via index
		OPTFLAG_SubqCoroutine = 0x0100,		// Evaluate subqueries as coroutines
		OPTFLAG_Transitive = 0x0200,		// Transitive constraints
		OPTFLAG_AllOpts = 0xffff,			// All optimizations
	};

	struct FuncDefHash
	{
		FuncDef *data[23]; // Hash table for functions
	};

	struct VTable;
	struct VTableContext;
	class Context : public BContext
	{
	public:
		struct InitInfo
		{
			int NewTid;						// Rootpage of table being initialized
			uint8 DB;						// Which db file is being initialized
			bool Busy;						// TRUE if currently initializing
			uint8 OrphanTrigger;			// Last statement is orphaned TEMP trigger
		};

		OPTFLAG OptFlags;
		VSystem *Vfs;						// OS Interface
		array_t<Vdbe> Vdbe;					// List of active virtual machines
		CollSeq *DefaultColl;				// The default collating sequence (BINARY)
		int64 LastRowID;					// ROWID of most recent insert (see above)
		unsigned int OpenFlags;				// Flags passed to sqlite3_vfs.xOpen()
		RC ErrCode;							// Most recent error code (RC_*)
		int ErrMask;						// & result codes with this before returning
		void(*CollNeeded)(void *, Context *, int textRep, const char *);
		void(*CollNeeded16)(void *, Context *, int textRep, const void *);
		void *CollNeededArg;
		Mem *Err;							// Most recent error message
		char *ErrMsg;						// Most recent error message (UTF-8 encoded)
		char *ErrMsg16;						// Most recent error message (UTF-16 encoded)
		union
		{
			volatile bool IsInterrupted;	// True if sqlite3_interrupt has been called
			double NotUsed1;				// Spacer
		} u1;
		Lookaside Lookaside;				// Lookaside malloc configuration

		bool MallocFailed;					// True if we have seen a malloc failure
		uint8 VTableOnConflict;				// Value to return for s3_vtab_on_conflict()
		uint8 IsTransactionSavepoint;		// True if the outermost savepoint is a TS
		int NextPagesize;					// Pagesize after VACUUM if >0
		MAGIC Magic;						// Magic number for detect library misuse
		int Limits[LIMIT_MAX_];				// Limits
		InitInfo Init;						// Information used during initialization
#ifndef OMIT_AUTHORIZATION
		ARC (*Auth)(void*,int,const char*,const char*,const char*,const char*); // Access authorization function
		void *AuthArg;						// 1st argument to the access auth function
#endif
#ifndef OMIT_VIRTUALTABLE
		Hash Modules;						// populated by sqlite3_create_module()
		VTableContext *VTableCtx;			// Context for active vtab connect/create
		array_t<VTable *> VTrans;			// Virtual tables with open transactions / Allocated size of aVTrans
		VTable *Disconnect;					// Disconnect these in next sqlite3_prepare()
#endif
		FuncDefHash Funcs;					// Hash table of connection functions
		Hash CollSeqs;						// All collating sequences

		DB DbStatics[2];					// Static space for the 2 default backends
		int Statements;						// Number of nested statement-transactions
		int64 DeferredCons;					// Net deferred constraints this transaction.
		int *BytesFreed;					// If not NULL, increment this in DbFree()
#ifdef ENABLE_UNLOCK_NOTIFY
		Context *BlockingConnection;			// Connection that caused SQLITE_LOCKED
		Context *UnlockConnection;				// Connection to watch for unlock
		void *UnlockArg;						// Argument to xUnlockNotify
		void (*UnlockNotify)(void **, int);		// Unlock notify callback
		Context *NextBlocked;					// Next in list of all blocked connections
#endif

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