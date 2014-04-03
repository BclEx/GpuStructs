using System;
using FuncDefHash = Core.array_t3<int, Core.FuncDef, object>;

namespace Core
{
    public enum LIMIT : byte
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
    }

    public enum MAGIC : uint
    {
        MAGIC_OPEN = 0xa029a697,	// Database is open
        MAGIC_CLOSED = 0x9f3c2d33,  // Database is closed
        MAGIC_SICK = 0x4b771290,	// Error and awaiting close
        MAGIC_BUSY = 0xf03b7906,	// Database currently in use
        MAGIC_ERROR = 0xb5357930,	// An SQLITE_MISUSE error occurred
        MAGIC_ZOMBIE = 0x64cffc7f,  // Close with last statement close
    }

    public class LookasideSlot
    {
        public LookasideSlot Next;    // Next buffer in the list of free buffers
    }
    public struct Lookaside
    {
        ushort Size;            // Size of each buffer in bytes
        bool Enabled;           // False to disable new lookaside allocations
        bool Malloced;          // True if pStart obtained from sqlite3_malloc()
        int Outs;               // Number of buffers currently checked out
        int MaxOuts;            // Highwater mark for nOut
        int[] Stats = new int[3];			// 0: hits.  1: size misses.  2: full misses
        LookasideSlot Free;	// List of available buffers
        object Start;			// First byte of available memory space
        object End;				// First byte past end of available space
    }

    public static class ContextEx
    {
#if !OMIT_BUILTIN_TEST
        public static bool CtxOptimizationDisabled(Context ctx, OPTFLAG mask) { return (((ctx).OptFlags & (mask)) != 0); }
        public static bool CtxOptimizationEnabled(Context ctx, OPTFLAG mask) { return (((ctx).OptFlags & (mask)) == 0); }
#else
            public static bool CtxOptimizationDisabled(Context ctx, OPTFLAG mask) { return false; }
            public static bool CtxOptimizationEnabled(Context ctx, OPTFLAG mask) { return true; }
#endif
    }

    [Flags]
    public enum OPTFLAG : ushort
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
    }

    public class Context : BContext
    {
        public struct InitInfo
        {
            public int NewTid;				// Rootpage of table being initialized
            public byte DB;                 // Which db file is being initialized
            public bool Busy;				// TRUE if currently initializing
            public byte OrphanTrigger;      // Last statement is orphaned TEMP trigger
        }

        public OPTFLAG OptFlags;
        public VSystem Vfs;					// OS Interface
        public array_t<Vdbe> Vdbe;			// List of active virtual machines
        public CollSeq DefaultColl;		    // The default collating sequence (BINARY)
        public long LastRowID;				// ROWID of most recent insert (see above)
        public uint OpenFlags;		        // Flags passed to sqlite3_vfs.xOpen()
        public RC ErrCode;					// Most recent error code (RC_*)
        public int ErrMask;					// & result codes with this before returning
        public Action<object, Context, TEXTENCODE, string> CollNeeded;
        public Action<object, Context, TEXTENCODE, string> CollNeeded16;
        public object CollNeededArg;
        public Mem Err;						// Most recent error message
        public string ErrMsg;				// Most recent error message (UTF-8 encoded)
        public string ErrMsg16;				// Most recent error message (UTF-16 encoded)
        public struct _u1
        {
            public bool IsInterrupted;      // True if sqlite3_interrupt has been called
            public double NotUsed1;         // Spacer
        }
        public _u1 u1;
        public Lookaside Lookaside;			// Lookaside malloc configuration

        public bool MallocFailed;					// True if we have seen a malloc failure
        public byte VTableOnConflict;				// Value to return for s3_vtab_on_conflict()
        public byte IsTransactionSavepoint;		// True if the outermost savepoint is a TS
        public MAGIC Magic;						// Magic number for detect library misuse
        public int[] Limits = new int[(int)LIMIT.LIMIT_MAX_];				// Limits
        public InitInfo Init;						// Information used during initialization
#if !OMIT_VIRTUALTABLE
        public Hash Modules;						// populated by sqlite3_create_module()
        public VTableContext VTableCtx;			// Context for active vtab connect/create
        public array_t<VTable> VTrans;			// Virtual tables with open transactions / Allocated size of aVTrans
        public VTable Disconnect;					// Disconnect these in next sqlite3_prepare()
#endif
        public FuncDefHash Funcs;					// Hash table of connection functions
        public Hash CollSeqs;						// All collating sequences
        public int* BytesFreed;					// If not NULL, increment this in DbFree()
    }
}