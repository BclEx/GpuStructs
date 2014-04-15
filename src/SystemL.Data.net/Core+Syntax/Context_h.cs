using System;

namespace Core
{
    public enum LIMIT : byte
    {
        LENGTH = 0,
        SQL_LENGTH = 1,
        COLUMN = 2,
        EXPR_DEPTH = 3,
        COMPOUND_SELECT = 4,
        VDBE_OP = 5,
        FUNCTION_ARG = 6,
        ATTACHED = 7,
        LIKE_PATTERN_LENGTH = 8,
        VARIABLE_NUMBER = 9,
        TRIGGER_DEPTH = 10,
        MAX_ = 11,
    }

    public enum MAGIC : uint
    {
        OPEN = 0xa029a697,	// Database is open
        CLOSED = 0x9f3c2d33,  // Database is closed
        SICK = 0x4b771290,	// Error and awaiting close
        BUSY = 0xf03b7906,	// Database currently in use
        ERROR = 0xb5357930,	// An SQLITE_MISUSE error occurred
        ZOMBIE = 0x64cffc7f,  // Close with last statement close
    }

    public class LookasideSlot
    {
        public LookasideSlot Next;    // Next buffer in the list of free buffers
    }
    public struct Lookaside
    {
        public ushort Size;            // Size of each buffer in bytes
        public bool Enabled;           // False to disable new lookaside allocations
        public bool Malloced;          // True if pStart obtained from sqlite3_malloc()
        public int Outs;               // Number of buffers currently checked out
        public int MaxOuts;            // Highwater mark for nOut
        public int[] Stats = new int[3];			// 0: hits.  1: size misses.  2: full misses
        public LookasideSlot Free;	// List of available buffers
        public int Start;			// First byte of available memory space
        public int End;				// First byte past end of available space
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
        QueryFlattener = 0x0001,	// Query flattening
        ColumnCache = 0x0002,		// Column cache
        GroupByOrder = 0x0004,		// GROUPBY cover of ORDERBY
        FactorOutConst = 0x0008,	// Constant factoring
        IdxRealAsInt = 0x0010,		// Store REAL as INT in indices
        DistinctOpt = 0x0020,		// DISTINCT using indexes
        CoverIdxScan = 0x0040,		// Covering index scans
        OrderByIdxJoin = 0x0080,	// ORDER BY of joins via index
        SubqCoroutine = 0x0100,		// Evaluate subqueries as coroutines
        Transitive = 0x0200,		// Transitive constraints
        AllOpts = 0xffff,			// All optimizations
    }

    public class FuncDefHash
    {
        FuncDef[] data = new FuncDef[23]; // Hash table for functions
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
        public byte IsTransactionSavepoint;		    // True if the outermost savepoint is a TS
        public MAGIC Magic;						    // Magic number for detect library misuse
        public int[] Limits = new int[(int)LIMIT.MAX_];				// Limits
        public InitInfo Init;						// Information used during initialization
#if !OMIT_AUTHORIZATION
        public Func<object, int, string, string, string, int, RC> Auth; // Access authorization function
        public object AuthArg;						// 1st argument to the access auth function
#endif
#if !OMIT_VIRTUALTABLE
        public Hash Modules;						// populated by sqlite3_create_module()
        public VTableContext VTableCtx;			    // Context for active vtab connect/create
        public array_t<VTable> VTrans;			    // Virtual tables with open transactions / Allocated size of aVTrans
        public VTable Disconnect;					// Disconnect these in next sqlite3_prepare()
#endif
        public FuncDefHash Funcs;					// Hash table of connection functions
        public Hash CollSeqs;						// All collating sequences
        public int* BytesFreed;					    // If not NULL, increment this in DbFree()

        internal static void Error(object tag, RC rc, string format, params object[] args)
        {
            throw new NotImplementedException();
        }
    }
}