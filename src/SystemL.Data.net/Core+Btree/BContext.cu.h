namespace Core
{
	class Btree;
	struct Savepoint;

#define CTXENCODE(ctx) ((ctx)->DBs[0].Schema->Encode)

	class BContext : public TagBase
	{
	public:
		static const int MAX_ATTACHED = 10;

		struct BusyHandlerType
		{
			int (*Func)(void *, int);	// The busy callback
			void *Arg;					// First arg to busy callback
			int Busys;					// Incremented with each busy call
		};

		struct DB
		{
			char *Name;					// Name of this database
			Btree *Bt;					// The B*Tree structure for this database file
			uint8 InTrans;				// 0: not writable.  1: Transaction.  2: Checkpoint
			uint8 SafetyLevel;			// How aggressive at syncing data to disk
			Schema *Schema;				// Pointer to database schema (possibly shared)
		};

		enum FLAG : uint32
		{
			FLAG_VdbeTrace      = 0x00000001,	// True to trace VDBE execution
			FLAG_InternChanges  = 0x00000002,	// Uncommitted Hash table changes
			FLAG_FullColNames   = 0x00000004,	// Show full column names on SELECT
			FLAG_ShortColNames  = 0x00000008,	// Show short columns names
			FLAG_CountRows      = 0x00000010,	// Count rows changed by INSERT, DELETE, or UPDATE and return the count using a callback.
			FLAG_NullCallback   = 0x00000020,	// Invoke the callback once if the result set is empty
			FLAG_SqlTrace       = 0x00000040,	// Debug print SQL as it executes
			FLAG_VdbeListing    = 0x00000080,	// Debug listings of VDBE programs
			FLAG_WriteSchema    = 0x00000100,	// OK to update SQLITE_MASTER
			FLAG_VdbeAddopTrace = 0x00000200,	// Trace sqlite3VdbeAddOp() calls
			FLAG_IgnoreChecks   = 0x00000400,	// Do not enforce check constraints
			FLAG_ReadUncommitted = 0x0000800,	// For shared-cache mode
			FLAG_LegacyFileFmt  = 0x00001000,	// Create new databases in format 1
			FLAG_FullFSync      = 0x00002000,	// Use full fsync on the backend
			FLAG_CkptFullFSync  = 0x00004000,	// Use full fsync for checkpoint
			FLAG_RecoveryMode   = 0x00008000,	// Ignore schema errors
			FLAG_ReverseOrder   = 0x00010000,	// Reverse unordered SELECTs
			FLAG_RecTriggers    = 0x00020000,	// Enable recursive triggers
			FLAG_ForeignKeys    = 0x00040000,	// Enforce foreign key constraints
			FLAG_AutoIndex      = 0x00080000,	// Enable automatic indexes
			FLAG_PreferBuiltin  = 0x00100000,	// Preference to built-in funcs
			FLAG_LoadExtension  = 0x00200000,	// Enable load_extension
			FLAG_EnableTrigger  = 0x00400000,	// True to enable triggers
		};

		array_t<DB> DBs;				// All backends / Number of backends currently in use
		FLAG Flags;						// Miscellaneous flags. See below
		int ActiveVdbeCnt;				// Number of VDBEs currently executing
		BusyHandlerType *BusyHandler;	// Busy callback
		DB DBStatics[2];				// Static space for the 2 default backends
		Savepoint *Savepoints;			// List of active savepoints
		int BusyTimeout;				// Busy handler timeout, in msec
		int SavepointsLength;			// Number of non-transaction savepoints

		__device__ inline int InvokeBusyHandler()
		{
			if (_NEVER(BusyHandler == nullptr) || BusyHandler->Func == nullptr || BusyHandler->Busys < 0)
				return 0;
			int rc = BusyHandler->Func(BusyHandler->Arg, BusyHandler->Busys);
			if (rc == 0)
				BusyHandler->Busys = -1;
			else
				BusyHandler->Busys++;
			return rc;
		}

		// HOOKS
#if ENABLE_UNLOCK_NOTIFY
		__device__ void ConnectionBlocked(BContext *a, BContext *b);
		__device__ void ConnectionUnlocked(BContext *a);
		__device__ void ConnectionClosed(BContext *a);
#else
		__device__ static void ConnectionBlocked(BContext *a, BContext *b) { }
		//__device__ static void ConnectionUnlocked(BContext *a) { }
		//__device__ static void ConnectionClosed(BContext *a) { }
#endif

		__device__ inline bool TempInMemory()
		{
			return true;
			//if (TEMP_STORE == 1) return (temp_store == 2);
			//if (TEMP_STORE == 2) return (temp_store != 1);
			//if (TEMP_STORE == 3) return true;
			//if (TEMP_STORE < 1 || TEMP_STORE > 3) return false;
			//return false;
		}
	};
	__device__ BContext::FLAG inline operator|=(BContext::FLAG a, int b) { return (BContext::FLAG)(a | b); }
	__device__ BContext::FLAG inline operator&=(BContext::FLAG a, int b) { return (BContext::FLAG)(a & b); }
}