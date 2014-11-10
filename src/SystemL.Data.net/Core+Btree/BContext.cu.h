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
			FLAG_VdbeTrace = 0x00000100,
			FLAG_InternChanges = 0x00000200,
			FLAG_FullColNames = 0x00000400,
			FLAG_ShortColNames = 0x00000800,
			FLAG_CountRows = 0x00001000,
			FLAG_NullCallback = 0x00002000,
			FLAG_SqlTrace = 0x00004000,
			FLAG_VdbeListing = 0x00008000,
			FLAG_WriteSchema = 0x00010000,
			FLAG_NoReadlock = 0x00020000,
			FLAG_IgnoreChecks = 0x00040000,
			FLAG_ReadUncommitted = 0x0080000,
			FLAG_LegacyFileFmt = 0x00100000,
			FLAG_FullFSync = 0x00200000,
			FLAG_CkptFullFSync = 0x00400000,
			FLAG_RecoveryMode = 0x00800000,
			FLAG_ReverseOrder = 0x01000000,
			FLAG_RecTriggers = 0x02000000,
			FLAG_ForeignKeys = 0x04000000,
			FLAG_AutoIndex = 0x08000000,
			FLAG_PreferBuiltin = 0x10000000,
			FLAG_LoadExtension = 0x20000000,
			FLAG_EnableTrigger = 0x40000000,
		};

		array_t<DB> DBs;				// All backends / Number of backends currently in use
		FLAG Flags;
		int ActiveVdbeCnt;
		BusyHandlerType *BusyHandler;
		Savepoint *Savepoints;			// List of active savepoints
		int BusyTimeout;				// Busy handler timeout, in msec
		int SavepointsLength;			// Number of non-transaction savepoints
		//bool IsTransactionSavepoint;    // True if the outermost savepoint is a TS

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