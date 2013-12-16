#include "../Core+Btree/Core+Btree.cu.h"
namespace Core
{

#pragma region Table

	struct VTable;
	struct VTableCursor;
	struct IndexInfo;
	struct FuncContext;

	class ITableModule
	{
	public:
		int iVersion;
		__device__ virtual int Create(Context *, void *aux, int argc, const char *const*argv, VTable **vtabs, char **);
		__device__ virtual int Connect(Context *, void *aux, int argc, const char *const*argv, VTable **vtabs, char **);
		__device__ virtual int BestIndex(VTable *vtab, IndexInfo *);
		__device__ virtual int Disconnect(VTable *vtab);
		__device__ virtual int Destroy(VTable *vtab);
		__device__ virtual int Open(VTable *vtab, VTableCursor **cursors);
		__device__ virtual int Close(VTableCursor*);
		__device__ virtual int Filter(VTableCursor*, int idxNum, const char *idxStr, int argc, Mem **argv);
		__device__ virtual int Next(VTableCursor*);
		__device__ virtual int Eof(VTableCursor*);
		__device__ virtual int Column(VTableCursor *, FuncContext *, int);
		__device__ virtual int Rowid(VTableCursor *, int64 *rowid);
		__device__ virtual int Update(VTable *, int, Mem **, int64 *);
		__device__ virtual int Begin(VTable *vtab);
		__device__ virtual int Sync(VTable *vtab);
		__device__ virtual int Commit(VTable *vtab);
		__device__ virtual int Rollback(VTable *vtab);
		__device__ virtual int FindFunction(VTable *vtab, int argsLength, const char *name, void (**func)(FuncContext *, int, Mem **), void **args);
		__device__ virtual int Rename(VTable *vtab, const char *new_);
		__device__ virtual int Savepoint(VTable *vtab, int);
		__device__ virtual int Release(VTable *vtab, int);
		__device__ virtual int RollbackTo(VTable *vtab, int);
	};

	enum INDEX_CONSTRAINT : uint8
	{
		INDEX_CONSTRAINT_EQ = 2,
		INDEX_CONSTRAINT_GT = 4,
		INDEX_CONSTRAINT_LE = 8,
		INDEX_CONSTRAINT_LT = 16,
		INDEX_CONSTRAINT_GE = 32,
		INDEX_CONSTRAINT_MATCH = 64,
	};

	struct IndexInfo
	{
		struct IndexInfo_Constraint
		{
			int Column;         // Column on left-hand side of constraint
			INDEX_CONSTRAINT OP;// Constraint operator
			bool Usable;		// True if this constraint is usable
			int TermOffset;     // Used internally - xBestIndex should ignore
		};
		struct IndexInfo_Orderby
		{
			int Column;			// Column number
			bool Desc;			// True for DESC.  False for ASC.
		};
		struct IndexInfo_Constraintusage
		{
			int argvIndex;		// if >0, constraint is part of argv to xFilter
			unsigned char Omit; // Do not code a test for this constraint
		};
		// INPUTS
		array_t<IndexInfo_Constraint> Constraints; // Table of WHERE clause constraints
		array_t<IndexInfo_Orderby> OrderBys; // The ORDER BY clause
		// OUTPUTS
		array_t<IndexInfo_Constraintusage> ConstraintUsage;
		int IdxNum;                // Number used to identify the index
		char *IdxStr;              // String, possibly obtained from sqlite3_malloc
		int NeedToFreeIdxStr;      // Free idxStr using sqlite3_free() if true
		int OrderByConsumed;       // True if output is already ordered
		double EstimatedCost;      // Estimated cost of using this index
	};

	int sqlite3_create_module(Context *db, const char *name, const ITableModule *p, void *clientData, void (*destroy)(void *));

	struct VTable
	{
		const ITableModule *Module;		// The module for this virtual table
		char *ErrMsg;					// Error message from sqlite3_mprintf()
		// Virtual table implementations will typically add additional fields
	};

	struct VTableCursor
	{
		VTable *VTable;		// Virtual table of this cursor
		// Virtual table implementations will typically add additional fields
	};

	int sqlite3_declare_vtab(Context *, const char *sql);
	int sqlite3_overload_function(Context *, const char *funcName, int args);

#pragma endregion

	__device__ RC sqlite3_exec(Context *, const char *sql, bool (*callback)(void*,int,char**,char**), void *, char **errmsg);
}