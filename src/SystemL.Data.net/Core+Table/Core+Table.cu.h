#include "../Core+Btree/Core+Btree.cu.h"
namespace Core
{
	struct VTable;
	struct Trigger;

	struct Module
	{
		const ITableModule *Module;       // Callback pointers
		const char *Name;                   // Name passed to create_module()
		void *Aux;                          // pAux passed to create_module()
		void (*Destroy)(void *);            // Module destructor function
	};

	enum COLFLAG : uint16
	{
		COLFLAG_PRIMKEY = 0x0001,    // Column is part of the primary key
		COLFLAG_HIDDEN = 0x0002,    // A hidden column in a virtual table
	};

	struct Column
	{
		char *Name;     // Name of this column
		Expr *Dflt;     // Default value of this column
		char *DfltName; // Original text of the default value
		char *zType;     // Data type for this column
		char *zColl;     // Collating sequence.  If NULL, use the default
		uint8 NotNull;      // An OE_ code for handling a NOT NULL constraint
		char Affinity;   // One of the SQLITE_AFF_... values
		COLFLAG ColFlags;    // Boolean properties.  See COLFLAG_ defines below
	};

	struct VTable
	{
		Context *Db;            // Database connection associated with this table
		Module *Module;			// Pointer to module implementation
		VTable *VTable;			// Pointer to vtab instance
		int Refs;               // Number of pointers to this structure
		bool Constraint;        // True if constraints are supported
		int Savepoint;          // Depth of the SAVEPOINT stack
		VTable *Next;			// Next in linked list (see above)
	};

	struct FKey
	{
		Table *pFrom;     /* Table containing the REFERENCES clause (aka: Child) */
		FKey *pNextFrom;  /* Next foreign key in pFrom */
		char *zTo;        /* Name of table that the key points to (aka: Parent) */
		FKey *pNextTo;    /* Next foreign key on table named zTo */
		FKey *pPrevTo;    /* Previous foreign key on table named zTo */
		int nCol;         /* Number of columns in this key */
		/* EV: R-30323-21917 */
		uint8 isDeferred;    /* True if constraint checking is deferred till COMMIT */
		uint8 aAction[2];          /* ON DELETE and ON UPDATE actions, respectively */
		Trigger *apTrigger[2];  /* Triggers for aAction[] actions */
		struct sColMap {  /* Mapping of columns in pFrom to columns in zTo */
			int iFrom;         /* Index of column in pFrom */
			char *zCol;        /* Name of column in zTo.  If 0 use PRIMARY KEY */
		} aCol[1];        /* One entry for each of nCol column s */
	};

	enum TF : uint8
	{
		TF_Readonly = 0x01,    // Read-only system table
		TF_Ephemeral = 0x02,    // An ephemeral table
		TF_HasPrimaryKey = 0x04,    // Table has a primary key
		TF_Autoincrement = 0x08,    // Integer primary key is autoincrement
		TF_Virtual = 0x10,    // Is a virtual table
	};

	struct Table
	{
		char *zName;         /* Name of the table or view */
		Column *aCol;        /* Information about each column */
		Index *pIndex;       /* List of SQL indexes on this table. */
		Select *pSelect;     /* NULL for tables.  Points to definition if a view. */
		FKey *pFKey;         /* Linked list of all foreign keys in this table */
		char *zColAff;       /* String defining the affinity of each column */
#ifndef SQLITE_OMIT_CHECK
		ExprList *pCheck;    /* All CHECK constraints */
#endif
		tRowcnt nRowEst;     /* Estimated rows in table - from sqlite_stat1 table */
		int tnum;            /* Root BTree node for this table (see note above) */
		i16 iPKey;           /* If not negative, use aCol[iPKey] as the primary key */
		i16 nCol;            /* Number of columns in this table */
		u16 nRef;            /* Number of pointers to this Table */
		TF tabFlags;         /* Mask of TF_* values */
		u8 keyConf;          /* What to do in case of uniqueness conflict on iPKey */
#ifndef OMIT_ALTERTABLE
		int addColOffset;    /* Offset in CREATE TABLE stmt to add a new column */
#endif
#ifndef OMIT_VIRTUALTABLE
		int nModuleArg;      /* Number of arguments to the module */
		char **azModuleArg;  /* Text of all module args. [0] is module name */
		VTable *pVTable;     /* List of VTable objects. */
#endif
		Trigger *pTrigger;   /* List of triggers stored in pSchema */
		Schema *pSchema;     /* Schema that contains this table */
		Table *pNextZombie;  /* Next on the Parse.pZombieTab list */
	};

#ifndef OMIT_VIRTUALTABLE
#define IsVirtual(X)      (((X)->tabFlags & TF_Virtual)!=0)
#define IsHiddenColumn(X) (((X)->colFlags & COLFLAG_HIDDEN)!=0)
#else
#define IsVirtual(X)      0
#define IsHiddenColumn(X) 0
#endif



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

#pragma region Rowset

	struct RowSet;
	__device__ RowSet *RowSet_Init(Context *db, void *space, unsigned int n);
	__device__ void RowSet_Clear(RowSet *p);
	__device__ void RowSet_Insert(RowSet *p, int64 rowid);
	__device__ bool RowSet_Test(RowSet *rowSet, uint8 batch, int64 rowid);
	__device__ bool RowSet_Next(RowSet *p, int64 *rowid);

#pragma endregion

	__device__ RC sqlite3_exec(Context *, const char *sql, bool (*callback)(void*,int,char**,char**), void *, char **errmsg);
}