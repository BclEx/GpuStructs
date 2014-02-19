#include "../Core+Syntax/Core+Syntax.cu.h"
namespace Core
{

#pragma region Internal

	struct VdbeCursor;
	struct VdbeFrame;
	struct Mem;
	struct VdbeFunc;
	struct Explain;

	typedef void (*Destructor_t)(void *);
#define DESTRUCTOR_STATIC ((Destructor_t)0)
#define DESTRUCTOR_TRANSIENT ((Destructor_t)-1)
#define DESTRUCTOR_DYNAMIC ((Destructor_t)SysEx::AllocSize)

#pragma endregion 

#pragma region Savepoint

	struct Savepoint
	{
		char *Name;				// Savepoint name (nul-terminated)
		int64 DeferredCons;		// Number of deferred fk violations
		Savepoint *Next;		// Parent savepoint (if any)
	};

#pragma endregion

#pragma region Column Affinity

	enum AFF
	{
		AFF_TEXT = 'a',
		AFF_NONE = 'b',
		AFF_NUMERIC = 'c',
		AFF_INTEGER = 'd',
		AFF_REAL = 'e',
		AFF_MASK = 0x67, // The SQLITE_AFF_MASK values masks off the significant bits of an affinity value. 
		// Additional bit values that can be ORed with an affinity without changing the affinity.
		AFF_BIT_JUMPIFNULL = 0x08, // jumps if either operand is NULL
		AFF_BIT_STOREP2 = 0x10,	// Store result in reg[P2] rather than jump
		AFF_BIT_NULLEQ = 0x80,  // NULL=NULL
	};

#define IsNumericAffinity(X) ((X) >= AFF_NUMERIC)

#pragma endregion

#pragma region Mem

	__device__ void Mem_ApplyAffinity(Mem *mem, uint8 affinity, TEXTENCODE encode);
	__device__ const void *Mem_Text(Mem *mem, TEXTENCODE encode);
	__device__ int Mem_Bytes(Mem *mem, TEXTENCODE encode);
	__device__ void Mem_SetStr(Mem *mem, int n, const void *z, TEXTENCODE encode, void (*del)(void *));
	__device__ void Mem_Free(Mem *mem);
	__device__ Mem *Mem_New(Context *db);
	__device__ RC Mem_FromExpr(Context *db, Expr *expr, TEXTENCODE encode, AFF affinity, Mem **value);

#pragma endregion

#pragma region Parse

	class Vdbe;
	struct TableLock;
	struct SubProgram;

#ifdef OMIT_VIRTUALTABLE
#define INDECLARE_VTABLE(x) false
#else
#define INDECLARE_VTABLE(x) (x->DeclareVTable)
#endif

	//enum OPFLAG
	//{
	//	OPFLAG_NCHANGE = 0x01,		// Set to update db->nChange
	//	OPFLAG_LASTROWID = 0x02,    // Set to update db->lastRowid
	//	OPFLAG_ISUPDATE = 0x04,		// This OP_Insert is an sql UPDATE
	//	OPFLAG_APPEND = 0x08,		// This is likely to be an append
	//	OPFLAG_USESEEKRESULT = 0x10,// Try to avoid a seek in BtreeInsert()
	//	OPFLAG_CLEARCACHE = 0x20,   // Clear pseudo-table cache in OP_Column
	//	OPFLAG_LENGTHARG = 0x40,    // OP_Column only used for length()
	//	OPFLAG_TYPEOFARG = 0x80,    // OP_Column only used for typeof()
	//	OPFLAG_BULKCSR = 0x01,		// OP_Open** used to open bulk cursor
	//	OPFLAG_P2ISREG = 0x02,		// P2 to OP_Open** is a register number
	//	OPFLAG_PERMUTE = 0x01,		// OP_Compare: use the permutation
	//};

	struct AutoincInfo
	{
		AutoincInfo *Next;		// Next info block in a list of them all
		Table *Table;			// Table this info block refers to
		int DB;					// Index in sqlite3.aDb[] of database holding pTab
		int RegCtr;				// Memory register holding the rowid counter
	};

	struct TriggerPrg
	{
		Trigger *Trigger;		// Trigger this program was coded from
		TriggerPrg *Next;		// Next entry in Parse.pTriggerPrg list
		SubProgram *Program;	// Program implementing pTrigger/orconf
		int Orconf;             // Default ON CONFLICT policy
		uint32 Colmasks[2];     // Masks of old.*, new.* columns accessed
	};

	struct Parse
	{
		struct yColCache
		{
			int Table;				// Table cursor number
			int Column;				// Table column number
			uint8 TempReg;			// iReg is a temp register that needs to be freed
			int Level;				// Nesting level
			int Reg;				// Reg with value of this column. 0 means none.
			int Lru;				// Least recently used entry has the smallest value
		};

		Context *Ctx;				// The main database structure
		char *ErrMsg;				// An error message
		Vdbe *V;					// An engine for executing database bytecode
		RC RC;						// Return code from execution
		uint8 ColNamesSet;			// TRUE after OP_ColumnName has been issued to pVdbe
		uint8 CheckSchema;			// Causes schema cookie check after an error
		uint8 Nested;				// Number of nested calls to the parser/code generator
		//uint8 TempReg;			// Number of temporary registers in aTempReg[]
		uint8 TempRegsInUse;		// Number of aTempReg[] currently checked out
		//uint8 ColCaches;			// Number of entries in aColCache[]
		uint8 ColCacheIdx;			// Next entry in aColCache[] to replace
		uint8 IsMultiWrite;			// True if statement may modify/insert multiple rows
		uint8 MayAbort;				// True if statement may throw an ABORT exception
		array_t3<uint8, int, 8> TempReg; // Holding area for temporary registers
		int RangeRegs;				// Size of the temporary register block
		int RangeRegIdx;			// First register in temporary register block
		int Errs;					// Number of errors seen
		int Tabs;					// Number of previously allocated VDBE cursors
		int Mems;					// Number of memory cells used so far
		int Sets;					// Number of sets used so far
		int Onces;					// Number of OP_Once instructions so far
		int CkBase;					// Base register of data during check constraints
		int CacheLevel;				// ColCache valid when aColCache[].iLevel<=iCacheLevel
		int CacheCnt;				// Counter used to generate aColCache[].lru values
		array_t3<uint8, yColCache, N_COLCACHE> ColCaches; // One for each column cache entry
		yDbMask WriteMask;			// Start a write transaction on these databases
		yDbMask CookieMask;			// Bitmask of schema verified databases
		int CookieGoto;				// Address of OP_Goto to cookie verifier subroutine
		int CookieValue[MAX_ATTACHED + 2];  // Values of cookies to verify
		int RegRowid;				// Register holding rowid of CREATE TABLE entry
		int RegRoot;				// Register holding root page number for new objects
		int MaxArgs;				// Max args passed to user function by sub-program
		Token ConstraintName;		// Name of the constraint currently being parsed
#ifndef OMIT_SHARED_CACHE
		// int TableLocks;			// Number of locks in aTableLock
		array_t<TableLock> TableLocks; // Required table locks for shared-cache mode
#endif
		AutoincInfo *Ainc;			// Information about AUTOINCREMENT counters

		// Information used while coding trigger programs.
		Parse *Toplevel;			// Parse structure for main program (or NULL)
		Table *TriggerTab;			// Table triggers are being coded for
		double QueryLoops;			// Estimated number of iterations of a query
		uint32 Oldmask;				// Mask of old.* columns referenced
		uint32 Newmask;				// Mask of new.* columns referenced
		uint8 TriggerOp;			// TK_UPDATE, TK_INSERT or TK_DELETE
		uint8 Orconf;				// Default ON CONFLICT policy for trigger steps
		uint8 DisableTriggers;		// True to disable triggers

		// Above is constant between recursions.  Below is reset before and after each recursion
		int VarsSeen;				// Number of '?' variables seen in the SQL so far
		//int nzVar;				// Number of available slots in azVar[]
		uint8 Explain;				// True if the EXPLAIN flag is found on the query
#ifndef OMIT_VIRTUALTABLE
		bool DeclareVTable;			// True if inside sqlite3_declare_vtab()
		//int nVtabLock;			// Number of virtual tables to lock
#endif
		//int nAlias;				// Number of aliased result set columns
		int nHeight;				// Expression tree height of current sub-select
#ifndef OMIT_EXPLAIN
		int iSelectId;				// ID of current select for EXPLAIN output
		int iNextSelectId;			// Next available select ID for EXPLAIN output
#endif
		array_t<char *>Vars;		// Pointers to names of parameters
		Core::Vdbe *Reprepare;		// VM being reprepared (sqlite3Reprepare())
		array_t<int> Alias;			// Register used to hold aliased result
		const char *Tail;			// All SQL text past the last semicolon parsed
		Table *NewTable;			// A table being constructed by CREATE TABLE
		Trigger *NewTrigger;		// Trigger under construct by a CREATE TRIGGER
		const char *AuthContext;	// The 6th parameter to db->xAuth callbacks
		Token NameToken;			// Token with unqualified schema object name
		Token LastToken;			// The last token parsed
#ifndef OMIT_VIRTUALTABLE
		Token Arg;					// Complete text of a module argument
		array_t<Table *> VTableLocks; // Pointer to virtual tables needing locking
#endif
		Table *ZombieTab;			// List of Table objects to delete after code gen
		TriggerPrg *TriggerPrg;		// Linked list of coded triggers

#pragma region FromBuild_c

		__device__ void BeginParse(bool explainFlag);
#ifndef OMIT_SHARED_CACHE
		__device__ void TableLock(int db, int table, bool isWriteLock, const char *name);
#endif
		__device__ void FinishCoding();
		__device__ void NestedParse(const char *format, void **args);
		__device__ static Table *FindTable(Context *ctx, const char *name, const char *database);
		__device__ Table *LocateTable(bool isView, const char *name, const char *database);
		__device__ Table *LocateTableItem(bool isView,  SrcList::SrcListItem *item);
		__device__ static Index *FindIndex(Context *ctx, const char *name, const char *database);
		__device__ static void UnlinkAndDeleteIndex(Context *ctx, int db, const char *indexName);
		__device__ static void CollapseDatabaseArray(Context *ctx);
		__device__ static void ResetOneSchema(Context *ctx, int db);
		__device__ static void ResetAllSchemasOfConnection(Context *ctx);
		__device__ static void CommitInternalChanges(Context *ctx);
		__device__ static void DeleteTable(Context *ctx, Table *table);
		__device__ static void UnlinkAndDeleteTable(Context *ctx, int db, const char *tableName);
		__device__ static char *NameFromToken(Context *ctx, Token *name);
		__device__ void OpenMasterTable(int db);
		__device__ static int FindDbName(Context *ctx, const char *name);
		__device__ static int FindDb(Context *ctx, Token *name);
		__device__ int TwoPartName(Token *name1, Token *name2, Token **unqual);
		__device__ Core::RC CheckObjectName(const char *name);
		__device__ void StartTable(Token *name1, Token *name2, bool isTemp, bool isView, bool isVirtual, bool noErr);
		__device__ void AddColumn(Token *name);
		__device__ void AddNotNull(uint8 onError);
		__device__ static AFF AffinityType(const char *data);
		__device__ void AddColumnType(Token *type);
		__device__ void AddDefaultValue(ExprSpan *span);
		__device__ void AddPrimaryKey(ExprList *list, uint8 onError, bool autoInc, int sortOrder);
		__device__ void AddCheckConstraint(Expr *checkExpr);
		__device__ void AddCollateType(Token *token);
		__device__ CollSeq *LocateCollSeq(const char *name);
		__device__ void ChangeCookie(int db);
		__device__ void EndTable(Token *cons, Token *end, Select *select);
#ifndef OMIT_VIEW
		__device__ void CreateView(Token *begin, Token *name1, Token *name2, Select *select, bool isTemp, bool noErr);
#endif
#if !defined(OMIT_VIEW) || !defined(OMIT_VIRTUALTABLE)
		__device__ int ViewGetColumnNames(Table *table);
#endif
#ifndef OMIT_VIEW
		__device__ static void ViewResetAll(Context *ctx, int db);
#endif
#ifndef OMIT_AUTOVACUUM
		__device__ void Parse::RootPageMoved(Context *ctx, int db, int from, int to);
#endif
		__device__ void ClearStatTables(int db, const char *type, const char *name);
		__device__ void CodeDropTable(Table *table, int db, bool isView);
		__device__ void DropTable(SrcList *name, bool isView, bool noErr);
		__device__ void CreateForeignKey(ExprList *fromCol, Token *to, ExprList *toCol, int flags);
		__device__ void DeferForeignKey(bool isDeferred);
		__device__ void RefillIndex(Index *index, int memRootPage);











#pragma endregion

	};

	struct AuthContext
	{
		const char *AuthCtx;		// Put saved Parse.zAuthContext here
		Parse *Parse;				// The Parse structure
	};

#pragma endregion

}
#include "Vdbe.cu.h"

