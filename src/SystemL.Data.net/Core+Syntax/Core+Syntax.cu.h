#include "../Parse.h"
#include "../Core+Btree/Core+Btree.cu.h"
#include "Context.cu.h"
namespace Core
{

#pragma region Limits

#ifndef MAX_ATTACHED
#define MAX_ATTACHED 10
#endif

#if MAX_ATTACHED > 30
	typedef uint64 yDbMask;
#else
	typedef unsigned int yDbMask;
#endif

#if MAX_VARIABLE_NUMBER <= 32767
	typedef int16 yVars;
#else
	typedef int yVars;
#endif

#ifndef N_COLCACHE
#define N_COLCACHE 10
#endif

#if _64BITSTATS
	typedef uint64 tRowcnt;    // 64-bit only if requested at compile-time
#else
	typedef uint32 tRowcnt;    // 32-bit is the default
#endif
	typedef uint64 Bitmask;
#define BMS ((int)(sizeof(Bitmask)*8))

#pragma endregion 

#pragma region Func

	struct FuncContext;

	enum FUNC : uint8
	{
		FUNC_LIKE = 0x01,		// Candidate for the LIKE optimization
		FUNC_CASE = 0x02,		// Case-sensitive LIKE-type function
		FUNC_EPHEM = 0x04,		// Ephemeral.  Delete with VDBE
		FUNC_NEEDCOLL = 0x08,	// sqlite3GetFuncCollSeq() might be called
		FUNC_COUNT = 0x10,		// Built-in count(*) aggregate
		FUNC_COALESCE = 0x20,	// Built-in coalesce() or ifnull() function
		FUNC_LENGTH = 0x40,		// Built-in length() function
		FUNC_TYPEOF = 0x80,		// Built-in typeof() function
	};
	__device__ FUNC inline operator|=(FUNC a, int b) { return (FUNC)(a | b); }

	struct FuncDestructor
	{
		int Refs;
		void (*Destroy)(void *);
		void *UserData;
	};

	struct FuncDef
	{
		int16 Args;				// Number of arguments.  -1 means unlimited
		uint8 PrefEnc;			// Preferred text encoding (SQLITE_UTF8, 16LE, 16BE)
		FUNC Flags;				// Some combination of SQLITE_FUNC_*
		void *UserData;			// User data parameter
		FuncDef *Next;	// Next function with same name */
		void (*Func)(FuncContext *, int, Mem**); // Regular function
		void (*Step)(FuncContext *, int, Mem**); // Aggregate step
		void (*Finalize)(FuncContext *); // Aggregate finalizer
		char *Name;				// SQL name of the function.
		FuncDef *Hash;			// Next with a different name but the same hash
		FuncDestructor *Destructor; // Reference counted destructor function
	};

	//   FUNCTION(zName, nArg, iArg, bNC, xFunc)
	//     Used to create a scalar function definition of a function zName implemented by C function xFunc that accepts nArg arguments. The
	//     value passed as iArg is cast to a (void*) and made available as the user-data (sqlite3_user_data()) for the function. If 
	//     argument bNC is true, then the SQLITE_FUNC_NEEDCOLL flag is set.
	//
	//   AGGREGATE(zName, nArg, iArg, bNC, xStep, xFinal)
	//     Used to create an aggregate function definition implemented by the C functions xStep and xFinal. The first four parameters
	//     are interpreted in the same way as the first 4 parameters to FUNCTION().
	//
	//   LIKEFUNC(zName, nArg, pArg, flags)
	//     Used to create a scalar function definition of a function zName that accepts nArg arguments and is implemented by a call to C 
	//     function likeFunc. Argument pArg is cast to a (void *) and made available as the function user-data (sqlite3_user_data()). The
	//     FuncDef.flags variable is set to the value passed as the flags parameter.
#define FUNCTION(zName, nArg, iArg, bNC, xFunc) {nArg, SQLITE_UTF8, (bNC*FUNC_NEEDCOLL), \
	INT_TO_PTR(iArg), 0, xFunc, 0, 0, #zName, 0, 0}
#define FUNCTION2(zName, nArg, iArg, bNC, xFunc, extraFlags) {nArg, SQLITE_UTF8, (bNC*FUNC_NEEDCOLL)|extraFlags, \
	INT_TO_PTR(iArg), 0, xFunc, 0, 0, #zName, 0, 0}
#define STR_FUNCTION(zName, nArg, pArg, bNC, xFunc) {nArg, SQLITE_UTF8, bNC*FUNC_NEEDCOLL, \
	pArg, 0, xFunc, 0, 0, #zName, 0, 0}
#define LIKEFUNC(zName, nArg, arg, flags) {nArg, TEXTENCODE_UTF8, flags, (void *)arg, 0, likeFunc, 0, 0, #zName, 0, 0}
#define AGGREGATE(zName, nArg, arg, nc, xStep, xFinal) {nArg, TEXTENCODE_UTF8, nc*SQLITE_FUNC_NEEDCOLL, \
	INT_TO_PTR(arg), 0, 0, xStep,xFinal,#zName,0,0}

#pragma endregion

#pragma region Table

	struct IVTable;
	struct IVTableCursor;
	struct IIndexInfo;

	class ITableModule //was:sqlite3_module
	{
	public:
		int Version;
		__device__ virtual RC Create(Context *, void *aux, int argc, const char *const *argv, IVTable **vtabs, char **);
		__device__ virtual RC Connect(Context *, void *aux, int argc, const char *const *argv, IVTable **vtabs, char **);
		__device__ virtual RC BestIndex(IVTable *vtab, IIndexInfo *);
		__device__ virtual RC Disconnect(IVTable *vtab);
		__device__ virtual RC Destroy(IVTable *vtab);
		__device__ virtual RC Open(IVTable *vtab, IVTableCursor **cursors);
		__device__ virtual RC Close(IVTableCursor*);
		__device__ virtual RC Filter(IVTableCursor*, int idxNum, const char *idxStr, int argc, Mem **argv);
		__device__ virtual RC Next(IVTableCursor*);
		__device__ virtual RC Eof(IVTableCursor*);
		__device__ virtual RC Column(IVTableCursor *, FuncContext *, int);
		__device__ virtual RC Rowid(IVTableCursor *, int64 *rowid);
		__device__ virtual RC Update(IVTable *, int, Mem **, int64 *);
		__device__ virtual RC Begin(IVTable *vtab);
		__device__ virtual RC Sync(IVTable *vtab);
		__device__ virtual RC Commit(IVTable *vtab);
		__device__ virtual RC Rollback(IVTable *vtab);
		__device__ virtual RC FindFunction(IVTable *vtab, int argsLength, const char *name, void (**func)(FuncContext *, int, Mem **), void **args);
		__device__ virtual RC Rename(IVTable *vtab, const char *new_);
		__device__ virtual RC Savepoint(IVTable *vtab, int);
		__device__ virtual RC Release(IVTable *vtab, int);
		__device__ virtual RC RollbackTo(IVTable *vtab, int);
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

	struct IIndexInfo // was:sqlite3_index_info
	{
		struct Constraint
		{
			int Column;				// Column on left-hand side of constraint
			INDEX_CONSTRAINT OP;	// Constraint operator
			bool Usable;			// True if this constraint is usable
			int TermOffset;			// Used internally - xBestIndex should ignore
		};
		struct Orderby
		{
			int Column;				// Column number
			bool Desc;				// True for DESC.  False for ASC.
		};
		struct ConstraintUsage
		{
			int ArgvIndex;			// if >0, constraint is part of argv to xFilter
			unsigned char Omit;		// Do not code a test for this constraint
		};
		// INPUTS
		array_t<Constraint> Constraints; // Table of WHERE clause constraints
		array_t<Orderby> OrderBys;	// The ORDER BY clause
		// OUTPUTS
		array_t<ConstraintUsage> ConstraintUsages;
		int IdxNum;					// Number used to identify the index
		char *IdxStr;				// String, possibly obtained from sqlite3_malloc
		int NeedToFreeIdxStr;		// Free idxStr using sqlite3_free() if true
		int OrderByConsumed;		// True if output is already ordered
		double EstimatedCost;		// Estimated cost of using this index
	};

	struct IVTable //was:sqlite3_vtab
	{
		const ITableModule *IModule;// The module for this virtual table
		char *ErrMsg;				// Error message from sqlite3_mprintf()
		// Virtual table implementations will typically add additional fields
	};

	struct IVTableCursor //was:sqlite3_vtab_cursor
	{
		IVTable *IVTable;			// Virtual table of this cursor
		// Virtual table implementations will typically add additional fields
	};

	enum OE : uint8
	{
		OE_None = 0,   // There is no constraint to check
		OE_Rollback = 1,   // Fail the operation and rollback the transaction
		OE_Abort = 2,   // Back out changes but do no rollback transaction
		OE_Fail = 3,   // Stop the operation but leave all prior changes
		OE_Ignore = 4,   // Ignore the error. Do not do the INSERT or UPDATE
		OE_Replace = 5,   // Delete existing record, then do INSERT or UPDATE
		OE_Restrict = 6,   // OE_Abort for IMMEDIATE, OE_Rollback for DEFERRED
		OE_SetNull = 7,   // Set the foreign key value to NULL
		OE_SetDflt = 8,   // Set the foreign key value to its default
		OE_Cascade = 9,   // Cascade the changes
		OE_Default = 99,  // Do whatever the default action is
	};

	enum CONFLICT : uint8
	{
		CONFLICT_ROLLBACK = 1,
		CONFLICT_IGNORE = 2,
		CONFLICT_FAIL = 3,
		CONFLICT_ABORT = 4,
		CONFLICT_REPLACE = 5
	};

#pragma endregion

#pragma region NAME1

	struct Expr;
	struct ExprList;

	enum TYPE : uint8
	{
		TYPE_INTEGER = 1,
		TYPE_FLOAT = 2,
		TYPE_BLOB = 4,
		TYPE_NULL = 5,
		TYPE_TEXT = 3,
	};

	struct Index
	{
		char *Name;					// Name of this index
		array_t2<uint16, int> Columns;   // Which columns are used by this index.  1st is 0 // Number of columns in table used by this index
		tRowcnt *RowEsts;			// From ANALYZE: Est. rows selected by each column
		Table *Table;				// The SQL table being indexed
		char *ColAff;				// String defining the affinity of each column
		Index *Next;				// The next index associated with the same table
		Schema *Schema;				// Schema containing this index
		uint8 *SortOrders;			// for each column: True==DESC, False==ASC
		char **CollNames;			// Array of collation sequence names for index
		int TNum;					// DB Page containing root of this index
		uint8 OnError;				// OE_Abort, OE_Ignore, OE_Replace, or OE_None
		unsigned AutoIndex:2;		// 1==UNIQUE, 2==PRIMARY KEY, 0==CREATE INDEX
		unsigned Unordered:1;		// Use this index for == or IN queries only
#ifdef ENABLE_STAT3		  
		tRowcnt AvgEq;					// Average nEq value for key values not in aSample
		array_t<IndexSample> Samples;   // Samples of the left-most key
#endif
	};

	struct IndexSample
	{
		union
		{
			char *Z;			// Value if eType is SQLITE_TEXT or SQLITE_BLOB */
			double R;			// Value if eType is SQLITE_FLOAT */
			int64 I;			// Value if eType is SQLITE_INTEGER */
		} u;
		TYPE Type;				// SQLITE_NULL, SQLITE_INTEGER ... etc.
		int Bytes;				// Size in byte of text or blob.
		tRowcnt Eqs;			// Est. number of rows where the key equals this sample
		tRowcnt Lts;			// Est. number of rows where key is less than this sample
		tRowcnt DLts;			// Est. number of distinct keys less than this sample
	};

	typedef array_t<char> Token;
	//struct Token
	//{
	//	const char *Z;     // Text of the token.  Not NULL-terminated!
	//	unsigned int N;    // Number of characters in this token
	//};

	struct AggInfo
	{
		struct AggInfoColumn
		{
			Table *Table;           // Source table
			int TableID;            // Cursor number of the source table
			int Column;				// Column number within the source table
			int SorterColumn;		// Column number in the sorting index
			int Mem;				// Memory location that acts as accumulator
			Expr *Expr;				// The original expression
		};
		struct AggInfoFunc
		{   
			Expr *Expr;             // Expression encoding the function
			FuncDef *Func;          // The aggregate function implementation
			int Mem;                // Memory location that acts as accumulator
			int Distinct;           // Ephemeral table used to enforce DISTINCT
		};
		uint8 DirectMode;			// Direct rendering mode means take data directly from source tables rather than from accumulators
		uint8 UseSortingIdx;		// In direct mode, reference the sorting index rather than the source table
		int SortingIdx;				// Cursor number of the sorting index
		int SortingIdxPTab;			// Cursor number of pseudo-table
		int SortingColumns;			// Number of columns in the sorting index
		ExprList *GroupBy;			// The group by clause
		array_t<AggInfoColumn> Columns; // For each column used in source tables
		int Accumulators;			// Number of columns that show through to the output. Additional columns are used only as parameters to aggregate functions
		array_t<AggInfoFunc> Funcs; // For each aggregate function
	};

#pragma endregion

#pragma region Select

	struct Parse;
	struct Select;
	struct WhereClause;

	struct IdList
	{
		struct IdListItem
		{
			char *Name;     // Name of the identifier
			int Idx;        // Index in some Table.aCol[] of a column named zName
		};
		array_t<IdListItem> s;
	};

	enum JT : uint8
	{
		JT_INNER = 0x0001,    // Any kind of inner or cross join
		JT_CROSS = 0x0002,    // Explicit use of the CROSS keyword
		JT_NATURAL = 0x0004,    // True for a "natural" join
		JT_LEFT = 0x0008,    // Left outer join
		JT_RIGHT = 0x0010,    // Right outer join
		JT_OUTER = 0x0020,    // The "OUTER" keyword is present
		JT_ERROR = 0x0040,    // unknown or unsupported join type
	};

	struct SrcList
	{
		struct SrcListItem
		{
			Schema *Schema;  // Schema to which this item is fixed
			char *Database;  // Name of database holding this table
			char *Name;      // Name of the table
			char *Alias;     // The "B" part of a "A AS B" phrase.  zName is the "A"
			Table *Tab;      // An SQL table corresponding to zName
			Select *Select;  // A SELECT statement used in place of a table name
			int AddrFillSub;  // Address of subroutine to manifest a subquery
			int RegReturn;    // Register holding return address of addrFillSub
			JT Jointype;      // Type of join between this able and the previous
			unsigned NotIndexed:1;    // True if there is a NOT INDEXED clause
			unsigned IsCorrelated:1;  // True if sub-query is correlated
			unsigned ViaCoroutine:1;  // Implemented as a co-routine
#ifndef OMIT_EXPLAIN
			uint8 SelectId;     // If pSelect!=0, the id of the sub-select in EQP
#endif
			int iCursor;      // The VDBE cursor number used to access this table
			Expr *On;        // The ON clause of a join
			IdList *Using;   // The USING clause of a join
			Bitmask ColUsed;  // Bit N (1<<N) set if column N of pTab is used
			char *IndexName;     // Identifier from "INDEXED BY <zIndex>" clause
			Index *Index;    // Index structure corresponding to zIndex, if any
		};
		int16 Srcs;        // Number of tables or subqueries in the FROM clause
		int16 Allocs;      // Number of entries allocated in a[] below
		SrcListItem a[1];             // One entry for each identifier on the list
	};

	enum WHERE : uint16
	{
		WHERE_ORDERBY_NORMAL = 0x0000, // No-op
		WHERE_ORDERBY_MIN = 0x0001, // ORDER BY processing for min() func
		WHERE_ORDERBY_MAX = 0x0002, // ORDER BY processing for max() func
		WHERE_ONEPASS_DESIRED = 0x0004, // Want to do one-pass UPDATE/DELETE
		WHERE_DUPLICATES_OK = 0x0008, // Ok to return a row more than once
		WHERE_OMIT_OPEN_CLOSE = 0x0010, // Table cursors are already open
		WHERE_FORCE_TABLE = 0x0020, // Do not use an index-only search
		WHERE_ONETABLE_ONLY = 0x0040, // Only code the 1st table in pTabList
		WHERE_AND_ONLY = 0x0080, // Don't use indices for OR terms
	};

	struct WherePlan
	{
		uint32 WsFlags;                   // WHERE_* flags that describe the strategy
		uint16 Eqs;                       // Number of == constraints
		uint16 OBSats;                    // Number of ORDER BY terms satisfied
		double Rows;                   // Estimated number of rows (for EQP)
		union
		{
			Index *Index;                   // Index when WHERE_INDEXED is true
			struct WhereTerm *Term;       // WHERE clause term for OR-search
			IIndexInfo *VTableIndex;  // Virtual table index to use
		} u;
	};

	struct WhereLevel
	{
		WherePlan Plan;       // query plan for this element of the FROM clause
		int iLeftJoin;        // Memory cell used to implement LEFT OUTER JOIN
		int iTabCur;          // The VDBE cursor used to access the table
		int iIdxCur;          // The VDBE cursor used to access pIdx
		int addrBrk;          // Jump here to break out of the loop
		int addrNxt;          // Jump here to start the next IN combination
		int addrCont;         // Jump here to continue with the next loop cycle
		int addrFirst;        // First instruction of interior of the loop
		uint8 iFrom;             // Which entry in the FROM clause
		uint8 op, p5;            // Opcode and P5 of the opcode that ends the loop
		int p1, p2;           // Operands of the opcode used to ends the loop
		union
		{               // Information that depends on plan.wsFlags
			struct
			{
				int nIn;              // Number of entries in aInLoop[]
				struct InLoop
				{
					int iCur;              // The VDBE cursor used by this IN operator
					int addrInTop;         // Top of the IN loop
					uint8 eEndLoopOp;         // IN Loop terminator. OP_Next or OP_Prev
				} *aInLoop;           // Information about each nested IN operator
			} in;                 // Used when plan.wsFlags&WHERE_IN_ABLE
			Index *pCovidx;       // Possible covering index for WHERE_MULTI_OR
		} u;
		double OptCost;      // "Optimal" cost for this level
		// The following field is really not part of the current level.  But we need a place to cache virtual table index information for each
		// virtual table in the FROM clause and the WhereLevel structure is a convenient place since there is one WhereLevel for each FROM clause element.
		IIndexInfo *IndexInfo;  // Index info for n-th source table
	};

	enum WHERE_DISTINCT : uint8
	{
		WHERE_DISTINCT_NOOP = 0,  // DISTINCT keyword not used
		WHERE_DISTINCT_UNIQUE = 1,  // No duplicates
		WHERE_DISTINCT_ORDERED = 2,  // All duplicates are adjacent
		WHERE_DISTINCT_UNORDERED = 3,  // Duplicates are scattered
	};

	struct WhereInfo
	{
		Parse *pParse;            // Parsing and code generating context
		SrcList *pTabList;        // List of tables in the join
		uint16 nOBSat;               // Number of ORDER BY terms satisfied by indices
		WHERE WctrlFlags;           // Flags originally passed to sqlite3WhereBegin()
		uint8 okOnePass;             // Ok to use one-pass algorithm for UPDATE/DELETE
		uint8 untestedTerms;         // Not all WHERE terms resolved by outer loop
		WHERE_DISTINCT eDistinct;             // One of the WHERE_DISTINCT_* values below
		int iTop;                 // The very beginning of the WHERE loop
		int iContinue;            // Jump here to continue with next record
		int iBreak;               // Jump here to break out of the loop
		int nLevel;               // Number of nested loop
		WhereClause *WC;			// Decomposition of the WHERE clause
		double SavedNQueryLoop;   // pParse->nQueryLoop outside the WHERE loop
		double RowOuts;           // Estimated number of output rows
		WhereLevel a[1];          // Information about each nest loop in WHERE
	};

	enum NC : uint8
	{
		NC_AllowAgg = 0x01,    // Aggregate functions are allowed here
		NC_HasAgg = 0x02,    // One or more aggregate functions seen
		NC_IsCheck = 0x04,    // True if resolving names in a CHECK constraint
		NC_InAggFunc = 0x08,    // True if analyzing arguments to an agg func
	};

	struct NameContext
	{
		Parse *Parse;				// The parser
		SrcList *SrcList;			// One or more tables used to resolve names
		ExprList *EList;			// Optional list of named expressions
		AggInfo *AggInfo;			// Information about aggregates at this level
		NameContext *Next;			// Next outer name context.  NULL for outermost
		int Refs;					// Number of names resolved by this context
		int Errs;					// Number of errors encountered while resolving names
		NC NcFlags;					// Zero or more NC_* flags defined below
	};

	enum SF : uint16
	{
		SF_Distinct = 0x0001,		// Output should be DISTINCT
		SF_Resolved = 0x0002,		// Identifiers have been resolved
		SF_Aggregate = 0x0004,		// Contains aggregate functions
		SF_UsesEphemeral = 0x0008,  // Uses the OpenEphemeral opcode
		SF_Expanded = 0x0010,		// sqlite3SelectExpand() called on this
		SF_HasTypeInfo = 0x0020,	// FROM subqueries have Table metadata
		SF_UseSorter = 0x0040,		// Sort using a sorter
		SF_Values = 0x0080,			// Synthesized from VALUES clause
		SF_Materialize = 0x0100,	// Force materialization of views
		SF_NestedFrom = 0x0200,		// Part of a parenthesized FROM clause
	};

	struct Select
	{
		ExprList *EList;			// The fields of the result
		uint8 OP;					// One of: TK_UNION TK_ALL TK_INTERSECT TK_EXCEPT
		SF SelFlags;				// Various SF_* values
		int iLimit, iOffset;		// Memory registers holding LIMIT & OFFSET counters
		int addrOpenEphm[3];		// OP_OpenEphem opcodes related to this select
		double nSelectRow;			// Estimated number of result rows
		SrcList *pSrc;				// The FROM clause
		Expr *pWhere;				// The WHERE clause
		ExprList *pGroupBy;			// The GROUP BY clause
		Expr *pHaving;				// The HAVING clause
		ExprList *pOrderBy;			// The ORDER BY clause
		Select *pPrior;				// Prior select in a compound select statement
		Select *pNext;				// Next select to the left in a compound
		Select *pRightmost;			// Right-most select in a compound select statement
		Expr *pLimit;				// LIMIT expression. NULL means not used.
		Expr *pOffset;				// OFFSET expression. NULL means not used.
	};

	enum SRT : uint8
	{
		SRT_Union = 1,				// Store result as keys in an index
		SRT_Except = 2,				// Remove result from a UNION index
		SRT_Exists = 3,				// Store 1 if the result is not empty
		SRT_Discard = 4,			// Do not save the results anywhere
		// IgnorableOrderby(x) : The ORDER BY clause is ignored for all of the above
		SRT_Output = 5,  // Output each row of result
		SRT_Mem = 6,  // Store result in a memory cell
		SRT_Set = 7,  // Store results as keys in an index
		SRT_Table = 8,  // Store result as data with an automatic rowid
		SRT_EphemTab = 9,  // Create transient tab and store like SRT_Table
		SRT_Coroutine = 10,  // Generate a single row of result
	};
#define IgnorableOrderby(x) ((x->Dest)<=SRT_Discard)

	struct SelectDest
	{
		uint8 eDest;         // How to dispose of the results.  On of SRT_* above.
		char affSdst;     // Affinity used when eDest==SRT_Set
		int iSDParm;      // A parameter used by the eDest disposal method
		int iSdst;        // Base register where results are written
		int nSdst;        // Number of registers allocated
	};

#pragma endregion

#pragma region Expr

	enum EP : uint16
	{
		EP_FromJoin = 0x0001,		// Originated in ON or USING clause of a join
		EP_Agg = 0x0002,			// Contains one or more aggregate functions
		EP_Resolved = 0x0004,		// IDs have been resolved to COLUMNs
		EP_Error = 0x0008,			// Expression contains one or more errors
		EP_Distinct = 0x0010,		// Aggregate function with DISTINCT keyword
		EP_VarSelect = 0x0020,		// pSelect is correlated, not constant
		EP_DblQuoted = 0x0040,		// token.z was originally in "..."
		EP_InfixFunc = 0x0080,		// True for an infix function: LIKE, GLOB, etc
		EP_Collate = 0x0100,		// Tree contains a TK_COLLATE opeartor
		EP_FixedDest = 0x0200,		// Result needed in a specific register
		EP_IntValue = 0x0400,		// Integer value contained in u.iValue
		EP_xIsSelect = 0x0800,		// x.pSelect is valid (otherwise x.pList is)
		EP_Hint = 0x1000,			// Not used
		EP_Reduced = 0x2000,		// Expr struct is EXPR_REDUCEDSIZE bytes only
		EP_TokenOnly = 0x4000,		// Expr struct is EXPR_TOKENONLYSIZE bytes only
		EP_Static = 0x8000,			// Held in memory not obtained from malloc()
	};

	enum EP2 : uint8
	{
		EP2_MallocedToken = 0x0001,	// Need to sqlite3DbFree() Expr.zToken
		EP2_Irreducible = 0x0002,	// Cannot EXPRDUP_REDUCE this Expr
	};

	struct ExprList;
	struct Expr
	{
		uint8 OP;					// Operation performed by this node
		uint8 Affinity;				// The affinity of the column or 0 if not a column
		EP Flags;					// Various flags.  EP_* See below
		union
		{
			char *Token;			// Token value. Zero terminated and dequoted
			int I;				// Non-negative integer value if EP_IntValue
		} u;

		// If the EP_TokenOnly flag is set in the Expr.flags mask, then no space is allocated for the fields below this point. An attempt to
		// access them will result in a segfault or malfunction. 
		Expr *Left;					// Left subnode
		Expr *Right;				// Right subnode
		union
		{
			ExprList *List;			// Function arguments or in "<expr> IN (<expr-list)"
			Select *Select;			// Used for sub-selects and "<expr> IN (<select>)"
		} x;

		// If the EP_Reduced flag is set in the Expr.flags mask, then no space is allocated for the fields below this point. An attempt to
		// access them will result in a segfault or malfunction.
#if MAX_EXPR_DEPTH>0
		int Height;					// Height of the tree headed by this node
#endif
		// TK_COLUMN: cursor number of table holding column
		// TK_REGISTER: register number
		// TK_TRIGGER: 1 -> new, 0 -> old
		int TableIdx;            
		// TK_COLUMN: column index.  -1 for rowid.
		// TK_VARIABLE: variable number (always >= 1).
		yVars ColumnIdx;         
		int16 Agg;					// Which entry in pAggInfo->aCol[] or ->aFunc[]
		int16 RightJoinTable;		// If EP_FromJoin, the right table of the join
		EP2 Flags2;					// Second set of flags.  EP2_...
		// TK_REGISTER: original value of Expr.op
		// TK_COLUMN: the value of p5 for OP_Column
		// TK_AGG_FUNCTION: nesting depth
		uint8 OP2;                
		// Used by TK_AGG_COLUMN and TK_AGG_FUNCTION
		AggInfo *AggInfo;
		// Table for TK_COLUMN expressions.
		Table *Table;
	};

#ifdef _DEBUG
# define ExprSetIrreducible(X) (X)->Flags2 |= EP2_Irreducible
#else
# define ExprSetIrreducible(X)
#endif

#define ExprHasProperty(E,P)     (((E)->Flags&(P))==(P))
#define ExprHasAnyProperty(E,P)  (((E)->Flags&(P))!=0)
#define ExprSetProperty(E,P)     (E)->Flags|=(P)
#define ExprClearProperty(E,P)   (E)->Flags&=~(P)

#define EXPR_FULLSIZE           sizeof(Expr)           // Full size
#define EXPR_REDUCEDSIZE        offsetof(Expr,iTable)  // Common features
#define EXPR_TOKENONLYSIZE      offsetof(Expr,Left)   // Fewer features

#define EXPRDUP_REDUCE         0x0001  // Used reduced-size Expr nodes

	struct ExprList
	{
		int Exprs;				// Number of expressions on the list
		int ECursor;			// VDBE Cursor associated with this ExprList
		// For each expression in the list
		struct ExprListItem
		{
			Expr *Expr;            // The list of expressions
			char *Name;            // Token associated with this expression
			char *Span;            // Original text of the expression
			uint8 SortOrder;           // 1 for DESC or 0 for ASC
			unsigned Done:1;       // A flag to indicate when processing is finished
			unsigned SpanIsTab:1; // zSpan holds DB.TABLE.COLUMN
			uint16 OrderByCol;        // For ORDER BY, column number in result set
			uint16 Alias;             // Index into Parse.aAlias[] for zName
		} *a;					// Alloc a power of two greater or equal to nExpr
	};

	struct ExprSpan
	{
		Expr *Expr;          // The expression parse tree
		const char *Start;   // First character of input text
		const char *End;     // One character past the end of input text
	};

#pragma endregion

#pragma region Table

	struct Trigger;
	struct Expr;

	struct TableModule
	{
		const ITableModule *IModule;    // Callback pointers
		const char *Name;               // Name passed to create_module()
		void *Aux;                      // pAux passed to create_module()
		void (*Destroy)(void *);        // Module destructor function
	};

	enum COLFLAG : uint16
	{
		COLFLAG_PRIMKEY = 0x0001,		// Column is part of the primary key
		COLFLAG_HIDDEN = 0x0002,		// A hidden column in a virtual table
	};
	__device__ COLFLAG inline operator|=(COLFLAG a, int b) { return (COLFLAG)(a | b); }

	struct Column
	{
		char *Name;					// Name of this column
		Expr *Dflt;					// Default value of this column
		char *DfltName;				// Original text of the default value
		char *Type;					// Data type for this column
		char *Coll;					// Collating sequence.  If NULL, use the default
		uint8 NotNull;				// An OE_ code for handling a NOT NULL constraint
		char Affinity;				// One of the SQLITE_AFF_... values
		COLFLAG ColFlags;			// Boolean properties.  See COLFLAG_ defines below
	};

	enum VTABLECONFIG : uint8
	{
		VTABLECONFIG_CONSTRAINT = 1,
	};

	struct VTable
	{
		Context *Ctx;				// Database connection associated with this table
		TableModule *Module;		// Pointer to module implementation
		IVTable *IVTable;			// Pointer to vtab instance
		int Refs;					// Number of pointers to this structure
		bool Constraint;			// True if constraints are supported
		int Savepoints;				// Depth of the SAVEPOINT stack
		VTable *Next;				// Next in linked list (see above)

#ifdef OMIT_VIRTUALTABLE
		__device__ inline static void Clear(Context *ctx, Table *table) {}
		__device__ inline static RC Sync(Context *ctx, char **error) { return RC_OK; }
		__device__ static RC Rollback(Context *ctx);
		__device__ static RC Commit(Context *ctx);
#define VTable_InSync(ctx) 0
		__device__ inline void Lock() {}
		__device__ inline void Unlock() {}
		__device__ inline static void UnlockList(Context *ctx) {}
		__device__ inline static RC Savepoint(Context *ctx, int op, IPager::SAVEPOINT savepoint) { return RC_OK; }
		__device__ inline static VTable *GetVTable(Context *ctx, Table *table) { return nullptr; }
#else
		__device__ static RC CreateModule(Context *ctx, const char *name, const ITableModule *imodule, void *aux, void (*destroy)(void *));
		__device__ void Lock();
		__device__ static VTable *GetVTable(Context *ctx, Table *table);
		__device__ void Unlock();
		__device__ static void Disconnect(Context *ctx, Table *table);
		__device__ static void UnlockList(Context *ctx);
		__device__ static void Clear(Context *ctx, Table *table);
		__device__ static void BeginParse(Parse *parse, Token *name1, Token *name2, Token *moduleName, int ifNotExists);
		__device__ static void FinishParse(Parse *parse, Token *end);
		__device__ static void ArgInit(Parse *parse);
		__device__ static void ArgExtend(Parse *parse, Token *token);
		__device__ static RC CallConnect(Parse *parse, Table *table);
		__device__ static RC CallCreate(Context *ctx, int dbidx, const char *tableName, char **error);
		__device__ static RC DeclareVTable(Context *ctx, const char *createTable);
		__device__ static RC CallDestroy(Context *ctx, int dbidx, const char *tableName);
		__device__ static RC Sync(Context *ctx, char **error);
		__device__ static RC Rollback(Context *ctx);
		__device__ static RC Commit(Context *ctx);
#define VTable_InSync(ctx) ((ctx)->VTrans.length>0 && (ctx)->VTrans==nullptr)
		__device__ static RC Begin(Context *ctx, VTable *vtable);
		__device__ static RC Savepoint(Context *ctx, int op, IPager::SAVEPOINT savepoint);
		__device__ static FuncDef *OverloadFunction(Context *ctx, FuncDef *def, int argsLength, Expr *expr);
		__device__ static void MakeWritable(Parse *parse, Table *table);
		__device__ static CONFLICT OnConflict(Context *ctx);
		__device__ static RC Config(Context *ctx, VTABLECONFIG op, void *arg1);
#endif
	};

	enum TF : uint8
	{
		TF_Readonly = 0x01,    // Read-only system table
		TF_Ephemeral = 0x02,    // An ephemeral table
		TF_HasPrimaryKey = 0x04,    // Table has a primary key
		TF_Autoincrement = 0x08,    // Integer primary key is autoincrement
		TF_Virtual = 0x10,    // Is a virtual table
	};
	__device__ TF inline operator|=(TF a, int b) { return (TF)(a | b); }

	struct Index;
	struct Select;
	struct FKey;
	struct ExprList;

	struct Table
	{
		char *Name;         // Name of the table or view
		array_t2<int16, Column> Cols;        // Information about each column
		Index *Index;       // List of SQL indexes on this table.
		Select *Select;     // NULL for tables.  Points to definition if a view.
		FKey *FKeys;         // Linked list of all foreign keys in this table
		char *ColAff;       // String defining the affinity of each column
#ifndef OMIT_CHECK
		ExprList *Check;    // All CHECK constraints
#endif
		tRowcnt RowEst;     // Estimated rows in table - from sqlite_stat1 table
		int TNum;            // Root BTree node for this table (see note above)
		int16 PKey;           // If not negative, use aCol[iPKey] as the primary key
		uint16 Refs;            // Number of pointers to this Table
		TF TabFlags;         // Mask of TF_* values
		uint8 KeyConf;          // What to do in case of uniqueness conflict on iPKey
#ifndef OMIT_ALTERTABLE
		int AddColOffset;		// Offset in CREATE TABLE stmt to add a new column
#endif
#ifndef OMIT_VIRTUALTABLE
		array_t<char *>ModuleArgs;  // Text of all module args. [0] is module name
		VTable *VTables;		// List of VTable objects.
#endif
		Trigger *Triggers;		// List of triggers stored in pSchema
		Schema *Schema;			// Schema that contains this table
		Table *NextZombie;		// Next on the Parse.pZombieTab list
	};

#ifndef OMIT_VIRTUALTABLE
#define IsVirtual(X)      (((X)->TabFlags & TF_Virtual)!=0)
#define IsHiddenColumn(X) (((X)->ColFlags & COLFLAG_HIDDEN)!=0)
#else
#define IsVirtual(X)      0
#define IsHiddenColumn(X) 0
#endif

	struct FKey
	{
		Table *pFrom;     /* Table containing the REFERENCES clause (aka: Child) */
		FKey *pNextFrom;  /* Next foreign key in pFrom */
		char *zTo;        /* Name of table that the key points to (aka: Parent) */
		FKey *pNextTo;    /* Next foreign key on table named zTo */
		FKey *pPrevTo;    /* Previous foreign key on table named zTo */
		int nCol;         /* Number of columns in this key */
		uint8 isDeferred;    /* True if constraint checking is deferred till COMMIT */
		uint8 aAction[2];          /* ON DELETE and ON UPDATE actions, respectively */
		Trigger *apTrigger[2];  /* Triggers for aAction[] actions */
		struct sColMap {  /* Mapping of columns in pFrom to columns in zTo */
			int iFrom;         /* Index of column in pFrom */
			char *zCol;        /* Name of column in zTo.  If 0 use PRIMARY KEY */
		} aCol[1];        /* One entry for each of nCol column s */
	};


#pragma endregion

#pragma region Rowset

	struct RowSet;
	__device__ RowSet *RowSet_Init(Context *ctx, void *space, unsigned int n);
	__device__ void RowSet_Clear(RowSet *p);
	__device__ void RowSet_Insert(RowSet *p, int64 rowid);
	__device__ bool RowSet_Test(RowSet *rowSet, uint8 batch, int64 rowid);
	__device__ bool RowSet_Next(RowSet *p, int64 *rowid);

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
		Core::Vdbe *Vdbe;			// An engine for executing database bytecode
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
		__device__ Table *LocateTableItem(bool isView,  SrcList_Item *p);
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

#pragma endregion

	};

	struct AuthContext
	{
		const char *AuthCtx;		// Put saved Parse.zAuthContext here
		Parse *Parse;				// The Parse structure
	};

#pragma endregion

	__device__ RC sqlite3_exec(Context *, const char *sql, bool (*callback)(void*,int,char**,char**), void *, char **errmsg);
}