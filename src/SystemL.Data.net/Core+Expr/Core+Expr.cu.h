#include "../Core+Table/Core+Table.cu.h"
namespace Core
{

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
			int Value;				// Non-negative integer value if EP_IntValue
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
		ynVar ColumnIdx;         
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
}