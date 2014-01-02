/*
To register
1. lib\Lalr regasm /codebase Contoso.VisualStudio.dll
2. ensure custom tool: LALR
*/

// All token codes are small integers with #defines that begin with "TK_"
%token_prefix TK_

// The type of the data attached to each token is Token.  This is also the default type for non-terminals.
%token_type {Token}
%default_type {Token}

// The generated parser function takes a 4th argument as follows:
%extra_argument {Parse *pParse}

// This code runs whenever there is a syntax error
%syntax_error {
	_assert(TOKEN.z[0]); // The tokenizer always gives us a token
	sqlite3ErrorMsg(parse, "near \"%T\": syntax error", &TOKEN);
}
%stack_overflow {
	sqlite3ErrorMsg(parse, "parser stack overflow");
}

// The name of the generated procedure that implements the parser is as follows:
%name sqlite3Parser

// The following text is included near the beginning of the C source code file that implements the parser.
%include {
	#include "sqliteInt.h"

	#define YYNOERRORRECOVERY 1 // Disable all error recovery processing in the parser push-down automaton.
	#define yytestcase(X) testcase(X) // Make yytestcase() the same as testcase()

	// An instance of this structure holds information about the LIMIT clause of a SELECT statement.
	struct LimitVal
	{
		Expr *Limit;	// The LIMIT expression.  NULL if there is no limit
		Expr *Offset;	// The OFFSET expression.  NULL if there is none
	};

	// An instance of this structure is used to store the LIKE, GLOB, NOT LIKE, and NOT GLOB operators.
	struct LikeOp
	{
		Token Operator;	// "like" or "glob" or "regexp"
		bool Not;		// True if the NOT keyword is present
	};

	// An instance of the following structure describes the event of a TRIGGER.  "a" is the event type, one of TK_UPDATE, TK_INSERT,
	// TK_DELETE, or TK_INSTEAD.  If the event is of the form
	//      UPDATE ON (a,b,c)
	// Then the "b" IdList records the list "a,b,c".
	struct TrigEvent { int a; IdList *b; };

	// An instance of this structure holds the ATTACH key and the key type.
	struct AttachKey { int type; Token key; };

	// One or more VALUES claues
	struct ValueList
	{
		ExprList *List;
		Select *Select;
	};
} // end %include

// Input is a single SQL command
input ::= cmdlist.
cmdlist ::= cmdlist ecmd.
cmdlist ::= ecmd.
ecmd ::= SEMI.
ecmd ::= explain cmdx SEMI.
explain ::= .           { sqlite3BeginParse(parse, 0); }
%ifndef SQLITE_OMIT_EXPLAIN
explain ::= EXPLAIN.              { sqlite3BeginParse(parse, 1); }
explain ::= EXPLAIN QUERY PLAN.   { sqlite3BeginParse(parse, 2); }
%endif  SQLITE_OMIT_EXPLAIN
cmdx ::= cmd.           { sqlite3FinishCoding(parse); }

///////////////////// Begin and end transactions. ////////////////////////////

cmd ::= BEGIN transtype(Y) trans_opt.  { sqlite3BeginTransaction(parse, Y); }
trans_opt ::= .
trans_opt ::= TRANSACTION.
trans_opt ::= TRANSACTION nm.
%type transtype {int}
transtype(A) ::= .             { A = TK_DEFERRED; }
transtype(A) ::= DEFERRED(X).  { A = @X; }
transtype(A) ::= IMMEDIATE(X). { A = @X; }
transtype(A) ::= EXCLUSIVE(X). { A = @X; }
cmd ::= COMMIT trans_opt.      { sqlite3CommitTransaction(parse); }
cmd ::= END trans_opt.         { sqlite3CommitTransaction(parse); }
cmd ::= ROLLBACK trans_opt.    { sqlite3RollbackTransaction(parse); }

savepoint_opt ::= SAVEPOINT.
savepoint_opt ::= .
cmd ::= SAVEPOINT nm(X). {
	sqlite3Savepoint(parse, SAVEPOINT_BEGIN, &X);
}
cmd ::= RELEASE savepoint_opt nm(X). {
	sqlite3Savepoint(parse, SAVEPOINT_RELEASE, &X);
}
cmd ::= ROLLBACK trans_opt TO savepoint_opt nm(X). {
	sqlite3Savepoint(parse, SAVEPOINT_ROLLBACK, &X);
}

///////////////////// The CREATE TABLE statement ////////////////////////////

cmd ::= create_table create_table_args.
create_table ::= createkw temp(T) TABLE ifnotexists(E) nm(Y) dbnm(Z). {
	sqlite3StartTable(parse,&Y,&Z,T,0,0,E);
}
createkw(A) ::= CREATE(X).  {
	parse->Db->Lookaside.Enabled = false;
	A = X;
}
%type ifnotexists {int}
ifnotexists(A) ::= .              { A = 0; }
ifnotexists(A) ::= IF NOT EXISTS. { A = 1; }
%type temp {int}
%ifndef SQLITE_OMIT_TEMPDB
temp(A) ::= TEMP.  { A = 1; }
%endif  SQLITE_OMIT_TEMPDB
temp(A) ::= .      { A = 0; }
create_table_args ::= LP columnlist conslist_opt(X) RP(Y). {
	sqlite3EndTable(parse, &X, &Y, 0);
}
create_table_args ::= AS select(S). {
	sqlite3EndTable(parse, 0, 0, S);
	sqlite3SelectDelete(parse->Db, S);
}
columnlist ::= columnlist COMMA column.
columnlist ::= column.

// A "column" is a complete description of a single column in a CREATE TABLE statement.  This includes the column name, its
// datatype, and other keywords such as PRIMARY KEY, UNIQUE, REFERENCES, NOT NULL and so forth.
column(A) ::= columnid(X) type carglist. {
	A.z = X.z;
	A.n = (int)(parse->LastToken.z - X.z) + parse->LastToken.n;
}
columnid(A) ::= nm(X). {
	sqlite3AddColumn(parse, &X);
	A = X;
	parse->ConstraintName.n = 0;
}


// An IDENTIFIER can be a generic identifier, or one of several keywords.  Any non-standard keyword can also be an identifier.
%type id {Token}
id(A) ::= ID(X).         { A = X; }
id(A) ::= INDEXED(X).    { A = X; }

// The following directive causes tokens ABORT, AFTER, ASC, etc. to fallback to ID if they will not parse as their original value.
// This obviates the need for the "id" nonterminal.
%fallback ID
	ABORT ACTION AFTER ANALYZE ASC ATTACH BEFORE BEGIN BY CASCADE CAST COLUMNKW
	CONFLICT DATABASE DEFERRED DESC DETACH EACH END EXCLUSIVE EXPLAIN FAIL FOR
	IGNORE IMMEDIATE INITIALLY INSTEAD LIKE_KW MATCH NO PLAN
	QUERY KEY OF OFFSET PRAGMA RAISE RELEASE REPLACE RESTRICT ROW ROLLBACK
	SAVEPOINT TEMP TRIGGER VACUUM VIEW VIRTUAL
%ifdef OMIT_COMPOUND_SELECT
	EXCEPT INTERSECT UNION
%endif
	REINDEX RENAME CTIME_KW IF
	.
%wildcard ANY.

// Define operator precedence early so that this is the first occurance of the operator tokens in the grammer.  Keeping the operators together
// causes them to be assigned integer values that are close together, which keeps parser tables smaller.
//
// The token values assigned to these symbols is determined by the order in which lemon first sees them.  It must be the case that ISNULL/NOTNULL,
// NE/EQ, GT/LE, and GE/LT are separated by only a single value.  See the sqlite3ExprIfFalse() routine for additional information on this constraint.
%left OR.
%left AND.
%right NOT.
%left IS MATCH LIKE_KW BETWEEN IN ISNULL NOTNULL NE EQ.
%left GT LE LT GE.
%right ESCAPE.
%left BITAND BITOR LSHIFT RSHIFT.
%left PLUS MINUS.
%left STAR SLASH REM.
%left CONCAT.
%left COLLATE.
%right BITNOT.

// And "ids" is an identifer-or-string.
%type ids {Token}
ids(A) ::= ID|STRING(X).   { A = X; }

// The name of a column or table can be any of the following:
%type nm {Token}
nm(A) ::= id(X).         { A = X; }
nm(A) ::= STRING(X).     { A = X; }
nm(A) ::= JOIN_KW(X).    { A = X; }

// A typetoken is really one or more tokens that form a type name such as can be found after the column name in a CREATE TABLE statement.
// Multiple tokens are concatenated to form the value of the typetoken.
%type typetoken {Token}
type ::= .
type ::= typetoken(X).                   { sqlite3AddColumnType(parse, &X); }
typetoken(A) ::= typename(X).   { A = X; }
typetoken(A) ::= typename(X) LP signed RP(Y). {
	A.z = X.z;
	A.n = (int)(&Y.z[Y.n] - X.z);
}
typetoken(A) ::= typename(X) LP signed COMMA signed RP(Y). {
	A.z = X.z;
	A.n = (int)(&Y.z[Y.n] - X.z);
}
%type typename {Token}
typename(A) ::= ids(X).             { A = X; }
typename(A) ::= typename(X) ids(Y). { A.z = X.z; A.n = Y.n+(int)(Y.z - X.z); }
signed ::= plus_num.
signed ::= minus_num.

// "carglist" is a list of additional constraints that come after the column name and column type in a CREATE TABLE statement.
carglist ::= carglist ccons.
carglist ::= .
ccons ::= CONSTRAINT nm(X).           {pParse->constraintName = X;}
ccons ::= DEFAULT term(X).            {sqlite3AddDefaultValue(pParse,&X);}
ccons ::= DEFAULT LP expr(X) RP.      {sqlite3AddDefaultValue(pParse,&X);}
ccons ::= DEFAULT PLUS term(X).       {sqlite3AddDefaultValue(pParse,&X);}
ccons ::= DEFAULT MINUS(A) term(X).      {
	ExprSpan v;
	v.Expr = sqlite3PExpr(parse, TK_UMINUS, X.Expr, 0, 0);
	v.Start = A.z;
	v.End = X.End;
	sqlite3AddDefaultValue(parse, &v);
}
ccons ::= DEFAULT id(X).              {
	ExprSpan v;
	spanExpr(&v, parse, TK_STRING, &X);
	sqlite3AddDefaultValue(parse, &v);
}

// In addition to the type name, we also care about the primary key and UNIQUE constraints.
ccons ::= NULL onconf.
ccons ::= NOT NULL onconf(R).    { sqlite3AddNotNull(parse, R); }
ccons ::= PRIMARY KEY sortorder(Z) onconf(R) autoinc(I).
                                 { sqlite3AddPrimaryKey(parse, 0, R, I, Z); }
ccons ::= UNIQUE onconf(R).      { sqlite3CreateIndex(parse, 0, 0, 0, 0, R, 0, 0, 0, 0); }
ccons ::= CHECK LP expr(X) RP.   { sqlite3AddCheckConstraint(parse, X.Expr); }
ccons ::= REFERENCES nm(T) idxlist_opt(TA) refargs(R).
                                 { sqlite3CreateForeignKey(parse, 0, &T, TA, R); }
ccons ::= defer_subclause(D).    { sqlite3DeferForeignKey(parse, D); }
ccons ::= COLLATE ids(C).        { sqlite3AddCollateType(parse, &C); }

// The optional AUTOINCREMENT keyword
%type autoinc {int}
autoinc(X) ::= .          { X = 0; }
autoinc(X) ::= AUTOINCR.  { X = 1; }

// The next group of rules parses the arguments to a REFERENCES clause that determine if the referential integrity checking is deferred or
// or immediate and which determine what action to take if a ref-integ check fails.
%type refargs {int}
refargs(A) ::= .                  { A = OE_None*0x0101; /* EV: R-19803-45884 */}
refargs(A) ::= refargs(X) refarg(Y). { A = (X & ~Y.mask) | Y.value; }
%type refarg {struct {int value; int mask;}}
refarg(A) ::= MATCH nm.              { A.value = 0;     A.mask = 0x000000; }
refarg(A) ::= ON INSERT refact.      { A.value = 0;     A.mask = 0x000000; }
refarg(A) ::= ON DELETE refact(X).   { A.value = X;     A.mask = 0x0000ff; }
refarg(A) ::= ON UPDATE refact(X).   { A.value = X<<8;  A.mask = 0x00ff00; }
%type refact {int}
refact(A) ::= SET NULL.              { A = OE_SetNull;  /* EV: R-33326-45252 */}
refact(A) ::= SET DEFAULT.           { A = OE_SetDflt;  /* EV: R-33326-45252 */}
refact(A) ::= CASCADE.               { A = OE_Cascade;  /* EV: R-33326-45252 */}
refact(A) ::= RESTRICT.              { A = OE_Restrict; /* EV: R-33326-45252 */}
refact(A) ::= NO ACTION.             { A = OE_None;     /* EV: R-33326-45252 */}
%type defer_subclause {int}
defer_subclause(A) ::= NOT DEFERRABLE init_deferred_pred_opt.     { A = 0; }
defer_subclause(A) ::= DEFERRABLE init_deferred_pred_opt(X).      { A = X; }
%type init_deferred_pred_opt {int}
init_deferred_pred_opt(A) ::= .                       { A = 0; }
init_deferred_pred_opt(A) ::= INITIALLY DEFERRED.     { A = 1; }
init_deferred_pred_opt(A) ::= INITIALLY IMMEDIATE.    { A = 0; }

conslist_opt(A) ::= .                         { A.n = 0; A.z = 0; }
conslist_opt(A) ::= COMMA(X) conslist.        { A = X; }
conslist ::= conslist tconscomma tcons.
conslist ::= tcons.
tconscomma ::= COMMA.            { parse->ConstraintName.n = 0;}
tconscomma ::= .
tcons ::= CONSTRAINT nm(X).      { parse->ConstraintName = X; }
tcons ::= PRIMARY KEY LP idxlist(X) autoinc(I) RP onconf(R). { sqlite3AddPrimaryKey(parse, X, R, I, 0); }
tcons ::= UNIQUE LP idxlist(X) RP onconf(R). { sqlite3CreateIndex(parse, 0, 0, 0, X, R, 0, 0, 0, 0); }
tcons ::= CHECK LP expr(E) RP onconf. { sqlite3AddCheckConstraint(parse, E.Expr); }
tcons ::= FOREIGN KEY LP idxlist(FA) RP
          REFERENCES nm(T) idxlist_opt(TA) refargs(R) defer_subclause_opt(D). {
		sqlite3CreateForeignKey(pParse, FA, &T, TA, R);
		sqlite3DeferForeignKey(pParse, D);
}
%type defer_subclause_opt {int}
defer_subclause_opt(A) ::= .                    { A = 0; }
defer_subclause_opt(A) ::= defer_subclause(X).  { A = X; }

// The following is a non-standard extension that allows us to declare the default behavior when there is a constraint conflict.
%type onconf { int }
%type orconf { uint8 }
%type resolvetype { int }
onconf(A) ::= .                              { A = OE_Default; }
onconf(A) ::= ON CONFLICT resolvetype(X).    { A = X; }
orconf(A) ::= .                              { A = OE_Default; }
orconf(A) ::= OR resolvetype(X).             { A = (uint8)X; }
resolvetype(A) ::= raisetype(X).             { A = X; }
resolvetype(A) ::= IGNORE.                   { A = OE_Ignore; }
resolvetype(A) ::= REPLACE.                  { A = OE_Replace; }

////////////////////////// The DROP TABLE /////////////////////////////////////
cmd ::= DROP TABLE ifexists(E) fullname(X). { sqlite3DropTable(parse, X, 0, E); }
%type ifexists {int}
ifexists(A) ::= IF EXISTS.   { A = 1; }
ifexists(A) ::= .            { A = 0; }

///////////////////// The CREATE VIEW statement /////////////////////////////
%ifndef OMIT_VIEW
cmd ::= createkw(X) temp(T) VIEW ifnotexists(E) nm(Y) dbnm(Z) AS select(S). { sqlite3CreateView(parse, &X, &Y, &Z, S, T, E); }
cmd ::= DROP VIEW ifexists(E) fullname(X). { sqlite3DropTable(parse, X, 1, E); }
%endif

//////////////////////// The SELECT statement /////////////////////////////////
//
cmd ::= select(X).  {
	SelectDest dest = { SRT_Output, 0, 0, 0, 0 };
	sqlite3Select(parse, X, &dest);
	sqlite3ExplainBegin(parse->Vdbe);
	sqlite3ExplainSelect(parse->Vdbe, X);
	sqlite3ExplainFinish(parse->Vdbe);
	sqlite3SelectDelete(parse->Db, X);
}

%type select {Select *}
%destructor select { sqlite3SelectDelete(pParse->db, $$); }
%type oneselect {Select *}
%destructor oneselect { sqlite3SelectDelete(pParse->db, $$); }

select(A) ::= oneselect(X).                      { A = X; }
%ifndef SQLITE_OMIT_COMPOUND_SELECT
select(A) ::= select(X) multiselect_op(Y) oneselect(Z).  {
	if (Z) { Z->op = (uint8)Y; Z->Prior = X; }
	else
		sqlite3SelectDelete(pParse->db, X);
	A = Z;
}
%type multiselect_op {int}
multiselect_op(A) ::= UNION(OP).             { A = @OP; }
multiselect_op(A) ::= UNION ALL.             { A = TK_ALL; }
multiselect_op(A) ::= EXCEPT|INTERSECT(OP).  { A = @OP; }
%endif SQLITE_OMIT_COMPOUND_SELECT
oneselect(A) ::= SELECT distinct(D) selcollist(W) from(X) where_opt(Y)
                 groupby_opt(P) having_opt(Q) orderby_opt(Z) limit_opt(L). {
	A = sqlite3SelectNew(parse, W, X, Y, P, Q, Z, D, L.Limit, L.Offset);
}

// The "distinct" nonterminal is true (1) if the DISTINCT keyword is present and false (0) if it is not.
%type distinct {uint16}
distinct(A) ::= DISTINCT.   { A = SF_Distinct; }
distinct(A) ::= ALL.        { A = 0; }
distinct(A) ::= .           { A = 0; }

// selcollist is a list of expressions that are to become the return values of the SELECT statement.  The "*" in statements like
// "SELECT * FROM ..." is encoded as a special expression with an opcode of TK_ALL.
%type selcollist {ExprList *}
%destructor selcollist { sqlite3ExprListDelete(parse->Db, $$); }
%type sclp {ExprList *}
%destructor sclp { sqlite3ExprListDelete(parse->Db, $$); }
sclp(A) ::= selcollist(X) COMMA.             { A = X; }
sclp(A) ::= .                                { A = 0; }
selcollist(A) ::= sclp(P) expr(X) as(Y).     {
	A = sqlite3ExprListAppend(parse, P, X.Expr);
	if (Y.n > 0) sqlite3ExprListSetName(parse, A, &Y, 1);
	sqlite3ExprListSetSpan(parse, A, &X);
}
selcollist(A) ::= sclp(P) STAR. {
	Expr *p = sqlite3Expr(parse->Db, TK_ALL, 0);
	A = sqlite3ExprListAppend(parse, P, p);
}
selcollist(A) ::= sclp(P) nm(X) DOT STAR(Y). {
	Expr *right = sqlite3PExpr(parse, TK_ALL, 0, 0, &Y);
	Expr *left = sqlite3PExpr(parse, TK_ID, 0, 0, &X);
	Expr *dot = sqlite3PExpr(parse, TK_DOT, left, right, 0);
	A = sqlite3ExprListAppend(parse, P, dot);
}

// An option "AS <id>" phrase that can follow one of the expressions that define the result set, or one of the tables in the FROM clause.
%type as {Token}
as(X) ::= AS nm(Y).    { X = Y; }
as(X) ::= ids(Y).      { X = Y; }
as(X) ::= .            { X.n = 0; }


%type seltablist {SrcList *}
%destructor seltablist { sqlite3SrcListDelete(pParse->db, $$); }
%type stl_prefix {SrcList *}
%destructor stl_prefix { sqlite3SrcListDelete(pParse->db, $$); }
%type from {SrcList *}
%destructor from { sqlite3SrcListDelete(pParse->db, $$); }

// A complete FROM clause.
from(A) ::= .                { A = sqlite3DbMallocZero(parse->db, sizeof(*A)); }
from(A) ::= FROM seltablist(X). {
	A = X;
	sqlite3SrcListShiftJoinType(A);
}

// "seltablist" is a "Select Table List" - the content of the FROM clause in a SELECT statement.  "stl_prefix" is a prefix of this list.
stl_prefix(A) ::= seltablist(X) joinop(Y).    {
	A = X;
	if (ALWAYS(A && A->nSrc > 0)) A->a[A->nSrc-1].jointype = (uint8)Y;
}
stl_prefix(A) ::= .                           { A = 0; }
seltablist(A) ::= stl_prefix(X) nm(Y) dbnm(D) as(Z) indexed_opt(I)
                  on_opt(N) using_opt(U). {
	A = sqlite3SrcListAppendFromTerm(parse, X, &Y, &D, &Z, 0, N, U);
	sqlite3SrcListIndexedBy(parse, A, &I);
}
%ifndef OMIT_SUBQUERY
seltablist(A) ::= stl_prefix(X) LP select(S) RP
                  as(Z) on_opt(N) using_opt(U). {
	A = sqlite3SrcListAppendFromTerm(pParse,X,0,0,&Z,S,N,U);
}
seltablist(A) ::= stl_prefix(X) LP seltablist(F) RP
                  as(Z) on_opt(N) using_opt(U). {
    if (X == 0 && Z.n == 0 && N == 0 && U == 0)
		A = F;
	else if (F->nSrc == 1)
	{
		A = sqlite3SrcListAppendFromTerm(parse, X, 0, 0, &Z, 0, N, U);
		if (A)
		{
			struct SrcList_item *new_ = &A->a[A->nSrc-1];
			struct SrcList_item *old = F->a;
			new_->zName = old->zName;
			new_->zDatabase = old->zDatabase;
			old->zName = old->zDatabase = 0;
		}
		sqlite3SrcListDelete(parse->Db, F);
	}
	else
	{
		sqlite3SrcListShiftJoinType(F);
		Select *subquery = sqlite3SelectNew(parse, 0, F, 0, 0, 0, 0, SF_NestedFrom, 0, 0);
		A = sqlite3SrcListAppendFromTerm(parse, X, 0, 0, &Z, subquery, N, U);
	}
}
%endif

%type dbnm {Token}
dbnm(A) ::= .          { A.z=0; A.n=0; }
dbnm(A) ::= DOT nm(X). { A = X; }

%type fullname {SrcList *}
%destructor fullname { sqlite3SrcListDelete(parse->Db, $$); }
fullname(A) ::= nm(X) dbnm(Y).	{ A = sqlite3SrcListAppend(parse->Db, 0, &X, &Y); }

%type joinop {int}
%type joinop2 {int}
joinop(X) ::= COMMA|JOIN.					{ X = JT_INNER; }
joinop(X) ::= JOIN_KW(A) JOIN.				{ X = sqlite3JoinType(parse, &A, 0, 0); }
joinop(X) ::= JOIN_KW(A) nm(B) JOIN.		{ X = sqlite3JoinType(parse, &A, &B, 0); }
joinop(X) ::= JOIN_KW(A) nm(B) nm(C) JOIN.	{ X = sqlite3JoinType(parse, &A, &B, &C); }

%type on_opt {Expr *}
%destructor on_opt { sqlite3ExprDelete(parse->Db, $$); }
on_opt(N) ::= ON expr(E).	{ N = E.pExpr; }
on_opt(N) ::= .             { N = 0; }

// Note that this block abuses the Token type just a little. If there is no "INDEXED BY" clause, the returned token is empty (z==0 && n==0). If
// there is an INDEXED BY clause, then the token is populated as per normal, with z pointing to the token data and n containing the number of bytes
// in the token.
//
// If there is a "NOT INDEXED" clause, then (z==0 && n==1), which is  normally illegal. The sqlite3SrcListIndexedBy() function 
// recognizes and interprets this as a special case.
%type indexed_opt {Token}
indexed_opt(A) ::= .					{ A.z=0; A.n=0; }
indexed_opt(A) ::= INDEXED BY nm(X).	{ A = X; }
indexed_opt(A) ::= NOT INDEXED.			{ A.z=0; A.n=1; }

%type using_opt {IdList*}
%destructor using_opt { sqlite3IdListDelete(parse->Db, $$); }
using_opt(U) ::= USING LP inscollist(L) RP. { U = L; }
using_opt(U) ::= .							{ U = 0; }


%type orderby_opt {ExprList *}
%destructor orderby_opt {sqlite3ExprListDelete(parse->Db, $$);}
%type sortlist {ExprList *}
%destructor sortlist { sqlite3ExprListDelete(parse->Db, $$); }

orderby_opt(A) ::= .                          { A = 0; }
orderby_opt(A) ::= ORDER BY sortlist(X).      { A = X; }
sortlist(A) ::= sortlist(X) COMMA expr(Y) sortorder(Z).	{ A = sqlite3ExprListAppend(parse, X, Y.pExpr); if (A) A->a[A->nExpr-1].sortOrder = (uint8)Z; }
sortlist(A) ::= expr(Y) sortorder(Z).					{ A = sqlite3ExprListAppend(parse, 0, Y.pExpr); if (A && ALWAYS(A->a)) A->a[0].sortOrder = (uint8)Z; }

%type sortorder {int}
sortorder(A) ::= ASC.           { A = SQLITE_SO_ASC; }
sortorder(A) ::= DESC.          { A = SQLITE_SO_DESC; }
sortorder(A) ::= .				{ A = SQLITE_SO_ASC; }

%type groupby_opt {ExprList*}
%destructor groupby_opt {sqlite3ExprListDelete(parse->Db, $$);}
groupby_opt(A) ::= .						{ A = 0; }
groupby_opt(A) ::= GROUP BY nexprlist(X).	{ A = X; }

%type having_opt {Expr *}
%destructor having_opt { sqlite3ExprDelete(pParse->db, $$); }
having_opt(A) ::= .					{ A = 0; }
having_opt(A) ::= HAVING expr(X).	{ A = X.pExpr; }

%type limit_opt {struct LimitVal}

// The destructor for limit_opt will never fire in the current grammar. The limit_opt non-terminal only occurs at the end of a single production
// rule for SELECT statements.  As soon as the rule that create the limit_opt non-terminal reduces, the SELECT statement rule will also
// reduce.  So there is never a limit_opt non-terminal on the stack except as a transient.  So there is never anything to destroy.
//
//%destructor limit_opt {
//		sqlite3ExprDelete(parse->Db, $$.pLimit);
//		sqlite3ExprDelete(parse->Db, $$.pOffset);
//}
limit_opt(A) ::= .								{ A.pLimit = 0; A.pOffset = 0; }
limit_opt(A) ::= LIMIT expr(X).					{ A.pLimit = X.pExpr; A.pOffset = 0; }
limit_opt(A) ::= LIMIT expr(X) OFFSET expr(Y).  { A.pLimit = X.pExpr; A.pOffset = Y.pExpr; }
limit_opt(A) ::= LIMIT expr(X) COMMA expr(Y).	{ A.pOffset = X.pExpr; A.pLimit = Y.pExpr; }

/////////////////////////// The DELETE statement /////////////////////////////
%ifdef ENABLE_UPDATE_DELETE_LIMIT
cmd ::= DELETE FROM fullname(X) indexed_opt(I) where_opt(W) 
        orderby_opt(O) limit_opt(L). {
	sqlite3SrcListIndexedBy(parse, X, &I);
	W = sqlite3LimitWhere(parse, X, W, O, L.pLimit, L.pOffset, "DELETE");
	sqlite3DeleteFrom(parse, X, W);
}
%endif
%ifndef ENABLE_UPDATE_DELETE_LIMIT
cmd ::= DELETE FROM fullname(X) indexed_opt(I) where_opt(W). {
		sqlite3SrcListIndexedBy(parse, X, &I);
		sqlite3DeleteFrom(parse, X, W);
}
%endif

%type where_opt {Expr *}
%destructor where_opt { sqlite3ExprDelete(parse->Db, $$); }
where_opt(A) ::= .              { A = 0; }
where_opt(A) ::= WHERE expr(X).	{ A = X.pExpr; }

////////////////////////// The UPDATE command ////////////////////////////////
%ifdef ENABLE_UPDATE_DELETE_LIMIT
cmd ::= UPDATE orconf(R) fullname(X) indexed_opt(I) SET setlist(Y) where_opt(W)
        orderby_opt(O) limit_opt(L).  {
	sqlite3SrcListIndexedBy(parse, X, &I);
	sqlite3ExprListCheckLength(parse, Y, "set list"); 
	W = sqlite3LimitWhere(parse, X, W, O, L.pLimit, L.pOffset, "UPDATE");
	sqlite3Update(parse, X, Y, W, R);
}
%endif
%ifndef SQLITE_ENABLE_UPDATE_DELETE_LIMIT
cmd ::= UPDATE orconf(R) fullname(X) indexed_opt(I) SET setlist(Y)
        where_opt(W).  {
	sqlite3SrcListIndexedBy(parse, X, &I);
	sqlite3ExprListCheckLength(parse, Y, "set list"); 
	sqlite3Update(parse, X, Y, W, R);
}
%endif

%type setlist {ExprList*}
%destructor setlist { sqlite3ExprListDelete(pParse->db, $$); }
setlist(A) ::= setlist(Z) COMMA nm(X) EQ expr(Y).	{ A = sqlite3ExprListAppend(parse, Z, Y.pExpr); sqlite3ExprListSetName(parse, A, &X, 1); }
setlist(A) ::= nm(X) EQ expr(Y).					{ A = sqlite3ExprListAppend(parse, 0, Y.pExpr); sqlite3ExprListSetName(parse, A, &X, 1); }

////////////////////////// The INSERT command /////////////////////////////////
cmd ::= insert_cmd(R) INTO fullname(X) inscollist_opt(F) valuelist(Y).		{ sqlite3Insert(pParse, X, Y.pList, Y.pSelect, F, R); }
cmd ::= insert_cmd(R) INTO fullname(X) inscollist_opt(F) select(S).			{ sqlite3Insert(pParse, X, 0, S, F, R); }
cmd ::= insert_cmd(R) INTO fullname(X) inscollist_opt(F) DEFAULT VALUES.	{ sqlite3Insert(pParse, X, 0, 0, F, R); }

%type insert_cmd {u8}
insert_cmd(A) ::= INSERT orconf(R). { A = R; }
insert_cmd(A) ::= REPLACE.          { A = OE_Replace; }

// A ValueList is either a single VALUES clause or a comma-separated list of VALUES clauses.  If it is a single VALUES clause then the
// ValueList.pList field points to the expression list of that clause. If it is a list of VALUES clauses, then those clauses are transformed
// into a set of SELECT statements without FROM clauses and connected by UNION ALL and the ValueList.pSelect points to the right-most SELECT in that compound.
%type valuelist {struct ValueList}
%destructor valuelist { sqlite3ExprListDelete(parse->Db, $$.pList); sqlite3SelectDelete(parse->Db, $$.pSelect); }
valuelist(A) ::= VALUES LP nexprlist(X) RP. { A.pList = X; A.pSelect = 0; }

// Since a list of VALUEs is inplemented as a compound SELECT, we have to disable the value list option if compound SELECTs are disabled.
%ifndef OMIT_COMPOUND_SELECT
valuelist(A) ::= valuelist(X) COMMA LP exprlist(Y) RP. {
	Select *right = sqlite3SelectNew(parse, Y, 0, 0, 0, 0, 0, 0, 0, 0);
	if (X.pList)
	{
		X.pSelect = sqlite3SelectNew(parse, X.pList, 0, 0, 0, 0, 0, 0, 0, 0);
		X.pList = 0;
	}
	A.pList = 0;
	if (X.pSelect == 0 || right == 0)
	{
		sqlite3SelectDelete(parse->Db, right);
		sqlite3SelectDelete(parse->Db, X.pSelect);
		A.pSelect = 0;
	}
	else
	{
		right->op = TK_ALL;
		right->pPrior = X.pSelect;
		right->selFlags |= SF_Values;
		right->pPrior->selFlags |= SF_Values;
		A.pSelect = right;
	}
}
%endif

%type inscollist_opt {IdList *}
%destructor inscollist_opt { sqlite3IdListDelete(parse->Db, $$); }
%type inscollist {IdList*}
%destructor inscollist { sqlite3IdListDelete(parse->Db, $$); }

inscollist_opt(A) ::= .                     { A = 0; }
inscollist_opt(A) ::= LP inscollist(X) RP.  { A = X; }
inscollist(A) ::= inscollist(X) COMMA nm(Y).	{ A = sqlite3IdListAppend(parse->Db, X, &Y); }
inscollist(A) ::= nm(Y).						{ A = sqlite3IdListAppend(parse->Db, 0, &Y); }

/////////////////////////// Expression Processing /////////////////////////////

%type expr {ExprSpan}
%destructor expr { sqlite3ExprDelete(parse->Db, $$.pExpr); }
%type term {ExprSpan}
%destructor term { sqlite3ExprDelete(parse->Db, $$.pExpr); }

%include {
	// This is a utility routine used to set the ExprSpan.zStart and ExprSpan.zEnd values of pOut so that the span covers the complete
	// range of text beginning with pStart and going to the end of pEnd.
	static void spanSet(ExprSpan *pOut, Token *pStart, Token *pEnd)
	{
		pOut->zStart = pStart->z;
		pOut->zEnd = &pEnd->z[pEnd->n];
	}

	// Construct a new Expr object from a single identifier.  Use the new Expr to populate pOut.  Set the span of pOut to be the identifier
	// that created the expression.
	static void spanExpr(ExprSpan *pOut, Parse *pParse, int op, Token *pValue)
	{
		pOut->pExpr = sqlite3PExpr(pParse, op, 0, 0, pValue);
		pOut->zStart = pValue->z;
		pOut->zEnd = &pValue->z[pValue->n];
	}
}

expr(A) ::= term(X).				{ A = X; }
expr(A) ::= LP(B) expr(X) RP(E).	{ A.pExpr = X.pExpr; spanSet(&A, &B, &E); }
term(A) ::= NULL(X).				{ spanExpr(&A, parse, @X, &X); }
expr(A) ::= id(X).					{ spanExpr(&A, parse, TK_ID, &X); }
expr(A) ::= JOIN_KW(X).				{ spanExpr(&A, parse, TK_ID, &X); }
expr(A) ::= nm(X) DOT nm(Y).		{
	Expr *temp1 = sqlite3PExpr(parse, TK_ID, 0, 0, &X);
	Expr *temp2 = sqlite3PExpr(parse, TK_ID, 0, 0, &Y);
	A.pExpr = sqlite3PExpr(parse, TK_DOT, temp1, temp2, 0);
	spanSet(&A, &X, &Y);
}
expr(A) ::= nm(X) DOT nm(Y) DOT nm(Z). {
	Expr *temp1 = sqlite3PExpr(parse, TK_ID, 0, 0, &X);
	Expr *temp2 = sqlite3PExpr(parse, TK_ID, 0, 0, &Y);
	Expr *temp3 = sqlite3PExpr(parse, TK_ID, 0, 0, &Z);
	Expr *temp4 = sqlite3PExpr(parse, TK_DOT, temp2, temp3, 0);
	A.pExpr = sqlite3PExpr(parse, TK_DOT, temp1, temp4, 0);
	spanSet(&A, &X, &Z);
}
term(A) ::= INTEGER|FLOAT|BLOB(X).	{ spanExpr(&A, parse, @X, &X); }
term(A) ::= STRING(X).              { spanExpr(&A, parse, @X, &X); }
expr(A) ::= REGISTER(X).			{
	// When doing a nested parse, one can include terms in an expression that look like this:   #1 #2 ...  These terms refer to registers
	// in the virtual machine.  #N is the N-th register.
	if (parse->Nested == 0)
	{
		sqlite3ErrorMsg(parse, "near \"%T\": syntax error", &X);
		A.pExpr = 0;
	}
	else
	{
		A.pExpr = sqlite3PExpr(parse, TK_REGISTER, 0, 0, &X);
		if (A.pExpr) sqlite3GetInt32(&X.z[1], &A.pExpr->iTable);
	}
	spanSet(&A, &X, &X);
}
expr(A) ::= VARIABLE(X).			{
	spanExpr(&A, pParse, TK_VARIABLE, &X);
	sqlite3ExprAssignVarNumber(pParse, A.pExpr);
	spanSet(&A, &X, &X);
}
expr(A) ::= expr(E) COLLATE ids(C).	{
	A.pExpr = sqlite3ExprAddCollateToken(parse, E.pExpr, &C);
	A.zStart = E.zStart;
	A.zEnd = &C.z[C.n];
}
%ifndef OMIT_CAST
expr(A) ::= CAST(X) LP expr(E) AS typetoken(T) RP(Y). {
  A.pExpr = sqlite3PExpr(parse, TK_CAST, E.pExpr, 0, &T);
  spanSet(&A, &X, &Y);
}
%endif
expr(A) ::= ID(X) LP distinct(D) exprlist(Y) RP(E). {
	if (Y && Y->nExpr > parse->Db->Limits[SQLITE_LIMIT_FUNCTION_ARG])
		sqlite3ErrorMsg(parse, "too many arguments on function %T", &X);
	A.pExpr = sqlite3ExprFunction(parse, Y, &X);
	spanSet(&A, &X, &E);
	if (D && A.pExpr) A.pExpr->flags |= EP_Distinct;
}
expr(A) ::= ID(X) LP STAR RP(E).	{
	A.pExpr = sqlite3ExprFunction(parse, 0, &X);
	spanSet(&A, &X, &E);
}
term(A) ::= CTIME_KW(OP).			{
  /// The CURRENT_TIME, CURRENT_DATE, and CURRENT_TIMESTAMP values are treated as functions that return constants
  A.pExpr = sqlite3ExprFunction(pParse, 0,&OP);
  if (A.pExpr) A.pExpr->op = TK_CONST_FUNC;  
  spanSet(&A, &OP, &OP);
}

%include {
	// This routine constructs a binary expression node out of two ExprSpan objects and uses the result to populate a new ExprSpan object.
	static void spanBinaryExpr(ExprSpan *pOut, Parse *parse, int op, ExprSpan *left, ExprSpan *right)
	{
		pOut->pExpr = sqlite3PExpr(parse, op, left->pExpr, right->pExpr, 0);
		pOut->zStart = left->zStart;
		pOut->zEnd = right->zEnd;
	}
}

expr(A) ::= expr(X) AND(OP) expr(Y).			{ spanBinaryExpr(&A, parse, @OP, &X, &Y); }
expr(A) ::= expr(X) OR(OP) expr(Y).				{ spanBinaryExpr(&A, parse, @OP, &X, &Y); }
expr(A) ::= expr(X) LT|GT|GE|LE(OP) expr(Y).	{ spanBinaryExpr(&A, parse, @OP, &X, &Y); }
expr(A) ::= expr(X) EQ|NE(OP) expr(Y).			{ spanBinaryExpr(&A, parse, @OP, &X, &Y); }
expr(A) ::= expr(X) BITAND|BITOR|LSHIFT|RSHIFT(OP) expr(Y). { spanBinaryExpr(&A, parse, @OP, &X, &Y); }
expr(A) ::= expr(X) PLUS|MINUS(OP) expr(Y).		{ spanBinaryExpr(&A, parse, @OP, &X, &Y); }
expr(A) ::= expr(X) STAR|SLASH|REM(OP) expr(Y).	{ spanBinaryExpr(&A, parse, @OP, &X, &Y); }
expr(A) ::= expr(X) CONCAT(OP) expr(Y).			{ spanBinaryExpr(&A, parse, @OP, &X, &Y); }
%type likeop {struct LikeOp}
likeop(A) ::= LIKE_KW(X).		{ A.eOperator = X; A.bNot = 0; }
likeop(A) ::= NOT LIKE_KW(X).	{ A.eOperator = X; A.bNot = 1; }
likeop(A) ::= MATCH(X).			{ A.eOperator = X; A.bNot = 0; }
likeop(A) ::= NOT MATCH(X).		{ A.eOperator = X; A.bNot = 1; }
expr(A) ::= expr(X) likeop(OP) expr(Y).  [LIKE_KW]  {
	ExprList *pList;
	pList = sqlite3ExprListAppend(parse,0, Y.pExpr);
	pList = sqlite3ExprListAppend(parse,pList, X.pExpr);
	A.pExpr = sqlite3ExprFunction(parse, pList, &OP.eOperator);
	if (OP.bNot) A.pExpr = sqlite3PExpr(parse, TK_NOT, A.pExpr, 0, 0);
	A.zStart = X.zStart;
	A.zEnd = Y.zEnd;
	if (A.pExpr) A.pExpr->flags |= EP_InfixFunc;
}
expr(A) ::= expr(X) likeop(OP) expr(Y) ESCAPE expr(E).  [LIKE_KW]  {
	ExprList *pList;
	pList = sqlite3ExprListAppend(pParse,0, Y.pExpr);
	pList = sqlite3ExprListAppend(pParse,pList, X.pExpr);
	pList = sqlite3ExprListAppend(pParse,pList, E.pExpr);
	A.pExpr = sqlite3ExprFunction(pParse, pList, &OP.eOperator);
	if( OP.bNot ) A.pExpr = sqlite3PExpr(pParse, TK_NOT, A.pExpr, 0, 0);
	A.zStart = X.zStart;
	A.zEnd = E.zEnd;
	if (A.pExpr) A.pExpr->flags |= EP_InfixFunc;
}

%include {
	// Construct an expression node for a unary postfix operator
	static void spanUnaryPostfix(ExprSpan *pOut, Parse *pParse, int op, ExprSpan *pOperand, Token *pPostOp)
	{
		pOut->pExpr = sqlite3PExpr(pParse, op, pOperand->pExpr, 0, 0);
		pOut->zStart = pOperand->zStart;
		pOut->zEnd = &pPostOp->z[pPostOp->n];
	}                           
}

expr(A) ::= expr(X) ISNULL|NOTNULL(E).  { spanUnaryPostfix(&A,pParse,@E,&X,&E); }
expr(A) ::= expr(X) NOT NULL(E).		{ spanUnaryPostfix(&A,pParse,TK_NOTNULL,&X,&E); }

%include {
	// A routine to convert a binary TK_IS or TK_ISNOT expression into a unary TK_ISNULL or TK_NOTNULL expression.
	static void binaryToUnaryIfNull(Parse *pParse, Expr *pY, Expr *pA, int op)
	{
		sqlite3 *db = pParse->db;
		if (db->mallocFailed==0 && pY->op==TK_NULL)
		{
			pA->op = (u8)op;
			sqlite3ExprDelete(db, pA->pRight);
			pA->pRight = 0;
		}
	}
}

//    expr1 IS expr2
//    expr1 IS NOT expr2
//
// If expr2 is NULL then code as TK_ISNULL or TK_NOTNULL.  If expr2 is any other expression, code as TK_IS or TK_ISNOT.
expr(A) ::= expr(X) IS expr(Y).			{ spanBinaryExpr(&A,pParse,TK_IS,&X,&Y); binaryToUnaryIfNull(pParse, Y.pExpr, A.pExpr, TK_ISNULL); }
expr(A) ::= expr(X) IS NOT expr(Y).		{ spanBinaryExpr(&A,pParse,TK_ISNOT,&X,&Y); binaryToUnaryIfNull(pParse, Y.pExpr, A.pExpr, TK_NOTNULL); }

%include {
	// Construct an expression node for a unary prefix operator
	static void spanUnaryPrefix(ExprSpan *pOut, Parse *pParse, int op, ExprSpan *pOperand, Token *pPreOp)
	{
		pOut->pExpr = sqlite3PExpr(pParse, op, pOperand->pExpr, 0, 0);
		pOut->zStart = pPreOp->z;
		pOut->zEnd = pOperand->zEnd;
	}
}

expr(A) ::= NOT(B) expr(X).				{ spanUnaryPrefix(&A, parse, @B, &X, &B); }
expr(A) ::= BITNOT(B) expr(X).			{ spanUnaryPrefix(&A, parse, @B, &X, &B); }
expr(A) ::= MINUS(B) expr(X). [BITNOT]	{ spanUnaryPrefix(&A, parse, TK_UMINUS, &X, &B); }
expr(A) ::= PLUS(B) expr(X). [BITNOT]	{ spanUnaryPrefix(&A, parse, TK_UPLUS, &X, &B); }

%type between_op {int}
between_op(A) ::= BETWEEN.		{ A = 0; }
between_op(A) ::= NOT BETWEEN.	{ A = 1; }
expr(A) ::= expr(W) between_op(N) expr(X) AND expr(Y). [BETWEEN] {
	ExprList *pList = sqlite3ExprListAppend(pParse,0, X.pExpr);
	pList = sqlite3ExprListAppend(pParse,pList, Y.pExpr);
	A.pExpr = sqlite3PExpr(pParse, TK_BETWEEN, W.pExpr, 0, 0);
	if (A.pExpr) A.pExpr->x.pList = pList;
	else sqlite3ExprListDelete(pParse->db, pList);
	if (N) A.pExpr = sqlite3PExpr(pParse, TK_NOT, A.pExpr, 0, 0);
	A.zStart = W.zStart;
	A.zEnd = Y.zEnd;
}
%ifndef OMIT_SUBQUERY
%type in_op {int}
in_op(A) ::= IN.      { A = 0; }
in_op(A) ::= NOT IN.  { A = 1; }
expr(A) ::= expr(X) in_op(N) LP exprlist(Y) RP(E). [IN] {
	if (Y == 0)
	{
		// Expressions of the form
		//      expr1 IN ()
		//      expr1 NOT IN ()
		// simplify to constants 0 (false) and 1 (true), respectively, regardless of the value of expr1.
		A.pExpr = sqlite3PExpr(parse, TK_INTEGER, 0, 0, &sqlite3IntTokens[N]);
		sqlite3ExprDelete(parse->db, X.pExpr);
	}
	else
	{
		A.pExpr = sqlite3PExpr(parse, TK_IN, X.pExpr, 0, 0);
		if (A.pExpr)
		{
			A.pExpr->x.pList = Y;
			sqlite3ExprSetHeight(parse, A.pExpr);
		}
		else sqlite3ExprListDelete(parse->db, Y);
		if (N) A.pExpr = sqlite3PExpr(parse, TK_NOT, A.pExpr, 0, 0);
	}
	A.zStart = X.zStart;
	A.zEnd = &E.z[E.n];
}
expr(A) ::= LP(B) select(X) RP(E). {
	A.pExpr = sqlite3PExpr(pParse, TK_SELECT, 0, 0, 0);
	if (A.pExpr)
	{
		A.pExpr->x.pSelect = X;
		ExprSetProperty(A.pExpr, EP_xIsSelect);
		sqlite3ExprSetHeight(pParse, A.pExpr);
	}
	else sqlite3SelectDelete(pParse->db, X);
	A.zStart = B.z;
	A.zEnd = &E.z[E.n];
}
expr(A) ::= expr(X) in_op(N) LP select(Y) RP(E).  [IN] {
	A.pExpr = sqlite3PExpr(pParse, TK_IN, X.pExpr, 0, 0);
	if (A.pExpr)
	{
		A.pExpr->x.pSelect = Y;
		ExprSetProperty(A.pExpr, EP_xIsSelect);
		sqlite3ExprSetHeight(pParse, A.pExpr);
	}
	else sqlite3SelectDelete(pParse->db, Y);
	if (N) A.pExpr = sqlite3PExpr(pParse, TK_NOT, A.pExpr, 0, 0);
	A.zStart = X.zStart;
	A.zEnd = &E.z[E.n];
}
expr(A) ::= expr(X) in_op(N) nm(Y) dbnm(Z). [IN] {
	SrcList *pSrc = sqlite3SrcListAppend(pParse->db, 0,&Y,&Z);
	A.pExpr = sqlite3PExpr(pParse, TK_IN, X.pExpr, 0, 0);
	if (A.pExpr)
	{
		A.pExpr->x.pSelect = sqlite3SelectNew(pParse, 0,pSrc,0,0,0,0,0,0,0);
		ExprSetProperty(A.pExpr, EP_xIsSelect);
		sqlite3ExprSetHeight(pParse, A.pExpr);
	}
	else sqlite3SrcListDelete(pParse->db, pSrc);
	if (N) A.pExpr = sqlite3PExpr(pParse, TK_NOT, A.pExpr, 0, 0);
	A.zStart = X.zStart;
	A.zEnd = Z.z ? &Z.z[Z.n] : &Y.z[Y.n];
}
expr(A) ::= EXISTS(B) LP select(Y) RP(E). {
	Expr *p = A.pExpr = sqlite3PExpr(pParse, TK_EXISTS, 0, 0, 0);
	if (p)
	{
		p->x.pSelect = Y;
		ExprSetProperty(p, EP_xIsSelect);
		sqlite3ExprSetHeight(pParse, p);
	}
	else sqlite3SelectDelete(pParse->db, Y);
	A.zStart = B.z;
	A.zEnd = &E.z[E.n];
}
%endif OMIT_SUBQUERY

// CASE expressions
expr(A) ::= CASE(C) case_operand(X) case_exprlist(Y) case_else(Z) END(E). {
	A.pExpr = sqlite3PExpr(pParse, TK_CASE, X, Z, 0);
	if (A.pExpr)
	{
		A.pExpr->x.pList = Y;
		sqlite3ExprSetHeight(pParse, A.pExpr);
	}
	else sqlite3ExprListDelete(pParse->db, Y);
	A.zStart = C.z;
	A.zEnd = &E.z[E.n];
}
%type case_exprlist {ExprList *}
%destructor case_exprlist { sqlite3ExprListDelete(parse->Db, $$); }
case_exprlist(A) ::= case_exprlist(X) WHEN expr(Y) THEN expr(Z). {
	A = sqlite3ExprListAppend(parse,X, Y.pExpr);
	A = sqlite3ExprListAppend(parse,A, Z.pExpr);
}
case_exprlist(A) ::= WHEN expr(Y) THEN expr(Z). {
	A = sqlite3ExprListAppend(parse,0, Y.pExpr);
	A = sqlite3ExprListAppend(parse,A, Z.pExpr);
}
%type case_else {Expr *}
%destructor case_else { sqlite3ExprDelete(parse->Db, $$); }
case_else(A) ::=  ELSE expr(X).     { A = X.pExpr; }
case_else(A) ::=  .                 { A = 0; }
%type case_operand {Expr *}
%destructor case_operand { sqlite3ExprDelete(parse->Db, $$); }
case_operand(A) ::= expr(X).        { A = X.pExpr; } 
case_operand(A) ::= .               { A = 0; } 

%type exprlist {ExprList *}
%destructor exprlist { sqlite3ExprListDelete(parse->Db, $$); }
%type nexprlist {ExprList*}
%destructor nexprlist { sqlite3ExprListDelete(parse->Db, $$); }

exprlist(A) ::= nexprlist(X).       { A = X; }
exprlist(A) ::= .                   { A = 0; }
nexprlist(A) ::= nexprlist(X) COMMA expr(Y).	{ A = sqlite3ExprListAppend(pParse, X, Y.pExpr); }
nexprlist(A) ::= expr(Y).						{ A = sqlite3ExprListAppend(pParse, 0, Y.pExpr); }


///////////////////////////// The CREATE INDEX command ///////////////////////
cmd ::= createkw(S) uniqueflag(U) INDEX ifnotexists(NE) nm(X) dbnm(D)
        ON nm(Y) LP idxlist(Z) RP(E). {
	sqlite3CreateIndex(parse, &X, &D, sqlite3SrcListAppend(parse->Db, 0, &Y, 0), Z, U, &S, &E, SQLITE_SO_ASC, NE);
}

%type uniqueflag {int}
uniqueflag(A) ::= UNIQUE.	{ A = OE_Abort; }
uniqueflag(A) ::= .			{ A = OE_None; }

%type idxlist {ExprList *}
%destructor idxlist { sqlite3ExprListDelete(parse->Db, $$); }
%type idxlist_opt {ExprList *}
%destructor idxlist_opt { sqlite3ExprListDelete(parse->Db, $$); }

idxlist_opt(A) ::= .                         { A = 0; }
idxlist_opt(A) ::= LP idxlist(X) RP.         { A = X; }
idxlist(A) ::= idxlist(X) COMMA nm(Y) collate(C) sortorder(Z).  {
	Expr *p = sqlite3ExprAddCollateToken(pParse, 0, &C);
	A = sqlite3ExprListAppend(pParse,X, p);
	sqlite3ExprListSetName(pParse,A,&Y,1);
	sqlite3ExprListCheckLength(pParse, A, "index");
	if (A) A->a[A->nExpr-1].sortOrder = (uint8)Z;
}
idxlist(A) ::= nm(Y) collate(C) sortorder(Z). {
	Expr *p = sqlite3ExprAddCollateToken(pParse, 0, &C);
	A = sqlite3ExprListAppend(pParse,0, p);
	sqlite3ExprListSetName(pParse, A, &Y, 1);
	sqlite3ExprListCheckLength(pParse, A, "index");
	if (A) A->a[A->nExpr-1].sortOrder = (uint8)Z;
}

%type collate {Token}
collate(C) ::= .					{ C.z = 0; C.n = 0; }
collate(C) ::= COLLATE ids(X).		{ C = X; }


///////////////////////////// The DROP INDEX command /////////////////////////
cmd ::= DROP INDEX ifexists(E) fullname(X).   { sqlite3DropIndex(pParse, X, E); }

///////////////////////////// The VACUUM command /////////////////////////////
%ifndef OMIT_VACUUM
%ifndef OMIT_ATTACH
cmd ::= VACUUM.					{ sqlite3Vacuum(parse); }
cmd ::= VACUUM nm.				{ sqlite3Vacuum(parse); }
%endif
%endif

///////////////////////////// The PRAGMA command /////////////////////////////
%ifndef OMIT_PRAGMA
cmd ::= PRAGMA nm(X) dbnm(Z).						{ sqlite3Pragma(parse, &X, &Z, 0, 0); }
cmd ::= PRAGMA nm(X) dbnm(Z) EQ nmnum(Y).			{ sqlite3Pragma(parse, &X, &Z, &Y, 0); }
cmd ::= PRAGMA nm(X) dbnm(Z) LP nmnum(Y) RP.		{ sqlite3Pragma(parse, &X, &Z, &Y, 0); }
cmd ::= PRAGMA nm(X) dbnm(Z) EQ minus_num(Y).		{ sqlite3Pragma(parse, &X, &Z, &Y, 1); }
cmd ::= PRAGMA nm(X) dbnm(Z) LP minus_num(Y) RP.	{ sqlite3Pragma(parse, &X, &Z, &Y, 1); }
nmnum(A) ::= plus_num(X).				{ A = X; }
nmnum(A) ::= nm(X).						{ A = X; }
nmnum(A) ::= ON(X).						{ A = X; }
nmnum(A) ::= DELETE(X).					{ A = X; }
nmnum(A) ::= DEFAULT(X).				{ A = X; }
%endif
plus_num(A) ::= PLUS number(X).			{ A = X; }
plus_num(A) ::= number(X).				{ A = X; }
minus_num(A) ::= MINUS number(X).		{ A = X; }
number(A) ::= INTEGER|FLOAT(X).			{ A = X; }

//////////////////////////// The CREATE TRIGGER command /////////////////////
%ifndef OMIT_TRIGGER

cmd ::= createkw trigger_decl(A) BEGIN trigger_cmd_list(S) END(Z). {
	Token all;
	all.z = A.z;
	all.n = (int)(Z.z - A.z) + Z.n;
	sqlite3FinishTrigger(pParse, S, &all);
}

trigger_decl(A) ::= temp(T) TRIGGER ifnotexists(NOERR) nm(B) dbnm(Z) 
                    trigger_time(C) trigger_event(D)
                    ON fullname(E) foreach_clause when_clause(G). {
	sqlite3BeginTrigger(pParse, &B, &Z, C, D.a, D.b, E, G, T, NOERR);
	A = (Z.n==0?B:Z);
}

%type trigger_time {int}
trigger_time(A) ::= BEFORE.      { A = TK_BEFORE; }
trigger_time(A) ::= AFTER.       { A = TK_AFTER;  }
trigger_time(A) ::= INSTEAD OF.  { A = TK_INSTEAD;}
trigger_time(A) ::= .            { A = TK_BEFORE; }

%type trigger_event {struct TrigEvent}
%destructor trigger_event { sqlite3IdListDelete(pParse->db, $$.b); }
trigger_event(A) ::= DELETE|INSERT(OP).       { A.a = @OP; A.b = 0; }
trigger_event(A) ::= UPDATE(OP).              { A.a = @OP; A.b = 0; }
trigger_event(A) ::= UPDATE OF inscollist(X). { A.a = TK_UPDATE; A.b = X; }

foreach_clause ::= .
foreach_clause ::= FOR EACH ROW.

%type when_clause {Expr *}
%destructor when_clause { sqlite3ExprDelete(pParse->db, $$); }
when_clause(A) ::= .             { A = 0; }
when_clause(A) ::= WHEN expr(X). { A = X.pExpr; }

%type trigger_cmd_list {TriggerStep *}
%destructor trigger_cmd_list { sqlite3DeleteTriggerStep(parse->db, $$); }
trigger_cmd_list(A) ::= trigger_cmd_list(Y) trigger_cmd(X) SEMI. {
	_assert(Y != 0);
	Y->pLast->pNext = X;
	Y->pLast = X;
	A = Y;
}
trigger_cmd_list(A) ::= trigger_cmd(X) SEMI. { 
	_assert(X != 0);
	X->pLast = X;
	A = X;
}

// Disallow qualified table names on INSERT, UPDATE, and DELETE statements within a trigger.  The table to INSERT, UPDATE, or DELETE is always in 
// the same database as the table that the trigger fires on.
%type trnm {Token}
trnm(A) ::= nm(X).			{ A = X; }
trnm(A) ::= nm DOT nm(X).	{ A = X; sqlite3ErrorMsg(parse, "qualified table names are not allowed on INSERT, UPDATE, and DELETE statements within triggers"); }

// Disallow the INDEX BY and NOT INDEXED clauses on UPDATE and DELETE statements within triggers.  We make a specific error message for this
// since it is an exception to the default grammar rules.
tridxby ::= .
tridxby ::= INDEXED BY nm.	{ sqlite3ErrorMsg(parse, "the INDEXED BY clause is not allowed on UPDATE or DELETE statements within triggers"); }
tridxby ::= NOT INDEXED.	{ sqlite3ErrorMsg(parse, "the NOT INDEXED clause is not allowed on UPDATE or DELETE statements within triggers"); }

%type trigger_cmd {TriggerStep *}
%destructor trigger_cmd { sqlite3DeleteTriggerStep(pParse->db, $$); }
// UPDATE 
trigger_cmd(A) ::= UPDATE orconf(R) trnm(X) tridxby SET setlist(Y) where_opt(Z).	{ A = sqlite3TriggerUpdateStep(pParse->db, &X, Y, Z, R); }
// INSERT
trigger_cmd(A) ::= insert_cmd(R) INTO trnm(X) inscollist_opt(F) valuelist(Y).		{ A = sqlite3TriggerInsertStep(pParse->db, &X, F, Y.pList, Y.pSelect, R); }
trigger_cmd(A) ::= insert_cmd(R) INTO trnm(X) inscollist_opt(F) select(S).			{ A = sqlite3TriggerInsertStep(pParse->db, &X, F, 0, S, R); }
// DELETE
trigger_cmd(A) ::= DELETE FROM trnm(X) tridxby where_opt(Y).						{ A = sqlite3TriggerDeleteStep(pParse->db, &X, Y); }
// SELECT
trigger_cmd(A) ::= select(X).														{ A = sqlite3TriggerSelectStep(pParse->db, X); }

// The special RAISE expression that may occur in trigger programs
expr(A) ::= RAISE(X) LP IGNORE RP(Y).  {
	A.pExpr = sqlite3PExpr(parse, TK_RAISE, 0, 0, 0); 
	if (A.pExpr) A.pExpr->affinity = OE_Ignore;
	A.zStart = X.z;
	A.zEnd = &Y.z[Y.n];
}
expr(A) ::= RAISE(X) LP raisetype(T) COMMA nm(Z) RP(Y).  {
	A.pExpr = sqlite3PExpr(parse, TK_RAISE, 0, 0, &Z); 
	if (A.pExpr) A.pExpr->affinity = (char)T;
	A.zStart = X.z;
	A.zEnd = &Y.z[Y.n];
}
%endif OMIT_TRIGGER

%type raisetype {int}
raisetype(A) ::= ROLLBACK.	{ A = OE_Rollback; }
raisetype(A) ::= ABORT.		{ A = OE_Abort; }
raisetype(A) ::= FAIL.      { A = OE_Fail; }

////////////////////////  DROP TRIGGER statement //////////////////////////////
%ifndef OMIT_TRIGGER
cmd ::= DROP TRIGGER ifexists(NOERR) fullname(X). { sqlite3DropTrigger(pParse,X,NOERR); }
%endif

//////////////////////// ATTACH DATABASE file AS name /////////////////////////
%ifndef OMIT_ATTACH
cmd ::= ATTACH database_kw_opt expr(F) AS expr(D) key_opt(K).	{ sqlite3Attach(pParse, F.pExpr, D.pExpr, K); }
cmd ::= DETACH database_kw_opt expr(D).							{ sqlite3Detach(pParse, D.pExpr); }

%type key_opt {Expr*}
%destructor key_opt {sqlite3ExprDelete(pParse->db, $$);}
key_opt(A) ::= .					{ A = 0; }
key_opt(A) ::= KEY expr(X).         { A = X.pExpr; }
database_kw_opt ::= DATABASE.
database_kw_opt ::= .
%endif

////////////////////////// REINDEX collation //////////////////////////////////
%ifndef OMIT_REINDEX
cmd ::= REINDEX.                { sqlite3Reindex(pParse, 0, 0); }
cmd ::= REINDEX nm(X) dbnm(Y).	{ sqlite3Reindex(pParse, &X, &Y); }
%endif

/////////////////////////////////// ANALYZE ///////////////////////////////////
%ifndef OMIT_ANALYZE
cmd ::= ANALYZE.                { sqlite3Analyze(pParse, 0, 0); }
cmd ::= ANALYZE nm(X) dbnm(Y).  { sqlite3Analyze(pParse, &X, &Y); }
%endif

//////////////////////// ALTER TABLE table ... ////////////////////////////////
%ifndef OMIT_ALTERTABLE
cmd ::= ALTER TABLE fullname(X) RENAME TO nm(Z).					{ sqlite3AlterRenameTable(parse, X, &Z); }
cmd ::= ALTER TABLE add_column_fullname ADD kwcolumn_opt column(Y). { sqlite3AlterFinishAddColumn(parse, &Y); }
add_column_fullname ::= fullname(X).	{ parse->Db->lookaside.bEnabled = 0; sqlite3AlterBeginAddColumn(parse, X); }
kwcolumn_opt ::= .
kwcolumn_opt ::= COLUMNKW.
%endif

//////////////////////// CREATE VIRTUAL TABLE ... /////////////////////////////
%ifndef OMIT_VIRTUALTABLE
cmd ::= create_vtab.						{ sqlite3VtabFinishParse(parse, 0); }
cmd ::= create_vtab LP vtabarglist RP(X).	{ sqlite3VtabFinishParse(parse, &X); }
create_vtab ::= createkw VIRTUAL TABLE ifnotexists(E)
                nm(X) dbnm(Y) USING nm(Z).	{ sqlite3VtabBeginParse(parse, &X, &Y, &Z, E); }
vtabarglist ::= vtabarg.
vtabarglist ::= vtabarglist COMMA vtabarg.
vtabarg ::= .								{ sqlite3VtabArgInit(parse);}
vtabarg ::= vtabarg vtabargtoken.
vtabargtoken ::= ANY(X).					{ sqlite3VtabArgExtend(parse, &X); }
vtabargtoken ::= lp anylist RP(X).			{ sqlite3VtabArgExtend(parse, &X); }
lp ::= LP(X).								{ sqlite3VtabArgExtend(parse, &X); }
anylist ::= .
anylist ::= anylist LP anylist RP.
anylist ::= anylist ANY.
%endif
