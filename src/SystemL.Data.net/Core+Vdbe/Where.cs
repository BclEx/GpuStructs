using System;
using System.Diagnostics;
using System.Text;
using Bitmask = System.UInt64;

namespace Core
{
    public partial class Where
    {
        const int BMS = ((int)(sizeof(Bitmask) * 8));

#if DEBUG
        static bool WhereTrace = false;
        static void WHERETRACE(string x, params object[] args) { if (WhereTrace) Console.WriteLine("p:" + string.Format(x, args)); }
#else
        static void WHERETRACE(string x, params object[] args) { }
#endif

        enum WO : ushort
        {
            IN = 0x001,
            EQ = 0x002,
            LT = (WO.EQ << (TK.LT - TK.EQ)),
            LE = (WO.EQ << (TK.LE - TK.EQ)),
            GT = (WO.EQ << (TK.GT - TK.EQ)),
            GE = (WO.EQ << (TK.GE - TK.EQ)),
            MATCH = 0x040,
            ISNULL = 0x080,
            OR = 0x100,			// Two or more OR-connected terms
            AND = 0x200,			// Two or more AND-connected terms
            EQUIV = 0x400,       // Of the form A==B, both columns
            NOOP = 0x800,		// This term does not restrict search space

            ALL = 0xfff,			// Mask of all possible WO_* values
            SINGLE = 0x0ff,      // Mask of all non-compound WO_* values
        }

        enum TERM : byte
        {
            DYNAMIC = 0x01,   // Need to call sqlite3ExprDelete(db, pExpr)
            VIRTUAL = 0x02,   // Added by the optimizer.  Do not code
            CODED = 0x04,   // This term is already coded
            COPIED = 0x08,   // Has a child
            ORINFO = 0x10,   // Need to free the WhereTerm.u.pOrInfo object
            ANDINFO = 0x20,   // Need to free the WhereTerm.u.pAndInfo obj
            OR_OK = 0x40,   // Used during OR-clause processing
#if !ENABLE_STAT3
            VNULL = 0x80,   // Manufactured x>NULL or x<=NULL term
#else
		    VNULL = 0x00,   // Disabled if not using stat3
#endif
        }

        public class WhereTerm
        {
            public Expr Expr;                   // Pointer to the subexpression that is this term
            public int Parent;                  // Disable pWC.a[iParent] when this term disabled
            public int LeftCursor;              // Cursor number of X in "X <op> <expr>"
            public class _u
            {
                public int LeftColumn;          // Column number of X in "X <op> <expr>"
                public WhereOrInfo OrInfo;      // Extra information if eOperator==WO_OR
                public WhereAndInfo AndInfo;    // Extra information if eOperator==WO_AND
            }
            public _u u = new _u();
            public WO EOperator;                // A WO_xx value describing <op> 
            public TERM WtFlags;                // TERM_xxx bit flags.  See below
            public byte Childs;                 // Number of children that must disable us
            public WhereClause WC;              // The clause this term is part of
            public Bitmask PrereqRight;         // Bitmask of tables used by pExpr.pRight
            public Bitmask PrereqAll;           // Bitmask of tables referenced by pExpr
        }

        public class WhereClause
        {
            public Parse Parse;                 // The parser context
            public WhereMaskSet MaskSet;        // Mapping of table cursor numbers to bitmasks
            public WhereClause Outer;			// Outer conjunction
            public byte OP;                     // Split operator.  TK_AND or TK_OR
            public ushort WctrlFlags;		    // Might include WHERE_AND_ONLY
            public int Terms;                   // Number of terms
            public array_t<WhereTerm> Slots;    // Each a[] describes a term of the WHERE cluase
#if SMALL_STACK
            public WhereTerm[] Statics = new WhereTerm[1];      // Initial static space for a[]
#else
            public WhereTerm[] Statics = new WhereTerm[8];      // Initial static space for a[]
#endif
            public void memcpy(WhereClause wc)
            {
                wc.Parse = Parse;
                wc.MaskSet = new WhereMaskSet();
                MaskSet.memcpy(wc.MaskSet);
                wc.Outer = new WhereClause();
                Outer.memcpy(wc.Outer);
                wc.OP = OP;
                wc.Terms = Terms;
                wc.Slots.length = Slots.length;
                wc.Slots.data = (WhereTerm[])Slots.data.Clone();
                wc.Statics = (WhereTerm[])Statics.Clone();
            }
        }

        public class WhereOrInfo
        {
            public WhereClause WC = new WhereClause();      // Decomposition into subterms
            public Bitmask Indexable;                       // Bitmask of all indexable tables in the clause
        }

        public class WhereAndInfo
        {
            public WhereClause WC = new WhereClause();      // The subexpression broken out
        }

        public class WhereMaskSet
        {
            public int n;                           // Number of Debug.Assigned cursor values
            public int[] ix = new int[BMS];         // Cursor Debug.Assigned to each bit

            public void memcpy(WhereMaskSet wms)
            {
                wms.n = n;
                wms.ix = (int[])ix.Clone();
            }
        }

        public class WhereCost
        {
            public WherePlan Plan = new WherePlan();    // The lookup strategy
            public double Cost;                         // Overall cost of pursuing this search strategy
            public Bitmask Used;                        // Bitmask of cursors used by this plan

            public void Clear()
            {
                Plan.Clear();
                Cost = 0;
                Used = 0;
            }
        }

        public enum WHERE
        {
            ROWID_EQ = 0x00001000,		// rowid=EXPR or rowid IN (...)
            ROWID_RANGE = 0x00002000,		// rowid<EXPR and/or rowid>EXPR
            COLUMN_EQ = 0x00010000,		// x=EXPR or x IN (...) or x IS NULL
            COLUMN_RANGE = 0x00020000,	// x<EXPR and/or x>EXPR
            COLUMN_IN = 0x00040000,		// x IN (...)
            COLUMN_NULL = 0x00080000,		// x IS NULL
            INDEXED = 0x000f0000,			// Anything that uses an index
            NOT_FULLSCAN = 0x100f3000,	// Does not do a full table scan
            IN_ABLE = 0x080f1000,			// Able to support an IN operator
            TOP_LIMIT = 0x00100000,		// x<EXPR or x<=EXPR constraint
            BTM_LIMIT = 0x00200000,		// x>EXPR or x>=EXPR constraint
            BOTH_LIMIT = 0x00300000,		// Both x>EXPR and x<EXPR
            IDX_ONLY = 0x00400000,		// Use index only - omit table
            ORDERED = 0x00800000,			// Output will appear in correct order
            REVERSE = 0x01000000,			// Scan in reverse order */
            UNIQUE = 0x02000000,			// Selects no more than one row
            ALL_UNIQUE = 0x04000000,		// This and all prior have one row
            OB_UNIQUE = 0x00004000,		// Values in ORDER BY columns are 
            // different for every output row
            VIRTUALTABLE = 0x08000000,	// Use virtual-table processing
            MULTI_OR = 0x10000000,		// OR using multiple indices
            TEMP_INDEX = 0x20000000,		// Uses an ephemeral index
            DISTINCT = 0x40000000,		// Correct order for DISTINCT
            COVER_SCAN = 0x80000000,		// Full scan of a covering index
        }

        public struct WhereBestIdx
        {
            public Parse Parse;					// Parser context
            public WhereClause WC;				// The WHERE clause
            public SrcList.SrcListItem Src;      // The FROM clause term to search
            public Bitmask NotReady;               // Mask of cursors not available
            public Bitmask NotValid;               // Cursors not available for any purpose
            public ExprList OrderBy;				// The ORDER BY clause
            public ExprList Distinct;				// The select-list if query is DISTINCT
            public IIndexInfo[] IdxInfo;	// Index information passed to xBestIndex
            public int i;							// Which loop is being coded
            public int n;							// # of loops
            public WhereLevel Levels;             // Info about outer loops
            public WhereCost Cost;					// Lowest cost query plan
        }

        static bool CompareCost(WhereCost probe, WhereCost baseline)
        {
            if (probe.Cost < baseline.Cost) return true;
            if (probe.Cost > baseline.Cost) return false;
            if (probe.Plan.OBSats > baseline.Plan.OBSats) return true;
            if (probe.Plan.Rows < baseline.Plan.Rows) return true;
            return false;
        }

        static void WhereClauseInit(WhereClause wc, Parse parse, WhereMaskSet maskSet, ushort wctrlFlags)
        {
            wc.Parse = parse;
            wc.MaskSet = maskSet;
            wc.Outer = null;
            wc.Terms = 0;
            wc.Slots.length = wc.Statics.Length - 1;
            wc.Slots.data = wc.Statics;
            wc.WctrlFlags = wctrlFlags;
        }

        static void WhereOrInfoDelete(Context ctx, WhereOrInfo p)
        {
            WhereClauseClear(p.WC);
            SysEx.TagFree(ctx, ref p);
        }

        static void WhereAndInfoDelete(Context ctx, WhereAndInfo p)
        {
            WhereClauseClear(p.WC);
            SysEx.TagFree(ctx, ref p);
        }

        static void WhereClauseClear(WhereClause wc)
        {
            Context ctx = wc.Parse.Ctx;
            int i;
            WhereTerm a;
            for (i = wc.Terms - 1; i >= 0; i--)
            {
                a = wc.Slots[i];
                if ((a.WtFlags & TERM.DYNAMIC) != 0)
                    Expr.Delete(ctx, ref a.Expr);
                if ((a.WtFlags & TERM.ORINFO) != 0)
                    WhereOrInfoDelete(ctx, a.u.OrInfo);
                else if ((a.WtFlags & TERM.ANDINFO) != 0)
                    WhereAndInfoDelete(ctx, a.u.AndInfo);
            }
            if (wc.Slots.data != wc.Statics)
                SysEx.TagFree(ctx, ref wc.Slots.data);
        }

        static int WhereClauseInsert(WhereClause wc, Expr p, TERM wtFlags)
        {
            SysEx.ASSERTCOVERAGE((wtFlags & TERM.VIRTUAL) != 0);  // EV: R-00211-15100
            if (wc.Terms >= wc.Slots.length)
            {
                Array.Resize(ref wc.Slots.data, wc.Slots.length * 2);
                wc.Slots.length = wc.Slots.data.Length - 1;
            }
            int idx;
            wc.Slots[idx = wc.Terms++] = new WhereTerm();
            WhereTerm term = wc.Slots[idx];
            term.Expr = p;
            term.WtFlags = wtFlags;
            term.WC = wc;
            term.Parent = -1;
            return idx;
        }

        static void WhereSplit(WhereClause wc, Expr expr, TK op)
        {
            wc.OP = (byte)op;
            if (expr == null) return;
            if (expr.OP != op)
                WhereClauseInsert(wc, expr, 0);
            else
            {
                WhereSplit(wc, expr.Left, op);
                WhereSplit(wc, expr.Right, op);
            }
        }

        static Bitmask GetMask(WhereMaskSet maskSet, int cursor)
        {
            Debug.Assert(maskSet.n <= (int)sizeof(Bitmask) * 8);
            for (int i = 0; i < maskSet.n; i++)
                if (maskSet.ix[i] == cursor)
                    return ((Bitmask)1) << i;
            return 0;
        }

        static void CreateMask(WhereMaskSet maskSet, int cursor)
        {
            Debug.Assert(maskSet.n < maskSet.ix.Length);
            maskSet.ix[maskSet.n++] = cursor;
        }

        static Bitmask ExprTableUsage(WhereMaskSet maskSet, Expr p)
        {
            Bitmask mask = 0;
            if (p == null) return 0;
            if (p.OP == TK.COLUMN)
            {
                mask = GetMask(maskSet, p.TableIdx);
                return mask;
            }
            mask = ExprTableUsage(maskSet, p.Right);
            mask |= ExprTableUsage(maskSet, p.Left);
            if (ExprHasProperty(p, EP.xIsSelect))
                mask |= ExprSelectTableUsage(maskSet, p.x.Select);
            else
                mask |= ExprListTableUsage(maskSet, p.x.List);
            return mask;
        }
        static Bitmask ExprListTableUsage(WhereMaskSet pMaskSet, ExprList list)
        {
            Bitmask mask = 0;
            if (list != null)
                for (int i = 0; i < list.Exprs; i++)
                    mask |= ExprTableUsage(pMaskSet, list.Ids[i].Expr);
            return mask;
        }
        static Bitmask ExprSelectTableUsage(WhereMaskSet maskSet, Select s)
        {
            Bitmask mask = 0;
            while (s != null)
            {
                mask |= ExprListTableUsage(maskSet, s.EList);
                mask |= ExprListTableUsage(maskSet, s.GroupBy);
                mask |= ExprListTableUsage(maskSet, s.OrderBy);
                mask |= ExprTableUsage(maskSet, s.Where);
                mask |= ExprTableUsage(maskSet, s.Having);
                SrcList src = s.Src;
                if (SysEx.ALWAYS(src != null))
                {
                    for (int i = 0; i < src.Srcs; i++)
                    {
                        mask |= ExprSelectTableUsage(maskSet, src.Ids[i].Select);
                        mask |= ExprTableUsage(maskSet, src.Ids[i].On);
                    }
                }
                s = s.Prior;
            }
            return mask;
        }

        static bool AllowedOp(int op)
        {
            Debug.Assert(TK.GT > TK.EQ && TK.GT < TK.GE);
            Debug.Assert(TK.LT > TK.EQ && TK.LT < TK.GE);
            Debug.Assert(TK.LE > TK.EQ && TK.LE < TK.GE);
            Debug.Assert(TK.GE == TK.EQ + 4);
            return (op == TK.IN || (op >= TK,EQ && op <= TK.GE) || op == TK.ISNULL);
        }

        static void ExprCommute(Parse parse, Expr expr)
        {
            ushort expRight = (ushort)(expr.Right.Flags & EP.ExpCollate);
            ushort expLeft = (ushort)(expr.Left.Flags & EP.ExpCollate);
            Debug.Assert(AllowedOp(expr.OP) && expr.OP != TK.IN);
            if (expRight == expLeft)
            {
                // Either X and Y both have COLLATE operator or neither do
                if (expRight != null)
                    expr.Right.Flags &= ~EP.Collate; // Both X and Y have COLLATE operators.  Make sure X is always used by clearing the EP_Collate flag from Y.
                else if (parse.ExprCollSeq(expr.Left))
                    expr.Left.Flags |= EP_Collate; // Neither X nor Y have COLLATE operators, but X has a non-default collating sequence.  So add the EP_Collate marker on X to cause it to be searched first.
            }
            SWAP(ref expr.Right, ref expr.Left);
            if (expr.OP >= TK.GT)
            {
                Debug.Assert(TK.LT == TK.GT + 2);
                Debug.Assert(TK.GE == TK.LE + 2);
                Debug.Assert(TK.GT > TK.EQ);
                Debug.Assert(TK.GT < TK.LE);
                Debug.Assert(expr.OP >= TK.GT && expr.OP <= TK.GE);
                expr.OP = (byte)(((expr.OP - TK.GT) ^ 2) + TK.GT);
            }
        }

        static WO OperatorMask(int op)
        {
            Debug.Assert(AllowedOp(op));
            WO c;
            if (op == TK.IN)
                c = WO.IN;
            else if (op == TK.ISNULL)
                c = WO.ISNULL;
            else
            {
                Debug.Assert((WO.EQ << (op - TK.EQ)) < 0x7fff);
                c = (WO)(WO.EQ << (op - TK.EQ));
            }
            Debug.Assert(op != TK.ISNULL || c == WO.ISNULL);
            Debug.Assert(op != TK.IN || c == WO.IN);
            Debug.Assert(op != TK.EQ || c == WO.EQ);
            Debug.Assert(op != TK.LT || c == WO.LT);
            Debug.Assert(op != TK.LE || c == WO.LE);
            Debug.Assert(op != TK.GT || c == WO.GT);
            Debug.Assert(op != TK.GE || c == WO.GE);
            return c;
        }

        static WhereTerm FindTerm(WhereClause wc, int cursor, int column, Bitmask notReady, uint op, Index idx)
        {
            WhereTerm result = null; // The answer to return
            int origColumn = column;  // Original value of iColumn
            int equivLength = 2; // Number of entires in aEquiv[]
            int equivId = 2; // Number of entries of aEquiv[] processed so far
            int[] equivs = new int[22]; // iCur,iColumn and up to 10 other equivalents

            Debug.Assert(cursor >= 0);
            equivs[0] = cursor;
            equivs[1] = column;
            WhereClause origWC = wc;  // Original pWC value
            WhereTerm term;            // Term being examined as possible result
            int j, k;
            for (; ; )
            {
                for (wc = origWC; wc; wc = wc.Outer)
                {
                    for (k = wc.Terms; k; k--)
                    {
                        term = wc.Slots[wc.Terms - k];
                        if (term.LeftCursor == cursor && term.u.LeftColumn == column)
                        {
                            if ((term.PrereqRight & notReady) == 0 && (term.EOperator & op & WO.ALL) != 0)
                            {
                                if (origColumn >= 0 && idx != null && (term.EOperator & WO.ISNULL) == 0)
                                {
                                    Expr x = term.Expr;
                                    Parse parse = wc.Parse;
                                    AFF idxaff = idx.Table.Cols[origColumn].Affinity;
                                    if (!x.ValidIndexAffinity(idxaff))
                                        continue;
                                    // Figure out the collation sequence required from an index for it to be useful for optimising expression pX. Store this
                                    // value in variable pColl.
                                    Debug.Assert(x.Left != null);
                                    CollSeq coll = sqlite3BinaryCompareCollSeq(parse, x.Left, x.Right);
                                    if (coll == null) coll = parse.Ctx.DefaultColl;
                                    for (j = 0; idx.Columns[j] != origColumn; j++)
                                        if (SysEx.NEVER(j >= idx.Columns.length)) return 0;
                                    if (!string.Compare(coll.Name, idx.CollNames[j], StringComparison.OrdinalIgnoreCase))
                                        continue;
                                }
                                if (term.PrereqRight == 0)
                                {
                                    result = term;
                                    goto findTerm_success;
                                }
                                else if (result == null)
                                    result = term;
                            }
                            if ((term.EOperator & WO.EQUIV) != 0 && equivsLength < equivs.Length)
                            {
                                Expr x = Expr.SkipCollate(term.Expr->Right);
                                Debug.Assert(x.OP == TK_COLUMN);
                                for (j = 0; j < equivsLength; j += 2)
                                    if (equivs[j] == x.TableIdx && equivs[j + 1] == x.ColumnIdx) break;
                                if (j == equivsLength)
                                {
                                    equivs[j] = x.TableIdx;
                                    equivs[j + 1] = x.ColumnIdx;
                                    equivsLength += 2;
                                }
                            }
                        }
                    }
                }
                if (equivId >= equivsLength) break;
                cursor = equivs[equivId++];
                column = equivs[equivId++];
            }
        findTerm_success:
            return result;
        }

        static void ExprAnalyzeAll(SrcList tabList, WhereClause wc)
        {
            for (int i = wc.Terms - 1; i >= 0; i--)
                ExprAnalyze(tabList, wc, i);
        }

#if  !OMIT_LIKE_OPTIMIZATION

        static int isLikeOrGlob(
        Parse pParse,         /* Parsing and code generating context */
        Expr pExpr,           /* Test this expression */
        ref Expr ppPrefix,    /* Pointer to TK_STRING expression with pattern prefix */
        ref bool pisComplete, /* True if the only wildcard is % in the last character */
        ref bool pnoCase      /* True if uppercase is equivalent to lowercase */
        )
        {
            string z = null;            /* String on RHS of LIKE operator */
            Expr pRight, pLeft;        /* Right and left size of LIKE operator */
            ExprList pList;            /* List of operands to the LIKE operator */
            int c = 0;                 /* One character in z[] */
            int cnt;                   /* Number of non-wildcard prefix characters */
            char[] wc = new char[3];   /* Wildcard characters */
            sqlite3 db = pParse.db;    /* Data_base connection */
            sqlite3_value pVal = null;
            int op;                    /* Opcode of pRight */

            if (!sqlite3IsLikeFunction(db, pExpr, ref pnoCase, wc))
            {
                return 0;
            }
            //#if SQLITE_EBCDIC
            //if( pnoCase ) return 0;
            //#endif
            pList = pExpr.x.pList;
            pLeft = pList.a[1].pExpr;
            if (pLeft.op != TK_COLUMN || sqlite3ExprAffinity(pLeft) != SQLITE_AFF_TEXT)
            {
                /* IMP: R-02065-49465 The left-hand side of the LIKE or GLOB operator must
                ** be the name of an indexed column with TEXT affinity. */
                return 0;
            }
            Debug.Assert(pLeft.iColumn != (-1)); /* Because IPK never has AFF_TEXT */

            pRight = pList.a[0].pExpr;
            op = pRight.op;
            if (op == TK_REGISTER)
            {
                op = pRight.op2;
            }
            if (op == TK_VARIABLE)
            {
                Vdbe pReprepare = pParse.pReprepare;
                int iCol = pRight.iColumn;
                pVal = sqlite3VdbeGetValue(pReprepare, iCol, (byte)SQLITE_AFF_NONE);
                if (pVal != null && sqlite3_value_type(pVal) == SQLITE_TEXT)
                {
                    z = sqlite3_value_text(pVal);
                }
                sqlite3VdbeSetVarmask(pParse.pVdbe, iCol); /* IMP: R-23257-02778 */
                Debug.Assert(pRight.op == TK_VARIABLE || pRight.op == TK_REGISTER);
            }
            else if (op == TK_STRING)
            {
                z = pRight.u.zToken;
            }
            if (!String.IsNullOrEmpty(z))
            {
                cnt = 0;
                while (cnt < z.Length && (c = z[cnt]) != 0 && c != wc[0] && c != wc[1] && c != wc[2])
                {
                    cnt++;
                }
                if (cnt != 0 && 255 != (u8)z[cnt - 1])
                {
                    Expr pPrefix;
                    pisComplete = c == wc[0] && cnt == z.Length - 1;
                    pPrefix = sqlite3Expr(db, TK_STRING, z);
                    if (pPrefix != null)
                        pPrefix.u.zToken = pPrefix.u.zToken.Substring(0, cnt);
                    ppPrefix = pPrefix;
                    if (op == TK_VARIABLE)
                    {
                        Vdbe v = pParse.pVdbe;
                        sqlite3VdbeSetVarmask(v, pRight.iColumn); /* IMP: R-23257-02778 */
                        if (pisComplete && pRight.u.zToken.Length > 1)
                        {
                            /* If the rhs of the LIKE expression is a variable, and the current
                            ** value of the variable means there is no need to invoke the LIKE
                            ** function, then no OP_Variable will be added to the program.
                            ** This causes problems for the sqlite3_bind_parameter_name()
                            ** API. To workaround them, add a dummy OP_Variable here.
                            */
                            int r1 = sqlite3GetTempReg(pParse);
                            sqlite3ExprCodeTarget(pParse, pRight, r1);
                            sqlite3VdbeChangeP3(v, sqlite3VdbeCurrentAddr(v) - 1, 0);
                            sqlite3ReleaseTempReg(pParse, r1);
                        }
                    }
                }
                else
                {
                    z = null;
                }
            }

            sqlite3ValueFree(ref pVal);
            return (z != null) ? 1 : 0;
        }
#endif //* SQLITE_OMIT_LIKE_OPTIMIZATION */


#if  !SQLITE_OMIT_VIRTUALTABLE
        /*
** Check to see if the given expression is of the form
**
**         column MATCH expr
**
** If it is then return TRUE.  If not, return FALSE.
*/
        static int isMatchOfColumn(
        Expr pExpr      /* Test this expression */
        )
        {
            ExprList pList;

            if (pExpr.op != TK_FUNCTION)
            {
                return 0;
            }
            if (!pExpr.u.zToken.Equals("match", StringComparison.OrdinalIgnoreCase))
            {
                return 0;
            }
            pList = pExpr.x.pList;
            if (pList.nExpr != 2)
            {
                return 0;
            }
            if (pList.a[1].pExpr.op != TK_COLUMN)
            {
                return 0;
            }
            return 1;
        }
#endif //* SQLITE_OMIT_VIRTUALTABLE */

        /*
** If the pBase expression originated in the ON or USING clause of
** a join, then transfer the appropriate markings over to derived.
*/
        static void transferJoinMarkings(Expr pDerived, Expr pBase)
        {
            pDerived.flags = (u16)(pDerived.flags | pBase.flags & EP_FromJoin);
            pDerived.iRightJoinTable = pBase.iRightJoinTable;
        }

#if  !(SQLITE_OMIT_OR_OPTIMIZATION) && !(SQLITE_OMIT_SUBQUERY)
        /*
** Analyze a term that consists of two or more OR-connected
** subterms.  So in:
**
**     ... WHERE  (a=5) AND (b=7 OR c=9 OR d=13) AND (d=13)
**                          ^^^^^^^^^^^^^^^^^^^^
**
** This routine analyzes terms such as the middle term in the above example.
** A WhereOrTerm object is computed and attached to the term under
** analysis, regardless of the outcome of the analysis.  Hence:
**
**     WhereTerm.wtFlags   |=  TERM_ORINFO
**     WhereTerm.u.pOrInfo  =  a dynamically allocated WhereOrTerm object
**
** The term being analyzed must have two or more of OR-connected subterms.
** A single subterm might be a set of AND-connected sub-subterms.
** Examples of terms under analysis:
**
**     (A)     t1.x=t2.y OR t1.x=t2.z OR t1.y=15 OR t1.z=t3.a+5
**     (B)     x=expr1 OR expr2=x OR x=expr3
**     (C)     t1.x=t2.y OR (t1.x=t2.z AND t1.y=15)
**     (D)     x=expr1 OR (y>11 AND y<22 AND z LIKE '*hello*')
**     (E)     (p.a=1 AND q.b=2 AND r.c=3) OR (p.x=4 AND q.y=5 AND r.z=6)
**
** CASE 1:
**
** If all subterms are of the form T.C=expr for some single column of C
** a single table T (as shown in example B above) then create a new virtual
** term that is an equivalent IN expression.  In other words, if the term
** being analyzed is:
**
**      x = expr1  OR  expr2 = x  OR  x = expr3
**
** then create a new virtual term like this:
**
**      x IN (expr1,expr2,expr3)
**
** CASE 2:
**
** If all subterms are indexable by a single table T, then set
**
**     WhereTerm.eOperator              =  WO_OR
**     WhereTerm.u.pOrInfo.indexable  |=  the cursor number for table T
**
** A subterm is "indexable" if it is of the form
** "T.C <op> <expr>" where C is any column of table T and
** <op> is one of "=", "<", "<=", ">", ">=", "IS NULL", or "IN".
** A subterm is also indexable if it is an AND of two or more
** subsubterms at least one of which is indexable.  Indexable AND
** subterms have their eOperator set to WO_AND and they have
** u.pAndInfo set to a dynamically allocated WhereAndTerm object.
**
** From another point of view, "indexable" means that the subterm could
** potentially be used with an index if an appropriate index exists.
** This analysis does not consider whether or not the index exists; that
** is something the bestIndex() routine will determine.  This analysis
** only looks at whether subterms appropriate for indexing exist.
**
** All examples A through E above all satisfy case 2.  But if a term
** also statisfies case 1 (such as B) we know that the optimizer will
** always prefer case 1, so in that case we pretend that case 2 is not
** satisfied.
**
** It might be the case that multiple tables are indexable.  For example,
** (E) above is indexable on tables P, Q, and R.
**
** Terms that satisfy case 2 are candidates for lookup by using
** separate indices to find rowids for each subterm and composing
** the union of all rowids using a RowSet object.  This is similar
** to "bitmap indices" in other data_base engines.
**
** OTHERWISE:
**
** If neither case 1 nor case 2 apply, then leave the eOperator set to
** zero.  This term is not useful for search.
*/
        static void exprAnalyzeOrTerm(
        SrcList pSrc,            /* the FROM clause */
        WhereClause pWC,         /* the complete WHERE clause */
        int idxTerm               /* Index of the OR-term to be analyzed */
        )
        {
            Parse pParse = pWC.pParse;            /* Parser context */
            sqlite3 db = pParse.db;               /* Data_base connection */
            WhereTerm pTerm = pWC.a[idxTerm];    /* The term to be analyzed */
            Expr pExpr = pTerm.pExpr;             /* The expression of the term */
            WhereMaskSet pMaskSet = pWC.pMaskSet; /* Table use masks */
            int i;                                  /* Loop counters */
            WhereClause pOrWc;        /* Breakup of pTerm into subterms */
            WhereTerm pOrTerm;        /* A Sub-term within the pOrWc */
            WhereOrInfo pOrInfo;      /* Additional information Debug.Associated with pTerm */
            Bitmask chngToIN;         /* Tables that might satisfy case 1 */
            Bitmask indexable;        /* Tables that are indexable, satisfying case 2 */

            /*
            ** Break the OR clause into its separate subterms.  The subterms are
            ** stored in a WhereClause structure containing within the WhereOrInfo
            ** object that is attached to the original OR clause term.
            */
            Debug.Assert((pTerm.wtFlags & (TERM_DYNAMIC | TERM_ORINFO | TERM_ANDINFO)) == 0);
            Debug.Assert(pExpr.op == TK_OR);
            pTerm.u.pOrInfo = pOrInfo = new WhereOrInfo();//sqlite3DbMallocZero(db, sizeof(*pOrInfo));
            if (pOrInfo == null)
                return;
            pTerm.wtFlags |= TERM_ORINFO;
            pOrWc = pOrInfo.wc;
            whereClauseInit(pOrWc, pWC.pParse, pMaskSet);
            whereSplit(pOrWc, pExpr, TK_OR);
            exprAnalyzeAll(pSrc, pOrWc);
            //      if ( db.mallocFailed != 0 ) return;
            Debug.Assert(pOrWc.nTerm >= 2);

            /*
            ** Compute the set of tables that might satisfy cases 1 or 2.
            */
            indexable = ~(Bitmask)0;
            chngToIN = ~(pWC.vmask);
            for (i = pOrWc.nTerm - 1; i >= 0 && indexable != 0; i--)//, pOrTerm++ )
            {
                pOrTerm = pOrWc.a[i];
                if ((pOrTerm.eOperator & WO_SINGLE) == 0)
                {
                    WhereAndInfo pAndInfo;
                    Debug.Assert(pOrTerm.eOperator == 0);
                    Debug.Assert((pOrTerm.wtFlags & (TERM_ANDINFO | TERM_ORINFO)) == 0);
                    chngToIN = 0;
                    pAndInfo = new WhereAndInfo();//sqlite3DbMallocRaw(db, sizeof(*pAndInfo));
                    if (pAndInfo != null)
                    {
                        WhereClause pAndWC;
                        WhereTerm pAndTerm;
                        int j;
                        Bitmask b = 0;
                        pOrTerm.u.pAndInfo = pAndInfo;
                        pOrTerm.wtFlags |= TERM_ANDINFO;
                        pOrTerm.eOperator = WO_AND;
                        pAndWC = pAndInfo.wc;
                        whereClauseInit(pAndWC, pWC.pParse, pMaskSet);
                        whereSplit(pAndWC, pOrTerm.pExpr, TK_AND);
                        exprAnalyzeAll(pSrc, pAndWC);
                        //testcase( db.mallocFailed );
                        ////if ( 0 == db.mallocFailed )
                        {
                            for (j = 0; j < pAndWC.nTerm; j++)//, pAndTerm++ )
                            {
                                pAndTerm = pAndWC.a[j];
                                Debug.Assert(pAndTerm.pExpr != null);
                                if (allowedOp(pAndTerm.pExpr.op))
                                {
                                    b |= getMask(pMaskSet, pAndTerm.leftCursor);
                                }
                            }
                        }
                        indexable &= b;
                    }
                }
                else if ((pOrTerm.wtFlags & TERM_COPIED) != 0)
                {
                    /* Skip this term for now.  We revisit it when we process the
                    ** corresponding TERM_VIRTUAL term */
                }
                else
                {
                    Bitmask b;
                    b = getMask(pMaskSet, pOrTerm.leftCursor);
                    if ((pOrTerm.wtFlags & TERM_VIRTUAL) != 0)
                    {
                        WhereTerm pOther = pOrWc.a[pOrTerm.iParent];
                        b |= getMask(pMaskSet, pOther.leftCursor);
                    }
                    indexable &= b;
                    if (pOrTerm.eOperator != WO_EQ)
                    {
                        chngToIN = 0;
                    }
                    else
                    {
                        chngToIN &= b;
                    }
                }
            }

            /*
            ** Record the set of tables that satisfy case 2.  The set might be
            ** empty.
            */
            pOrInfo.indexable = indexable;
            pTerm.eOperator = (u16)(indexable == 0 ? 0 : WO_OR);

            /*
            ** chngToIN holds a set of tables that *might* satisfy case 1.  But
            ** we have to do some additional checking to see if case 1 really
            ** is satisfied.
            **
            ** chngToIN will hold either 0, 1, or 2 bits.  The 0-bit case means
            ** that there is no possibility of transforming the OR clause into an
            ** IN operator because one or more terms in the OR clause contain
            ** something other than == on a column in the single table.  The 1-bit
            ** case means that every term of the OR clause is of the form
            ** "table.column=expr" for some single table.  The one bit that is set
            ** will correspond to the common table.  We still need to check to make
            ** sure the same column is used on all terms.  The 2-bit case is when
            ** the all terms are of the form "table1.column=table2.column".  It
            ** might be possible to form an IN operator with either table1.column
            ** or table2.column as the LHS if either is common to every term of
            ** the OR clause.
            **
            ** Note that terms of the form "table.column1=table.column2" (the
            ** same table on both sizes of the ==) cannot be optimized.
            */
            if (chngToIN != 0)
            {
                int okToChngToIN = 0;     /* True if the conversion to IN is valid */
                int iColumn = -1;         /* Column index on lhs of IN operator */
                int iCursor = -1;         /* Table cursor common to all terms */
                int j = 0;                /* Loop counter */

                /* Search for a table and column that appears on one side or the
                ** other of the == operator in every subterm.  That table and column
                ** will be recorded in iCursor and iColumn.  There might not be any
                ** such table and column.  Set okToChngToIN if an appropriate table
                ** and column is found but leave okToChngToIN false if not found.
                */
                for (j = 0; j < 2 && 0 == okToChngToIN; j++)
                {
                    //pOrTerm = pOrWc.a;
                    for (i = pOrWc.nTerm - 1; i >= 0; i--)//, pOrTerm++)
                    {
                        pOrTerm = pOrWc.a[pOrWc.nTerm - 1 - i];
                        Debug.Assert(pOrTerm.eOperator == WO_EQ);
                        pOrTerm.wtFlags = (u8)(pOrTerm.wtFlags & ~TERM_OR_OK);
                        if (pOrTerm.leftCursor == iCursor)
                        {
                            /* This is the 2-bit case and we are on the second iteration and
                            ** current term is from the first iteration.  So skip this term. */
                            Debug.Assert(j == 1);
                            continue;
                        }
                        if ((chngToIN & getMask(pMaskSet, pOrTerm.leftCursor)) == 0)
                        {
                            /* This term must be of the form t1.a==t2.b where t2 is in the
                            ** chngToIN set but t1 is not.  This term will be either preceeded
                            ** or follwed by an inverted copy (t2.b==t1.a).  Skip this term
                            ** and use its inversion. */
                            testcase(pOrTerm.wtFlags & TERM_COPIED);
                            testcase(pOrTerm.wtFlags & TERM_VIRTUAL);
                            Debug.Assert((pOrTerm.wtFlags & (TERM_COPIED | TERM_VIRTUAL)) != 0);
                            continue;
                        }
                        iColumn = pOrTerm.u.leftColumn;
                        iCursor = pOrTerm.leftCursor;
                        break;
                    }
                    if (i < 0)
                    {
                        /* No candidate table+column was found.  This can only occur
                        ** on the second iteration */
                        Debug.Assert(j == 1);
                        Debug.Assert((chngToIN & (chngToIN - 1)) == 0);
                        Debug.Assert(chngToIN == getMask(pMaskSet, iCursor));
                        break;
                    }
                    testcase(j == 1);

                    /* We have found a candidate table and column.  Check to see if that
                    ** table and column is common to every term in the OR clause */
                    okToChngToIN = 1;
                    for (; i >= 0 && okToChngToIN != 0; i--)//, pOrTerm++)
                    {
                        pOrTerm = pOrWc.a[pOrWc.nTerm - 1 - i];
                        Debug.Assert(pOrTerm.eOperator == WO_EQ);
                        if (pOrTerm.leftCursor != iCursor)
                        {
                            pOrTerm.wtFlags = (u8)(pOrTerm.wtFlags & ~TERM_OR_OK);
                        }
                        else if (pOrTerm.u.leftColumn != iColumn)
                        {
                            okToChngToIN = 0;
                        }
                        else
                        {
                            int affLeft, affRight;
                            /* If the right-hand side is also a column, then the affinities
                            ** of both right and left sides must be such that no type
                            ** conversions are required on the right.  (Ticket #2249)
                            */
                            affRight = sqlite3ExprAffinity(pOrTerm.pExpr.pRight);
                            affLeft = sqlite3ExprAffinity(pOrTerm.pExpr.pLeft);
                            if (affRight != 0 && affRight != affLeft)
                            {
                                okToChngToIN = 0;
                            }
                            else
                            {
                                pOrTerm.wtFlags |= TERM_OR_OK;
                            }
                        }
                    }
                }

                /* At this point, okToChngToIN is true if original pTerm satisfies
                ** case 1.  In that case, construct a new virtual term that is
                ** pTerm converted into an IN operator.
                **
                ** EV: R-00211-15100
                */
                if (okToChngToIN != 0)
                {
                    Expr pDup;            /* A transient duplicate expression */
                    ExprList pList = null;   /* The RHS of the IN operator */
                    Expr pLeft = null;       /* The LHS of the IN operator */
                    Expr pNew;            /* The complete IN operator */

                    for (i = pOrWc.nTerm - 1; i >= 0; i--)//, pOrTerm++)
                    {
                        pOrTerm = pOrWc.a[pOrWc.nTerm - 1 - i];
                        if ((pOrTerm.wtFlags & TERM_OR_OK) == 0)
                            continue;
                        Debug.Assert(pOrTerm.eOperator == WO_EQ);
                        Debug.Assert(pOrTerm.leftCursor == iCursor);
                        Debug.Assert(pOrTerm.u.leftColumn == iColumn);
                        pDup = sqlite3ExprDup(db, pOrTerm.pExpr.pRight, 0);
                        pList = sqlite3ExprListAppend(pWC.pParse, pList, pDup);
                        pLeft = pOrTerm.pExpr.pLeft;
                    }
                    Debug.Assert(pLeft != null);
                    pDup = sqlite3ExprDup(db, pLeft, 0);
                    pNew = sqlite3PExpr(pParse, TK_IN, pDup, null, null);
                    if (pNew != null)
                    {
                        int idxNew;
                        transferJoinMarkings(pNew, pExpr);
                        Debug.Assert(!ExprHasProperty(pNew, EP_xIsSelect));
                        pNew.x.pList = pList;
                        idxNew = whereClauseInsert(pWC, pNew, TERM_VIRTUAL | TERM_DYNAMIC);
                        testcase(idxNew == 0);
                        exprAnalyze(pSrc, pWC, idxNew);
                        pTerm = pWC.a[idxTerm];
                        pWC.a[idxNew].iParent = idxTerm;
                        pTerm.nChild = 1;
                    }
                    else
                    {
                        sqlite3ExprListDelete(db, ref pList);
                    }
                    pTerm.eOperator = WO_NOOP;  /* case 1 trumps case 2 */
                }
            }
        }
#endif //* !SQLITE_OMIT_OR_OPTIMIZATION && !SQLITE_OMIT_SUBQUERY */


        /*
** The input to this routine is an WhereTerm structure with only the
** "pExpr" field filled in.  The job of this routine is to analyze the
** subexpression and populate all the other fields of the WhereTerm
** structure.
**
** If the expression is of the form "<expr> <op> X" it gets commuted
** to the standard form of "X <op> <expr>".
**
** If the expression is of the form "X <op> Y" where both X and Y are
** columns, then the original expression is unchanged and a new virtual
** term of the form "Y <op> X" is added to the WHERE clause and
** analyzed separately.  The original term is marked with TERM_COPIED
** and the new term is marked with TERM_DYNAMIC (because it's pExpr
** needs to be freed with the WhereClause) and TERM_VIRTUAL (because it
** is a commuted copy of a prior term.)  The original term has nChild=1
** and the copy has idxParent set to the index of the original term.
*/
        static void exprAnalyze(
        SrcList pSrc,            /* the FROM clause */
        WhereClause pWC,         /* the WHERE clause */
        int idxTerm               /* Index of the term to be analyzed */
        )
        {
            WhereTerm pTerm;                 /* The term to be analyzed */
            WhereMaskSet pMaskSet;           /* Set of table index masks */
            Expr pExpr;                      /* The expression to be analyzed */
            Bitmask prereqLeft;              /* Prerequesites of the pExpr.pLeft */
            Bitmask prereqAll;               /* Prerequesites of pExpr */
            Bitmask extraRight = 0;           /* Extra dependencies on LEFT JOIN */
            Expr pStr1 = null;               /* RHS of LIKE/GLOB operator */
            bool isComplete = false;         /* RHS of LIKE/GLOB ends with wildcard */
            bool noCase = false;             /* LIKE/GLOB distinguishes case */
            int op;                          /* Top-level operator.  pExpr.op */
            Parse pParse = pWC.pParse;       /* Parsing context */
            sqlite3 db = pParse.db;          /* Data_base connection */

            //if ( db.mallocFailed != 0 )
            //{
            //  return;
            //}
            pTerm = pWC.a[idxTerm];
            pMaskSet = pWC.pMaskSet;
            pExpr = pTerm.pExpr;
            prereqLeft = exprTableUsage(pMaskSet, pExpr.pLeft);
            op = pExpr.op;
            if (op == TK_IN)
            {
                Debug.Assert(pExpr.pRight == null);
                if (ExprHasProperty(pExpr, EP_xIsSelect))
                {
                    pTerm.prereqRight = exprSelectTableUsage(pMaskSet, pExpr.x.pSelect);
                }
                else
                {
                    pTerm.prereqRight = exprListTableUsage(pMaskSet, pExpr.x.pList);
                }
            }
            else if (op == TK_ISNULL)
            {
                pTerm.prereqRight = 0;
            }
            else
            {
                pTerm.prereqRight = exprTableUsage(pMaskSet, pExpr.pRight);
            }
            prereqAll = exprTableUsage(pMaskSet, pExpr);
            if (ExprHasProperty(pExpr, EP_FromJoin))
            {
                Bitmask x = getMask(pMaskSet, pExpr.iRightJoinTable);
                prereqAll |= x;
                extraRight = x - 1;  /* ON clause terms may not be used with an index
** on left table of a LEFT JOIN.  Ticket #3015 */
            }
            pTerm.prereqAll = prereqAll;
            pTerm.leftCursor = -1;
            pTerm.iParent = -1;
            pTerm.eOperator = 0;
            if (allowedOp(op) && (pTerm.prereqRight & prereqLeft) == 0)
            {
                Expr pLeft = pExpr.pLeft;
                Expr pRight = pExpr.pRight;
                if (pLeft.op == TK_COLUMN)
                {
                    pTerm.leftCursor = pLeft.iTable;
                    pTerm.u.leftColumn = pLeft.iColumn;
                    pTerm.eOperator = operatorMask(op);
                }
                if (pRight != null && pRight.op == TK_COLUMN)
                {
                    WhereTerm pNew;
                    Expr pDup;
                    if (pTerm.leftCursor >= 0)
                    {
                        int idxNew;
                        pDup = sqlite3ExprDup(db, pExpr, 0);
                        //if ( db.mallocFailed != 0 )
                        //{
                        //  sqlite3ExprDelete( db, ref pDup );
                        //  return;
                        //}
                        idxNew = whereClauseInsert(pWC, pDup, TERM_VIRTUAL | TERM_DYNAMIC);
                        if (idxNew == 0)
                            return;
                        pNew = pWC.a[idxNew];
                        pNew.iParent = idxTerm;
                        pTerm = pWC.a[idxTerm];
                        pTerm.nChild = 1;
                        pTerm.wtFlags |= TERM_COPIED;
                    }
                    else
                    {
                        pDup = pExpr;
                        pNew = pTerm;
                    }
                    exprCommute(pParse, pDup);
                    pLeft = pDup.pLeft;
                    pNew.leftCursor = pLeft.iTable;
                    pNew.u.leftColumn = pLeft.iColumn;
                    testcase((prereqLeft | extraRight) != prereqLeft);
                    pNew.prereqRight = prereqLeft | extraRight;
                    pNew.prereqAll = prereqAll;
                    pNew.eOperator = operatorMask(pDup.op);
                }
            }

#if  !SQLITE_OMIT_BETWEEN_OPTIMIZATION
            /* If a term is the BETWEEN operator, create two new virtual terms
** that define the range that the BETWEEN implements.  For example:
**
**      a BETWEEN b AND c
**
** is converted into:
**
**      (a BETWEEN b AND c) AND (a>=b) AND (a<=c)
**
** The two new terms are added onto the end of the WhereClause object.
** The new terms are "dynamic" and are children of the original BETWEEN
** term.  That means that if the BETWEEN term is coded, the children are
** skipped.  Or, if the children are satisfied by an index, the original
** BETWEEN term is skipped.
*/
            else if (pExpr.op == TK_BETWEEN && pWC.op == TK_AND)
            {
                ExprList pList = pExpr.x.pList;
                int i;
                u8[] ops = new u8[] { TK_GE, TK_LE };
                Debug.Assert(pList != null);
                Debug.Assert(pList.nExpr == 2);
                for (i = 0; i < 2; i++)
                {
                    Expr pNewExpr;
                    int idxNew;
                    pNewExpr = sqlite3PExpr(pParse, ops[i],
                    sqlite3ExprDup(db, pExpr.pLeft, 0),
                    sqlite3ExprDup(db, pList.a[i].pExpr, 0), null);
                    idxNew = whereClauseInsert(pWC, pNewExpr, TERM_VIRTUAL | TERM_DYNAMIC);
                    testcase(idxNew == 0);
                    exprAnalyze(pSrc, pWC, idxNew);
                    pTerm = pWC.a[idxTerm];
                    pWC.a[idxNew].iParent = idxTerm;
                }
                pTerm.nChild = 2;
            }
#endif //* SQLITE_OMIT_BETWEEN_OPTIMIZATION */

#if  !(SQLITE_OMIT_OR_OPTIMIZATION) && !(SQLITE_OMIT_SUBQUERY)
            /* Analyze a term that is composed of two or more subterms connected by
** an OR operator.
*/
            else if (pExpr.op == TK_OR)
            {
                Debug.Assert(pWC.op == TK_AND);
                exprAnalyzeOrTerm(pSrc, pWC, idxTerm);
                pTerm = pWC.a[idxTerm];
            }
#endif //* SQLITE_OMIT_OR_OPTIMIZATION */

#if  !SQLITE_OMIT_LIKE_OPTIMIZATION
            /* Add constraints to reduce the search space on a LIKE or GLOB
** operator.
**
** A like pattern of the form "x LIKE 'abc%'" is changed into constraints
**
**          x>='abc' AND x<'abd' AND x LIKE 'abc%'
**
** The last character of the prefix "abc" is incremented to form the
** termination condition "abd".
*/
            if (pWC.op == TK_AND
            && isLikeOrGlob(pParse, pExpr, ref pStr1, ref isComplete, ref noCase) != 0
            )
            {
                Expr pLeft;       /* LHS of LIKE/GLOB operator */
                Expr pStr2;       /* Copy of pStr1 - RHS of LIKE/GLOB operator */
                Expr pNewExpr1;
                Expr pNewExpr2;
                int idxNew1;
                int idxNew2;
                CollSeq pColl;    /* Collating sequence to use */

                pLeft = pExpr.x.pList.a[1].pExpr;
                pStr2 = sqlite3ExprDup(db, pStr1, 0);
                ////if ( 0 == db.mallocFailed )
                {
                    int c, pC;    /* Last character before the first wildcard */
                    pC = pStr2.u.zToken[sqlite3Strlen30(pStr2.u.zToken) - 1];
                    c = pC;
                    if (noCase)
                    {
                        /* The point is to increment the last character before the first
                        ** wildcard.  But if we increment '@', that will push it into the
                        ** alphabetic range where case conversions will mess up the
                        ** inequality.  To avoid this, make sure to also run the full
                        ** LIKE on all candidate expressions by clearing the isComplete flag
                        */
                        if (c == 'A' - 1)
                            isComplete = false;   /* EV: R-64339-08207 */
                        c = sqlite3UpperToLower[c];
                    }
                    pStr2.u.zToken = pStr2.u.zToken.Substring(0, sqlite3Strlen30(pStr2.u.zToken) - 1) + (char)(c + 1);// pC = c + 1;
                }
                pColl = sqlite3FindCollSeq(db, SQLITE_UTF8, noCase ? "NOCASE" : "BINARY", 0);
                pNewExpr1 = sqlite3PExpr(pParse, TK_GE,
                sqlite3ExprSetColl(sqlite3ExprDup(db, pLeft, 0), pColl),
                pStr1, 0);
                idxNew1 = whereClauseInsert(pWC, pNewExpr1, TERM_VIRTUAL | TERM_DYNAMIC);
                testcase(idxNew1 == 0);
                exprAnalyze(pSrc, pWC, idxNew1);
                pNewExpr2 = sqlite3PExpr(pParse, TK_LT,
                         sqlite3ExprSetColl(sqlite3ExprDup(db, pLeft, 0), pColl),
                         pStr2, null);
                idxNew2 = whereClauseInsert(pWC, pNewExpr2, TERM_VIRTUAL | TERM_DYNAMIC);
                testcase(idxNew2 == 0);
                exprAnalyze(pSrc, pWC, idxNew2);
                pTerm = pWC.a[idxTerm];
                if (isComplete)
                {
                    pWC.a[idxNew1].iParent = idxTerm;
                    pWC.a[idxNew2].iParent = idxTerm;
                    pTerm.nChild = 2;
                }
            }
#endif //* SQLITE_OMIT_LIKE_OPTIMIZATION */

#if  !SQLITE_OMIT_VIRTUALTABLE
            /* Add a WO_MATCH auxiliary term to the constraint set if the
** current expression is of the form:  column MATCH expr.
** This information is used by the xBestIndex methods of
** virtual tables.  The native query optimizer does not attempt
** to do anything with MATCH functions.
*/
            if (isMatchOfColumn(pExpr) != 0)
            {
                int idxNew;
                Expr pRight, pLeft;
                WhereTerm pNewTerm;
                Bitmask prereqColumn, prereqExpr;

                pRight = pExpr.x.pList.a[0].pExpr;
                pLeft = pExpr.x.pList.a[1].pExpr;
                prereqExpr = exprTableUsage(pMaskSet, pRight);
                prereqColumn = exprTableUsage(pMaskSet, pLeft);
                if ((prereqExpr & prereqColumn) == 0)
                {
                    Expr pNewExpr;
                    pNewExpr = sqlite3PExpr(pParse, TK_MATCH,
                    null, sqlite3ExprDup(db, pRight, 0), null);
                    idxNew = whereClauseInsert(pWC, pNewExpr, TERM_VIRTUAL | TERM_DYNAMIC);
                    testcase(idxNew == 0);
                    pNewTerm = pWC.a[idxNew];
                    pNewTerm.prereqRight = prereqExpr;
                    pNewTerm.leftCursor = pLeft.iTable;
                    pNewTerm.u.leftColumn = pLeft.iColumn;
                    pNewTerm.eOperator = WO_MATCH;
                    pNewTerm.iParent = idxTerm;
                    pTerm = pWC.a[idxTerm];
                    pTerm.nChild = 1;
                    pTerm.wtFlags |= TERM_COPIED;
                    pNewTerm.prereqAll = pTerm.prereqAll;
                }
            }
#endif //* SQLITE_OMIT_VIRTUALTABLE */

#if SQLITE_ENABLE_STAT2
      /* When sqlite_stat2 histogram data is available an operator of the
  ** form "x IS NOT NULL" can sometimes be evaluated more efficiently
  ** as "x>NULL" if x is not an INTEGER PRIMARY KEY.  So construct a
  ** virtual term of that form.
  **
  ** Note that the virtual term must be tagged with TERM_VNULL.  This
  ** TERM_VNULL tag will suppress the not-null check at the beginning
  ** of the loop.  Without the TERM_VNULL flag, the not-null check at
  ** the start of the loop will prevent any results from being returned.
  */
      if ( pExpr.op == TK_NOTNULL
       && pExpr.pLeft.op == TK_COLUMN
       && pExpr.pLeft.iColumn >= 0
      )
      {
        Expr pNewExpr;
        Expr pLeft = pExpr.pLeft;
        int idxNew;
        WhereTerm pNewTerm;

        pNewExpr = sqlite3PExpr( pParse, TK_GT,
                                sqlite3ExprDup( db, pLeft, 0 ),
                                sqlite3PExpr( pParse, TK_NULL, 0, 0, 0 ), 0 );

        idxNew = whereClauseInsert( pWC, pNewExpr,
                                  TERM_VIRTUAL | TERM_DYNAMIC | TERM_VNULL );
        if ( idxNew != 0 )
        {
          pNewTerm = pWC.a[idxNew];
          pNewTerm.prereqRight = 0;
          pNewTerm.leftCursor = pLeft.iTable;
          pNewTerm.u.leftColumn = pLeft.iColumn;
          pNewTerm.eOperator = WO_GT;
          pNewTerm.iParent = idxTerm;
          pTerm = pWC.a[idxTerm];
          pTerm.nChild = 1;
          pTerm.wtFlags |= TERM_COPIED;
          pNewTerm.prereqAll = pTerm.prereqAll;
        }
      }
#endif //* SQLITE_ENABLE_STAT2 */

            /* Prevent ON clause terms of a LEFT JOIN from being used to drive
** an index for tables to the left of the join.
*/
            pTerm.prereqRight |= extraRight;
        }

        /*
        ** Return TRUE if any of the expressions in pList.a[iFirst...] contain
        ** a reference to any table other than the iBase table.
        */
        static bool referencesOtherTables(
        ExprList pList,          /* Search expressions in ths list */
        WhereMaskSet pMaskSet,   /* Mapping from tables to bitmaps */
        int iFirst,               /* Be searching with the iFirst-th expression */
        int iBase                 /* Ignore references to this table */
        )
        {
            Bitmask allowed = ~getMask(pMaskSet, iBase);
            while (iFirst < pList.nExpr)
            {
                if ((exprTableUsage(pMaskSet, pList.a[iFirst++].pExpr) & allowed) != 0)
                {
                    return true;
                }
            }
            return false;
        }


        /*
        ** This routine decides if pIdx can be used to satisfy the ORDER BY
        ** clause.  If it can, it returns 1.  If pIdx cannot satisfy the
        ** ORDER BY clause, this routine returns 0.
        **
        ** pOrderBy is an ORDER BY clause from a SELECT statement.  pTab is the
        ** left-most table in the FROM clause of that same SELECT statement and
        ** the table has a cursor number of "_base".  pIdx is an index on pTab.
        **
        ** nEqCol is the number of columns of pIdx that are used as equality
        ** constraints.  Any of these columns may be missing from the ORDER BY
        ** clause and the match can still be a success.
        **
        ** All terms of the ORDER BY that match against the index must be either
        ** ASC or DESC.  (Terms of the ORDER BY clause past the end of a UNIQUE
        ** index do not need to satisfy this constraint.)  The pbRev value is
        ** set to 1 if the ORDER BY clause is all DESC and it is set to 0 if
        ** the ORDER BY clause is all ASC.
        */
        static bool isSortingIndex(
        Parse pParse,           /* Parsing context */
        WhereMaskSet pMaskSet,  /* Mapping from table cursor numbers to bitmaps */
        Index pIdx,             /* The index we are testing */
        int _base,              /* Cursor number for the table to be sorted */
        ExprList pOrderBy,      /* The ORDER BY clause */
        int nEqCol,             /* Number of index columns with == constraints */
        int wsFlags,            /* Index usages flags */
        ref int pbRev           /* Set to 1 if ORDER BY is DESC */
        )
        {
            int i, j;                       /* Loop counters */
            int sortOrder = 0;              /* XOR of index and ORDER BY sort direction */
            int nTerm;                      /* Number of ORDER BY terms */
            ExprList_item pTerm;            /* A term of the ORDER BY clause */
            sqlite3 db = pParse.db;

            Debug.Assert(pOrderBy != null);
            nTerm = pOrderBy.nExpr;
            Debug.Assert(nTerm > 0);

            /* Argument pIdx must either point to a 'real' named index structure, 
            ** or an index structure allocated on the stack by bestBtreeIndex() to
            ** represent the rowid index that is part of every table.  */
            Debug.Assert(!String.IsNullOrEmpty(pIdx.zName) || (pIdx.nColumn == 1 && pIdx.aiColumn[0] == -1));

            /* Match terms of the ORDER BY clause against columns of
            ** the index.
            **
            ** Note that indices have pIdx.nColumn regular columns plus
            ** one additional column containing the rowid.  The rowid column
            ** of the index is also allowed to match against the ORDER BY
            ** clause.
            */
            for (i = j = 0; j < nTerm && i <= pIdx.nColumn; i++)
            {
                pTerm = pOrderBy.a[j];
                Expr pExpr;        /* The expression of the ORDER BY pTerm */
                CollSeq pColl;     /* The collating sequence of pExpr */
                int termSortOrder; /* Sort order for this term */
                int iColumn;       /* The i-th column of the index.  -1 for rowid */
                int iSortOrder;    /* 1 for DESC, 0 for ASC on the i-th index term */
                string zColl;      /* Name of the collating sequence for i-th index term */

                pExpr = pTerm.pExpr;
                if (pExpr.op != TK_COLUMN || pExpr.iTable != _base)
                {
                    /* Can not use an index sort on anything that is not a column in the
                    ** left-most table of the FROM clause */
                    break;
                }
                pColl = sqlite3ExprCollSeq(pParse, pExpr);
                if (null == pColl)
                {
                    pColl = db.pDfltColl;
                }
                if (!String.IsNullOrEmpty(pIdx.zName) && i < pIdx.nColumn)
                {
                    iColumn = pIdx.aiColumn[i];
                    if (iColumn == pIdx.pTable.iPKey)
                    {
                        iColumn = -1;
                    }
                    iSortOrder = pIdx.aSortOrder[i];
                    zColl = pIdx.azColl[i];
                }
                else
                {
                    iColumn = -1;
                    iSortOrder = 0;
                    zColl = pColl.zName;
                }
                if (pExpr.iColumn != iColumn || !pColl.zName.Equals(zColl, StringComparison.OrdinalIgnoreCase))
                {
                    /* Term j of the ORDER BY clause does not match column i of the index */
                    if (i < nEqCol)
                    {
                        /* If an index column that is constrained by == fails to match an
                        ** ORDER BY term, that is OK.  Just ignore that column of the index
                        */
                        continue;
                    }
                    else if (i == pIdx.nColumn)
                    {
                        /* Index column i is the rowid.  All other terms match. */
                        break;
                    }
                    else
                    {
                        /* If an index column fails to match and is not constrained by ==
                        ** then the index cannot satisfy the ORDER BY constraint.
                        */
                        return false;
                    }
                }
                Debug.Assert(pIdx.aSortOrder != null || iColumn == -1);
                Debug.Assert(pTerm.sortOrder == 0 || pTerm.sortOrder == 1);
                Debug.Assert(iSortOrder == 0 || iSortOrder == 1);
                termSortOrder = iSortOrder ^ pTerm.sortOrder;
                if (i > nEqCol)
                {
                    if (termSortOrder != sortOrder)
                    {
                        /* Indices can only be used if all ORDER BY terms past the
                        ** equality constraints are all either DESC or ASC. */
                        return false;
                    }
                }
                else
                {
                    sortOrder = termSortOrder;
                }
                j++;
                //pTerm++;
                if (iColumn < 0 && !referencesOtherTables(pOrderBy, pMaskSet, j, _base))
                {
                    /* If the indexed column is the primary key and everything matches
                    ** so far and none of the ORDER BY terms to the right reference other
                    ** tables in the join, then we are Debug.Assured that the index can be used
                    ** to sort because the primary key is unique and so none of the other
                    ** columns will make any difference
                    */
                    j = nTerm;
                }
            }

            pbRev = sortOrder != 0 ? 1 : 0;
            if (j >= nTerm)
            {
                /* All terms of the ORDER BY clause are covered by this index so
                ** this index can be used for sorting. */
                return true;
            }
            if (pIdx.onError != OE_None && i == pIdx.nColumn
              && (wsFlags & WHERE_COLUMN_NULL) == 0
              && !referencesOtherTables(pOrderBy, pMaskSet, j, _base))
            {
                /* All terms of this index match some prefix of the ORDER BY clause
                ** and the index is UNIQUE and no terms on the tail of the ORDER BY
                ** clause reference other tables in a join.  If this is all true then
                ** the order by clause is superfluous.  Not that if the matching
                ** condition is IS NULL then the result is not necessarily unique
                ** even on a UNIQUE index, so disallow those cases. */
                return true;
            }
            return false;
        }

        /*
        ** Prepare a crude estimate of the logarithm of the input value.
        ** The results need not be exact.  This is only used for estimating
        ** the total cost of performing operations with O(logN) or O(NlogN)
        ** complexity.  Because N is just a guess, it is no great tragedy if
        ** logN is a little off.
        */
        static double estLog(double N)
        {
            double logN = 1;
            double x = 10;
            while (N > x)
            {
                logN += 1;
                x *= 10;
            }
            return logN;
        }

        /*
        ** Two routines for printing the content of an sqlite3_index_info
        ** structure.  Used for testing and debugging only.  If neither
        ** SQLITE_TEST or SQLITE_DEBUG are defined, then these routines
        ** are no-ops.
        */
#if  !(SQLITE_OMIT_VIRTUALTABLE) && (SQLITE_DEBUG)
static void TRACE_IDX_INPUTS( sqlite3_index_info p )
{
int i;
if ( !sqlite3WhereTrace ) return;
for ( i = 0 ; i < p.nConstraint ; i++ )
{
sqlite3DebugPrintf( "  constraint[%d]: col=%d termid=%d op=%d usabled=%d\n",
i,
p.aConstraint[i].iColumn,
p.aConstraint[i].iTermOffset,
p.aConstraint[i].op,
p.aConstraint[i].usable );
}
for ( i = 0 ; i < p.nOrderBy ; i++ )
{
sqlite3DebugPrintf( "  orderby[%d]: col=%d desc=%d\n",
i,
p.aOrderBy[i].iColumn,
p.aOrderBy[i].desc );
}
}
static void TRACE_IDX_OUTPUTS( sqlite3_index_info p )
{
int i;
if ( !sqlite3WhereTrace ) return;
for ( i = 0 ; i < p.nConstraint ; i++ )
{
sqlite3DebugPrintf( "  usage[%d]: argvIdx=%d omit=%d\n",
i,
p.aConstraintUsage[i].argvIndex,
p.aConstraintUsage[i].omit );
}
sqlite3DebugPrintf( "  idxNum=%d\n", p.idxNum );
sqlite3DebugPrintf( "  idxStr=%s\n", p.idxStr );
sqlite3DebugPrintf( "  orderByConsumed=%d\n", p.orderByConsumed );
sqlite3DebugPrintf( "  estimatedCost=%g\n", p.estimatedCost );
}
#else
        //#define TRACE_IDX_INPUTS(A)
        static void TRACE_IDX_INPUTS(sqlite3_index_info p) { }
        //#define TRACE_IDX_OUTPUTS(A)
        static void TRACE_IDX_OUTPUTS(sqlite3_index_info p) { }
#endif

        /*
** Required because bestIndex() is called by bestOrClauseIndex()
*/
        //static void bestIndex(
        //Parse*, WhereClause*, struct SrcList_item*, 
        //Bitmask, ExprList*, WhereCost);

        /*
        ** This routine attempts to find an scanning strategy that can be used
        ** to optimize an 'OR' expression that is part of a WHERE clause.
        **
        ** The table associated with FROM clause term pSrc may be either a
        ** regular B-Tree table or a virtual table.
        */
        static void bestOrClauseIndex(
        Parse pParse,               /* The parsing context */
        WhereClause pWC,            /* The WHERE clause */
        SrcList_item pSrc,          /* The FROM clause term to search */
        Bitmask notReady,           /* Mask of cursors not available for indexing */
        Bitmask notValid,           /* Cursors not available for any purpose */
        ExprList pOrderBy,          /* The ORDER BY clause */
        WhereCost pCost             /* Lowest cost query plan */
        )
        {
#if !SQLITE_OMIT_OR_OPTIMIZATION
            int iCur = pSrc.iCursor;   /* The cursor of the table to be accessed */
            Bitmask maskSrc = getMask(pWC.pMaskSet, iCur);  /* Bitmask for pSrc */
            WhereTerm pWCEnd = pWC.a[pWC.nTerm];        /* End of pWC.a[] */
            WhereTerm pTerm;                 /* A single term of the WHERE clause */

            /* No OR-clause optimization allowed if the INDEXED BY or NOT INDEXED clauses
            ** are used */
            if (pSrc.notIndexed != 0 || pSrc.pIndex != null)
            {
                return;
            }

            /* Search the WHERE clause terms for a usable WO_OR term. */
            for (int _pt = 0; _pt < pWC.nTerm; _pt++)//<pWCEnd; pTerm++)
            {
                pTerm = pWC.a[_pt];
                if (pTerm.eOperator == WO_OR
                && ((pTerm.prereqAll & ~maskSrc) & notReady) == 0
                && (pTerm.u.pOrInfo.indexable & maskSrc) != 0
                )
                {
                    WhereClause pOrWC = pTerm.u.pOrInfo.wc;
                    WhereTerm pOrWCEnd = pOrWC.a[pOrWC.nTerm];
                    WhereTerm pOrTerm;
                    int flags = WHERE_MULTI_OR;
                    double rTotal = 0;
                    double nRow = 0;
                    Bitmask used = 0;

                    for (int _pOrWC = 0; _pOrWC < pOrWC.nTerm; _pOrWC++)//pOrTerm = pOrWC.a ; pOrTerm < pOrWCEnd ; pOrTerm++ )
                    {
                        pOrTerm = pOrWC.a[_pOrWC];
                        WhereCost sTermCost = null;
#if  (SQLITE_TEST) && (SQLITE_DEBUG)
            WHERETRACE( "... Multi-index OR testing for term %d of %d....\n",
            _pOrWC, pOrWC.nTerm - _pOrWC//( pOrTerm - pOrWC.a ), ( pTerm - pWC.a )
            );
#endif
                        if (pOrTerm.eOperator == WO_AND)
                        {
                            WhereClause pAndWC = pOrTerm.u.pAndInfo.wc;
                            bestIndex(pParse, pAndWC, pSrc, notReady, notValid, null, ref sTermCost);
                        }
                        else if (pOrTerm.leftCursor == iCur)
                        {
                            WhereClause tempWC = new WhereClause();
                            tempWC.pParse = pWC.pParse;
                            tempWC.pMaskSet = pWC.pMaskSet;
                            tempWC.op = TK_AND;
                            tempWC.a = new WhereTerm[2];
                            tempWC.a[0] = pOrTerm;
                            tempWC.nTerm = 1;
                            bestIndex(pParse, tempWC, pSrc, notReady, notValid, null, ref sTermCost);
                        }
                        else
                        {
                            continue;
                        }
                        rTotal += sTermCost.rCost;
                        nRow += sTermCost.plan.nRow;
                        used |= sTermCost.used;
                        if (rTotal >= pCost.rCost)
                            break;
                    }

                    /* If there is an ORDER BY clause, increase the scan cost to account
                    ** for the cost of the sort. */
                    if (pOrderBy != null)
                    {
#if  (SQLITE_TEST) && (SQLITE_DEBUG)
            WHERETRACE( "... sorting increases OR cost %.9g to %.9g\n",
            rTotal, rTotal + nRow * estLog( nRow ) );
#endif
                        rTotal += nRow * estLog(nRow);
                    }

                    /* If the cost of scanning using this OR term for optimization is
                    ** less than the current cost stored in pCost, replace the contents
                    ** of pCost. */
#if  (SQLITE_TEST) && (SQLITE_DEBUG)
          WHERETRACE( "... multi-index OR cost=%.9g nrow=%.9g\n", rTotal, nRow );
#endif
                    if (rTotal < pCost.rCost)
                    {
                        pCost.rCost = rTotal;
                        pCost.used = used;
                        pCost.plan.nRow = nRow;
                        pCost.plan.wsFlags = (uint)flags;
                        pCost.plan.u.pTerm = pTerm;
                    }
                }
            }
#endif //* SQLITE_OMIT_OR_OPTIMIZATION */
        }

#if !SQLITE_OMIT_AUTOMATIC_INDEX
        /*
** Return TRUE if the WHERE clause term pTerm is of a form where it
** could be used with an index to access pSrc, assuming an appropriate
** index existed.
*/
        static int termCanDriveIndex(
        WhereTerm pTerm,              /* WHERE clause term to check */
        SrcList_item pSrc,     /* Table we are trying to access */
        Bitmask notReady               /* Tables in outer loops of the join */
        )
        {
            char aff;
            if (pTerm.leftCursor != pSrc.iCursor)
                return 0;
            if (pTerm.eOperator != WO_EQ)
                return 0;
            if ((pTerm.prereqRight & notReady) != 0)
                return 0;
            aff = pSrc.pTab.aCol[pTerm.u.leftColumn].affinity;
            if (!sqlite3IndexAffinityOk(pTerm.pExpr, aff))
                return 0;
            return 1;
        }
#endif

#if !SQLITE_OMIT_AUTOMATIC_INDEX
        /*
** If the query plan for pSrc specified in pCost is a full table scan
** and indexing is allows (if there is no NOT INDEXED clause) and it
** possible to construct a transient index that would perform better
** than a full table scan even when the cost of constructing the index
** is taken into account, then alter the query plan to use the
** transient index.
*/
        static void bestAutomaticIndex(
        Parse pParse,              /* The parsing context */
        WhereClause pWC,           /* The WHERE clause */
        SrcList_item pSrc,         /* The FROM clause term to search */
        Bitmask notReady,          /* Mask of cursors that are not available */
        WhereCost pCost            /* Lowest cost query plan */
        )
        {
            double nTableRow;          /* Rows in the input table */
            double logN;               /* log(nTableRow) */
            double costTempIdx;        /* per-query cost of the transient index */
            WhereTerm pTerm;           /* A single term of the WHERE clause */
            WhereTerm pWCEnd;          /* End of pWC.a[] */
            Table pTable;              /* Table that might be indexed */

            if (pParse.nQueryLoop <= (double)1)
            {
                /* There is no point in building an automatic index for a single scan */
                return;
            }
            if ((pParse.db.flags & SQLITE_AutoIndex) == 0)
            {
                /* Automatic indices are disabled at run-time */
                return;
            }
            if ((pCost.plan.wsFlags & WHERE_NOT_FULLSCAN) != 0)
            {
                /* We already have some kind of index in use for this query. */
                return;
            }
            if (pSrc.notIndexed != 0)
            {
                /* The NOT INDEXED clause appears in the SQL. */
                return;
            }

            Debug.Assert(pParse.nQueryLoop >= (double)1);
            pTable = pSrc.pTab;
            nTableRow = pTable.nRowEst;
            logN = estLog(nTableRow);
            costTempIdx = 2 * logN * (nTableRow / pParse.nQueryLoop + 1);
            if (costTempIdx >= pCost.rCost)
            {
                /* The cost of creating the transient table would be greater than
                ** doing the full table scan */
                return;
            }

            /* Search for any equality comparison term */
            //pWCEnd = pWC.a[pWC.nTerm];
            for (int ipTerm = 0; ipTerm < pWC.nTerm; ipTerm++)//; pTerm<pWCEnd; pTerm++)
            {
                pTerm = pWC.a[ipTerm];
                if (termCanDriveIndex(pTerm, pSrc, notReady) != 0)
                {
#if  (SQLITE_TEST) && (SQLITE_DEBUG)
          WHERETRACE( "auto-index reduces cost from %.2f to %.2f\n",
          pCost.rCost, costTempIdx );
#endif
                    pCost.rCost = costTempIdx;
                    pCost.plan.nRow = logN + 1;
                    pCost.plan.wsFlags = WHERE_TEMP_INDEX;
                    pCost.used = pTerm.prereqRight;
                    break;
                }
            }
        }
#else
//# define bestAutomaticIndex(A,B,C,D,E)  /* no-op */
static void bestAutomaticIndex(
Parse pParse,              /* The parsing context */
WhereClause pWC,           /* The WHERE clause */
SrcList_item pSrc,         /* The FROM clause term to search */
Bitmask notReady,          /* Mask of cursors that are not available */
WhereCost pCost            /* Lowest cost query plan */
){}
#endif //* SQLITE_OMIT_AUTOMATIC_INDEX */


#if !SQLITE_OMIT_AUTOMATIC_INDEX
        /*
** Generate code to construct the Index object for an automatic index
** and to set up the WhereLevel object pLevel so that the code generator
** makes use of the automatic index.
*/
        static void constructAutomaticIndex(
        Parse pParse,              /* The parsing context */
        WhereClause pWC,           /* The WHERE clause */
        SrcList_item pSrc,         /* The FROM clause term to get the next index */
        Bitmask notReady,          /* Mask of cursors that are not available */
        WhereLevel pLevel          /* Write new index here */
        )
        {
            int nColumn;               /* Number of columns in the constructed index */
            WhereTerm pTerm;           /* A single term of the WHERE clause */
            WhereTerm pWCEnd;          /* End of pWC.a[] */
            int nByte;                 /* Byte of memory needed for pIdx */
            Index pIdx;                /* Object describing the transient index */
            Vdbe v;                    /* Prepared statement under construction */
            int regIsInit;             /* Register set by initialization */
            int addrInit;              /* Address of the initialization bypass jump */
            Table pTable;              /* The table being indexed */
            KeyInfo pKeyinfo;          /* Key information for the index */
            int addrTop;               /* Top of the index fill loop */
            int regRecord;             /* Register holding an index record */
            int n;                     /* Column counter */
            int i;                     /* Loop counter */
            int mxBitCol;              /* Maximum column in pSrc.colUsed */
            CollSeq pColl;             /* Collating sequence to on a column */
            Bitmask idxCols;           /* Bitmap of columns used for indexing */
            Bitmask extraCols;         /* Bitmap of additional columns */

            /* Generate code to skip over the creation and initialization of the
            ** transient index on 2nd and subsequent iterations of the loop. */
            v = pParse.pVdbe;
            Debug.Assert(v != null);
            regIsInit = ++pParse.nMem;
            addrInit = sqlite3VdbeAddOp1(v, OP_If, regIsInit);
            sqlite3VdbeAddOp2(v, OP_Integer, 1, regIsInit);

            /* Count the number of columns that will be added to the index
            ** and used to match WHERE clause constraints */
            nColumn = 0;
            pTable = pSrc.pTab;
            //pWCEnd = pWC.a[pWC.nTerm];
            idxCols = 0;
            for (int ipTerm = 0; ipTerm < pWC.nTerm; ipTerm++)//; pTerm<pWCEnd; pTerm++)
            {
                pTerm = pWC.a[ipTerm];
                if (termCanDriveIndex(pTerm, pSrc, notReady) != 0)
                {
                    int iCol = pTerm.u.leftColumn;
                    Bitmask cMask = iCol >= BMS ? ((Bitmask)1) << (BMS - 1) : ((Bitmask)1) << iCol;
                    testcase(iCol == BMS);
                    testcase(iCol == BMS - 1);
                    if ((idxCols & cMask) == 0)
                    {
                        nColumn++;
                        idxCols |= cMask;
                    }
                }
            }
            Debug.Assert(nColumn > 0);
            pLevel.plan.nEq = (u32)nColumn;

            /* Count the number of additional columns needed to create a
            ** covering index.  A "covering index" is an index that contains all
            ** columns that are needed by the query.  With a covering index, the
            ** original table never needs to be accessed.  Automatic indices must
            ** be a covering index because the index will not be updated if the
            ** original table changes and the index and table cannot both be used
            ** if they go out of sync.
            */
            extraCols = pSrc.colUsed & (~idxCols | (((Bitmask)1) << (BMS - 1)));
            mxBitCol = (pTable.nCol >= BMS - 1) ? BMS - 1 : pTable.nCol;
            testcase(pTable.nCol == BMS - 1);
            testcase(pTable.nCol == BMS - 2);
            for (i = 0; i < mxBitCol; i++)
            {
                if ((extraCols & (((Bitmask)1) << i)) != 0)
                    nColumn++;
            }
            if ((pSrc.colUsed & (((Bitmask)1) << (BMS - 1))) != 0)
            {
                nColumn += pTable.nCol - BMS + 1;
            }
            pLevel.plan.wsFlags |= WHERE_COLUMN_EQ | WHERE_IDX_ONLY | WO_EQ;

            /* Construct the Index object to describe this index */
            //nByte = sizeof(Index);
            //nByte += nColumn*sizeof(int);     /* Index.aiColumn */
            //nByte += nColumn*sizeof(char);   /* Index.azColl */
            //nByte += nColumn;                 /* Index.aSortOrder */
            //pIdx = sqlite3DbMallocZero(pParse.db, nByte);
            //if( pIdx==null) return;
            pIdx = new Index();
            pLevel.plan.u.pIdx = pIdx;
            pIdx.azColl = new string[nColumn + 1];// pIdx[1];
            pIdx.aiColumn = new int[nColumn + 1];// pIdx.azColl[nColumn];
            pIdx.aSortOrder = new u8[nColumn + 1];// pIdx.aiColumn[nColumn];
            pIdx.zName = "auto-index";
            pIdx.nColumn = nColumn;
            pIdx.pTable = pTable;
            n = 0;
            idxCols = 0;
            //for(pTerm=pWC.a; pTerm<pWCEnd; pTerm++){
            for (int ipTerm = 0; ipTerm < pWC.nTerm; ipTerm++)
            {
                pTerm = pWC.a[ipTerm];
                if (termCanDriveIndex(pTerm, pSrc, notReady) != 0)
                {
                    int iCol = pTerm.u.leftColumn;
                    Bitmask cMask = iCol >= BMS ? ((Bitmask)1) << (BMS - 1) : ((Bitmask)1) << iCol;
                    if ((idxCols & cMask) == 0)
                    {
                        Expr pX = pTerm.pExpr;
                        idxCols |= cMask;
                        pIdx.aiColumn[n] = pTerm.u.leftColumn;
                        pColl = sqlite3BinaryCompareCollSeq(pParse, pX.pLeft, pX.pRight);
                        pIdx.azColl[n] = ALWAYS(pColl != null) ? pColl.zName : "BINARY";
                        n++;
                    }
                }
            }
            Debug.Assert((u32)n == pLevel.plan.nEq);

            /* Add additional columns needed to make the automatic index into
            ** a covering index */
            for (i = 0; i < mxBitCol; i++)
            {
                if ((extraCols & (((Bitmask)1) << i)) != 0)
                {
                    pIdx.aiColumn[n] = i;
                    pIdx.azColl[n] = "BINARY";
                    n++;
                }
            }
            if ((pSrc.colUsed & (((Bitmask)1) << (BMS - 1))) != 0)
            {
                for (i = BMS - 1; i < pTable.nCol; i++)
                {
                    pIdx.aiColumn[n] = i;
                    pIdx.azColl[n] = "BINARY";
                    n++;
                }
            }
            Debug.Assert(n == nColumn);

            /* Create the automatic index */
            pKeyinfo = sqlite3IndexKeyinfo(pParse, pIdx);
            Debug.Assert(pLevel.iIdxCur >= 0);
            sqlite3VdbeAddOp4(v, OP_OpenAutoindex, pLevel.iIdxCur, nColumn + 1, 0,
            pKeyinfo, P4_KEYINFO_HANDOFF);
            VdbeComment(v, "for %s", pTable.zName);

            /* Fill the automatic index with content */
            addrTop = sqlite3VdbeAddOp1(v, OP_Rewind, pLevel.iTabCur);
            regRecord = sqlite3GetTempReg(pParse);
            sqlite3GenerateIndexKey(pParse, pIdx, pLevel.iTabCur, regRecord, true);
            sqlite3VdbeAddOp2(v, OP_IdxInsert, pLevel.iIdxCur, regRecord);
            sqlite3VdbeChangeP5(v, OPFLAG_USESEEKRESULT);
            sqlite3VdbeAddOp2(v, OP_Next, pLevel.iTabCur, addrTop + 1);
            sqlite3VdbeChangeP5(v, SQLITE_STMTSTATUS_AUTOINDEX);
            sqlite3VdbeJumpHere(v, addrTop);
            sqlite3ReleaseTempReg(pParse, regRecord);

            /* Jump here when skipping the initialization */
            sqlite3VdbeJumpHere(v, addrInit);
        }
#endif //* SQLITE_OMIT_AUTOMATIC_INDEX */

#if !SQLITE_OMIT_VIRTUALTABLE
        /*
** Allocate and populate an sqlite3_index_info structure. It is the
** responsibility of the caller to eventually release the structure
** by passing the pointer returned by this function to //sqlite3_free().
*/
        static sqlite3_index_info allocateIndexInfo(
        Parse pParse,
        WhereClause pWC,
        SrcList_item pSrc,
        ExprList pOrderBy
        )
        {
            int i, j;
            int nTerm;
            sqlite3_index_constraint[] pIdxCons;
            sqlite3_index_orderby[] pIdxOrderBy;
            sqlite3_index_constraint_usage[] pUsage;
            WhereTerm pTerm;
            int nOrderBy;
            sqlite3_index_info pIdxInfo;

#if  (SQLITE_TEST) && (SQLITE_DEBUG)
      WHERETRACE( "Recomputing index info for %s...\n", pSrc.pTab.zName );
#endif

            /* Count the number of possible WHERE clause constraints referring
** to this virtual table */
            for (i = nTerm = 0; i < pWC.nTerm; i++)//, pTerm++ )
            {
                pTerm = pWC.a[i];
                if (pTerm.leftCursor != pSrc.iCursor)
                    continue;
                Debug.Assert((pTerm.eOperator & (pTerm.eOperator - 1)) == 0);
                testcase(pTerm.eOperator == WO_IN);
                testcase(pTerm.eOperator == WO_ISNULL);
                if ((pTerm.eOperator & (WO_IN | WO_ISNULL)) != 0)
                    continue;
                nTerm++;
            }

            /* If the ORDER BY clause contains only columns in the current
            ** virtual table then allocate space for the aOrderBy part of
            ** the sqlite3_index_info structure.
            */
            nOrderBy = 0;
            if (pOrderBy != null)
            {
                for (i = 0; i < pOrderBy.nExpr; i++)
                {
                    Expr pExpr = pOrderBy.a[i].pExpr;
                    if (pExpr.op != TK_COLUMN || pExpr.iTable != pSrc.iCursor)
                        break;
                }
                if (i == pOrderBy.nExpr)
                {
                    nOrderBy = pOrderBy.nExpr;
                }
            }

            /* Allocate the sqlite3_index_info structure
            */
            pIdxInfo = new sqlite3_index_info();
            //sqlite3DbMallocZero(pParse.db, sizeof(*pIdxInfo)
            //+ (sizeof(*pIdxCons) + sizeof(*pUsage))*nTerm
            //+ sizeof(*pIdxOrderBy)*nOrderBy );
            //if ( pIdxInfo == null )
            //{
            //  sqlite3ErrorMsg( pParse, "out of memory" );
            //  /* (double)0 In case of SQLITE_OMIT_FLOATING_POINT... */
            //  return null;
            //}

            /* Initialize the structure.  The sqlite3_index_info structure contains
            ** many fields that are declared "const" to prevent xBestIndex from
            ** changing them.  We have to do some funky casting in order to
            ** initialize those fields.
            */
            pIdxCons = new sqlite3_index_constraint[nTerm];//(sqlite3_index_constraint)pIdxInfo[1];
            pIdxOrderBy = new sqlite3_index_orderby[nOrderBy];//(sqlite3_index_orderby)pIdxCons[nTerm];
            pUsage = new sqlite3_index_constraint_usage[nTerm];//(sqlite3_index_constraint_usage)pIdxOrderBy[nOrderBy];
            pIdxInfo.nConstraint = nTerm;
            pIdxInfo.nOrderBy = nOrderBy;
            pIdxInfo.aConstraint = pIdxCons;
            pIdxInfo.aOrderBy = pIdxOrderBy;
            pIdxInfo.aConstraintUsage =
            pUsage;

            for (i = j = 0; i < pWC.nTerm; i++)//, pTerm++ )
            {
                pTerm = pWC.a[i];
                if (pTerm.leftCursor != pSrc.iCursor)
                    continue;
                Debug.Assert((pTerm.eOperator & (pTerm.eOperator - 1)) == 0);
                testcase(pTerm.eOperator == WO_IN);
                testcase(pTerm.eOperator == WO_ISNULL);
                if ((pTerm.eOperator & (WO_IN | WO_ISNULL)) != 0)
                    continue;
                if (pIdxCons[j] == null)
                    pIdxCons[j] = new sqlite3_index_constraint();
                pIdxCons[j].iColumn = pTerm.u.leftColumn;
                pIdxCons[j].iTermOffset = i;
                pIdxCons[j].op = (u8)pTerm.eOperator;
                /* The direct Debug.Assignment in the previous line is possible only because
                ** the WO_ and SQLITE_INDEX_CONSTRAINT_ codes are identical.  The
                ** following Debug.Asserts verify this fact. */
                Debug.Assert(WO_EQ == SQLITE_INDEX_CONSTRAINT_EQ);
                Debug.Assert(WO_LT == SQLITE_INDEX_CONSTRAINT_LT);
                Debug.Assert(WO_LE == SQLITE_INDEX_CONSTRAINT_LE);
                Debug.Assert(WO_GT == SQLITE_INDEX_CONSTRAINT_GT);
                Debug.Assert(WO_GE == SQLITE_INDEX_CONSTRAINT_GE);
                Debug.Assert(WO_MATCH == SQLITE_INDEX_CONSTRAINT_MATCH);
                Debug.Assert((pTerm.eOperator & (WO_EQ | WO_LT | WO_LE | WO_GT | WO_GE | WO_MATCH)) != 0);
                j++;
            }
            for (i = 0; i < nOrderBy; i++)
            {
                Expr pExpr = pOrderBy.a[i].pExpr;
                if (pIdxOrderBy[i] == null)
                    pIdxOrderBy[i] = new sqlite3_index_orderby();
                pIdxOrderBy[i].iColumn = pExpr.iColumn;
                pIdxOrderBy[i].desc = pOrderBy.a[i].sortOrder != 0;
            }

            return pIdxInfo;
        }

        /*
        ** The table object reference passed as the second argument to this function
        ** must represent a virtual table. This function invokes the xBestIndex()
        ** method of the virtual table with the sqlite3_index_info pointer passed
        ** as the argument.
        **
        ** If an error occurs, pParse is populated with an error message and a
        ** non-zero value is returned. Otherwise, 0 is returned and the output
        ** part of the sqlite3_index_info structure is left populated.
        **
        ** Whether or not an error is returned, it is the responsibility of the
        ** caller to eventually free p.idxStr if p.needToFreeIdxStr indicates
        ** that this is required.
        */
        static int vtabBestIndex(Parse pParse, Table pTab, sqlite3_index_info p)
        {
            sqlite3_vtab pVtab = sqlite3GetVTable(pParse.db, pTab).pVtab;
            int i;
            int rc;

#if  (SQLITE_TEST) && (SQLITE_DEBUG)
      WHERETRACE( "xBestIndex for %s\n", pTab.zName );
#endif
            TRACE_IDX_INPUTS(p);
            rc = pVtab.pModule.xBestIndex(pVtab, ref p);
            TRACE_IDX_OUTPUTS(p);

            if (rc != SQLITE_OK)
            {
                //if ( rc == SQLITE_NOMEM )
                //{
                //  pParse.db.mallocFailed = 1;
                //}
                // else 
                if (String.IsNullOrEmpty(pVtab.zErrMsg))
                {
                    sqlite3ErrorMsg(pParse, "%s", sqlite3ErrStr(rc));
                }
                else
                {
                    sqlite3ErrorMsg(pParse, "%s", pVtab.zErrMsg);
                }
            }
            //sqlite3_free( pVtab.zErrMsg );
            pVtab.zErrMsg = null;

            for (i = 0; i < p.nConstraint; i++)
            {
                if (!p.aConstraint[i].usable && p.aConstraintUsage[i].argvIndex > 0)
                {
                    sqlite3ErrorMsg(pParse,
                    "table %s: xBestIndex returned an invalid plan", pTab.zName);
                }
            }

            return pParse.nErr;
        }


        /*
        ** Compute the best index for a virtual table.
        **
        ** The best index is computed by the xBestIndex method of the virtual
        ** table module.  This routine is really just a wrapper that sets up
        ** the sqlite3_index_info structure that is used to communicate with
        ** xBestIndex.
        **
        ** In a join, this routine might be called multiple times for the
        ** same virtual table.  The sqlite3_index_info structure is created
        ** and initialized on the first invocation and reused on all subsequent
        ** invocations.  The sqlite3_index_info structure is also used when
        ** code is generated to access the virtual table.  The whereInfoDelete()
        ** routine takes care of freeing the sqlite3_index_info structure after
        ** everybody has finished with it.
        */
        static void bestVirtualIndex(
        Parse pParse,                  /* The parsing context */
        WhereClause pWC,               /* The WHERE clause */
        SrcList_item pSrc,             /* The FROM clause term to search */
        Bitmask notReady,              /* Mask of cursors not available for index */
        Bitmask notValid,              /* Cursors not valid for any purpose */
        ExprList pOrderBy,             /* The order by clause */
        ref WhereCost pCost,           /* Lowest cost query plan */
        ref sqlite3_index_info ppIdxInfo /* Index information passed to xBestIndex */
        )
        {
            Table pTab = pSrc.pTab;
            sqlite3_index_info pIdxInfo;
            sqlite3_index_constraint pIdxCons;
            sqlite3_index_constraint_usage[] pUsage = null;
            WhereTerm pTerm;
            int i, j;
            int nOrderBy;
            double rCost;


            /* Make sure wsFlags is initialized to some sane value. Otherwise, if the
            ** malloc in allocateIndexInfo() fails and this function returns leaving
            ** wsFlags in an uninitialized state, the caller may behave unpredictably.
            */
            pCost = new WhereCost();//memset(pCost, 0, sizeof(*pCost));
            pCost.plan.wsFlags = WHERE_VIRTUALTABLE;

            /* If the sqlite3_index_info structure has not been previously
            ** allocated and initialized, then allocate and initialize it now.
            */
            pIdxInfo = ppIdxInfo;
            if (pIdxInfo == null)
            {
                ppIdxInfo = pIdxInfo = allocateIndexInfo(pParse, pWC, pSrc, pOrderBy);
            }
            if (pIdxInfo == null)
            {
                return;
            }

            /* At this point, the sqlite3_index_info structure that pIdxInfo points
            ** to will have been initialized, either during the current invocation or
            ** during some prior invocation.  Now we just have to customize the
            ** details of pIdxInfo for the current invocation and pDebug.Ass it to
            ** xBestIndex.
            */

            /* The module name must be defined. Also, by this point there must
            ** be a pointer to an sqlite3_vtab structure. Otherwise
            ** sqlite3ViewGetColumnNames() would have picked up the error.
            */
            Debug.Assert(pTab.azModuleArg != null && pTab.azModuleArg[0] != null);
            Debug.Assert(sqlite3GetVTable(pParse.db, pTab) != null);

            /* Set the aConstraint[].usable fields and initialize all
            ** output variables to zero.
            **
            ** aConstraint[].usable is true for constraints where the right-hand
            ** side contains only references to tables to the left of the current
            ** table.  In other words, if the constraint is of the form:
            **
            **           column = expr
            **
            ** and we are evaluating a join, then the constraint on column is
            ** only valid if all tables referenced in expr occur to the left
            ** of the table containing column.
            **
            ** The aConstraints[] array contains entries for all constraints
            ** on the current table.  That way we only have to compute it once
            ** even though we might try to pick the best index multiple times.
            ** For each attempt at picking an index, the order of tables in the
            ** join might be different so we have to recompute the usable flag
            ** each time.
            */
            //pIdxCons = *(struct sqlite3_index_constraint**)&pIdxInfo->aConstraint;
            //pUsage = pIdxInfo->aConstraintUsage;
            for (i = 0; i < pIdxInfo.nConstraint; i++)
            {
                pIdxCons = pIdxInfo.aConstraint[i];
                pUsage = pIdxInfo.aConstraintUsage;
                j = pIdxCons.iTermOffset;
                pTerm = pWC.a[j];
                pIdxCons.usable = (pTerm.prereqRight & notReady) == 0;
                pUsage[i] = new sqlite3_index_constraint_usage();
            }
            // memset(pUsage, 0, sizeof(pUsage[0])*pIdxInfo.nConstraint);
            if (pIdxInfo.needToFreeIdxStr != 0)
            {
                //sqlite3_free(ref pIdxInfo.idxStr);
            }
            pIdxInfo.idxStr = null;
            pIdxInfo.idxNum = 0;
            pIdxInfo.needToFreeIdxStr = 0;
            pIdxInfo.orderByConsumed = false;
            /* ((double)2) In case of SQLITE_OMIT_FLOATING_POINT... */
            pIdxInfo.estimatedCost = SQLITE_BIG_DBL / ((double)2);
            nOrderBy = pIdxInfo.nOrderBy;
            if (null == pOrderBy)
            {
                pIdxInfo.nOrderBy = 0;
            }

            if (vtabBestIndex(pParse, pTab, pIdxInfo) != 0)
            {
                return;
            }

            //pIdxCons = (sqlite3_index_constraint)pIdxInfo.aConstraint;
            for (i = 0; i < pIdxInfo.nConstraint; i++)
            {
                if (pUsage[i].argvIndex > 0)
                {
                    //pCost.used |= pWC.a[pIdxCons[i].iTermOffset].prereqRight;
                    pCost.used |= pWC.a[pIdxInfo.aConstraint[i].iTermOffset].prereqRight;
                }
            }

            /* If there is an ORDER BY clause, and the selected virtual table index
            ** does not satisfy it, increase the cost of the scan accordingly. This
            ** matches the processing for non-virtual tables in bestBtreeIndex().
            */
            rCost = pIdxInfo.estimatedCost;
            if (pOrderBy != null && !pIdxInfo.orderByConsumed)
            {
                rCost += estLog(rCost) * rCost;
            }

            /* The cost is not allowed to be larger than SQLITE_BIG_DBL (the
            ** inital value of lowestCost in this loop. If it is, then the
            ** (cost<lowestCost) test below will never be true.
            **
            ** Use "(double)2" instead of "2.0" in case OMIT_FLOATING_POINT
            ** is defined.
            */
            if ((SQLITE_BIG_DBL / ((double)2)) < rCost)
            {
                pCost.rCost = (SQLITE_BIG_DBL / ((double)2));
            }
            else
            {
                pCost.rCost = rCost;
            }
            pCost.plan.u.pVtabIdx = pIdxInfo;
            if (pIdxInfo.orderByConsumed)
            {
                pCost.plan.wsFlags |= WHERE_ORDERBY;
            }
            pCost.plan.nEq = 0;
            pIdxInfo.nOrderBy = nOrderBy;

            /* Try to find a more efficient access pattern by using multiple indexes
            ** to optimize an OR expression within the WHERE clause.
            */
            bestOrClauseIndex(pParse, pWC, pSrc, notReady, notValid, pOrderBy, pCost);
        }
#endif //* SQLITE_OMIT_VIRTUALTABLE */

        /*
** Argument pIdx is a pointer to an index structure that has an array of
** SQLITE_INDEX_SAMPLES evenly spaced samples of the first indexed column
** stored in Index.aSample. These samples divide the domain of values stored
** the index into (SQLITE_INDEX_SAMPLES+1) regions.
** Region 0 contains all values less than the first sample value. Region
** 1 contains values between the first and second samples.  Region 2 contains
** values between samples 2 and 3.  And so on.  Region SQLITE_INDEX_SAMPLES
** contains values larger than the last sample.
**
** If the index contains many duplicates of a single value, then it is
** possible that two or more adjacent samples can hold the same value.
** When that is the case, the smallest possible region code is returned
** when roundUp is false and the largest possible region code is returned
** when roundUp is true.
**
** If successful, this function determines which of the regions value 
** pVal lies in, sets *piRegion to the region index (a value between 0
** and SQLITE_INDEX_SAMPLES+1, inclusive) and returns SQLITE_OK.
** Or, if an OOM occurs while converting text values between encodings,
** SQLITE_NOMEM is returned and *piRegion is undefined.
*/
#if SQLITE_ENABLE_STAT2
    static int whereRangeRegion(
    Parse pParse,               /* Database connection */
    Index pIdx,                 /* Index to consider domain of */
    sqlite3_value pVal,         /* Value to consider */
    int roundUp,                /* Return largest valid region if true */
    out int piRegion            /* OUT: Region of domain in which value lies */
    )
    {
      piRegion = 0;
      Debug.Assert( roundUp == 0 || roundUp == 1 );
      if ( ALWAYS( pVal ) )
      {
        IndexSample[] aSample = pIdx.aSample;
        int i = 0;
        int eType = sqlite3_value_type( pVal );

        if ( eType == SQLITE_INTEGER || eType == SQLITE_FLOAT )
        {
          double r = sqlite3_value_double( pVal );
          for ( i = 0; i < SQLITE_INDEX_SAMPLES; i++ )
          {
            if ( aSample[i].eType == SQLITE_NULL )
              continue;
            if ( aSample[i].eType >= SQLITE_TEXT )
              break;
            if ( roundUp != 0 )
            {
              if ( aSample[i].u.r > r )
                break;
            }
            else
            {
              if ( aSample[i].u.r >= r )
                break;
            }
          }
        }
        else if ( eType == SQLITE_NULL )
        {
          i = 0;
          if ( roundUp != 0 )
          {
            while ( i < SQLITE_INDEX_SAMPLES && aSample[i].eType == SQLITE_NULL )
              i++;
          }
        }
        else
        {
          sqlite3 db = pParse.db;
          CollSeq pColl;
          string z;
          int n;

          /* pVal comes from sqlite3ValueFromExpr() so the type cannot be NULL */
          Debug.Assert( eType == SQLITE_TEXT || eType == SQLITE_BLOB );

          if ( eType == SQLITE_BLOB )
          {
            byte[] blob = sqlite3_value_blob( pVal );
            z = Encoding.UTF8.GetString( blob, 0, blob.Length );
            pColl = db.pDfltColl;
            Debug.Assert( pColl.enc == SQLITE_UTF8 );
          }
          else
          {
            pColl = sqlite3GetCollSeq( db, SQLITE_UTF8, null, pIdx.azColl[0] );
            if ( pColl == null )
            {
              sqlite3ErrorMsg( pParse, "no such collation sequence: %s",
                  pIdx.azColl );
              return SQLITE_ERROR;
            }
            z = sqlite3ValueText( pVal, pColl.enc );
            //if( null==z ){
            //  return SQLITE_NOMEM;
            //}
            Debug.Assert( z != "" && pColl != null && pColl.xCmp != null );
          }
          n = sqlite3ValueBytes( pVal, pColl.enc );

          for ( i = 0; i < SQLITE_INDEX_SAMPLES; i++ )
          {
            int c;
            int eSampletype = aSample[i].eType;
            if ( eSampletype == SQLITE_NULL || eSampletype < eType )
              continue;
            if ( ( eSampletype != eType ) )
              break;
#if !SQLITE_OMIT_UTF16
if( pColl.enc!=SQLITE_UTF8 ){
int nSample;
string zSample;
zSample = sqlite3Utf8to16(
db, pColl.enc, aSample[i].u.z, aSample[i].nByte, ref nSample
);
zSample = aSample[i].u.z;
nSample = aSample[i].u.z.Length;
//if( null==zSample ){
//  Debug.Assert( db.mallocFailed );
//  return SQLITE_NOMEM;
//}
c = pColl.xCmp(pColl.pUser, nSample, zSample, n, z);
sqlite3DbFree(db, ref zSample);
}else
#endif
            {
              c = pColl.xCmp( pColl.pUser, aSample[i].nByte, aSample[i].u.z, n, z );
            }
            if ( c - roundUp >= 0 )
              break;
          }
        }

        Debug.Assert( i >= 0 && i <= SQLITE_INDEX_SAMPLES );
        piRegion = i;
      }
      return SQLITE_OK;
    }
#endif   //* #if SQLITE_ENABLE_STAT2 */

        /*
** If expression pExpr represents a literal value, set *pp to point to
** an sqlite3_value structure containing the same value, with affinity
** aff applied to it, before returning. It is the responsibility of the 
** caller to eventually release this structure by passing it to 
** sqlite3ValueFree().
**
** If the current parse is a recompile (sqlite3Reprepare()) and pExpr
** is an SQL variable that currently has a non-NULL value bound to it,
** create an sqlite3_value structure containing this value, again with
** affinity aff applied to it, instead.
**
** If neither of the above apply, set *pp to NULL.
**
** If an error occurs, return an error code. Otherwise, SQLITE_OK.
*/
#if SQLITE_ENABLE_STAT2
    static int valueFromExpr(
    Parse pParse,
    Expr pExpr,
    char aff,
    ref sqlite3_value pp
    )
    {
      if ( pExpr.op == TK_VARIABLE
      || ( pExpr.op == TK_REGISTER && pExpr.op2 == TK_VARIABLE )
      )
      {
        int iVar = pExpr.iColumn;
        sqlite3VdbeSetVarmask( pParse.pVdbe, iVar ); /* IMP: R-23257-02778 */
        pp = sqlite3VdbeGetValue( pParse.pReprepare, iVar, (u8)aff );
        return SQLITE_OK;
      }
      return sqlite3ValueFromExpr( pParse.db, pExpr, SQLITE_UTF8, aff, ref pp );
    }
#endif

        /*
** This function is used to estimate the number of rows that will be visited
** by scanning an index for a range of values. The range may have an upper
** bound, a lower bound, or both. The WHERE clause terms that set the upper
** and lower bounds are represented by pLower and pUpper respectively. For
** example, assuming that index p is on t1(a):
**
**   ... FROM t1 WHERE a > ? AND a < ? ...
**                    |_____|   |_____|
**                       |         |
**                     pLower    pUpper
**
** If either of the upper or lower bound is not present, then NULL is passed in
** place of the corresponding WhereTerm.
**
** The nEq parameter is passed the index of the index column subject to the
** range constraint. Or, equivalently, the number of equality constraints
** optimized by the proposed index scan. For example, assuming index p is
** on t1(a, b), and the SQL query is:
**
**   ... FROM t1 WHERE a = ? AND b > ? AND b < ? ...
**
** then nEq should be passed the value 1 (as the range restricted column,
** b, is the second left-most column of the index). Or, if the query is:
**
**   ... FROM t1 WHERE a > ? AND a < ? ...
**
** then nEq should be passed 0.
**
** The returned value is an integer between 1 and 100, inclusive. A return
** value of 1 indicates that the proposed range scan is expected to visit
** approximately 1/100th (1%) of the rows selected by the nEq equality
** constraints (if any). A return value of 100 indicates that it is expected
** that the range scan will visit every row (100%) selected by the equality
** constraints.
**
** In the absence of sqlite_stat2 ANALYZE data, each range inequality
** reduces the search space by 3/4ths.  Hence a single constraint (x>?)
** results in a return of 25 and a range constraint (x>? AND x<?) results
** in a return of 6.
*/
        static int whereRangeScanEst(
        Parse pParse,       /* Parsing & code generating context */
        Index p,            /* The index containing the range-compared column; "x" */
        int nEq,            /* index into p.aCol[] of the range-compared column */
        WhereTerm pLower,   /* Lower bound on the range. ex: "x>123" Might be NULL */
        WhereTerm pUpper,   /* Upper bound on the range. ex: "x<455" Might be NULL */
        out int piEst       /* OUT: Return value */
        )
        {
            int rc = SQLITE_OK;

#if SQLITE_ENABLE_STAT2

      if ( nEq == 0 && p.aSample != null )
      {
        sqlite3_value pLowerVal = null;
        sqlite3_value pUpperVal = null;
        int iEst;
        int iLower = 0;
        int iUpper = SQLITE_INDEX_SAMPLES;
        int roundUpUpper = 0;
        int roundUpLower = 0;
        char aff = p.pTable.aCol[p.aiColumn[0]].affinity;

        if ( pLower != null )
        {
          Expr pExpr = pLower.pExpr.pRight;
          rc = valueFromExpr( pParse, pExpr, aff, ref pLowerVal );
          Debug.Assert( pLower.eOperator == WO_GT || pLower.eOperator == WO_GE );
          roundUpLower = ( pLower.eOperator == WO_GT ) ? 1 : 0;
        }
        if ( rc == SQLITE_OK && pUpper != null )
        {
          Expr pExpr = pUpper.pExpr.pRight;
          rc = valueFromExpr( pParse, pExpr, aff, ref pUpperVal );
          Debug.Assert( pUpper.eOperator == WO_LT || pUpper.eOperator == WO_LE );
          roundUpUpper = ( pUpper.eOperator == WO_LE ) ? 1 : 0;
        }

        if ( rc != SQLITE_OK || ( pLowerVal == null && pUpperVal == null ) )
        {
          sqlite3ValueFree( ref pLowerVal );
          sqlite3ValueFree( ref pUpperVal );
          goto range_est_fallback;
        }
        else if ( pLowerVal == null )
        {
          rc = whereRangeRegion( pParse, p, pUpperVal, roundUpUpper, out iUpper );
          if ( pLower != null )
            iLower = iUpper / 2;
        }
        else if ( pUpperVal == null )
        {
          rc = whereRangeRegion( pParse, p, pLowerVal, roundUpLower, out iLower );
          if ( pUpper != null )
            iUpper = ( iLower + SQLITE_INDEX_SAMPLES + 1 ) / 2;
        }
        else
        {
          rc = whereRangeRegion( pParse, p, pUpperVal, roundUpUpper, out iUpper );
          if ( rc == SQLITE_OK )
          {
            rc = whereRangeRegion( pParse, p, pLowerVal, roundUpLower, out iLower );
          }
        }
        WHERETRACE( "range scan regions: %d..%d\n", iLower, iUpper );

        iEst = iUpper - iLower;
        testcase( iEst == SQLITE_INDEX_SAMPLES );
        Debug.Assert( iEst <= SQLITE_INDEX_SAMPLES );
        if ( iEst < 1 )
        {
          piEst = 50 / SQLITE_INDEX_SAMPLES;
        }
        else
        {
          piEst = ( iEst * 100 ) / SQLITE_INDEX_SAMPLES;
        }

        sqlite3ValueFree( ref pLowerVal );
        sqlite3ValueFree( ref pUpperVal );
        return rc;
      }
range_est_fallback:
#else
            UNUSED_PARAMETER(pParse);
            UNUSED_PARAMETER(p);
            UNUSED_PARAMETER(nEq);
#endif
            Debug.Assert(pLower != null || pUpper != null);
            piEst = 100;
            if (pLower != null && (pLower.wtFlags & TERM_VNULL) == 0)
                piEst /= 4;
            if (pUpper != null)
                piEst /= 4;
            return rc;
        }

#if SQLITE_ENABLE_STAT2
    /*
** Estimate the number of rows that will be returned based on
** an equality constraint x=VALUE and where that VALUE occurs in
** the histogram data.  This only works when x is the left-most
** column of an index and sqlite_stat2 histogram data is available
** for that index.  When pExpr==NULL that means the constraint is
** "x IS NULL" instead of "x=VALUE".
**
** Write the estimated row count into *pnRow and return SQLITE_OK. 
** If unable to make an estimate, leave *pnRow unchanged and return
** non-zero.
**
** This routine can fail if it is unable to load a collating sequence
** required for string comparison, or if unable to allocate memory
** for a UTF conversion required for comparison.  The error is stored
** in the pParse structure.
*/
    static int whereEqualScanEst(
      Parse pParse,       /* Parsing & code generating context */
      Index p,            /* The index whose left-most column is pTerm */
      Expr pExpr,         /* Expression for VALUE in the x=VALUE constraint */
      ref double pnRow    /* Write the revised row estimate here */
    )
    {
      sqlite3_value pRhs = null;/* VALUE on right-hand side of pTerm */
      int iLower = 0;
      int iUpper = 0;           /* Range of histogram regions containing pRhs */
      char aff;                 /* Column affinity */
      int rc;                   /* Subfunction return code */
      double nRowEst;           /* New estimate of the number of rows */

      Debug.Assert( p.aSample != null );
      aff = p.pTable.aCol[p.aiColumn[0]].affinity;
      if ( pExpr != null )
      {
        rc = valueFromExpr( pParse, pExpr, aff, ref pRhs );
        if ( rc != 0 )
          goto whereEqualScanEst_cancel;
      }
      else
      {
        pRhs = sqlite3ValueNew( pParse.db );
      }
      if ( pRhs == null )
        return SQLITE_NOTFOUND;
      rc = whereRangeRegion( pParse, p, pRhs, 0, out iLower );
      if ( rc != 0 )
        goto whereEqualScanEst_cancel;
      rc = whereRangeRegion( pParse, p, pRhs, 1, out iUpper );
      if ( rc != 0 )
        goto whereEqualScanEst_cancel;
      WHERETRACE( "equality scan regions: %d..%d\n", iLower, iUpper );
      if ( iLower >= iUpper )
      {
        nRowEst = p.aiRowEst[0] / ( SQLITE_INDEX_SAMPLES * 2 );
        if ( nRowEst < pnRow )
          pnRow = nRowEst;
      }
      else
      {
        nRowEst = ( iUpper - iLower ) * p.aiRowEst[0] / SQLITE_INDEX_SAMPLES;
        pnRow = nRowEst;
      }

whereEqualScanEst_cancel:
      sqlite3ValueFree( ref pRhs );
      return rc;
    }
#endif //* defined(SQLITE_ENABLE_STAT2) */

#if SQLITE_ENABLE_STAT2
    /*
** Estimate the number of rows that will be returned based on
** an IN constraint where the right-hand side of the IN operator
** is a list of values.  Example:
**
**        WHERE x IN (1,2,3,4)
**
** Write the estimated row count into *pnRow and return SQLITE_OK. 
** If unable to make an estimate, leave *pnRow unchanged and return
** non-zero.
**
** This routine can fail if it is unable to load a collating sequence
** required for string comparison, or if unable to allocate memory
** for a UTF conversion required for comparison.  The error is stored
** in the pParse structure.
*/
    static int whereInScanEst(
      Parse pParse,       /* Parsing & code generating context */
      Index p,            /* The index whose left-most column is pTerm */
      ExprList pList,     /* The value list on the RHS of "x IN (v1,v2,v3,...)" */
      ref double pnRow    /* Write the revised row estimate here */
    )
    {
      sqlite3_value pVal = null;/* One value from list */
      int iLower = 0;
      int iUpper = 0;           /* Range of histogram regions containing pRhs */
      char aff;                 /* Column affinity */
      int rc = SQLITE_OK;       /* Subfunction return code */
      double nRowEst;           /* New estimate of the number of rows */
      int nSpan = 0;            /* Number of histogram regions spanned */
      int nSingle = 0;          /* Histogram regions hit by a single value */
      int nNotFound = 0;        /* Count of values that are not constants */
      int i;                               /* Loop counter */
      u8[] aSpan = new u8[SQLITE_INDEX_SAMPLES + 1];    /* Histogram regions that are spanned */
      u8[] aSingle = new u8[SQLITE_INDEX_SAMPLES + 1];  /* Histogram regions hit once */

      Debug.Assert( p.aSample != null );
      aff = p.pTable.aCol[p.aiColumn[0]].affinity;
      //memset(aSpan, 0, sizeof(aSpan));
      //memset(aSingle, 0, sizeof(aSingle));
      for ( i = 0; i < pList.nExpr; i++ )
      {
        sqlite3ValueFree( ref pVal );
        rc = valueFromExpr( pParse, pList.a[i].pExpr, aff, ref pVal );
        if ( rc != 0 )
          break;
        if ( pVal == null || sqlite3_value_type( pVal ) == SQLITE_NULL )
        {
          nNotFound++;
          continue;
        }
        rc = whereRangeRegion( pParse, p, pVal, 0, out iLower );
        if ( rc != 0 )
          break;
        rc = whereRangeRegion( pParse, p, pVal, 1, out iUpper );
        if ( rc != 0 )
          break;
        if ( iLower >= iUpper )
        {
          aSingle[iLower] = 1;
        }
        else
        {
          Debug.Assert( iLower >= 0 && iUpper <= SQLITE_INDEX_SAMPLES );
          while ( iLower < iUpper )
            aSpan[iLower++] = 1;
        }
      }
      if ( rc == SQLITE_OK )
      {
        for ( i = nSpan = 0; i <= SQLITE_INDEX_SAMPLES; i++ )
        {
          if ( aSpan[i] != 0 )
          {
            nSpan++;
          }
          else if ( aSingle[i] != 0 )
          {
            nSingle++;
          }
        }
        nRowEst = ( nSpan * 2 + nSingle ) * p.aiRowEst[0] / ( 2 * SQLITE_INDEX_SAMPLES )
                   + nNotFound * p.aiRowEst[1];
        if ( nRowEst > p.aiRowEst[0] )
          nRowEst = p.aiRowEst[0];
        pnRow = nRowEst;
        WHERETRACE( "IN row estimate: nSpan=%d, nSingle=%d, nNotFound=%d, est=%g\n",
                     nSpan, nSingle, nNotFound, nRowEst );
      }
      sqlite3ValueFree( ref pVal );
      return rc;
    }
#endif //* defined(SQLITE_ENABLE_STAT2) */


        /*
** Find the best query plan for accessing a particular table.  Write the
** best query plan and its cost into the WhereCost object supplied as the
** last parameter.
**
** The lowest cost plan wins.  The cost is an estimate of the amount of
** CPU and disk I/O needed to process the requested result.
** Factors that influence cost include:
**
**    *  The estimated number of rows that will be retrieved.  (The
**       fewer the better.)
**
**    *  Whether or not sorting must occur.
**
**    *  Whether or not there must be separate lookups in the
**       index and in the main table.
**
** If there was an INDEXED BY clause (pSrc->pIndex) attached to the table in
** the SQL statement, then this function only considers plans using the 
** named index. If no such plan is found, then the returned cost is
** SQLITE_BIG_DBL. If a plan is found that uses the named index, 
** then the cost is calculated in the usual way.
**
** If a NOT INDEXED clause (pSrc->notIndexed!=0) was attached to the table 
** in the SELECT statement, then no indexes are considered. However, the 
** selected plan may still take advantage of the built-in rowid primary key
** index.
*/
        static void bestBtreeIndex(
        Parse pParse,              /* The parsing context */
        WhereClause pWC,           /* The WHERE clause */
        SrcList_item pSrc,         /* The FROM clause term to search */
        Bitmask notReady,          /* Mask of cursors not available for indexing */
        Bitmask notValid,          /* Cursors not available for any purpose */
        ExprList pOrderBy,         /* The ORDER BY clause */
        ref WhereCost pCost        /* Lowest cost query plan */
        )
        {
            int iCur = pSrc.iCursor;    /* The cursor of the table to be accessed */
            Index pProbe;               /* An index we are evaluating */
            Index pIdx;                 /* Copy of pProbe, or zero for IPK index */
            u32 eqTermMask;             /* Current mask of valid equality operators */
            u32 idxEqTermMask;          /* Index mask of valid equality operators */
            Index sPk;                  /* A fake index object for the primary key */
            int[] aiRowEstPk = new int[2]; /* The aiRowEst[] value for the sPk index */
            int aiColumnPk = -1;        /* The aColumn[] value for the sPk index */
            int wsFlagMask;             /* Allowed flags in pCost.plan.wsFlag */

            /* Initialize the cost to a worst-case value */
            if (pCost == null)
                pCost = new WhereCost();
            else
                pCost.Clear();  //memset(pCost, 0, sizeof(*pCost));
            pCost.rCost = SQLITE_BIG_DBL;

            /* If the pSrc table is the right table of a LEFT JOIN then we may not
            ** use an index to satisfy IS NULL constraints on that table.  This is
            ** because columns might end up being NULL if the table does not match -
            ** a circumstance which the index cannot help us discover.  Ticket #2177.
            */
            if ((pSrc.jointype & JT_LEFT) != 0)
            {
                idxEqTermMask = WO_EQ | WO_IN;
            }
            else
            {
                idxEqTermMask = WO_EQ | WO_IN | WO_ISNULL;
            }

            if (pSrc.pIndex != null)
            {
                /* An INDEXED BY clause specifies a particular index to use */
                pIdx = pProbe = pSrc.pIndex;
                wsFlagMask = ~(WHERE_ROWID_EQ | WHERE_ROWID_RANGE);
                eqTermMask = idxEqTermMask;
            }
            else
            {
                /* There is no INDEXED BY clause.  Create a fake Index object in local
                ** variable sPk to represent the rowid primary key index.  Make this
                ** fake index the first in a chain of Index objects with all of the real
                ** indices to follow */
                Index pFirst;                  /* First of real indices on the table */
                sPk = new Index(); // memset( &sPk, 0, sizeof( Index ) );
                sPk.aSortOrder = new byte[1];
                sPk.azColl = new string[1];
                sPk.azColl[0] = "";
                sPk.nColumn = 1;
                sPk.aiColumn = new int[1];
                sPk.aiColumn[0] = aiColumnPk;
                sPk.aiRowEst = aiRowEstPk;
                sPk.onError = OE_Replace;
                sPk.pTable = pSrc.pTab;
                aiRowEstPk[0] = (int)pSrc.pTab.nRowEst;
                aiRowEstPk[1] = 1;
                pFirst = pSrc.pTab.pIndex;
                if (pSrc.notIndexed == 0)
                {
                    /* The real indices of the table are only considered if the
                    ** NOT INDEXED qualifier is omitted from the FROM clause */
                    sPk.pNext = pFirst;
                }
                pProbe = sPk;
                wsFlagMask = ~(
                WHERE_COLUMN_IN | WHERE_COLUMN_EQ | WHERE_COLUMN_NULL | WHERE_COLUMN_RANGE
                );
                eqTermMask = WO_EQ | WO_IN;
                pIdx = null;
            }

            /* Loop over all indices looking for the best one to use
            */
            for (; pProbe != null; pIdx = pProbe = pProbe.pNext)
            {
                int[] aiRowEst = pProbe.aiRowEst;
                double cost;                /* Cost of using pProbe */
                double nRow;                /* Estimated number of rows in result set */
                double log10N = 0;          /* base-10 logarithm of nRow (inexact) */
                int rev = 0;                /* True to scan in reverse order */
                int wsFlags = 0;
                Bitmask used = 0;

                /* The following variables are populated based on the properties of
                ** index being evaluated. They are then used to determine the expected
                ** cost and number of rows returned.
                **
                **  nEq: 
                **    Number of equality terms that can be implemented using the index.
                **    In other words, the number of initial fields in the index that
                **    are used in == or IN or NOT NULL constraints of the WHERE clause.
                **
                **  nInMul:  
                **    The "in-multiplier". This is an estimate of how many seek operations 
                **    SQLite must perform on the index in question. For example, if the 
                **    WHERE clause is:
                **
                **      WHERE a IN (1, 2, 3) AND b IN (4, 5, 6)
                **
                **    SQLite must perform 9 lookups on an index on (a, b), so nInMul is 
                **    set to 9. Given the same schema and either of the following WHERE 
                **    clauses:
                **
                **      WHERE a =  1
                **      WHERE a >= 2
                **
                **    nInMul is set to 1.
                **
                **    If there exists a WHERE term of the form "x IN (SELECT ...)", then 
                **    the sub-select is assumed to return 25 rows for the purposes of 
                **    determining nInMul.
                **
                **  bInEst:  
                **    Set to true if there was at least one "x IN (SELECT ...)" term used 
                **    in determining the value of nInMul.  Note that the RHS of the
                **    IN operator must be a SELECT, not a value list, for this variable
                **    to be true.
                **
                **  estBound:
                **    An estimate on the amount of the table that must be searched.  A
                **    value of 100 means the entire table is searched.  Range constraints
                **    might reduce this to a value less than 100 to indicate that only
                **    a fraction of the table needs searching.  In the absence of
                **    sqlite_stat2 ANALYZE data, a single inequality reduces the search
                **    space to 1/4rd its original size.  So an x>? constraint reduces
                **    estBound to 25.  Two constraints (x>? AND x<?) reduce estBound to 6.
                **
                **  bSort:   
                **    Boolean. True if there is an ORDER BY clause that will require an 
                **    external sort (i.e. scanning the index being evaluated will not 
                **    correctly order records).
                **
                **  bLookup: 
                **    Boolean. True if a table lookup is required for each index entry
                **    visited.  In other words, true if this is not a covering index.
                **    This is always false for the rowid primary key index of a table.
                **    For other indexes, it is true unless all the columns of the table
                **    used by the SELECT statement are present in the index (such an
                **    index is sometimes described as a covering index).
                **    For example, given the index on (a, b), the second of the following 
                **    two queries requires table b-tree lookups in order to find the value
                **    of column c, but the first does not because columns a and b are
                **    both available in the index.
                **
                **             SELECT a, b    FROM tbl WHERE a = 1;
                **             SELECT a, b, c FROM tbl WHERE a = 1;
                */
                int nEq;                      /* Number of == or IN terms matching index */
                int bInEst = 0;               /* True if "x IN (SELECT...)" seen */
                int nInMul = 1;               /* Number of distinct equalities to lookup */
                int estBound = 100;           /* Estimated reduction in search space */
                int nBound = 0;               /* Number of range constraints seen */
                int bSort = 0;                /* True if external sort required */
                int bLookup = 0;              /* True if not a covering index */
                WhereTerm pTerm;              /* A single term of the WHERE clause */
#if SQLITE_ENABLE_STAT2
        WhereTerm pFirstTerm = null;  /* First term matching the index */
#endif

                /* Determine the values of nEq and nInMul */
                for (nEq = 0; nEq < pProbe.nColumn; nEq++)
                {
                    int j = pProbe.aiColumn[nEq];
                    pTerm = FindTerm(pWC, iCur, j, notReady, eqTermMask, pIdx);
                    if (pTerm == null)
                        break;
                    wsFlags |= (WHERE_COLUMN_EQ | WHERE_ROWID_EQ);
                    if ((pTerm.eOperator & WO_IN) != 0)
                    {
                        Expr pExpr = pTerm.pExpr;
                        wsFlags |= WHERE_COLUMN_IN;
                        if (ExprHasProperty(pExpr, EP_xIsSelect))
                        {
                            /* "x IN (SELECT ...)":  Assume the SELECT returns 25 rows */
                            nInMul *= 25;
                            bInEst = 1;
                        }
                        else if (ALWAYS(pExpr.x.pList != null) && pExpr.x.pList.nExpr != 0)
                        {
                            /* "x IN (value, value, ...)" */
                            nInMul *= pExpr.x.pList.nExpr;
                        }
                    }
                    else if ((pTerm.eOperator & WO_ISNULL) != 0)
                    {
                        wsFlags |= WHERE_COLUMN_NULL;
                    }
#if SQLITE_ENABLE_STAT2
          if ( nEq == 0 && pProbe.aSample != null )
            pFirstTerm = pTerm;
#endif
                    used |= pTerm.prereqRight;
                }

                /* Determine the value of estBound. */
                if (nEq < pProbe.nColumn && pProbe.bUnordered == 0)
                {
                    int j = pProbe.aiColumn[nEq];
                    if (FindTerm(pWC, iCur, j, notReady, WO_LT | WO_LE | WO_GT | WO_GE, pIdx) != null)
                    {
                        WhereTerm pTop = FindTerm(pWC, iCur, j, notReady, WO_LT | WO_LE, pIdx);
                        WhereTerm pBtm = FindTerm(pWC, iCur, j, notReady, WO_GT | WO_GE, pIdx);
                        whereRangeScanEst(pParse, pProbe, nEq, pBtm, pTop, out estBound);
                        if (pTop != null)
                        {
                            nBound = 1;
                            wsFlags |= WHERE_TOP_LIMIT;
                            used |= pTop.prereqRight;
                        }
                        if (pBtm != null)
                        {
                            nBound++;
                            wsFlags |= WHERE_BTM_LIMIT;
                            used |= pBtm.prereqRight;
                        }
                        wsFlags |= (WHERE_COLUMN_RANGE | WHERE_ROWID_RANGE);
                    }
                }
                else if (pProbe.onError != OE_None)
                {
                    testcase(wsFlags & WHERE_COLUMN_IN);
                    testcase(wsFlags & WHERE_COLUMN_NULL);
                    if ((wsFlags & (WHERE_COLUMN_IN | WHERE_COLUMN_NULL)) == 0)
                    {
                        wsFlags |= WHERE_UNIQUE;
                    }
                }

                /* If there is an ORDER BY clause and the index being considered will
                ** naturally scan rows in the required order, set the appropriate flags
                ** in wsFlags. Otherwise, if there is an ORDER BY clause but the index
                ** will scan rows in a different order, set the bSort variable.  */
                if (pOrderBy != null)
                {
                    if ((wsFlags & WHERE_COLUMN_IN) == 0
                      && pProbe.bUnordered == 0
                      && isSortingIndex(pParse, pWC.pMaskSet, pProbe, iCur, pOrderBy,
                                        nEq, wsFlags, ref rev)
                    )
                    {
                        wsFlags |= WHERE_ROWID_RANGE | WHERE_COLUMN_RANGE | WHERE_ORDERBY;
                        wsFlags |= (rev != 0 ? WHERE_REVERSE : 0);
                    }
                    else
                    {
                        bSort = 1;
                    }
                }

                /* If currently calculating the cost of using an index (not the IPK
                ** index), determine if all required column data may be obtained without 
                ** using the main table (i.e. if the index is a covering
                ** index for this query). If it is, set the WHERE_IDX_ONLY flag in
                ** wsFlags. Otherwise, set the bLookup variable to true.  */
                if (pIdx != null && wsFlags != 0)
                {
                    Bitmask m = pSrc.colUsed;
                    int j;
                    for (j = 0; j < pIdx.nColumn; j++)
                    {
                        int x = pIdx.aiColumn[j];
                        if (x < BMS - 1)
                        {
                            m &= ~(((Bitmask)1) << x);
                        }
                    }
                    if (m == 0)
                    {
                        wsFlags |= WHERE_IDX_ONLY;
                    }
                    else
                    {
                        bLookup = 1;
                    }
                }

                /*
                ** Estimate the number of rows of output.  For an "x IN (SELECT...)"
                ** constraint, do not let the estimate exceed half the rows in the table.
                */
                nRow = (double)(aiRowEst[nEq] * nInMul);
                if (bInEst != 0 && nRow * 2 > aiRowEst[0])
                {
                    nRow = aiRowEst[0] / 2;
                    nInMul = (int)(nRow / aiRowEst[nEq]);
                }

#if SQLITE_ENABLE_STAT2
        /* If the constraint is of the form x=VALUE and histogram
    ** data is available for column x, then it might be possible
    ** to get a better estimate on the number of rows based on
    ** VALUE and how common that value is according to the histogram.
    */
        if ( nRow > (double)1 && nEq == 1 && pFirstTerm != null )
        {
          if ( ( pFirstTerm.eOperator & ( WO_EQ | WO_ISNULL ) ) != 0 )
          {
            testcase( pFirstTerm.eOperator == WO_EQ );
            testcase( pFirstTerm.eOperator == WO_ISNULL );
            whereEqualScanEst( pParse, pProbe, pFirstTerm.pExpr.pRight, ref nRow );
          }
          else if ( pFirstTerm.eOperator == WO_IN && bInEst == 0 )
          {
            whereInScanEst( pParse, pProbe, pFirstTerm.pExpr.x.pList, ref nRow );
          }
        }
#endif //* SQLITE_ENABLE_STAT2 */

                /* Adjust the number of output rows and downward to reflect rows
    ** that are excluded by range constraints.
    */
                nRow = (nRow * (double)estBound) / (double)100;
                if (nRow < 1)
                    nRow = 1;

                /* Experiments run on real SQLite databases show that the time needed
                ** to do a binary search to locate a row in a table or index is roughly
                ** log10(N) times the time to move from one row to the next row within
                ** a table or index.  The actual times can vary, with the size of
                ** records being an important factor.  Both moves and searches are
                ** slower with larger records, presumably because fewer records fit
                ** on one page and hence more pages have to be fetched.
                **
                ** The ANALYZE command and the sqlite_stat1 and sqlite_stat2 tables do
                ** not give us data on the relative sizes of table and index records.
                ** So this computation assumes table records are about twice as big
                ** as index records
                */
                if ((wsFlags & WHERE_NOT_FULLSCAN) == 0)
                {
                    /* The cost of a full table scan is a number of move operations equal
                    ** to the number of rows in the table.
                    **
                    ** We add an additional 4x penalty to full table scans.  This causes
                    ** the cost function to err on the side of choosing an index over
                    ** choosing a full scan.  This 4x full-scan penalty is an arguable
                    ** decision and one which we expect to revisit in the future.  But
                    ** it seems to be working well enough at the moment.
                    */
                    cost = aiRowEst[0] * 4;
                }
                else
                {
                    log10N = estLog(aiRowEst[0]);
                    cost = nRow;
                    if (pIdx != null)
                    {
                        if (bLookup != 0)
                        {
                            /* For an index lookup followed by a table lookup:
                            **    nInMul index searches to find the start of each index range
                            **  + nRow steps through the index
                            **  + nRow table searches to lookup the table entry using the rowid
                            */
                            cost += (nInMul + nRow) * log10N;
                        }
                        else
                        {
                            /* For a covering index:
                            **     nInMul index searches to find the initial entry 
                            **   + nRow steps through the index
                            */
                            cost += nInMul * log10N;
                        }
                    }
                    else
                    {
                        /* For a rowid primary key lookup:
                        **    nInMult table searches to find the initial entry for each range
                        **  + nRow steps through the table
                        */
                        cost += nInMul * log10N;
                    }
                }

                /* Add in the estimated cost of sorting the result.  Actual experimental
                ** measurements of sorting performance in SQLite show that sorting time
                ** adds C*N*log10(N) to the cost, where N is the number of rows to be 
                ** sorted and C is a factor between 1.95 and 4.3.  We will split the
                ** difference and select C of 3.0.
                */
                if (bSort != 0)
                {
                    cost += nRow * estLog(nRow) * 3;
                }

                /**** Cost of using this index has now been computed ****/

                /* If there are additional constraints on this table that cannot
                ** be used with the current index, but which might lower the number
                ** of output rows, adjust the nRow value accordingly.  This only 
                ** matters if the current index is the least costly, so do not bother
                ** with this step if we already know this index will not be chosen.
                ** Also, never reduce the output row count below 2 using this step.
                **
                ** It is critical that the notValid mask be used here instead of
                ** the notReady mask.  When computing an "optimal" index, the notReady
                ** mask will only have one bit set - the bit for the current table.
                ** The notValid mask, on the other hand, always has all bits set for
                ** tables that are not in outer loops.  If notReady is used here instead
                ** of notValid, then a optimal index that depends on inner joins loops
                ** might be selected even when there exists an optimal index that has
                ** no such dependency.
                */
                if (nRow > 2 && cost <= pCost.rCost)
                {
                    //int k;                       /* Loop counter */
                    int nSkipEq = nEq;           /* Number of == constraints to skip */
                    int nSkipRange = nBound;     /* Number of < constraints to skip */
                    Bitmask thisTab;             /* Bitmap for pSrc */

                    thisTab = getMask(pWC.pMaskSet, iCur);
                    for (int ipTerm = 0, k = pWC.nTerm; nRow > 2 && k != 0; k--, ipTerm++)//pTerm++)
                    {
                        pTerm = pWC.a[ipTerm];
                        if ((pTerm.wtFlags & TERM_VIRTUAL) != 0)
                            continue;
                        if ((pTerm.prereqAll & notValid) != thisTab)
                            continue;
                        if ((pTerm.eOperator & (WO_EQ | WO_IN | WO_ISNULL)) != 0)
                        {
                            if (nSkipEq != 0)
                            {
                                /* Ignore the first nEq equality matches since the index
                                ** has already accounted for these */
                                nSkipEq--;
                            }
                            else
                            {
                                /* Assume each additional equality match reduces the result
                                ** set size by a factor of 10 */
                                nRow /= 10;
                            }
                        }
                        else if ((pTerm.eOperator & (WO_LT | WO_LE | WO_GT | WO_GE)) != 0)
                        {
                            if (nSkipRange != 0)
                            {
                                /* Ignore the first nSkipRange range constraints since the index
                                ** has already accounted for these */
                                nSkipRange--;
                            }
                            else
                            {
                                /* Assume each additional range constraint reduces the result
                                ** set size by a factor of 3.  Indexed range constraints reduce
                                ** the search space by a larger factor: 4.  We make indexed range
                                ** more selective intentionally because of the subjective 
                                ** observation that indexed range constraints really are more
                                ** selective in practice, on average. */
                                nRow /= 3;
                            }
                        }
                        else if (pTerm.eOperator != WO_NOOP)
                        {
                            /* Any other expression lowers the output row count by half */
                            nRow /= 2;
                        }
                    }
                    if (nRow < 2)
                        nRow = 2;
                }

#if  (SQLITE_TEST) && (SQLITE_DEBUG)
        WHERETRACE(
        "%s(%s): nEq=%d nInMul=%d estBound=%d bSort=%d bLookup=%d wsFlags=0x%x\n" +
      "         notReady=0x%llx log10N=%.1f nRow=%.1f cost=%.1f used=0x%llx\n",
        pSrc.pTab.zName, ( pIdx != null ? pIdx.zName : "ipk" ),
        nEq, nInMul, estBound, bSort, bLookup, wsFlags,
        notReady, log10N, cost, used
        );
#endif
                /* If this index is the best we have seen so far, then record this
** index and its cost in the pCost structure.
*/
                if ((null == pIdx || wsFlags != 0)
                && (cost < pCost.rCost || (cost <= pCost.rCost && nRow < pCost.plan.nRow))
                )
                {
                    pCost.rCost = cost;
                    pCost.used = used;
                    pCost.plan.nRow = nRow;
                    pCost.plan.wsFlags = (uint)(wsFlags & wsFlagMask);
                    pCost.plan.nEq = (uint)nEq;
                    pCost.plan.u.pIdx = pIdx;
                }

                /* If there was an INDEXED BY clause, then only that one index is
                ** considered. */
                if (pSrc.pIndex != null)
                    break;

                /* Reset masks for the next index in the loop */
                wsFlagMask = ~(WHERE_ROWID_EQ | WHERE_ROWID_RANGE);
                eqTermMask = idxEqTermMask;
            }

            /* If there is no ORDER BY clause and the SQLITE_ReverseOrder flag
            ** is set, then reverse the order that the index will be scanned
            ** in. This is used for application testing, to help find cases
            ** where application behaviour depends on the (undefined) order that
            ** SQLite outputs rows in in the absence of an ORDER BY clause.  */
            if (null == pOrderBy && (pParse.db.flags & SQLITE_ReverseOrder) != 0)
            {
                pCost.plan.wsFlags |= WHERE_REVERSE;
            }

            Debug.Assert(pOrderBy != null || (pCost.plan.wsFlags & WHERE_ORDERBY) == 0);
            Debug.Assert(pCost.plan.u.pIdx == null || (pCost.plan.wsFlags & WHERE_ROWID_EQ) == 0);
            Debug.Assert(pSrc.pIndex == null
            || pCost.plan.u.pIdx == null
            || pCost.plan.u.pIdx == pSrc.pIndex
            );

#if  (SQLITE_TEST) && (SQLITE_DEBUG)
      WHERETRACE( "best index is: %s\n",
      ( ( pCost.plan.wsFlags & WHERE_NOT_FULLSCAN ) == 0 ? "none" :
      pCost.plan.u.pIdx != null ? pCost.plan.u.pIdx.zName : "ipk" )
      );
#endif
            bestOrClauseIndex(pParse, pWC, pSrc, notReady, notValid, pOrderBy, pCost);
            bestAutomaticIndex(pParse, pWC, pSrc, notReady, pCost);
            pCost.plan.wsFlags |= (u32)eqTermMask;
        }


        /*
        ** Find the query plan for accessing table pSrc.pTab. Write the
        ** best query plan and its cost into the WhereCost object supplied
        ** as the last parameter. This function may calculate the cost of
        ** both real and virtual table scans.
        */
        static void bestIndex(
        Parse pParse,               /* The parsing context */
        WhereClause pWC,            /* The WHERE clause */
        SrcList_item pSrc,          /* The FROM clause term to search */
        Bitmask notReady,           /* Mask of cursors not available for indexing */
        Bitmask notValid,           /* Cursors not available for any purpose */
        ExprList pOrderBy,          /* The ORDER BY clause */
        ref WhereCost pCost         /* Lowest cost query plan */
        )
        {
#if !SQLITE_OMIT_VIRTUALTABLE
            if (IsVirtual(pSrc.pTab))
            {
                sqlite3_index_info p = null;
                bestVirtualIndex(pParse, pWC, pSrc, notReady, notValid, pOrderBy, ref pCost, ref p);
                if (p.needToFreeIdxStr != 0)
                {
                    //sqlite3_free(ref p.idxStr);
                }
                sqlite3DbFree(pParse.db, ref p);
            }
            else
#endif
            {
                bestBtreeIndex(pParse, pWC, pSrc, notReady, notValid, pOrderBy, ref pCost);
            }
        }

        /*
        ** Disable a term in the WHERE clause.  Except, do not disable the term
        ** if it controls a LEFT OUTER JOIN and it did not originate in the ON
        ** or USING clause of that join.
        **
        ** Consider the term t2.z='ok' in the following queries:
        **
        **   (1)  SELECT * FROM t1 LEFT JOIN t2 ON t1.a=t2.x WHERE t2.z='ok'
        **   (2)  SELECT * FROM t1 LEFT JOIN t2 ON t1.a=t2.x AND t2.z='ok'
        **   (3)  SELECT * FROM t1, t2 WHERE t1.a=t2.x AND t2.z='ok'
        **
        ** The t2.z='ok' is disabled in the in (2) because it originates
        ** in the ON clause.  The term is disabled in (3) because it is not part
        ** of a LEFT OUTER JOIN.  In (1), the term is not disabled.
        **
        ** IMPLEMENTATION-OF: R-24597-58655 No tests are done for terms that are
        ** completely satisfied by indices.
        **
        ** Disabling a term causes that term to not be tested in the inner loop
        ** of the join.  Disabling is an optimization.  When terms are satisfied
        ** by indices, we disable them to prevent redundant tests in the inner
        ** loop.  We would get the correct results if nothing were ever disabled,
        ** but joins might run a little slower.  The trick is to disable as much
        ** as we can without disabling too much.  If we disabled in (1), we'd get
        ** the wrong answer.  See ticket #813.
        */
        static void disableTerm(WhereLevel pLevel, WhereTerm pTerm)
        {
            if (pTerm != null
            && (pTerm.wtFlags & TERM_CODED) == 0
            && (pLevel.iLeftJoin == 0 || ExprHasProperty(pTerm.pExpr, EP_FromJoin)))
            {
                pTerm.wtFlags |= TERM_CODED;
                if (pTerm.iParent >= 0)
                {
                    WhereTerm pOther = pTerm.pWC.a[pTerm.iParent];
                    if ((--pOther.nChild) == 0)
                    {
                        disableTerm(pLevel, pOther);
                    }
                }
            }
        }

        /*
        ** Code an OP_Affinity opcode to apply the column affinity string zAff
        ** to the n registers starting at base. 
        **
        ** As an optimization, SQLITE_AFF_NONE entries (which are no-ops) at the
        ** beginning and end of zAff are ignored.  If all entries in zAff are
        ** SQLITE_AFF_NONE, then no code gets generated.
        **
        ** This routine makes its own copy of zAff so that the caller is free
        ** to modify zAff after this routine returns.
        */
        static void codeApplyAffinity(Parse pParse, int _base, int n, string zAff)
        {
            Vdbe v = pParse.pVdbe;
            //if (zAff == 0)
            //{
            //  Debug.Assert(pParse.db.mallocFailed);
            //  return;
            //}
            Debug.Assert(v != null);
            /* Adjust base and n to skip over SQLITE_AFF_NONE entries at the beginning
            ** and end of the affinity string.
            */
            while (n > 0 && zAff[0] == SQLITE_AFF_NONE)
            {
                n--;
                _base++;
                zAff = zAff.Substring(1);// zAff++;
            }
            while (n > 1 && zAff[n - 1] == SQLITE_AFF_NONE)
            {
                n--;
            }

            /* Code the OP_Affinity opcode if there is anything left to do. */
            if (n > 0)
            {
                sqlite3VdbeAddOp2(v, OP_Affinity, _base, n);
                sqlite3VdbeChangeP4(v, -1, zAff, n);
                sqlite3ExprCacheAffinityChange(pParse, _base, n);
            }
        }

        /*
        ** Generate code for a single equality term of the WHERE clause.  An equality
        ** term can be either X=expr or X IN (...).   pTerm is the term to be
        ** coded.
        **
        ** The current value for the constraint is left in register iReg.
        **
        ** For a constraint of the form X=expr, the expression is evaluated and its
        ** result is left on the stack.  For constraints of the form X IN (...)
        ** this routine sets up a loop that will iterate over all values of X.
        */
        static int codeEqualityTerm(
        Parse pParse,      /* The parsing context */
        WhereTerm pTerm,   /* The term of the WHERE clause to be coded */
        WhereLevel pLevel, /* When level of the FROM clause we are working on */
        int iTarget         /* Attempt to leave results in this register */
        )
        {
            Expr pX = pTerm.pExpr;
            Vdbe v = pParse.pVdbe;
            int iReg;                  /* Register holding results */

            Debug.Assert(iTarget > 0);
            if (pX.op == TK_EQ)
            {
                iReg = sqlite3ExprCodeTarget(pParse, pX.pRight, iTarget);
            }
            else if (pX.op == TK_ISNULL)
            {
                iReg = iTarget;
                sqlite3VdbeAddOp2(v, OP_Null, 0, iReg);
#if  !SQLITE_OMIT_SUBQUERY
            }
            else
            {
                int eType;
                int iTab;
                InLoop pIn;

                Debug.Assert(pX.op == TK_IN);
                iReg = iTarget;
                int iDummy = -1;
                eType = sqlite3FindInIndex(pParse, pX, ref iDummy);
                iTab = pX.iTable;
                sqlite3VdbeAddOp2(v, OP_Rewind, iTab, 0);
                Debug.Assert((pLevel.plan.wsFlags & WHERE_IN_ABLE) != 0);
                if (pLevel.u._in.nIn == 0)
                {
                    pLevel.addrNxt = sqlite3VdbeMakeLabel(v);
                }
                pLevel.u._in.nIn++;
                if (pLevel.u._in.aInLoop == null)
                    pLevel.u._in.aInLoop = new InLoop[pLevel.u._in.nIn];
                else
                    Array.Resize(ref pLevel.u._in.aInLoop, pLevel.u._in.nIn);
                //sqlite3DbReallocOrFree(pParse.db, pLevel.u._in.aInLoop,
                //                       sizeof(pLevel.u._in.aInLoop[0])*pLevel.u._in.nIn);
                //pIn = pLevel.u._in.aInLoop;
                if (pLevel.u._in.aInLoop != null)//(pIn )
                {
                    pLevel.u._in.aInLoop[pLevel.u._in.nIn - 1] = new InLoop();
                    pIn = pLevel.u._in.aInLoop[pLevel.u._in.nIn - 1];//pIn++
                    pIn.iCur = iTab;
                    if (eType == IN_INDEX_ROWID)
                    {
                        pIn.addrInTop = sqlite3VdbeAddOp2(v, OP_Rowid, iTab, iReg);
                    }
                    else
                    {
                        pIn.addrInTop = sqlite3VdbeAddOp3(v, OP_Column, iTab, 0, iReg);
                    }
                    sqlite3VdbeAddOp1(v, OP_IsNull, iReg);
                }
                else
                {
                    pLevel.u._in.nIn = 0;
                }
#endif
            }
            disableTerm(pLevel, pTerm);
            return iReg;
        }

        /*
        ** Generate code for a single equality term of the WHERE clause.  An equality
        ** term can be either X=expr or X IN (...).   pTerm is the term to be 
        ** coded.
        **
        ** For example, consider table t1(a,b,c,d,e,f) with index i1(a,b,c).
        ** Suppose the WHERE clause is this:  a==5 AND b IN (1,2,3) AND c>5 AND c<10
        ** The index has as many as three equality constraints, but in this
        ** example, the third "c" value is an inequality.  So only two
        ** constraints are coded.  This routine will generate code to evaluate
        ** a==5 and b IN (1,2,3).  The current values for a and b will be stored
        ** in consecutive registers and the index of the first register is returned.
        **
        ** In the example above nEq==2.  But this subroutine works for any value
        ** of nEq including 0.  If nEq==null, this routine is nearly a no-op.
        ** The only thing it does is allocate the pLevel.iMem memory cell and
        ** compute the affinity string.
        **
        ** This routine always allocates at least one memory cell and returns
        ** the index of that memory cell. The code that
        ** calls this routine will use that memory cell to store the termination
        ** key value of the loop.  If one or more IN operators appear, then
        ** this routine allocates an additional nEq memory cells for internal
        ** use.
        **
        ** Before returning, *pzAff is set to point to a buffer containing a
        ** copy of the column affinity string of the index allocated using
        ** sqlite3DbMalloc(). Except, entries in the copy of the string associated
        ** with equality constraints that use NONE affinity are set to
        ** SQLITE_AFF_NONE. This is to deal with SQL such as the following:
        **
        **   CREATE TABLE t1(a TEXT PRIMARY KEY, b);
        **   SELECT ... FROM t1 AS t2, t1 WHERE t1.a = t2.b;
        **
        ** In the example above, the index on t1(a) has TEXT affinity. But since
        ** the right hand side of the equality constraint (t2.b) has NONE affinity,
        ** no conversion should be attempted before using a t2.b value as part of
        ** a key to search the index. Hence the first byte in the returned affinity
        ** string in this example would be set to SQLITE_AFF_NONE.
        */
        static int codeAllEqualityTerms(
        Parse pParse,        /* Parsing context */
        WhereLevel pLevel,   /* Which nested loop of the FROM we are coding */
        WhereClause pWC,     /* The WHERE clause */
        Bitmask notReady,    /* Which parts of FROM have not yet been coded */
        int nExtraReg,       /* Number of extra registers to allocate */
        out StringBuilder pzAff /* OUT: Set to point to affinity string */
        )
        {
            int nEq = (int)pLevel.plan.nEq;   /* The number of == or IN constraints to code */
            Vdbe v = pParse.pVdbe;       /* The vm under construction */
            Index pIdx;                  /* The index being used for this loop */
            int iCur = pLevel.iTabCur;   /* The cursor of the table */
            WhereTerm pTerm;             /* A single constraint term */
            int j;                       /* Loop counter */
            int regBase;                 /* Base register */
            int nReg;                    /* Number of registers to allocate */
            StringBuilder zAff;          /* Affinity string to return */

            /* This module is only called on query plans that use an index. */
            Debug.Assert((pLevel.plan.wsFlags & WHERE_INDEXED) != 0);
            pIdx = pLevel.plan.u.pIdx;

            /* Figure out how many memory cells we will need then allocate them.
            */
            regBase = pParse.nMem + 1;
            nReg = (int)(pLevel.plan.nEq + nExtraReg);
            pParse.nMem += nReg;

            zAff = new StringBuilder(sqlite3IndexAffinityStr(v, pIdx));//sqlite3DbStrDup(pParse.db, sqlite3IndexAffinityStr(v, pIdx));
            //if( null==zAff ){
            //  pParse.db.mallocFailed = 1;
            //}

            /* Evaluate the equality constraints
            */
            Debug.Assert(pIdx.nColumn >= nEq);
            for (j = 0; j < nEq; j++)
            {
                int r1;
                int k = pIdx.aiColumn[j];
                pTerm = FindTerm(pWC, iCur, k, notReady, pLevel.plan.wsFlags, pIdx);
                if (NEVER(pTerm == null))
                    break;
                /* The following true for indices with redundant columns. 
                ** Ex: CREATE INDEX i1 ON t1(a,b,a); SELECT * FROM t1 WHERE a=0 AND b=0; */
                testcase((pTerm.wtFlags & TERM_CODED) != 0);
                testcase(pTerm.wtFlags & TERM_VIRTUAL); /* EV: R-30575-11662 */
                r1 = codeEqualityTerm(pParse, pTerm, pLevel, regBase + j);
                if (r1 != regBase + j)
                {
                    if (nReg == 1)
                    {
                        sqlite3ReleaseTempReg(pParse, regBase);
                        regBase = r1;
                    }
                    else
                    {
                        sqlite3VdbeAddOp2(v, OP_SCopy, r1, regBase + j);
                    }
                }
                testcase(pTerm.eOperator & WO_ISNULL);
                testcase(pTerm.eOperator & WO_IN);
                if ((pTerm.eOperator & (WO_ISNULL | WO_IN)) == 0)
                {
                    Expr pRight = pTerm.pExpr.pRight;
                    sqlite3ExprCodeIsNullJump(v, pRight, regBase + j, pLevel.addrBrk);
                    if (zAff.Length > 0)
                    {
                        if (sqlite3CompareAffinity(pRight, zAff[j]) == SQLITE_AFF_NONE)
                        {
                            zAff[j] = SQLITE_AFF_NONE;
                        }
                        if ((sqlite3ExprNeedsNoAffinityChange(pRight, zAff[j])) != 0)
                        {
                            zAff[j] = SQLITE_AFF_NONE;
                        }
                    }
                }
            }
            pzAff = zAff;
            return regBase;
        }

#if !SQLITE_OMIT_EXPLAIN
        /*
** This routine is a helper for explainIndexRange() below
**
** pStr holds the text of an expression that we are building up one term
** at a time.  This routine adds a new term to the end of the expression.
** Terms are separated by AND so add the "AND" text for second and subsequent
** terms only.
*/
        static void explainAppendTerm(
        StrAccum pStr,              /* The text expression being built */
        int iTerm,                  /* Index of this term.  First is zero */
        string zColumn,             /* Name of the column */
        string zOp                  /* Name of the operator */
        )
        {
            if (iTerm != 0)
                sqlite3StrAccumAppend(pStr, " AND ", 5);
            sqlite3StrAccumAppend(pStr, zColumn, -1);
            sqlite3StrAccumAppend(pStr, zOp, 1);
            sqlite3StrAccumAppend(pStr, "?", 1);
        }

        /*
        ** Argument pLevel describes a strategy for scanning table pTab. This 
        ** function returns a pointer to a string buffer containing a description
        ** of the subset of table rows scanned by the strategy in the form of an
        ** SQL expression. Or, if all rows are scanned, NULL is returned.
        **
        ** For example, if the query:
        **
        **   SELECT * FROM t1 WHERE a=1 AND b>2;
        **
        ** is run and there is an index on (a, b), then this function returns a
        ** string similar to:
        **
        **   "a=? AND b>?"
        **
        ** The returned pointer points to memory obtained from sqlite3DbMalloc().
        ** It is the responsibility of the caller to free the buffer when it is
        ** no longer required.
        */
        static string explainIndexRange(sqlite3 db, WhereLevel pLevel, Table pTab)
        {
            WherePlan pPlan = pLevel.plan;
            Index pIndex = pPlan.u.pIdx;
            uint nEq = pPlan.nEq;
            int i, j;
            Column[] aCol = pTab.aCol;
            int[] aiColumn = pIndex.aiColumn;
            StrAccum txt = new StrAccum(100);

            if (nEq == 0 && (pPlan.wsFlags & (WHERE_BTM_LIMIT | WHERE_TOP_LIMIT)) == 0)
            {
                return null;
            }
            sqlite3StrAccumInit(txt, null, 0, SQLITE_MAX_LENGTH);
            txt.db = db;
            sqlite3StrAccumAppend(txt, " (", 2);
            for (i = 0; i < nEq; i++)
            {
                explainAppendTerm(txt, i, aCol[aiColumn[i]].zName, "=");
            }

            j = i;
            if ((pPlan.wsFlags & WHERE_BTM_LIMIT) != 0)
            {
                explainAppendTerm(txt, i++, aCol[aiColumn[j]].zName, ">");
            }
            if ((pPlan.wsFlags & WHERE_TOP_LIMIT) != 0)
            {
                explainAppendTerm(txt, i, aCol[aiColumn[j]].zName, "<");
            }
            sqlite3StrAccumAppend(txt, ")", 1);
            return sqlite3StrAccumFinish(txt);
        }

        /*
        ** This function is a no-op unless currently processing an EXPLAIN QUERY PLAN
        ** command. If the query being compiled is an EXPLAIN QUERY PLAN, a single
        ** record is added to the output to describe the table scan strategy in 
        ** pLevel.
        */
        static void explainOneScan(
        Parse pParse,                   /* Parse context */
        SrcList pTabList,               /* Table list this loop refers to */
        WhereLevel pLevel,              /* Scan to write OP_Explain opcode for */
        int iLevel,                     /* Value for "level" column of output */
        int iFrom,                      /* Value for "from" column of output */
        u16 wctrlFlags                  /* Flags passed to sqlite3WhereBegin() */
        )
        {
            if (pParse.explain == 2)
            {
                u32 flags = pLevel.plan.wsFlags;
                SrcList_item pItem = pTabList.a[pLevel.iFrom];
                Vdbe v = pParse.pVdbe;       /* VM being constructed */
                sqlite3 db = pParse.db;      /* Database handle */
                StringBuilder zMsg = new StringBuilder(1000); /* Text to add to EQP output */
                sqlite3_int64 nRow;          /* Expected number of rows visited by scan */
                int iId = pParse.iSelectId;  /* Select id (left-most output column) */
                bool isSearch;               /* True for a SEARCH. False for SCAN. */

                if ((flags & WHERE_MULTI_OR) != 0 || (wctrlFlags & WHERE_ONETABLE_ONLY) != 0)
                    return;

                isSearch = (pLevel.plan.nEq > 0)
                || (flags & (WHERE_BTM_LIMIT | WHERE_TOP_LIMIT)) != 0
                || (wctrlFlags & (WHERE_ORDERBY_MIN | WHERE_ORDERBY_MAX)) != 0;

                zMsg.Append(sqlite3MPrintf(db, "%s", isSearch ? "SEARCH" : "SCAN"));
                if (pItem.pSelect != null)
                {
                    zMsg.Append(sqlite3MAppendf(db, null, " SUBQUERY %d", pItem.iSelectId));
                }
                else
                {
                    zMsg.Append(sqlite3MAppendf(db, null, " TABLE %s", pItem.zName));
                }

                if (pItem.zAlias != null)
                {
                    zMsg.Append(sqlite3MAppendf(db, null, " AS %s", pItem.zAlias));
                }
                if ((flags & WHERE_INDEXED) != 0)
                {
                    string zWhere = explainIndexRange(db, pLevel, pItem.pTab);
                    zMsg.Append(sqlite3MAppendf(db, null, " USING %s%sINDEX%s%s%s",
                    ((flags & WHERE_TEMP_INDEX) != 0 ? "AUTOMATIC " : ""),
                    ((flags & WHERE_IDX_ONLY) != 0 ? "COVERING " : ""),
                    ((flags & WHERE_TEMP_INDEX) != 0 ? "" : " "),
                    ((flags & WHERE_TEMP_INDEX) != 0 ? "" : pLevel.plan.u.pIdx.zName),
                    zWhere != null ? zWhere : ""
                    ));
                    sqlite3DbFree(db, ref zWhere);
                }
                else if ((flags & (WHERE_ROWID_EQ | WHERE_ROWID_RANGE)) != 0)
                {
                    zMsg.Append(" USING INTEGER PRIMARY KEY");

                    if ((flags & WHERE_ROWID_EQ) != 0)
                    {
                        zMsg.Append(" (rowid=?)");
                    }
                    else if ((flags & WHERE_BOTH_LIMIT) == WHERE_BOTH_LIMIT)
                    {
                        zMsg.Append(" (rowid>? AND rowid<?)");
                    }
                    else if ((flags & WHERE_BTM_LIMIT) != 0)
                    {
                        zMsg.Append(" (rowid>?)");
                    }
                    else if ((flags & WHERE_TOP_LIMIT) != 0)
                    {
                        zMsg.Append(" (rowid<?)");
                    }
                }
#if !SQLITE_OMIT_VIRTUALTABLE
                else if ((flags & WHERE_VIRTUALTABLE) != 0)
                {
                    sqlite3_index_info pVtabIdx = pLevel.plan.u.pVtabIdx;
                    zMsg.Append(sqlite3MAppendf(db, null, " VIRTUAL TABLE INDEX %d:%s",
                    pVtabIdx.idxNum, pVtabIdx.idxStr));
                }
#endif
                if ((wctrlFlags & (WHERE_ORDERBY_MIN | WHERE_ORDERBY_MAX)) != 0)
                {
                    testcase(wctrlFlags & WHERE_ORDERBY_MIN);
                    nRow = 1;
                }
                else
                {
                    nRow = (sqlite3_int64)pLevel.plan.nRow;
                }
                zMsg.Append(sqlite3MAppendf(db, null, " (~%lld rows)", nRow));
                sqlite3VdbeAddOp4(v, OP_Explain, iId, iLevel, iFrom, zMsg, P4_DYNAMIC);
            }
        }
#else
//# define explainOneScan(u,v,w,x,y,z)
static void explainOneScan(  Parse u,  SrcList v,  WhereLevel w,  int x,  int y,  u16 z){}
#endif //* SQLITE_OMIT_EXPLAIN */


        /*
** Generate code for the start of the iLevel-th loop in the WHERE clause
** implementation described by pWInfo.
*/
        static Bitmask codeOneLoopStart(
        WhereInfo pWInfo,     /* Complete information about the WHERE clause */
        int iLevel,           /* Which level of pWInfo.a[] should be coded */
        u16 wctrlFlags,       /* One of the WHERE_* flags defined in sqliteInt.h */
        Bitmask notReady      /* Which tables are currently available */
        )
        {
            int j, k;                 /* Loop counters */
            int iCur;                 /* The VDBE cursor for the table */
            int addrNxt;              /* Where to jump to continue with the next IN case */
            int omitTable;            /* True if we use the index only */
            int bRev;                 /* True if we need to scan in reverse order */
            WhereLevel pLevel;        /* The where level to be coded */
            WhereClause pWC;          /* Decomposition of the entire WHERE clause */
            WhereTerm pTerm;          /* A WHERE clause term */
            Parse pParse;             /* Parsing context */
            Vdbe v;                   /* The prepared stmt under constructions */
            SrcList_item pTabItem;    /* FROM clause term being coded */
            int addrBrk;              /* Jump here to break out of the loop */
            int addrCont;             /* Jump here to continue with next cycle */
            int iRowidReg = 0;        /* Rowid is stored in this register, if not zero */
            int iReleaseReg = 0;      /* Temp register to free before returning */

            pParse = pWInfo.pParse;
            v = pParse.pVdbe;
            pWC = pWInfo.pWC;
            pLevel = pWInfo.a[iLevel];
            pTabItem = pWInfo.pTabList.a[pLevel.iFrom];
            iCur = pTabItem.iCursor;
            bRev = (pLevel.plan.wsFlags & WHERE_REVERSE) != 0 ? 1 : 0;
            omitTable = ((pLevel.plan.wsFlags & WHERE_IDX_ONLY) != 0
            && (wctrlFlags & WHERE_FORCE_TABLE) == 0) ? 1 : 0;

            /* Create labels for the "break" and "continue" instructions
            ** for the current loop.  Jump to addrBrk to break out of a loop.
            ** Jump to cont to go immediately to the next iteration of the
            ** loop.
            **
            ** When there is an IN operator, we also have a "addrNxt" label that
            ** means to continue with the next IN value combination.  When
            ** there are no IN operators in the constraints, the "addrNxt" label
            ** is the same as "addrBrk".
            */
            addrBrk = pLevel.addrBrk = pLevel.addrNxt = sqlite3VdbeMakeLabel(v);
            addrCont = pLevel.addrCont = sqlite3VdbeMakeLabel(v);

            /* If this is the right table of a LEFT OUTER JOIN, allocate and
            ** initialize a memory cell that records if this table matches any
            ** row of the left table of the join.
            */
            if (pLevel.iFrom > 0 && (pTabItem.jointype & JT_LEFT) != 0)// Check value of pTabItem[0].jointype
            {
                pLevel.iLeftJoin = ++pParse.nMem;
                sqlite3VdbeAddOp2(v, OP_Integer, 0, pLevel.iLeftJoin);
#if SQLITE_DEBUG
        VdbeComment( v, "init LEFT JOIN no-match flag" );
#endif
            }

#if  !SQLITE_OMIT_VIRTUALTABLE
            if ((pLevel.plan.wsFlags & WHERE_VIRTUALTABLE) != 0)
            {
                /* Case 0:  The table is a virtual-table.  Use the VFilter and VNext
                **          to access the data.
                */
                int iReg;   /* P3 Value for OP_VFilter */
                sqlite3_index_info pVtabIdx = pLevel.plan.u.pVtabIdx;
                int nConstraint = pVtabIdx.nConstraint;
                sqlite3_index_constraint_usage[] aUsage = pVtabIdx.aConstraintUsage;
                sqlite3_index_constraint[] aConstraint = pVtabIdx.aConstraint;

                sqlite3ExprCachePush(pParse);
                iReg = sqlite3GetTempRange(pParse, nConstraint + 2);
                for (j = 1; j <= nConstraint; j++)
                {
                    for (k = 0; k < nConstraint; k++)
                    {
                        if (aUsage[k].argvIndex == j)
                        {
                            int iTerm = aConstraint[k].iTermOffset;
                            sqlite3ExprCode(pParse, pWC.a[iTerm].pExpr.pRight, iReg + j + 1);
                            break;
                        }
                    }
                    if (k == nConstraint)
                        break;
                }
                sqlite3VdbeAddOp2(v, OP_Integer, pVtabIdx.idxNum, iReg);
                sqlite3VdbeAddOp2(v, OP_Integer, j - 1, iReg + 1);
                sqlite3VdbeAddOp4(v, OP_VFilter, iCur, addrBrk, iReg, pVtabIdx.idxStr,
                pVtabIdx.needToFreeIdxStr != 0 ? P4_MPRINTF : P4_STATIC);
                pVtabIdx.needToFreeIdxStr = 0;
                for (j = 0; j < nConstraint; j++)
                {
                    if (aUsage[j].omit != false)
                    {
                        int iTerm = aConstraint[j].iTermOffset;
                        disableTerm(pLevel, pWC.a[iTerm]);
                    }
                }
                pLevel.op = OP_VNext;
                pLevel.p1 = iCur;
                pLevel.p2 = sqlite3VdbeCurrentAddr(v);
                sqlite3ReleaseTempRange(pParse, iReg, nConstraint + 2);
                sqlite3ExprCachePop(pParse, 1);
            }
            else
#endif //* SQLITE_OMIT_VIRTUALTABLE */

                if ((pLevel.plan.wsFlags & WHERE_ROWID_EQ) != 0)
                {
                    /* Case 1:  We can directly reference a single row using an
                    **          equality comparison against the ROWID field.  Or
                    **          we reference multiple rows using a "rowid IN (...)"
                    **          construct.
                    */
                    iReleaseReg = sqlite3GetTempReg(pParse);
                    pTerm = FindTerm(pWC, iCur, -1, notReady, WO_EQ | WO_IN, null);
                    Debug.Assert(pTerm != null);
                    Debug.Assert(pTerm.pExpr != null);
                    Debug.Assert(pTerm.leftCursor == iCur);
                    Debug.Assert(omitTable == 0);
                    testcase(pTerm.wtFlags & TERM_VIRTUAL); /* EV: R-30575-11662 */
                    iRowidReg = codeEqualityTerm(pParse, pTerm, pLevel, iReleaseReg);
                    addrNxt = pLevel.addrNxt;
                    sqlite3VdbeAddOp2(v, OP_MustBeInt, iRowidReg, addrNxt);
                    sqlite3VdbeAddOp3(v, OP_NotExists, iCur, addrNxt, iRowidReg);
                    sqlite3ExprCacheStore(pParse, iCur, -1, iRowidReg);
#if SQLITE_DEBUG
          VdbeComment( v, "pk" );
#endif
                    pLevel.op = OP_Noop;
                }
                else if ((pLevel.plan.wsFlags & WHERE_ROWID_RANGE) != 0)
                {
                    /* Case 2:  We have an inequality comparison against the ROWID field.
                    */
                    int testOp = OP_Noop;
                    int start;
                    int memEndValue = 0;
                    WhereTerm pStart, pEnd;

                    Debug.Assert(omitTable == 0);
                    pStart = FindTerm(pWC, iCur, -1, notReady, WO_GT | WO_GE, null);
                    pEnd = FindTerm(pWC, iCur, -1, notReady, WO_LT | WO_LE, null);
                    if (bRev != 0)
                    {
                        pTerm = pStart;
                        pStart = pEnd;
                        pEnd = pTerm;
                    }
                    if (pStart != null)
                    {
                        Expr pX;             /* The expression that defines the start bound */
                        int r1, rTemp = 0;        /* Registers for holding the start boundary */

                        /* The following constant maps TK_xx codes into corresponding
                        ** seek opcodes.  It depends on a particular ordering of TK_xx
                        */
                        u8[] aMoveOp = new u8[]{
/* TK_GT */  OP_SeekGt,
/* TK_LE */  OP_SeekLe,
/* TK_LT */  OP_SeekLt,
/* TK_GE */  OP_SeekGe
};
                        Debug.Assert(TK_LE == TK_GT + 1);      /* Make sure the ordering.. */
                        Debug.Assert(TK_LT == TK_GT + 2);      /*  ... of the TK_xx values... */
                        Debug.Assert(TK_GE == TK_GT + 3);      /*  ... is correcct. */

                        testcase(pStart.wtFlags & TERM_VIRTUAL); /* EV: R-30575-11662 */
                        pX = pStart.pExpr;
                        Debug.Assert(pX != null);
                        Debug.Assert(pStart.leftCursor == iCur);
                        r1 = sqlite3ExprCodeTemp(pParse, pX.pRight, ref rTemp);
                        sqlite3VdbeAddOp3(v, aMoveOp[pX.op - TK_GT], iCur, addrBrk, r1);
#if SQLITE_DEBUG
            VdbeComment( v, "pk" );
#endif
                        sqlite3ExprCacheAffinityChange(pParse, r1, 1);
                        sqlite3ReleaseTempReg(pParse, rTemp);
                        disableTerm(pLevel, pStart);
                    }
                    else
                    {
                        sqlite3VdbeAddOp2(v, bRev != 0 ? OP_Last : OP_Rewind, iCur, addrBrk);
                    }
                    if (pEnd != null)
                    {
                        Expr pX;
                        pX = pEnd.pExpr;
                        Debug.Assert(pX != null);
                        Debug.Assert(pEnd.leftCursor == iCur);
                        testcase(pEnd.wtFlags & TERM_VIRTUAL); /* EV: R-30575-11662 */
                        memEndValue = ++pParse.nMem;
                        sqlite3ExprCode(pParse, pX.pRight, memEndValue);
                        if (pX.op == TK_LT || pX.op == TK_GT)
                        {
                            testOp = bRev != 0 ? OP_Le : OP_Ge;
                        }
                        else
                        {
                            testOp = bRev != 0 ? OP_Lt : OP_Gt;
                        }
                        disableTerm(pLevel, pEnd);
                    }
                    start = sqlite3VdbeCurrentAddr(v);
                    pLevel.op = (u8)(bRev != 0 ? OP_Prev : OP_Next);
                    pLevel.p1 = iCur;
                    pLevel.p2 = start;
                    if (pStart == null && pEnd == null)
                    {
                        pLevel.p5 = SQLITE_STMTSTATUS_FULLSCAN_STEP;
                    }
                    else
                    {
                        Debug.Assert(pLevel.p5 == 0);
                    }
                    if (testOp != OP_Noop)
                    {
                        iRowidReg = iReleaseReg = sqlite3GetTempReg(pParse);
                        sqlite3VdbeAddOp2(v, OP_Rowid, iCur, iRowidReg);
                        sqlite3ExprCacheStore(pParse, iCur, -1, iRowidReg);
                        sqlite3VdbeAddOp3(v, testOp, memEndValue, addrBrk, iRowidReg);
                        sqlite3VdbeChangeP5(v, SQLITE_AFF_NUMERIC | SQLITE_JUMPIFNULL);
                    }
                }
                else if ((pLevel.plan.wsFlags & (WHERE_COLUMN_RANGE | WHERE_COLUMN_EQ)) != 0)
                {
                    /* Case 3: A scan using an index.
                    **
                    **         The WHERE clause may contain zero or more equality
                    **         terms ("==" or "IN" operators) that refer to the N
                    **         left-most columns of the index. It may also contain
                    **         inequality constraints (>, <, >= or <=) on the indexed
                    **         column that immediately follows the N equalities. Only
                    **         the right-most column can be an inequality - the rest must
                    **         use the "==" and "IN" operators. For example, if the
                    **         index is on (x,y,z), then the following clauses are all
                    **         optimized:
                    **
                    **            x=5
                    **            x=5 AND y=10
                    **            x=5 AND y<10
                    **            x=5 AND y>5 AND y<10
                    **            x=5 AND y=5 AND z<=10
                    **
                    **         The z<10 term of the following cannot be used, only
                    **         the x=5 term:
                    **
                    **            x=5 AND z<10
                    **
                    **         N may be zero if there are inequality constraints.
                    **         If there are no inequality constraints, then N is at
                    **         least one.
                    **
                    **         This case is also used when there are no WHERE clause
                    **         constraints but an index is selected anyway, in order
                    **         to force the output order to conform to an ORDER BY.
                    */
                    u8[] aStartOp = new u8[]  {
0,
0,
OP_Rewind,           /* 2: (!start_constraints && startEq &&  !bRev) */
OP_Last,             /* 3: (!start_constraints && startEq &&   bRev) */
OP_SeekGt,           /* 4: (start_constraints  && !startEq && !bRev) */
OP_SeekLt,           /* 5: (start_constraints  && !startEq &&  bRev) */
OP_SeekGe,           /* 6: (start_constraints  &&  startEq && !bRev) */
OP_SeekLe            /* 7: (start_constraints  &&  startEq &&  bRev) */
};
                    u8[] aEndOp = new u8[]  {
OP_Noop,             /* 0: (!end_constraints) */
OP_IdxGE,            /* 1: (end_constraints && !bRev) */
OP_IdxLT             /* 2: (end_constraints && bRev) */
};
                    int nEq = (int)pLevel.plan.nEq; /* Number of == or IN terms */
                    int isMinQuery = 0;          /* If this is an optimized SELECT min(x).. */
                    int regBase;                 /* Base register holding constraint values */
                    int r1;                      /* Temp register */
                    WhereTerm pRangeStart = null;  /* Inequality constraint at range start */
                    WhereTerm pRangeEnd = null;    /* Inequality constraint at range end */
                    int startEq;                   /* True if range start uses ==, >= or <= */
                    int endEq;                     /* True if range end uses ==, >= or <= */
                    int start_constraints;         /* Start of range is constrained */
                    int nConstraint;               /* Number of constraint terms */
                    Index pIdx;                    /* The index we will be using */
                    int iIdxCur;                   /* The VDBE cursor for the index */
                    int nExtraReg = 0;             /* Number of extra registers needed */
                    int op;                        /* Instruction opcode */
                    StringBuilder zStartAff = new StringBuilder("");
                    ;/* Affinity for start of range constraint */
                    StringBuilder zEndAff;         /* Affinity for end of range constraint */

                    pIdx = pLevel.plan.u.pIdx;
                    iIdxCur = pLevel.iIdxCur;
                    k = pIdx.aiColumn[nEq];        /* Column for inequality constraints */

                    /* If this loop satisfies a sort order (pOrderBy) request that
                    ** was pDebug.Assed to this function to implement a "SELECT min(x) ..."
                    ** query, then the caller will only allow the loop to run for
                    ** a single iteration. This means that the first row returned
                    ** should not have a NULL value stored in 'x'. If column 'x' is
                    ** the first one after the nEq equality constraints in the index,
                    ** this requires some special handling.
                    */
                    if ((wctrlFlags & WHERE_ORDERBY_MIN) != 0
                    && ((pLevel.plan.wsFlags & WHERE_ORDERBY) != 0)
                    && (pIdx.nColumn > nEq)
                    )
                    {
                        /* Debug.Assert( pOrderBy.nExpr==1 ); */
                        /* Debug.Assert( pOrderBy.a[0].pExpr.iColumn==pIdx.aiColumn[nEq] ); */
                        isMinQuery = 1;
                        nExtraReg = 1;
                    }

                    /* Find any inequality constraint terms for the start and end
                    ** of the range.
                    */
                    if ((pLevel.plan.wsFlags & WHERE_TOP_LIMIT) != 0)
                    {
                        pRangeEnd = FindTerm(pWC, iCur, k, notReady, (WO_LT | WO_LE), pIdx);
                        nExtraReg = 1;
                    }
                    if ((pLevel.plan.wsFlags & WHERE_BTM_LIMIT) != 0)
                    {
                        pRangeStart = FindTerm(pWC, iCur, k, notReady, (WO_GT | WO_GE), pIdx);
                        nExtraReg = 1;
                    }

                    /* Generate code to evaluate all constraint terms using == or IN
                    ** and store the values of those terms in an array of registers
                    ** starting at regBase.
                    */
                    regBase = codeAllEqualityTerms(
                    pParse, pLevel, pWC, notReady, nExtraReg, out zStartAff
                    );
                    zEndAff = new StringBuilder(zStartAff.ToString());//sqlite3DbStrDup(pParse.db, zStartAff);
                    addrNxt = pLevel.addrNxt;

                    /* If we are doing a reverse order scan on an ascending index, or
                    ** a forward order scan on a descending index, interchange the
                    ** start and end terms (pRangeStart and pRangeEnd).
                    */
                    if (nEq < pIdx.nColumn && bRev == (pIdx.aSortOrder[nEq] == SQLITE_SO_ASC ? 1 : 0))
                    {
                        SWAP(ref pRangeEnd, ref pRangeStart);
                    }

                    testcase(pRangeStart != null && (pRangeStart.eOperator & WO_LE) != 0);
                    testcase(pRangeStart != null && (pRangeStart.eOperator & WO_GE) != 0);
                    testcase(pRangeEnd != null && (pRangeEnd.eOperator & WO_LE) != 0);
                    testcase(pRangeEnd != null && (pRangeEnd.eOperator & WO_GE) != 0);
                    startEq = (null == pRangeStart || (pRangeStart.eOperator & (WO_LE | WO_GE)) != 0) ? 1 : 0;
                    endEq = (null == pRangeEnd || (pRangeEnd.eOperator & (WO_LE | WO_GE)) != 0) ? 1 : 0;
                    start_constraints = (pRangeStart != null || nEq > 0) ? 1 : 0;

                    /* Seek the index cursor to the start of the range. */
                    nConstraint = nEq;
                    if (pRangeStart != null)
                    {
                        Expr pRight = pRangeStart.pExpr.pRight;
                        sqlite3ExprCode(pParse, pRight, regBase + nEq);
                        if ((pRangeStart.wtFlags & TERM_VNULL) == 0)
                        {
                            sqlite3ExprCodeIsNullJump(v, pRight, regBase + nEq, addrNxt);
                        }
                        if (zStartAff.Length != 0)
                        {
                            if (sqlite3CompareAffinity(pRight, zStartAff[nEq]) == SQLITE_AFF_NONE)
                            {
                                /* Since the comparison is to be performed with no conversions
                                ** applied to the operands, set the affinity to apply to pRight to 
                                ** SQLITE_AFF_NONE.  */
                                zStartAff[nEq] = SQLITE_AFF_NONE;
                            }
                            if ((sqlite3ExprNeedsNoAffinityChange(pRight, zStartAff[nEq])) != 0)
                            {
                                zStartAff[nEq] = SQLITE_AFF_NONE;
                            }
                        }
                        nConstraint++;
                        testcase(pRangeStart.wtFlags & TERM_VIRTUAL); /* EV: R-30575-11662 */
                    }
                    else if (isMinQuery != 0)
                    {
                        sqlite3VdbeAddOp2(v, OP_Null, 0, regBase + nEq);
                        nConstraint++;
                        startEq = 0;
                        start_constraints = 1;
                    }
                    codeApplyAffinity(pParse, regBase, nConstraint, zStartAff.ToString());
                    op = aStartOp[(start_constraints << 2) + (startEq << 1) + bRev];
                    Debug.Assert(op != 0);
                    testcase(op == OP_Rewind);
                    testcase(op == OP_Last);
                    testcase(op == OP_SeekGt);
                    testcase(op == OP_SeekGe);
                    testcase(op == OP_SeekLe);
                    testcase(op == OP_SeekLt);
                    sqlite3VdbeAddOp4Int(v, op, iIdxCur, addrNxt, regBase, nConstraint);

                    /* Load the value for the inequality constraint at the end of the
                    ** range (if any).
                    */
                    nConstraint = nEq;
                    if (pRangeEnd != null)
                    {
                        Expr pRight = pRangeEnd.pExpr.pRight;
                        sqlite3ExprCacheRemove(pParse, regBase + nEq, 1);
                        sqlite3ExprCode(pParse, pRight, regBase + nEq);
                        if ((pRangeEnd.wtFlags & TERM_VNULL) == 0)
                        {
                            sqlite3ExprCodeIsNullJump(v, pRight, regBase + nEq, addrNxt);
                        }
                        if (zEndAff.Length > 0)
                        {
                            if (sqlite3CompareAffinity(pRight, zEndAff[nEq]) == SQLITE_AFF_NONE)
                            {
                                /* Since the comparison is to be performed with no conversions
                                ** applied to the operands, set the affinity to apply to pRight to 
                                ** SQLITE_AFF_NONE.  */
                                zEndAff[nEq] = SQLITE_AFF_NONE;
                            }
                            if ((sqlite3ExprNeedsNoAffinityChange(pRight, zEndAff[nEq])) != 0)
                            {
                                zEndAff[nEq] = SQLITE_AFF_NONE;
                            }
                        }
                        codeApplyAffinity(pParse, regBase, nEq + 1, zEndAff.ToString());
                        nConstraint++;
                        testcase(pRangeEnd.wtFlags & TERM_VIRTUAL); /* EV: R-30575-11662 */
                    }
                    sqlite3DbFree(pParse.db, ref zStartAff);
                    sqlite3DbFree(pParse.db, ref zEndAff);

                    /* Top of the loop body */
                    pLevel.p2 = sqlite3VdbeCurrentAddr(v);

                    /* Check if the index cursor is past the end of the range. */
                    op = aEndOp[((pRangeEnd != null || nEq != 0) ? 1 : 0) * (1 + bRev)];
                    testcase(op == OP_Noop);
                    testcase(op == OP_IdxGE);
                    testcase(op == OP_IdxLT);
                    if (op != OP_Noop)
                    {
                        sqlite3VdbeAddOp4Int(v, op, iIdxCur, addrNxt, regBase, nConstraint);
                        sqlite3VdbeChangeP5(v, (u8)(endEq != bRev ? 1 : 0));
                    }

                    /* If there are inequality constraints, check that the value
                    ** of the table column that the inequality contrains is not NULL.
                    ** If it is, jump to the next iteration of the loop.
                    */
                    r1 = sqlite3GetTempReg(pParse);
                    testcase(pLevel.plan.wsFlags & WHERE_BTM_LIMIT);
                    testcase(pLevel.plan.wsFlags & WHERE_TOP_LIMIT);
                    if ((pLevel.plan.wsFlags & (WHERE_BTM_LIMIT | WHERE_TOP_LIMIT)) != 0)
                    {
                        sqlite3VdbeAddOp3(v, OP_Column, iIdxCur, nEq, r1);
                        sqlite3VdbeAddOp2(v, OP_IsNull, r1, addrCont);
                    }
                    sqlite3ReleaseTempReg(pParse, r1);

                    /* Seek the table cursor, if required */
                    disableTerm(pLevel, pRangeStart);
                    disableTerm(pLevel, pRangeEnd);
                    if (0 == omitTable)
                    {
                        iRowidReg = iReleaseReg = sqlite3GetTempReg(pParse);
                        sqlite3VdbeAddOp2(v, OP_IdxRowid, iIdxCur, iRowidReg);
                        sqlite3ExprCacheStore(pParse, iCur, -1, iRowidReg);
                        sqlite3VdbeAddOp2(v, OP_Seek, iCur, iRowidReg);  /* Deferred seek */
                    }

                    /* Record the instruction used to terminate the loop. Disable
                    ** WHERE clause terms made redundant by the index range scan.
                    */
                    if ((pLevel.plan.wsFlags & WHERE_UNIQUE) != 0)
                    {
                        pLevel.op = OP_Noop;
                    }
                    else if (bRev != 0)
                    {
                        pLevel.op = OP_Prev;
                    }
                    else
                    {
                        pLevel.op = OP_Next;
                    }
                    pLevel.p1 = iIdxCur;
                }
                else

#if  !SQLITE_OMIT_OR_OPTIMIZATION
                    if ((pLevel.plan.wsFlags & WHERE_MULTI_OR) != 0)
                    {
                        /* Case 4:  Two or more separately indexed terms connected by OR
                        **
                        ** Example:
                        **
                        **   CREATE TABLE t1(a,b,c,d);
                        **   CREATE INDEX i1 ON t1(a);
                        **   CREATE INDEX i2 ON t1(b);
                        **   CREATE INDEX i3 ON t1(c);
                        **
                        **   SELECT * FROM t1 WHERE a=5 OR b=7 OR (c=11 AND d=13)
                        **
                        ** In the example, there are three indexed terms connected by OR.
                        ** The top of the loop looks like this:
                        **
                        **          Null       1                # Zero the rowset in reg 1
                        **
                        ** Then, for each indexed term, the following. The arguments to
                        ** RowSetTest are such that the rowid of the current row is inserted
                        ** into the RowSet. If it is already present, control skips the
                        ** Gosub opcode and jumps straight to the code generated by WhereEnd().
                        **
                        **        sqlite3WhereBegin(<term>)
                        **          RowSetTest                  # Insert rowid into rowset
                        **          Gosub      2 A
                        **        sqlite3WhereEnd()
                        **
                        ** Following the above, code to terminate the loop. Label A, the target
                        ** of the Gosub above, jumps to the instruction right after the Goto.
                        **
                        **          Null       1                # Zero the rowset in reg 1
                        **          Goto       B                # The loop is finished.
                        **
                        **       A: <loop body>                 # Return data, whatever.
                        **
                        **          Return     2                # Jump back to the Gosub
                        **
                        **       B: <after the loop>
                        **
                        */
                        WhereClause pOrWc;    /* The OR-clause broken out into subterms */
                        SrcList pOrTab;       /* Shortened table list or OR-clause generation */

                        int regReturn = ++pParse.nMem;            /* Register used with OP_Gosub */
                        int regRowset = 0;                        /* Register for RowSet object */
                        int regRowid = 0;                         /* Register holding rowid */
                        int iLoopBody = sqlite3VdbeMakeLabel(v);/* Start of loop body */
                        int iRetInit;                             /* Address of regReturn init */
                        int untestedTerms = 0;                    /* Some terms not completely tested */
                        int ii;
                        pTerm = pLevel.plan.u.pTerm;
                        Debug.Assert(pTerm != null);
                        Debug.Assert(pTerm.eOperator == WO_OR);
                        Debug.Assert((pTerm.wtFlags & TERM_ORINFO) != 0);
                        pOrWc = pTerm.u.pOrInfo.wc;
                        pLevel.op = OP_Return;
                        pLevel.p1 = regReturn;

                        /* Set up a new SrcList in pOrTab containing the table being scanned
                        ** by this loop in the a[0] slot and all notReady tables in a[1..] slots.
                        ** This becomes the SrcList in the recursive call to sqlite3WhereBegin().
                        */
                        if (pWInfo.nLevel > 1)
                        {
                            int nNotReady;                 /* The number of notReady tables */
                            SrcList_item[] origSrc;         /* Original list of tables */
                            nNotReady = pWInfo.nLevel - iLevel - 1;
                            //sqlite3StackAllocRaw(pParse.db,
                            //sizeof(*pOrTab)+ nNotReady*sizeof(pOrTab.a[0]));
                            pOrTab = new SrcList();
                            pOrTab.a = new SrcList_item[nNotReady + 1];
                            //if( pOrTab==0 ) return notReady;
                            pOrTab.nAlloc = (i16)(nNotReady + 1);
                            pOrTab.nSrc = pOrTab.nAlloc;
                            pOrTab.a[0] = pTabItem;//memcpy(pOrTab.a, pTabItem, sizeof(*pTabItem));
                            origSrc = pWInfo.pTabList.a;
                            for (k = 1; k <= nNotReady; k++)
                            {
                                pOrTab.a[k] = origSrc[pWInfo.a[iLevel + k].iFrom];// memcpy(&pOrTab.a[k], &origSrc[pLevel[k].iFrom], sizeof(pOrTab.a[k]));
                            }
                        }
                        else
                        {
                            pOrTab = pWInfo.pTabList;
                        }

                        /* Initialize the rowset register to contain NULL. An SQL NULL is
                        ** equivalent to an empty rowset.
                        **
                        ** Also initialize regReturn to contain the address of the instruction
                        ** immediately following the OP_Return at the bottom of the loop. This
                        ** is required in a few obscure LEFT JOIN cases where control jumps
                        ** over the top of the loop into the body of it. In this case the
                        ** correct response for the end-of-loop code (the OP_Return) is to
                        ** fall through to the next instruction, just as an OP_Next does if
                        ** called on an uninitialized cursor.
                        */
                        if ((wctrlFlags & WHERE_DUPLICATES_OK) == 0)
                        {
                            regRowset = ++pParse.nMem;
                            regRowid = ++pParse.nMem;
                            sqlite3VdbeAddOp2(v, OP_Null, 0, regRowset);
                        }
                        iRetInit = sqlite3VdbeAddOp2(v, OP_Integer, 0, regReturn);

                        for (ii = 0; ii < pOrWc.nTerm; ii++)
                        {
                            WhereTerm pOrTerm = pOrWc.a[ii];
                            if (pOrTerm.leftCursor == iCur || pOrTerm.eOperator == WO_AND)
                            {
                                WhereInfo pSubWInfo;          /* Info for single OR-term scan */

                                /* Loop through table entries that match term pOrTerm. */
                                ExprList elDummy = null;
                                pSubWInfo = sqlite3WhereBegin(pParse, pOrTab, pOrTerm.pExpr, ref elDummy,
                                    WHERE_OMIT_OPEN | WHERE_OMIT_CLOSE |
                                    WHERE_FORCE_TABLE | WHERE_ONETABLE_ONLY);
                                if (pSubWInfo != null)
                                {
                                    explainOneScan(
                                    pParse, pOrTab, pSubWInfo.a[0], iLevel, pLevel.iFrom, 0
                                    );
                                    if ((wctrlFlags & WHERE_DUPLICATES_OK) == 0)
                                    {
                                        int iSet = ((ii == pOrWc.nTerm - 1) ? -1 : ii);
                                        int r;
                                        r = sqlite3ExprCodeGetColumn(pParse, pTabItem.pTab, -1, iCur,
                                        regRowid);
                                        sqlite3VdbeAddOp4Int(v, OP_RowSetTest, regRowset,
                                                       sqlite3VdbeCurrentAddr(v) + 2, r, iSet);
                                    }
                                    sqlite3VdbeAddOp2(v, OP_Gosub, regReturn, iLoopBody);

                                    /* The pSubWInfo.untestedTerms flag means that this OR term
                                    ** contained one or more AND term from a notReady table.  The
                                    ** terms from the notReady table could not be tested and will
                                    ** need to be tested later.
                                    */
                                    if (pSubWInfo.untestedTerms != 0)
                                        untestedTerms = 1;

                                    /* Finish the loop through table entries that match term pOrTerm. */
                                    sqlite3WhereEnd(pSubWInfo);
                                }
                            }
                        }
                        sqlite3VdbeChangeP1(v, iRetInit, sqlite3VdbeCurrentAddr(v));
                        sqlite3VdbeAddOp2(v, OP_Goto, 0, pLevel.addrBrk);
                        sqlite3VdbeResolveLabel(v, iLoopBody);

                        if (pWInfo.nLevel > 1)
                            sqlite3DbFree(pParse.db, ref pOrTab);//sqlite3DbFree(pParse.db, pOrTab)
                        if (0 == untestedTerms)
                            disableTerm(pLevel, pTerm);
                    }
                    else
#endif //* SQLITE_OMIT_OR_OPTIMIZATION */

                    {
                        /* Case 5:  There is no usable index.  We must do a complete
                        **          scan of the entire table.
                        */
                        u8[] aStep = new u8[] { OP_Next, OP_Prev };
                        u8[] aStart = new u8[] { OP_Rewind, OP_Last };
                        Debug.Assert(bRev == 0 || bRev == 1);
                        Debug.Assert(omitTable == 0);
                        pLevel.op = aStep[bRev];
                        pLevel.p1 = iCur;
                        pLevel.p2 = 1 + sqlite3VdbeAddOp2(v, aStart[bRev], iCur, addrBrk);
                        pLevel.p5 = SQLITE_STMTSTATUS_FULLSCAN_STEP;
                    }
            notReady &= ~getMask(pWC.pMaskSet, iCur);

            /* Insert code to test every subexpression that can be completely
            ** computed using the current set of tables.
            **
            ** IMPLEMENTATION-OF: R-49525-50935 Terms that cannot be satisfied through
            ** the use of indices become tests that are evaluated against each row of
            ** the relevant input tables.
            */
            for (j = pWC.nTerm; j > 0; j--)//, pTerm++)
            {
                pTerm = pWC.a[pWC.nTerm - j];
                Expr pE;
                testcase(pTerm.wtFlags & TERM_VIRTUAL); /* IMP: R-30575-11662 */
                testcase(pTerm.wtFlags & TERM_CODED);
                if ((pTerm.wtFlags & (TERM_VIRTUAL | TERM_CODED)) != 0)
                    continue;
                if ((pTerm.prereqAll & notReady) != 0)
                {
                    testcase(pWInfo.untestedTerms == 0
                    && (pWInfo.wctrlFlags & WHERE_ONETABLE_ONLY) != 0);
                    pWInfo.untestedTerms = 1;
                    continue;
                }
                pE = pTerm.pExpr;
                Debug.Assert(pE != null);
                if (pLevel.iLeftJoin != 0 && !((pE.flags & EP_FromJoin) == EP_FromJoin))// !ExprHasProperty(pE, EP_FromJoin) ){
                {
                    continue;
                }
                sqlite3ExprIfFalse(pParse, pE, addrCont, SQLITE_JUMPIFNULL);
                pTerm.wtFlags |= TERM_CODED;
            }

            /* For a LEFT OUTER JOIN, generate code that will record the fact that
            ** at least one row of the right table has matched the left table.
            */
            if (pLevel.iLeftJoin != 0)
            {
                pLevel.addrFirst = sqlite3VdbeCurrentAddr(v);
                sqlite3VdbeAddOp2(v, OP_Integer, 1, pLevel.iLeftJoin);
#if SQLITE_DEBUG
        VdbeComment( v, "record LEFT JOIN hit" );
#endif
                sqlite3ExprCacheClear(pParse);
                for (j = 0; j < pWC.nTerm; j++)//, pTerm++)
                {
                    pTerm = pWC.a[j];
                    testcase(pTerm.wtFlags & TERM_VIRTUAL);  /* IMP: R-30575-11662 */
                    testcase(pTerm.wtFlags & TERM_CODED);
                    if ((pTerm.wtFlags & (TERM_VIRTUAL | TERM_CODED)) != 0)
                        continue;
                    if ((pTerm.prereqAll & notReady) != 0)
                    {
                        Debug.Assert(pWInfo.untestedTerms != 0);
                        continue;
                    }
                    Debug.Assert(pTerm.pExpr != null);
                    sqlite3ExprIfFalse(pParse, pTerm.pExpr, addrCont, SQLITE_JUMPIFNULL);
                    pTerm.wtFlags |= TERM_CODED;
                }
            }

            sqlite3ReleaseTempReg(pParse, iReleaseReg);
            return notReady;
        }

#if  (SQLITE_TEST)
    /*
** The following variable holds a text description of query plan generated
** by the most recent call to sqlite3WhereBegin().  Each call to WhereBegin
** overwrites the previous.  This information is used for testing and
** analysis only.
*/
#if !TCLSH
    //char sqlite3_query_plan[BMS*2*40];  /* Text of the join */
    static StringBuilder sqlite3_query_plan;
#else
    static tcl.lang.Var.SQLITE3_GETSET sqlite3_query_plan = new tcl.lang.Var.SQLITE3_GETSET( "sqlite3_query_plan" );
#endif
    static int nQPlan = 0;              /* Next free slow in _query_plan[] */

#endif //* SQLITE_TEST */


        /*
** Free a WhereInfo structure
*/
        static void whereInfoFree(sqlite3 db, WhereInfo pWInfo)
        {
            if (ALWAYS(pWInfo != null))
            {
                int i;
                for (i = 0; i < pWInfo.nLevel; i++)
                {
                    sqlite3_index_info pInfo = pWInfo.a[i] != null ? pWInfo.a[i].pIdxInfo : null;
                    if (pInfo != null)
                    {
                        /* Debug.Assert( pInfo.needToFreeIdxStr==0 || db.mallocFailed ); */
                        if (pInfo.needToFreeIdxStr != 0)
                        {
                            //sqlite3_free( ref pInfo.idxStr );
                        }
                        sqlite3DbFree(db, ref pInfo);
                    }
                    if (pWInfo.a[i] != null && (pWInfo.a[i].plan.wsFlags & WHERE_TEMP_INDEX) != 0)
                    {
                        Index pIdx = pWInfo.a[i].plan.u.pIdx;
                        if (pIdx != null)
                        {
                            sqlite3DbFree(db, ref pIdx.zColAff);
                            sqlite3DbFree(db, ref pIdx);
                        }
                    }
                }
                whereClauseClear(pWInfo.pWC);
                sqlite3DbFree(db, ref pWInfo);
            }
        }


        /*
        ** Generate the beginning of the loop used for WHERE clause processing.
        ** The return value is a pointer to an opaque structure that contains
        ** information needed to terminate the loop.  Later, the calling routine
        ** should invoke sqlite3WhereEnd() with the return value of this function
        ** in order to complete the WHERE clause processing.
        **
        ** If an error occurs, this routine returns NULL.
        **
        ** The basic idea is to do a nested loop, one loop for each table in
        ** the FROM clause of a select.  (INSERT and UPDATE statements are the
        ** same as a SELECT with only a single table in the FROM clause.)  For
        ** example, if the SQL is this:
        **
        **       SELECT * FROM t1, t2, t3 WHERE ...;
        **
        ** Then the code generated is conceptually like the following:
        **
        **      foreach row1 in t1 do       \    Code generated
        **        foreach row2 in t2 do      |-- by sqlite3WhereBegin()
        **          foreach row3 in t3 do   /
        **            ...
        **          end                     \    Code generated
        **        end                        |-- by sqlite3WhereEnd()
        **      end                         /
        **
        ** Note that the loops might not be nested in the order in which they
        ** appear in the FROM clause if a different order is better able to make
        ** use of indices.  Note also that when the IN operator appears in
        ** the WHERE clause, it might result in additional nested loops for
        ** scanning through all values on the right-hand side of the IN.
        **
        ** There are Btree cursors Debug.Associated with each table.  t1 uses cursor
        ** number pTabList.a[0].iCursor.  t2 uses the cursor pTabList.a[1].iCursor.
        ** And so forth.  This routine generates code to open those VDBE cursors
        ** and sqlite3WhereEnd() generates the code to close them.
        **
        ** The code that sqlite3WhereBegin() generates leaves the cursors named
        ** in pTabList pointing at their appropriate entries.  The [...] code
        ** can use OP_Column and OP_Rowid opcodes on these cursors to extract
        ** data from the various tables of the loop.
        **
        ** If the WHERE clause is empty, the foreach loops must each scan their
        ** entire tables.  Thus a three-way join is an O(N^3) operation.  But if
        ** the tables have indices and there are terms in the WHERE clause that
        ** refer to those indices, a complete table scan can be avoided and the
        ** code will run much faster.  Most of the work of this routine is checking
        ** to see if there are indices that can be used to speed up the loop.
        **
        ** Terms of the WHERE clause are also used to limit which rows actually
        ** make it to the "..." in the middle of the loop.  After each "foreach",
        ** terms of the WHERE clause that use only terms in that loop and outer
        ** loops are evaluated and if false a jump is made around all subsequent
        ** inner loops (or around the "..." if the test occurs within the inner-
        ** most loop)
        **
        ** OUTER JOINS
        **
        ** An outer join of tables t1 and t2 is conceptally coded as follows:
        **
        **    foreach row1 in t1 do
        **      flag = 0
        **      foreach row2 in t2 do
        **        start:
        **          ...
        **          flag = 1
        **      end
        **      if flag==null then
        **        move the row2 cursor to a null row
        **        goto start
        **      fi
        **    end
        **
        ** ORDER BY CLAUSE PROCESSING
        **
        ** ppOrderBy is a pointer to the ORDER BY clause of a SELECT statement,
        ** if there is one.  If there is no ORDER BY clause or if this routine
        ** is called from an UPDATE or DELETE statement, then ppOrderBy is NULL.
        **
        ** If an index can be used so that the natural output order of the table
        ** scan is correct for the ORDER BY clause, then that index is used and
        ** ppOrderBy is set to NULL.  This is an optimization that prevents an
        ** unnecessary sort of the result set if an index appropriate for the
        ** ORDER BY clause already exists.
        **
        ** If the where clause loops cannot be arranged to provide the correct
        ** output order, then the ppOrderBy is unchanged.
        */
        static WhereInfo sqlite3WhereBegin(
        Parse pParse,           /* The parser context */
        SrcList pTabList,       /* A list of all tables to be scanned */
        Expr pWhere,            /* The WHERE clause */
        ref ExprList ppOrderBy, /* An ORDER BY clause, or NULL */
        u16 wctrlFlags          /* One of the WHERE_* flags defined in sqliteInt.h */
        )
        {
            int i;                     /* Loop counter */
            int nByteWInfo;            /* Num. bytes allocated for WhereInfo struct */
            int nTabList;              /* Number of elements in pTabList */
            WhereInfo pWInfo;          /* Will become the return value of this function */
            Vdbe v = pParse.pVdbe;     /* The virtual data_base engine */
            Bitmask notReady;          /* Cursors that are not yet positioned */
            WhereMaskSet pMaskSet;     /* The expression mask set */
            WhereClause pWC = new WhereClause();               /* Decomposition of the WHERE clause */
            SrcList_item pTabItem;     /* A single entry from pTabList */
            WhereLevel pLevel;         /* A single level in the pWInfo list */
            int iFrom;                 /* First unused FROM clause element */
            int andFlags;              /* AND-ed combination of all pWC.a[].wtFlags */
            sqlite3 db;                /* Data_base connection */

            /* The number of tables in the FROM clause is limited by the number of
            ** bits in a Bitmask
            */
            testcase(pTabList.nSrc == BMS);
            if (pTabList.nSrc > BMS)
            {
                sqlite3ErrorMsg(pParse, "at most %d tables in a join", BMS);
                return null;
            }

            /* This function normally generates a nested loop for all tables in 
            ** pTabList.  But if the WHERE_ONETABLE_ONLY flag is set, then we should
            ** only generate code for the first table in pTabList and assume that
            ** any cursors associated with subsequent tables are uninitialized.
            */
            nTabList = ((wctrlFlags & WHERE_ONETABLE_ONLY) != 0) ? 1 : (int)pTabList.nSrc;

            /* Allocate and initialize the WhereInfo structure that will become the
            ** return value. A single allocation is used to store the WhereInfo
            ** struct, the contents of WhereInfo.a[], the WhereClause structure
            ** and the WhereMaskSet structure. Since WhereClause contains an 8-byte
            ** field (type Bitmask) it must be aligned on an 8-byte boundary on
            ** some architectures. Hence the ROUND8() below.
            */
            db = pParse.db;
            pWInfo = new WhereInfo();
            //nByteWInfo = ROUND8(sizeof(WhereInfo)+(nTabList-1)*sizeof(WhereLevel));
            //pWInfo = sqlite3DbMallocZero( db,
            //    nByteWInfo +
            //    sizeof( WhereClause ) +
            //    sizeof( WhereMaskSet )
            //);
            pWInfo.a = new WhereLevel[pTabList.nSrc];
            for (int ai = 0; ai < pWInfo.a.Length; ai++)
            {
                pWInfo.a[ai] = new WhereLevel();
            }
            //if ( db.mallocFailed != 0 )
            //{
            //sqlite3DbFree(db, pWInfo);
            //pWInfo = 0;
            //  goto whereBeginError;
            //}
            pWInfo.nLevel = nTabList;
            pWInfo.pParse = pParse;
            pWInfo.pTabList = pTabList;
            pWInfo.iBreak = sqlite3VdbeMakeLabel(v);
            pWInfo.pWC = pWC = new WhereClause();// (WhereClause )((u8 )pWInfo)[nByteWInfo];
            pWInfo.wctrlFlags = wctrlFlags;
            pWInfo.savedNQueryLoop = pParse.nQueryLoop;
            //pMaskSet = (WhereMaskSet)pWC[1];

            /* Split the WHERE clause into separate subexpressions where each
            ** subexpression is separated by an AND operator.
            */
            pMaskSet = new WhereMaskSet();//initMaskSet(pMaskSet);
            whereClauseInit(pWC, pParse, pMaskSet);
            sqlite3ExprCodeConstants(pParse, pWhere);
            whereSplit(pWC, pWhere, TK_AND);   /* IMP: R-15842-53296 */

            /* Special case: a WHERE clause that is constant.  Evaluate the
            ** expression and either jump over all of the code or fall thru.
            */
            if (pWhere != null && (nTabList == 0 || sqlite3ExprIsConstantNotJoin(pWhere) != 0))
            {
                sqlite3ExprIfFalse(pParse, pWhere, pWInfo.iBreak, SQLITE_JUMPIFNULL);
                pWhere = null;
            }

            /* Assign a bit from the bitmask to every term in the FROM clause.
            **
            ** When assigning bitmask values to FROM clause cursors, it must be
            ** the case that if X is the bitmask for the N-th FROM clause term then
            ** the bitmask for all FROM clause terms to the left of the N-th term
            ** is (X-1).   An expression from the ON clause of a LEFT JOIN can use
            ** its Expr.iRightJoinTable value to find the bitmask of the right table
            ** of the join.  Subtracting one from the right table bitmask gives a
            ** bitmask for all tables to the left of the join.  Knowing the bitmask
            ** for all tables to the left of a left join is important.  Ticket #3015.
            **
            ** Configure the WhereClause.vmask variable so that bits that correspond
            ** to virtual table cursors are set. This is used to selectively disable
            ** the OR-to-IN transformation in exprAnalyzeOrTerm(). It is not helpful
            ** with virtual tables.
            **
            ** Note that bitmasks are created for all pTabList.nSrc tables in
            ** pTabList, not just the first nTabList tables.  nTabList is normally
            ** equal to pTabList.nSrc but might be shortened to 1 if the
            ** WHERE_ONETABLE_ONLY flag is set.
            */
            Debug.Assert(pWC.vmask == 0 && pMaskSet.n == 0);
            for (i = 0; i < pTabList.nSrc; i++)
            {
                createMask(pMaskSet, pTabList.a[i].iCursor);
#if !SQLITE_OMIT_VIRTUALTABLE
                if (ALWAYS(pTabList.a[i].pTab) && IsVirtual(pTabList.a[i].pTab))
                {
                    pWC.vmask |= ((Bitmask)1 << i);
                }
#endif
            }
#if  !NDEBUG
            {
                Bitmask toTheLeft = 0;
                for (i = 0; i < pTabList.nSrc; i++)
                {
                    Bitmask m = getMask(pMaskSet, pTabList.a[i].iCursor);
                    Debug.Assert((m - 1) == toTheLeft);
                    toTheLeft |= m;
                }
            }
#endif

            /* Analyze all of the subexpressions.  Note that exprAnalyze() might
** add new virtual terms onto the end of the WHERE clause.  We do not
** want to analyze these virtual terms, so start analyzing at the end
** and work forward so that the added virtual terms are never processed.
*/
            exprAnalyzeAll(pTabList, pWC);
            //if ( db.mallocFailed != 0 )
            //{
            //  goto whereBeginError;
            //}

            /* Chose the best index to use for each table in the FROM clause.
            **
            ** This loop fills in the following fields:
            **
            **   pWInfo.a[].pIdx      The index to use for this level of the loop.
            **   pWInfo.a[].wsFlags   WHERE_xxx flags Debug.Associated with pIdx
            **   pWInfo.a[].nEq       The number of == and IN constraints
            **   pWInfo.a[].iFrom     Which term of the FROM clause is being coded
            **   pWInfo.a[].iTabCur   The VDBE cursor for the data_base table
            **   pWInfo.a[].iIdxCur   The VDBE cursor for the index
            **   pWInfo.a[].pTerm     When wsFlags==WO_OR, the OR-clause term
            **
            ** This loop also figures out the nesting order of tables in the FROM
            ** clause.
            */
            notReady = ~(Bitmask)0;
            andFlags = ~0;
#if (SQLITE_TEST) && (SQLITE_DEBUG)
      WHERETRACE( "*** Optimizer Start ***\n" );
#endif
            for (i = iFrom = 0; i < nTabList; i++)//, pLevel++ )
            {
                pLevel = pWInfo.a[i];
                WhereCost bestPlan;         /* Most efficient plan seen so far */
                Index pIdx;                 /* Index for FROM table at pTabItem */
                int j;                      /* For looping over FROM tables */
                int bestJ = -1;             /* The value of j */
                Bitmask m;                  /* Bitmask value for j or bestJ */
                int isOptimal;              /* Iterator for optimal/non-optimal search */
                int nUnconstrained;         /* Number tables without INDEXED BY */
                Bitmask notIndexed;         /* Mask of tables that cannot use an index */

                bestPlan = new WhereCost();// memset( &bestPlan, 0, sizeof( bestPlan ) );
                bestPlan.rCost = SQLITE_BIG_DBL;
#if  (SQLITE_TEST) && (SQLITE_DEBUG)
        WHERETRACE( "*** Begin search for loop %d ***\n", i );
#endif

                /* Loop through the remaining entries in the FROM clause to find the
** next nested loop. The loop tests all FROM clause entries
** either once or twice. 
**
** The first test is always performed if there are two or more entries
** remaining and never performed if there is only one FROM clause entry
** to choose from.  The first test looks for an "optimal" scan.  In
** this context an optimal scan is one that uses the same strategy
** for the given FROM clause entry as would be selected if the entry
** were used as the innermost nested loop.  In other words, a table
** is chosen such that the cost of running that table cannot be reduced
** by waiting for other tables to run first.  This "optimal" test works
** by first assuming that the FROM clause is on the inner loop and finding
** its query plan, then checking to see if that query plan uses any
** other FROM clause terms that are notReady.  If no notReady terms are
** used then the "optimal" query plan works.
**
** Note that the WhereCost.nRow parameter for an optimal scan might
** not be as small as it would be if the table really were the innermost
** join.  The nRow value can be reduced by WHERE clause constraints
** that do not use indices.  But this nRow reduction only happens if the
** table really is the innermost join.  
**
** The second loop iteration is only performed if no optimal scan
** strategies were found by the first iteration. This second iteration
** is used to search for the lowest cost scan overall.
**
** Previous versions of SQLite performed only the second iteration -
** the next outermost loop was always that with the lowest overall
** cost. However, this meant that SQLite could select the wrong plan
** for scripts such as the following:
**   
**   CREATE TABLE t1(a, b); 
**   CREATE TABLE t2(c, d);
**   SELECT * FROM t2, t1 WHERE t2.rowid = t1.a;
**
** The best strategy is to iterate through table t1 first. However it
** is not possible to determine this with a simple greedy algorithm.
** Since the cost of a linear scan through table t2 is the same 
** as the cost of a linear scan through table t1, a simple greedy 
** algorithm may choose to use t2 for the outer loop, which is a much
** costlier approach.
*/
                nUnconstrained = 0;
                notIndexed = 0;
                for (isOptimal = (iFrom < nTabList - 1) ? 1 : 0; isOptimal >= 0 && bestJ < 0; isOptimal--)
                {
                    Bitmask mask;  /* Mask of tables not yet ready */
                    for (j = iFrom; j < nTabList; j++)//, pTabItem++)
                    {
                        pTabItem = pTabList.a[j];
                        int doNotReorder;       /* True if this table should not be reordered */
                        WhereCost sCost = new WhereCost(); /* Cost information from best[Virtual]Index() */
                        ExprList pOrderBy;      /* ORDER BY clause for index to optimize */

                        doNotReorder = (pTabItem.jointype & (JT_LEFT | JT_CROSS)) != 0 ? 1 : 0;
                        if ((j != iFrom && doNotReorder != 0))
                            break;
                        m = getMask(pMaskSet, pTabItem.iCursor);
                        if ((m & notReady) == 0)
                        {
                            if (j == iFrom)
                                iFrom++;
                            continue;
                        }
                        mask = (isOptimal != 0 ? m : notReady);
                        pOrderBy = ((i == 0 && ppOrderBy != null) ? ppOrderBy : null);
                        if (pTabItem.pIndex == null)
                            nUnconstrained++;

#if  (SQLITE_TEST) && (SQLITE_DEBUG)
            WHERETRACE( "=== trying table %d with isOptimal=%d ===\n",
            j, isOptimal );
#endif
                        Debug.Assert(pTabItem.pTab != null);
#if  !SQLITE_OMIT_VIRTUALTABLE
                        if (IsVirtual(pTabItem.pTab))
                        {
                            sqlite3_index_info pp = pWInfo.a[j].pIdxInfo;
                            bestVirtualIndex(pParse, pWC, pTabItem, mask, notReady, pOrderBy,
                             ref sCost, ref pp);
                        }
                        else
#endif
                        {
                            bestBtreeIndex(pParse, pWC, pTabItem, mask, notReady, pOrderBy,
                            ref sCost);
                        }
                        Debug.Assert(isOptimal != 0 || (sCost.used & notReady) == 0);

                        /* If an INDEXED BY clause is present, then the plan must use that
                        ** index if it uses any index at all */
                        Debug.Assert(pTabItem.pIndex == null
                        || (sCost.plan.wsFlags & WHERE_NOT_FULLSCAN) == 0
                        || sCost.plan.u.pIdx == pTabItem.pIndex);

                        if (isOptimal != 0 && (sCost.plan.wsFlags & WHERE_NOT_FULLSCAN) == 0)
                        {
                            notIndexed |= m;
                        }

                        /* Conditions under which this table becomes the best so far:
                        **
                        **   (1) The table must not depend on other tables that have not
                        **       yet run.
                        **
                        **   (2) A full-table-scan plan cannot supercede indexed plan unless
                        **       the full-table-scan is an "optimal" plan as defined above.
                        **
                        **   (3) All tables have an INDEXED BY clause or this table lacks an
                        **       INDEXED BY clause or this table uses the specific
                        **       index specified by its INDEXED BY clause.  This rule ensures
                        **       that a best-so-far is always selected even if an impossible
                        **       combination of INDEXED BY clauses are given.  The error
                        **       will be detected and relayed back to the application later.
                        **       The NEVER() comes about because rule (2) above prevents
                        **       An indexable full-table-scan from reaching rule (3).
                        **
                        **   (4) The plan cost must be lower than prior plans or else the
                        **       cost must be the same and the number of rows must be lower.
                        */
                        if ((sCost.used & notReady) == 0                       /* (1) */
                            && (bestJ < 0 || (notIndexed & m) != 0               /* (2) */
                                || (bestPlan.plan.wsFlags & WHERE_NOT_FULLSCAN) == 0
                                || (sCost.plan.wsFlags & WHERE_NOT_FULLSCAN) != 0)
                            && (nUnconstrained == 0 || pTabItem.pIndex == null   /* (3) */
                            || NEVER((sCost.plan.wsFlags & WHERE_NOT_FULLSCAN) != 0))
                        && (bestJ < 0 || sCost.rCost < bestPlan.rCost          /* (4) */
                        || (sCost.rCost <= bestPlan.rCost
                        && sCost.plan.nRow < bestPlan.plan.nRow))
                        )
                        {
#if  (SQLITE_TEST) && (SQLITE_DEBUG)
              WHERETRACE( "=== table %d is best so far" +
              " with cost=%g and nRow=%g\n",
              j, sCost.rCost, sCost.plan.nRow );
#endif
                            bestPlan = sCost;
                            bestJ = j;
                        }
                        if (doNotReorder != 0)
                            break;
                    }
                }
                Debug.Assert(bestJ >= 0);
                Debug.Assert((notReady & getMask(pMaskSet, pTabList.a[bestJ].iCursor)) != 0);
#if (SQLITE_TEST) && (SQLITE_DEBUG)
        WHERETRACE( "*** Optimizer selects table %d for loop %d" +
        " with cost=%g and nRow=%g\n",
        bestJ, i,//pLevel-pWInfo.a,
        bestPlan.rCost, bestPlan.plan.nRow );
#endif
                if ((bestPlan.plan.wsFlags & WHERE_ORDERBY) != 0)
                {
                    ppOrderBy = null;
                }
                andFlags = (int)(andFlags & bestPlan.plan.wsFlags);
                pLevel.plan = bestPlan.plan;
                testcase(bestPlan.plan.wsFlags & WHERE_INDEXED);
                testcase(bestPlan.plan.wsFlags & WHERE_TEMP_INDEX);
                if ((bestPlan.plan.wsFlags & (WHERE_INDEXED | WHERE_TEMP_INDEX)) != 0)
                {
                    pLevel.iIdxCur = pParse.nTab++;
                }
                else
                {
                    pLevel.iIdxCur = -1;
                }
                notReady &= ~getMask(pMaskSet, pTabList.a[bestJ].iCursor);
                pLevel.iFrom = (u8)bestJ;
                if (bestPlan.plan.nRow >= (double)1)
                {
                    pParse.nQueryLoop *= bestPlan.plan.nRow;
                }

                /* Check that if the table scanned by this loop iteration had an
                ** INDEXED BY clause attached to it, that the named index is being
                ** used for the scan. If not, then query compilation has failed.
                ** Return an error.
                */
                pIdx = pTabList.a[bestJ].pIndex;
                if (pIdx != null)
                {
                    if ((bestPlan.plan.wsFlags & WHERE_INDEXED) == 0)
                    {
                        sqlite3ErrorMsg(pParse, "cannot use index: %s", pIdx.zName);
                        goto whereBeginError;
                    }
                    else
                    {
                        /* If an INDEXED BY clause is used, the bestIndex() function is
                        ** guaranteed to find the index specified in the INDEXED BY clause
                        ** if it find an index at all. */
                        Debug.Assert(bestPlan.plan.u.pIdx == pIdx);
                    }
                }
            }
#if (SQLITE_TEST) && (SQLITE_DEBUG)
      WHERETRACE( "*** Optimizer Finished ***\n" );
#endif
            if (pParse.nErr != 0 /*|| db.mallocFailed != 0 */ )
            {
                goto whereBeginError;
            }

            /* If the total query only selects a single row, then the ORDER BY
            ** clause is irrelevant.
            */
            if ((andFlags & WHERE_UNIQUE) != 0 && ppOrderBy != null)
            {
                ppOrderBy = null;
            }

            /* If the caller is an UPDATE or DELETE statement that is requesting
            ** to use a one-pDebug.Ass algorithm, determine if this is appropriate.
            ** The one-pass algorithm only works if the WHERE clause constraints
            ** the statement to update a single row.
            */
            Debug.Assert((wctrlFlags & WHERE_ONEPASS_DESIRED) == 0 || pWInfo.nLevel == 1);
            if ((wctrlFlags & WHERE_ONEPASS_DESIRED) != 0 && (andFlags & WHERE_UNIQUE) != 0)
            {
                pWInfo.okOnePass = 1;
                pWInfo.a[0].plan.wsFlags = (u32)(pWInfo.a[0].plan.wsFlags & ~WHERE_IDX_ONLY);
            }

            /* Open all tables in the pTabList and any indices selected for
            ** searching those tables.
            */
            sqlite3CodeVerifySchema(pParse, -1); /* Insert the cookie verifier Goto */
            notReady = ~(Bitmask)0;
            pWInfo.nRowOut = (double)1;
            for (i = 0; i < nTabList; i++)//, pLevel++ )
            {
                pLevel = pWInfo.a[i];
                Table pTab;     /* Table to open */
                int iDb;         /* Index of data_base containing table/index */

                pTabItem = pTabList.a[pLevel.iFrom];
                pTab = pTabItem.pTab;
                pLevel.iTabCur = pTabItem.iCursor;
                pWInfo.nRowOut *= pLevel.plan.nRow;
                iDb = sqlite3SchemaToIndex(db, pTab.pSchema);
                if ((pTab.tabFlags & TF_Ephemeral) != 0 || pTab.pSelect != null)
                {
                    /* Do nothing */
                }
                else
#if  !SQLITE_OMIT_VIRTUALTABLE
                    if ((pLevel.plan.wsFlags & WHERE_VIRTUALTABLE) != 0)
                    {
                        VTable pVTab = sqlite3GetVTable(db, pTab);
                        int iCur = pTabItem.iCursor;
                        sqlite3VdbeAddOp4(v, OP_VOpen, iCur, 0, 0,
                        pVTab, P4_VTAB);
                    }
                    else
#endif
                        if ((pLevel.plan.wsFlags & WHERE_IDX_ONLY) == 0
                        && (wctrlFlags & WHERE_OMIT_OPEN) == 0)
                        {
                            int op = pWInfo.okOnePass != 0 ? OP_OpenWrite : OP_OpenRead;
                            sqlite3OpenTable(pParse, pTabItem.iCursor, iDb, pTab, op);
                            testcase(pTab.nCol == BMS - 1);
                            testcase(pTab.nCol == BMS);
                            if (0 == pWInfo.okOnePass && pTab.nCol < BMS)
                            {
                                Bitmask b = pTabItem.colUsed;
                                int n = 0;
                                for (; b != 0; b = b >> 1, n++)
                                {
                                }
                                sqlite3VdbeChangeP4(v, sqlite3VdbeCurrentAddr(v) - 1,
                                        n, P4_INT32);//SQLITE_INT_TO_PTR(n)
                                Debug.Assert(n <= pTab.nCol);
                            }
                        }
                        else
                        {
                            sqlite3TableLock(pParse, iDb, pTab.tnum, 0, pTab.zName);
                        }
#if !SQLITE_OMIT_AUTOMATIC_INDEX
                if ((pLevel.plan.wsFlags & WHERE_TEMP_INDEX) != 0)
                {
                    constructAutomaticIndex(pParse, pWC, pTabItem, notReady, pLevel);
                }
                else
#endif
                    if ((pLevel.plan.wsFlags & WHERE_INDEXED) != 0)
                    {
                        Index pIx = pLevel.plan.u.pIdx;
                        KeyInfo pKey = sqlite3IndexKeyinfo(pParse, pIx);
                        int iIdxCur = pLevel.iIdxCur;
                        Debug.Assert(pIx.pSchema == pTab.pSchema);
                        Debug.Assert(iIdxCur >= 0);
                        sqlite3VdbeAddOp4(v, OP_OpenRead, iIdxCur, pIx.tnum, iDb,
                        pKey, P4_KEYINFO_HANDOFF);
#if SQLITE_DEBUG
            VdbeComment( v, "%s", pIx.zName );
#endif
                    }
                sqlite3CodeVerifySchema(pParse, iDb);
                notReady &= ~getMask(pWC.pMaskSet, pTabItem.iCursor);
            }
            pWInfo.iTop = sqlite3VdbeCurrentAddr(v);
            //if( db.mallocFailed ) goto whereBeginError;

            /* Generate the code to do the search.  Each iteration of the for
            ** loop below generates code for a single nested loop of the VM
            ** program.
            */
            notReady = ~(Bitmask)0;
            for (i = 0; i < nTabList; i++)
            {
                pLevel = pWInfo.a[i];
                explainOneScan(pParse, pTabList, pLevel, i, pLevel.iFrom, wctrlFlags);
                notReady = codeOneLoopStart(pWInfo, i, wctrlFlags, notReady);
                pWInfo.iContinue = pLevel.addrCont;
            }

#if SQLITE_TEST  //* For testing and debugging use only */
      /* Record in the query plan information about the current table
** and the index used to access it (if any).  If the table itself
** is not used, its name is just '{}'.  If no index is used
** the index is listed as "{}".  If the primary key is used the
** index name is '*'.
*/
#if !TCLSH
      sqlite3_query_plan.Length = 0;
#else
      sqlite3_query_plan.sValue = "";
#endif
      for ( i = 0; i < nTabList; i++ )
      {
        string z;
        int n;
        pLevel = pWInfo.a[i];
        pTabItem = pTabList.a[pLevel.iFrom];
        z = pTabItem.zAlias;
        if ( z == null )
          z = pTabItem.pTab.zName;
        n = sqlite3Strlen30( z );
        if ( true ) //n+nQPlan < sizeof(sqlite3_query_plan)-10 )
        {
          if ( ( pLevel.plan.wsFlags & WHERE_IDX_ONLY ) != 0 )
          {
            sqlite3_query_plan.Append( "{}" ); //memcpy( &sqlite3_query_plan[nQPlan], "{}", 2 );
            nQPlan += 2;
          }
          else
          {
            sqlite3_query_plan.Append( z ); //memcpy( &sqlite3_query_plan[nQPlan], z, n );
            nQPlan += n;
          }
          sqlite3_query_plan.Append( " " );
          nQPlan++; //sqlite3_query_plan[nQPlan++] = ' ';
        }
        testcase( pLevel.plan.wsFlags & WHERE_ROWID_EQ );
        testcase( pLevel.plan.wsFlags & WHERE_ROWID_RANGE );
        if ( ( pLevel.plan.wsFlags & ( WHERE_ROWID_EQ | WHERE_ROWID_RANGE ) ) != 0 )
        {
          sqlite3_query_plan.Append( "* " ); //memcpy(&sqlite3_query_plan[nQPlan], "* ", 2);
          nQPlan += 2;
        }
        else if ( ( pLevel.plan.wsFlags & WHERE_INDEXED ) != 0 )
        {
          n = sqlite3Strlen30( pLevel.plan.u.pIdx.zName );
          if ( true ) //n+nQPlan < sizeof(sqlite3_query_plan)-2 )//if( n+nQPlan < sizeof(sqlite3_query_plan)-2 )
          {
            sqlite3_query_plan.Append( pLevel.plan.u.pIdx.zName ); //memcpy(&sqlite3_query_plan[nQPlan], pLevel.plan.u.pIdx.zName, n);
            nQPlan += n;
            sqlite3_query_plan.Append( " " ); //sqlite3_query_plan[nQPlan++] = ' ';
          }
        }
        else
        {
          sqlite3_query_plan.Append( "{} " ); //memcpy( &sqlite3_query_plan[nQPlan], "{} ", 3 );
          nQPlan += 3;
        }
      }
      //while( nQPlan>0 && sqlite3_query_plan[nQPlan-1]==' ' ){
      //  sqlite3_query_plan[--nQPlan] = 0;
      //}
      //sqlite3_query_plan[nQPlan] = 0;
#if !TCLSH
      sqlite3_query_plan = new StringBuilder( sqlite3_query_plan.ToString().Trim() );
#else
      sqlite3_query_plan.Trim();
#endif
      nQPlan = 0;
#endif //* SQLITE_TEST // Testing and debugging use only */

            /* Record the continuation address in the WhereInfo structure.  Then
** clean up and return.
*/
            return pWInfo;

        /* Jump here if malloc fails */
        whereBeginError:
            if (pWInfo != null)
            {
                pParse.nQueryLoop = pWInfo.savedNQueryLoop;
                whereInfoFree(db, pWInfo);
            }
            return null;
        }

        /*
        ** Generate the end of the WHERE loop.  See comments on
        ** sqlite3WhereBegin() for additional information.
        */
        static void sqlite3WhereEnd(WhereInfo pWInfo)
        {
            Parse pParse = pWInfo.pParse;
            Vdbe v = pParse.pVdbe;
            int i;
            WhereLevel pLevel;
            SrcList pTabList = pWInfo.pTabList;
            sqlite3 db = pParse.db;

            /* Generate loop termination code.
            */
            sqlite3ExprCacheClear(pParse);
            for (i = pWInfo.nLevel - 1; i >= 0; i--)
            {
                pLevel = pWInfo.a[i];
                sqlite3VdbeResolveLabel(v, pLevel.addrCont);
                if (pLevel.op != OP_Noop)
                {
                    sqlite3VdbeAddOp2(v, pLevel.op, pLevel.p1, pLevel.p2);
                    sqlite3VdbeChangeP5(v, pLevel.p5);
                }
                if ((pLevel.plan.wsFlags & WHERE_IN_ABLE) != 0 && pLevel.u._in.nIn > 0)
                {
                    InLoop pIn;
                    int j;
                    sqlite3VdbeResolveLabel(v, pLevel.addrNxt);
                    for (j = pLevel.u._in.nIn; j > 0; j--)//, pIn--)
                    {
                        pIn = pLevel.u._in.aInLoop[j - 1];
                        sqlite3VdbeJumpHere(v, pIn.addrInTop + 1);
                        sqlite3VdbeAddOp2(v, OP_Next, pIn.iCur, pIn.addrInTop);
                        sqlite3VdbeJumpHere(v, pIn.addrInTop - 1);
                    }
                    sqlite3DbFree(db, ref pLevel.u._in.aInLoop);
                }
                sqlite3VdbeResolveLabel(v, pLevel.addrBrk);
                if (pLevel.iLeftJoin != 0)
                {
                    int addr;
                    addr = sqlite3VdbeAddOp1(v, OP_IfPos, pLevel.iLeftJoin);
                    Debug.Assert((pLevel.plan.wsFlags & WHERE_IDX_ONLY) == 0
                    || (pLevel.plan.wsFlags & WHERE_INDEXED) != 0);
                    if ((pLevel.plan.wsFlags & WHERE_IDX_ONLY) == 0)
                    {
                        sqlite3VdbeAddOp1(v, OP_NullRow, pTabList.a[i].iCursor);
                    }
                    if (pLevel.iIdxCur >= 0)
                    {
                        sqlite3VdbeAddOp1(v, OP_NullRow, pLevel.iIdxCur);
                    }
                    if (pLevel.op == OP_Return)
                    {
                        sqlite3VdbeAddOp2(v, OP_Gosub, pLevel.p1, pLevel.addrFirst);
                    }
                    else
                    {
                        sqlite3VdbeAddOp2(v, OP_Goto, 0, pLevel.addrFirst);
                    }
                    sqlite3VdbeJumpHere(v, addr);
                }
            }

            /* The "break" point is here, just past the end of the outer loop.
            ** Set it.
            */
            sqlite3VdbeResolveLabel(v, pWInfo.iBreak);

            /* Close all of the cursors that were opened by sqlite3WhereBegin.
            */
            Debug.Assert(pWInfo.nLevel == 1 || pWInfo.nLevel == pTabList.nSrc);
            for (i = 0; i < pWInfo.nLevel; i++)//  for(i=0, pLevel=pWInfo.a; i<pWInfo.nLevel; i++, pLevel++){
            {
                pLevel = pWInfo.a[i];
                SrcList_item pTabItem = pTabList.a[pLevel.iFrom];
                Table pTab = pTabItem.pTab;
                Debug.Assert(pTab != null);
                if ((pTab.tabFlags & TF_Ephemeral) == 0
                && pTab.pSelect == null
                && (pWInfo.wctrlFlags & WHERE_OMIT_CLOSE) == 0
                )
                {
                    u32 ws = pLevel.plan.wsFlags;
                    if (0 == pWInfo.okOnePass && (ws & WHERE_IDX_ONLY) == 0)
                    {
                        sqlite3VdbeAddOp1(v, OP_Close, pTabItem.iCursor);
                    }
                    if ((ws & WHERE_INDEXED) != 0 && (ws & WHERE_TEMP_INDEX) == 0)
                    {
                        sqlite3VdbeAddOp1(v, OP_Close, pLevel.iIdxCur);
                    }
                }

                /* If this scan uses an index, make code substitutions to read data
                ** from the index in preference to the table. Sometimes, this means
                ** the table need never be read from. This is a performance boost,
                ** as the vdbe level waits until the table is read before actually
                ** seeking the table cursor to the record corresponding to the current
                ** position in the index.
                **
                ** Calls to the code generator in between sqlite3WhereBegin and
                ** sqlite3WhereEnd will have created code that references the table
                ** directly.  This loop scans all that code looking for opcodes
                ** that reference the table and converts them into opcodes that
                ** reference the index.
                */
                if ((pLevel.plan.wsFlags & WHERE_INDEXED) != 0)///* && 0 == db.mallocFailed */ )
                {
                    int k, j, last;
                    VdbeOp pOp;
                    Index pIdx = pLevel.plan.u.pIdx;

                    Debug.Assert(pIdx != null);
                    //pOp = sqlite3VdbeGetOp( v, pWInfo.iTop );
                    last = sqlite3VdbeCurrentAddr(v);
                    for (k = pWInfo.iTop; k < last; k++)//, pOp++ )
                    {
                        pOp = sqlite3VdbeGetOp(v, k);
                        if (pOp.p1 != pLevel.iTabCur)
                            continue;
                        if (pOp.opcode == OP_Column)
                        {
                            for (j = 0; j < pIdx.nColumn; j++)
                            {
                                if (pOp.p2 == pIdx.aiColumn[j])
                                {
                                    pOp.p2 = j;
                                    pOp.p1 = pLevel.iIdxCur;
                                    break;
                                }
                            }
                            Debug.Assert((pLevel.plan.wsFlags & WHERE_IDX_ONLY) == 0
                            || j < pIdx.nColumn);

                        }
                        else if (pOp.opcode == OP_Rowid)
                        {
                            pOp.p1 = pLevel.iIdxCur;
                            pOp.opcode = OP_IdxRowid;
                        }
                    }
                }
            }

            /* Final cleanup
            */
            pParse.nQueryLoop = pWInfo.savedNQueryLoop;
            whereInfoFree(db, pWInfo);
            return;
        }
    }
}
