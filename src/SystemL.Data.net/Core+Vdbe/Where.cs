using System;
using System.Diagnostics;
using System.Text;
using Bitmask = System.UInt64;
using tRowcnt = System.UInt32;

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
            public TK OP;                       // Split operator.  TK_AND or TK_OR
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

            internal void memset()
            {
                Plan.memset();
                Cost = 0;
                Used = 0;
            }
        }

        const uint WHERE_ROWID_EQ = 0x00001000;		// rowid=EXPR or rowid IN (...)
        const uint WHERE_ROWID_RANGE = 0x00002000;		// rowid<EXPR and/or rowid>EXPR
        const uint WHERE_COLUMN_EQ = 0x00010000;		// x=EXPR or x IN (...) or x IS NULL
        const uint WHERE_COLUMN_RANGE = 0x00020000;	// x<EXPR and/or x>EXPR
        const uint WHERE_COLUMN_IN = 0x00040000;		// x IN (...)
        const uint WHERE_COLUMN_NULL = 0x00080000;		// x IS NULL
        const uint WHERE_INDEXED = 0x000f0000;			// Anything that uses an index
        const uint WHERE_NOT_FULLSCAN = 0x100f3000;	// Does not do a full table scan
        const uint WHERE_IN_ABLE = 0x080f1000;			// Able to support an IN operator
        const uint WHERE_TOP_LIMIT = 0x00100000;		// x<EXPR or x<=EXPR constraint
        const uint WHERE_BTM_LIMIT = 0x00200000;		// x>EXPR or x>=EXPR constraint
        const uint WHERE_BOTH_LIMIT = 0x00300000;		// Both x>EXPR and x<EXPR
        const uint WHERE_IDX_ONLY = 0x00400000;		// Use index only - omit table
        const uint WHERE_ORDERED = 0x00800000;			// Output will appear in correct order
        const uint WHERE_REVERSE = 0x01000000;			// Scan in reverse order */
        const uint WHERE_UNIQUE = 0x02000000;			// Selects no more than one row
        const uint WHERE_ALL_UNIQUE = 0x04000000;		// This and all prior have one row
        const uint WHERE_OB_UNIQUE = 0x00004000;		// Values in ORDER BY columns are 
        // different for every output row
        const uint WHERE_VIRTUALTABLE = 0x08000000;	// Use virtual-table processing
        const uint WHERE_MULTI_OR = 0x10000000;		// OR using multiple indices
        const uint WHERE_TEMP_INDEX = 0x20000000;		// Uses an ephemeral index
        const uint WHERE_DISTINCT_ = 0x40000000;		// Correct order for DISTINCT
        const uint WHERE_COVER_SCAN = 0x80000000;		// Full scan of a covering index

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
            public WhereLevel[] Levels;             // Info about outer loops
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
            wc.OP = op;
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
            if (E.ExprHasProperty(p, EP.xIsSelect))
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

        static bool AllowedOp(TK op)
        {
            Debug.Assert(TK.GT > TK.EQ && TK.GT < TK.GE);
            Debug.Assert(TK.LT > TK.EQ && TK.LT < TK.GE);
            Debug.Assert(TK.LE > TK.EQ && TK.LE < TK.GE);
            Debug.Assert(TK.GE == TK.EQ + 4);
            return (op == TK.IN || (op >= TK.EQ && op <= TK.GE) || op == TK.ISNULL);
        }

        static void SWAP<T>(T A, T B) { T t = A; A = B; B = t; }

        static void ExprCommute(Parse parse, Expr expr)
        {
            ushort expRight = (ushort)(expr.Right.Flags & EP.Collate);
            ushort expLeft = (ushort)(expr.Left.Flags & EP.Collate);
            Debug.Assert(AllowedOp(expr.OP) && expr.OP != TK.IN);
            if (expRight == expLeft)
            {
                // Either X and Y both have COLLATE operator or neither do
                if (expRight != null)
                    expr.Right.Flags &= ~EP.Collate; // Both X and Y have COLLATE operators.  Make sure X is always used by clearing the EP_Collate flag from Y.
                else if (expr.Left.CollSeq(parse) != null)
                    expr.Left.Flags |= EP.Collate; // Neither X nor Y have COLLATE operators, but X has a non-default collating sequence.  So add the EP_Collate marker on X to cause it to be searched first.
            }
            SWAP<Expr>(expr.Right, expr.Left);
            if (expr.OP >= TK.GT)
            {
                Debug.Assert(TK.LT == TK.GT + 2);
                Debug.Assert(TK.GE == TK.LE + 2);
                Debug.Assert(TK.GT > TK.EQ);
                Debug.Assert(TK.GT < TK.LE);
                Debug.Assert(expr.OP >= TK.GT && expr.OP <= TK.GE);
                expr.OP = (TK)(((expr.OP - TK.GT) ^ 2) + TK.GT);
            }
        }

        static WO OperatorMask(TK op)
        {
            Debug.Assert(AllowedOp(op));
            WO c;
            if (op == TK.IN)
                c = WO.IN;
            else if (op == TK.ISNULL)
                c = WO.ISNULL;
            else
            {
                Debug.Assert(((int)WO.EQ << (op - TK.EQ)) < 0x7fff);
                c = (WO)((int)WO.EQ << (op - TK.EQ));
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

        static WhereTerm FindTerm(WhereClause wc, int cursor, int column, Bitmask notReady, WO op, Index idx)
        {
            WhereTerm result = null; // The answer to return
            int origColumn = column;  // Original value of iColumn
            int equivsLength = 2; // Number of entires in aEquiv[]
            int equivId = 2; // Number of entries of aEquiv[] processed so far
            int[] equivs = new int[22]; // iCur,iColumn and up to 10 other equivalents

            Debug.Assert(cursor >= 0);
            equivs[0] = cursor;
            equivs[1] = column;
            WhereClause origWC = wc; // Original pWC value
            WhereTerm term; // Term being examined as possible result
            int j, k;
            for (; ; )
            {
                for (wc = origWC; wc != null; wc = wc.Outer)
                {
                    for (term = wc.Slots[0], k = wc.Terms; k != 0; k--, term = wc.Slots[wc.Terms - k])
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
                                    CollSeq coll = Expr.BinaryCompareCollSeq(parse, x.Left, x.Right);
                                    if (coll == null) coll = parse.Ctx.DefaultColl;
                                    for (j = 0; idx.Columns[j] != origColumn; j++)
                                        if (SysEx.NEVER(j >= idx.Columns.length)) return null;
                                    if (!string.Equals(coll.Name, idx.CollNames[j], StringComparison.OrdinalIgnoreCase))
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
                                Expr x = term.Expr.Right.SkipCollate();
                                Debug.Assert(x.OP == TK.COLUMN);
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

#if !OMIT_LIKE_OPTIMIZATION
        static int IsLikeOrGlob(Parse parse, Expr expr, ref Expr prefix, ref bool isComplete, ref bool noCase)
        {
            int cnt; // Number of non-wildcard prefix characters
            char[] wc = new char[3]; // Wildcard characters
            Context ctx = parse.Ctx; // Data_base connection
            Mem val = null;

            if (!sqlite3IsLikeFunction(ctx, expr, ref noCase, wc))
                return 0;
            ExprList list = expr.x.List; // List of operands to the LIKE operator
            Expr left = list.Ids[1].Expr; // Right and left size of LIKE operator
            if (left.OP != TK.COLUMN || left.Affinity() != AFF.TEXT || E.IsVirtual(left.Table))
                return 0; // IMP: R-02065-49465 The left-hand side of the LIKE or GLOB operator must be the name of an indexed column with TEXT affinity.
            Debug.Assert(left.ColumnIdx != -1); // Because IPK never has AFF_TEXT

            Expr right = list.Ids[0].Expr; // Right and left size of LIKE operator
            TK op = right.OP; // Opcode of pRight
            if (op == TK.REGISTER)
                op = right.OP2;
            string z = null; // String on RHS of LIKE operator
            if (op == TK.VARIABLE)
            {
                Vdbe reprepare = parse.Reprepare;
                int column = right.ColumnIdx;
                val = Vdbe.GetValue(reprepare, column, (byte)AFF.NONE);
                if (val != null && sqlite3_value_type(val) == TYPE.TEXT)
                    z = sqlite3_value_text(val);
                sqlite3VdbeSetVarmask(parse.V, column); // IMP: R-23257-02778
                Debug.Assert(right.OP == TK.VARIABLE || right.OP == TK.REGISTER);
            }
            else if (op == TK.STRING)
                z = right.u.Token;
            if (z != null)
            {
                cnt = 0;
                int c; // One character in z[]
                while (cnt < z.Length && (c = z[cnt]) != 0 && c != wc[0] && c != wc[1] && c != wc[2]) cnt++;
                if (cnt != 0 && (byte)z[cnt - 1] != 255)
                {
                    isComplete = (c == wc[0] && cnt == z.Length - 1);
                    Expr Prefix = Expr.Expr_(ctx, TK.STRING, z);
                    if (Prefix != null)
                        Prefix.u.Token = Prefix.u.Token.Substring(0, cnt);
                    prefix = Prefix;
                    if (op == TK.VARIABLE)
                    {
                        Vdbe v = parse.V;
                        sqlite3VdbeSetVarmask(v, right.ColumnIdx); // IMP: R-23257-02778
                        if (isComplete && right.u.Token.Length > 1)
                        {
                            // If the rhs of the LIKE expression is a variable, and the current value of the variable means there is no need to invoke the LIKE
                            // function, then no OP_Variable will be added to the program. This causes problems for the sqlite3_bind_parameter_name()
                            // API. To workaround them, add a dummy OP_Variable here.
                            int r1 = parse.GetTempReg();
                            Expr.CodeTarget(parse, right, r1);
                            v.ChangeP3(v.CurrentAddr() - 1, 0);
                            parse.ReleaseTempReg(r1);
                        }
                    }
                }
                else
                    z = null;
            }
            sqlite3ValueFree(ref val);
            return (z != null ? 1 : 0);
        }
#endif


#if !OMIT_VIRTUALTABLE
        static bool IsMatchOfColumn(Expr expr)
        {
            if (expr.OP != TK.FUNCTION || !string.Equals(expr.u.Token, "match", StringComparison.OrdinalIgnoreCase))
                return false;
            ExprList list = expr.x.List;
            if (list.Exprs != 2 || list.Ids[1].Expr.OP != TK.COLUMN)
                return false;
            return true;
        }
#endif

        static void TransferJoinMarkings(Expr derived, Expr base_)
        {
            derived.Flags = (EP)(derived.Flags | base_.Flags & EP.FromJoin);
            derived.RightJoinTable = base_.RightJoinTable;
        }

#if !OMIT_OR_OPTIMIZATION && !OMIT_SUBQUERY
        static void ExprAnalyzeOrTerm(SrcList src, WhereClause wc, int idxTerm)
        {
            Parse parse = wc.Parse; // Parser context
            Context ctx = parse.Ctx; // Data_base connection
            WhereTerm term = wc.Slots[idxTerm]; // The term to be analyzed
            Expr expr = term.Expr; // The expression of the term
            WhereMaskSet maskSet = wc.MaskSet; // Table use masks

            // Break the OR clause into its separate subterms.  The subterms are stored in a WhereClause structure containing within the WhereOrInfo
            // object that is attached to the original OR clause term.
            Debug.Assert((term.WtFlags & (TERM.DYNAMIC | TERM.ORINFO | TERM.ANDINFO)) == 0);
            Debug.Assert(expr.OP == TK.OR);
            WhereOrInfo orInfo; // Additional information Debug.Associated with pTerm
            term.u.OrInfo = orInfo = new WhereOrInfo();
            if (orInfo == null) return;
            term.WtFlags |= TERM.ORINFO;
            WhereClause orWc = orInfo.WC; // Breakup of pTerm into subterms
            WhereClauseInit(orWc, wc.Parse, maskSet, wc.WctrlFlags);
            WhereSplit(orWc, expr, TK.OR);
            ExprAnalyzeAll(src, orWc);
            if (ctx.MallocFailed) return;
            Debug.Assert(orWc.Terms >= 2);

            // Compute the set of tables that might satisfy cases 1 or 2.
            int i;
            WhereTerm orTerm; // A Sub-term within the pOrWc
            Bitmask indexable = ~(Bitmask)0; // Tables that are indexable, satisfying case 2
            Bitmask chngToIN = ~(Bitmask)0; // Tables that might satisfy case 1
            for (i = orWc.Terms - 1, orTerm = orWc.Slots[0]; i >= 0 && indexable != 0; i--, orTerm = orWc.Slots[i])
            {
                if ((orTerm.EOperator & WO.SINGLE) == 0)
                {
                    Debug.Assert((orTerm.WtFlags & (TERM.ANDINFO | TERM.ORINFO)) == 0);
                    chngToIN = 0;
                    WhereAndInfo andInfo = new WhereAndInfo();
                    if (andInfo != null)
                    {
                        WhereTerm andTerm;
                        int j;
                        Bitmask b = 0;
                        orTerm.u.AndInfo = andInfo;
                        orTerm.WtFlags |= TERM.ANDINFO;
                        orTerm.EOperator = WO.AND;
                        WhereClause andWC = andInfo.WC;
                        WhereClauseInit(andWC, wc.Parse, maskSet, wc.WctrlFlags);
                        WhereSplit(andWC, orTerm.Expr, TK.AND);
                        ExprAnalyzeAll(src, andWC);
                        SysEx.ASSERTCOVERAGE(ctx.MallocFailed);
                        if (!ctx.MallocFailed)
                        {
                            for (j = 0, andTerm = andWC.Slots[0]; j < andWC.Terms; j++, andTerm = andWC.Slots[j])
                            {
                                Debug.Assert(andTerm.Expr != null);
                                if (AllowedOp(andTerm.Expr.OP))
                                    b |= GetMask(maskSet, andTerm.LeftCursor);
                            }
                        }
                        indexable &= b;
                    }
                }
                else if ((orTerm.WtFlags & TERM.COPIED) != 0) { } // Skip this term for now.  We revisit it when we process the corresponding TERM_VIRTUAL term */
                else
                {
                    Bitmask b = GetMask(maskSet, orTerm.LeftCursor);
                    if ((orTerm.WtFlags & TERM.VIRTUAL) != 0)
                    {
                        WhereTerm pOther = orWc.Slots[orTerm.Parent];
                        b |= GetMask(maskSet, pOther.LeftCursor);
                    }
                    indexable &= b;
                    if (orTerm.EOperator != WO.EQ)
                        chngToIN = 0;
                    else
                        chngToIN &= b;
                }
            }

            // Record the set of tables that satisfy case 2.  The set might be empty.
            orInfo.Indexable = indexable;
            term.EOperator = (WO)(indexable == 0 ? 0 : WO.OR);

            // chngToIN holds a set of tables that *might* satisfy case 1.  But we have to do some additional checking to see if case 1 really
            // is satisfied.
            //
            // chngToIN will hold either 0, 1, or 2 bits.  The 0-bit case means that there is no possibility of transforming the OR clause into an
            // IN operator because one or more terms in the OR clause contain something other than == on a column in the single table.  The 1-bit
            // case means that every term of the OR clause is of the form "table.column=expr" for some single table.  The one bit that is set
            // will correspond to the common table.  We still need to check to make sure the same column is used on all terms.  The 2-bit case is when
            // the all terms are of the form "table1.column=table2.column".  It might be possible to form an IN operator with either table1.column
            // or table2.column as the LHS if either is common to every term of the OR clause.
            //
            // Note that terms of the form "table.column1=table.column2" (the same table on both sizes of the ==) cannot be optimized.
            if (chngToIN != 0)
            {
                bool okToChngToIN = false; // True if the conversion to IN is valid
                int columnId = -1; // Column index on lhs of IN operator
                int cursorId = -1; // Table cursor common to all terms
                int j = 0;

                // Search for a table and column that appears on one side or the other of the == operator in every subterm.  That table and column
                // will be recorded in iCursor and iColumn.  There might not be any such table and column.  Set okToChngToIN if an appropriate table
                // and column is found but leave okToChngToIN false if not found.
                for (j = 0; j < 2 && !okToChngToIN; j++)
                {
                    for (i = orWc.Terms - 1, orTerm = orWc.Slots[0]; i >= 0; i--, orTerm = orWc.Slots[orWc.Terms - 1 - i])
                    {
                        Debug.Assert(orTerm.EOperator == WO.EQ);
                        orTerm.WtFlags = (TERM)(orTerm.WtFlags & ~TERM.OR_OK);
                        if (orTerm.LeftCursor == cursorId)
                        {
                            // This is the 2-bit case and we are on the second iteration and current term is from the first iteration.  So skip this term.
                            Debug.Assert(j == 1);
                            continue;
                        }
                        if ((chngToIN & GetMask(maskSet, orTerm.LeftCursor)) == 0)
                        {
                            // This term must be of the form t1.a==t2.b where t2 is in the chngToIN set but t1 is not.  This term will be either preceeded
                            // or follwed by an inverted copy (t2.b==t1.a).  Skip this term and use its inversion.
                            SysEx.ASSERTCOVERAGE((orTerm.WtFlags & TERM.COPIED) != 0);
                            SysEx.ASSERTCOVERAGE((orTerm.WtFlags & TERM.VIRTUAL) != 0);
                            Debug.Assert((orTerm.WtFlags & (TERM.COPIED | TERM.VIRTUAL)) != 0);
                            continue;
                        }
                        columnId = orTerm.u.LeftColumn;
                        cursorId = orTerm.LeftCursor;
                        break;
                    }
                    if (i < 0)
                    {
                        // No candidate table+column was found.  This can only occur on the second iteration
                        Debug.Assert(j == 1);
                        Debug.Assert(IsPowerOfTwo(chngToIN));
                        Debug.Assert(chngToIN == GetMask(maskSet, cursorId));
                        break;
                    }
                    SysEx.ASSERTCOVERAGE(j == 1);

                    // We have found a candidate table and column.  Check to see if that table and column is common to every term in the OR clause
                    okToChngToIN = true;
                    for (; i >= 0 && okToChngToIN; i--, orTerm = orWc.Slots[orWc.Terms - 1 - i])
                    {
                        Debug.Assert(orTerm.EOperator == WO.EQ);
                        if (orTerm.LeftCursor != cursorId)
                            orTerm.WtFlags = (TERM)(orTerm.WtFlags & ~TERM.OR_OK);
                        else if (orTerm.u.LeftColumn != columnId)
                            okToChngToIN = false;
                        else
                        {
                            // If the right-hand side is also a column, then the affinities of both right and left sides must be such that no type
                            // conversions are required on the right.  (Ticket #2249)
                            AFF affRight = orTerm.Expr.Right.Affinity();
                            AFF affLeft = orTerm.Expr.Left.Affinity();
                            if (affRight != 0 && affRight != affLeft)
                                okToChngToIN = false;
                            else
                                orTerm.WtFlags |= TERM.OR_OK;
                        }
                    }
                }

                // At this point, okToChngToIN is true if original pTerm satisfies
                // case 1.  In that case, construct a new virtual term that is pTerm converted into an IN operator. EV: R-00211-15100
                if (okToChngToIN)
                {
                    Expr dup; // A transient duplicate expression
                    ExprList list = null; // The RHS of the IN operator
                    Expr left = null; // The LHS of the IN operator
                    for (i = orWc.Terms - 1, orTerm = orWc.Slots[0]; i >= 0; i--, orTerm = orWc.Slots[i])
                    {
                        if ((orTerm.WtFlags & TERM.OR_OK) == 0) continue;
                        Debug.Assert((orTerm.EOperator & WO.EQ) != 0);
                        Debug.Assert(orTerm.LeftCursor == cursorId);
                        Debug.Assert(orTerm.u.LeftColumn == columnId);
                        dup = Expr.Dup(ctx, orTerm.Expr.Right, 0);
                        list = Expr.ListAppend(wc.Parse, list, dup);
                        left = orTerm.Expr.Left;
                    }
                    Debug.Assert(left != null);
                    dup = Expr.Dup(ctx, left, 0);
                    Expr newExpr = Expr.PExpr_(parse, TK.IN, dup, null, null); // The complete IN operator
                    if (newExpr != null)
                    {
                        TransferJoinMarkings(newExpr, expr);
                        Debug.Assert(!E.ExprHasProperty(newExpr, EP.xIsSelect));
                        newExpr.x.List = list;
                        int idxNew = WhereClauseInsert(wc, newExpr, TERM.VIRTUAL | TERM.DYNAMIC);
                        SysEx.ASSERTCOVERAGE(idxNew == 0);
                        ExprAnalyze(src, wc, idxNew);
                        term = wc.Slots[idxTerm];
                        wc.Slots[idxNew].Parent = idxTerm;
                        term.Childs = 1;
                    }
                    else
                        Expr.ListDelete(ctx, ref list);
                    term.EOperator = WO.NOOP; // case 1 trumps case 2
                }
            }
        }
#endif

        static void ExprAnalyze(SrcList src, WhereClause wc, int idxTerm)
        {
            Parse parse = wc.Parse; // Parsing context
            Context ctx = parse.Ctx; // Data_base connection

            if (ctx.MallocFailed) return;
            WhereTerm term = wc.Slots[idxTerm]; // The term to be analyzed
            WhereMaskSet maskSet = wc.MaskSet; // Set of table index masks
            Expr expr = term.Expr; // The expression to be analyzed
            Bitmask prereqLeft = ExprTableUsage(maskSet, expr.Left); // Prerequesites of the pExpr.pLeft
            TK op = expr.OP; // Top-level operator.  pExpr.op
            if (op == TK.IN)
            {
                Debug.Assert(expr.Right == null);
                if (E.ExprHasProperty(expr, EP.xIsSelect))
                    term.PrereqRight = ExprSelectTableUsage(maskSet, expr.x.Select);
                else
                    term.PrereqRight = ExprListTableUsage(maskSet, expr.x.List);
            }
            else if (op == TK.ISNULL)
                term.PrereqRight = 0;
            else
                term.PrereqRight = ExprTableUsage(maskSet, expr.Right);
            Bitmask prereqAll = ExprTableUsage(maskSet, expr); // Prerequesites of pExpr
            Bitmask extraRight = 0; // Extra dependencies on LEFT JOIN
            if (E.ExprHasProperty(expr, EP.FromJoin))
            {
                Bitmask x = GetMask(maskSet, expr.RightJoinTable);
                prereqAll |= x;
                extraRight = x - 1; // ON clause terms may not be used with an index on left table of a LEFT JOIN.  Ticket #3015
            }
            term.PrereqAll = prereqAll;
            term.LeftCursor = -1;
            term.Parent = -1;
            term.EOperator = 0;
            if (AllowedOp(op) && (term.PrereqRight & prereqLeft) == 0)
            {
                Expr left = expr.Left;
                Expr right = expr.Right;
                WO opMask = ((term.PrereqRight & prereqLeft) == 0 ? WO.ALL : WO.EQUIV);
                if (left.OP == TK.COLUMN)
                {
                    term.LeftCursor = left.TableIdx;
                    term.u.LeftColumn = left.ColumnIdx;
                    term.EOperator = OperatorMask(op) & opMask;
                }
                if (right != null && right.OP == TK.COLUMN)
                {
                    WhereTerm newTerm;
                    Expr dup;
                    WO extraOp = 0; // Extra bits for newTerm->eOperator
                    if (term.LeftCursor >= 0)
                    {
                        dup = Expr.Dup(ctx, expr, 0);
                        if (ctx.MallocFailed)
                        {
                            Expr.Delete(ctx, ref dup);
                            return;
                        }
                        int idxNew = WhereClauseInsert(wc, dup, TERM.VIRTUAL | TERM.DYNAMIC);
                        if (idxNew == 0) return;
                        newTerm = wc.Slots[idxNew];
                        newTerm.Parent = idxTerm;
                        term = wc.Slots[idxTerm];
                        term.Childs = 1;
                        term.WtFlags |= TERM.COPIED;
                        if (expr.OP == TK.EQ && !E.ExprHasProperty(expr, EP.FromJoin) && E.CtxOptimizationEnabled(ctx, OPTFLAG.Transitive))
                        {
                            term.EOperator |= WO.EQUIV;
                            extraOp = WO.EQUIV;
                        }
                    }
                    else
                    {
                        dup = expr;
                        newTerm = term;
                    }
                    ExprCommute(parse, dup);
                    left = dup.Left;
                    newTerm.LeftCursor = left.TableIdx;
                    newTerm.u.LeftColumn = left.ColumnIdx;
                    SysEx.ASSERTCOVERAGE((prereqLeft | extraRight) != prereqLeft);
                    newTerm.PrereqRight = prereqLeft | extraRight;
                    newTerm.PrereqAll = prereqAll;
                    newTerm.EOperator = (OperatorMask(dup.OP) | extraOp) & opMask;
                }
            }

#if  !OMIT_BETWEEN_OPTIMIZATION
            // If a term is the BETWEEN operator, create two new virtual terms that define the range that the BETWEEN implements.  For example:
            //      a BETWEEN b AND c
            //
            // is converted into:
            //      (a BETWEEN b AND c) AND (a>=b) AND (a<=c)
            //
            // The two new terms are added onto the end of the WhereClause object. The new terms are "dynamic" and are children of the original BETWEEN
            // term.  That means that if the BETWEEN term is coded, the children are skipped.  Or, if the children are satisfied by an index, the original
            // BETWEEN term is skipped.
            else if (expr.OP == TK.BETWEEN && wc.OP == TK.AND)
            {
                ExprList list = expr.x.List;
                TK[] ops = new[] { TK.GE, TK.LE };
                Debug.Assert(list != null);
                Debug.Assert(list.Exprs == 2);
                for (int i = 0; i < 2; i++)
                {
                    Expr newExpr = Expr.PExpr_(parse, ops[i], Expr.Dup(ctx, expr.Left, 0), Expr.Dup(ctx, list.Ids[i].Expr, 0), null);
                    int idxNew = WhereClauseInsert(wc, newExpr, TERM.VIRTUAL | TERM.DYNAMIC);
                    SysEx.ASSERTCOVERAGE(idxNew == 0);
                    ExprAnalyze(src, wc, idxNew);
                    term = wc.Slots[idxTerm];
                    wc.Slots[idxNew].Parent = idxTerm;
                }
                term.Childs = 2;
            }
#endif
#if !OMIT_OR_OPTIMIZATION && !OMIT_SUBQUERY
            // Analyze a term that is composed of two or more subterms connected by an OR operator.
            else if (expr.OP == TK.OR)
            {
                Debug.Assert(wc.OP == TK.AND);
                ExprAnalyzeOrTerm(src, wc, idxTerm);
                term = wc.Slots[idxTerm];
            }
#endif
#if !OMIT_LIKE_OPTIMIZATION
            // Add constraints to reduce the search space on a LIKE or GLOB operator.
            //
            // A like pattern of the form "x LIKE 'abc%'" is changed into constraints
            //          x>='abc' AND x<'abd' AND x LIKE 'abc%'
            // The last character of the prefix "abc" is incremented to form the termination condition "abd".
            Expr str1 = null; // RHS of LIKE/GLOB operator
            bool isComplete = false; // RHS of LIKE/GLOB ends with wildcard
            bool noCase = false; // LIKE/GLOB distinguishes case
            if (wc.OP == TK.AND && IsLikeOrGlob(parse, expr, ref str1, ref isComplete, ref noCase) != 0)
            {
                Expr left = expr.x.List.Ids[1].Expr; // LHS of LIKE/GLOB operator
                Expr str2 = Expr.Dup(ctx, str1, 0); // Copy of pStr1 - RHS of LIKE/GLOB operator
                if (!ctx.MallocFailed)
                {
                    char cRef = str2.u.Token[str2.u.Token.Length - 1];
                    char c = cRef; // Last character before the first wildcard
                    if (noCase)
                    {
                        // The point is to increment the last character before the first wildcard.  But if we increment '@', that will push it into the
                        // alphabetic range where case conversions will mess up the inequality.  To avoid this, make sure to also run the full
                        // LIKE on all candidate expressions by clearing the isComplete flag
                        if (c == 'A' - 1) isComplete = false; // EV: R-64339-08207
                        c = char.ToLowerInvariant(c);
                    }
                    str2.u.Token = str2.u.Token.Substring(0, str2.u.Token.Length - 1) + (char)(c + 1); //: *cRef = c + 1;
                }
                Token sCollSeqName; // Name of collating sequence
                sCollSeqName.data = (noCase ? "NOCASE" : "BINARY");
                sCollSeqName.length = 6;
                Expr newExpr1 = Expr.Dup(ctx, left, 0);
                newExpr1 = Expr.PExpr_(parse, TK.GE, newExpr1.AddCollateToken(parse, sCollSeqName), str1, 0);
                int idxNew1 = WhereClauseInsert(wc, newExpr1, TERM.VIRTUAL | TERM.DYNAMIC);
                SysEx.ASSERTCOVERAGE(idxNew1 == 0);
                ExprAnalyze(src, wc, idxNew1);
                Expr newExpr2 = Expr.Dup(ctx, left, 0);
                newExpr2 = Expr.PExpr_(parse, TK.LT, newExpr2.AddCollateToken(parse, sCollSeqName), str2, null);
                int idxNew2 = WhereClauseInsert(wc, newExpr2, TERM.VIRTUAL | TERM.DYNAMIC);
                SysEx.ASSERTCOVERAGE(idxNew2 == 0);
                ExprAnalyze(src, wc, idxNew2);
                term = wc.Slots[idxTerm];
                if (isComplete)
                {
                    wc.Slots[idxNew1].Parent = idxTerm;
                    wc.Slots[idxNew2].Parent = idxTerm;
                    term.Childs = 2;
                }
            }
#endif
#if !OMIT_VIRTUALTABLE
            // Add a WO_MATCH auxiliary term to the constraint set if the current expression is of the form:  column MATCH expr.
            // This information is used by the xBestIndex methods of virtual tables.  The native query optimizer does not attempt
            // to do anything with MATCH functions.
            if (IsMatchOfColumn(expr))
            {
                Expr right = expr.x.List.Ids[0].Expr;
                Expr left = expr.x.List.Ids[1].Expr;
                Bitmask prereqExpr = ExprTableUsage(maskSet, right);
                Bitmask prereqColumn = ExprTableUsage(maskSet, left);
                if ((prereqExpr & prereqColumn) == 0)
                {
                    Expr newExpr = Expr.PExpr_(parse, TK.MATCH, null, Expr.Dup(ctx, right, 0), null);
                    int idxNew = WhereClauseInsert(wc, newExpr, TERM.VIRTUAL | TERM.DYNAMIC);
                    SysEx.ASSERTCOVERAGE(idxNew == 0);
                    WhereTerm newTerm = wc.Slots[idxNew];
                    newTerm.PrereqRight = prereqExpr;
                    newTerm.LeftCursor = left.TableIdx;
                    newTerm.u.LeftColumn = left.ColumnIdx;
                    newTerm.EOperator = WO.MATCH;
                    newTerm.Parent = idxTerm;
                    term = wc.Slots[idxTerm];
                    term.Childs = 1;
                    term.WtFlags |= TERM.COPIED;
                    newTerm.PrereqAll = term.PrereqAll;
                }
            }
#endif
#if ENABLE_STAT3
            // When sqlite_stat3 histogram data is available an operator of the form "x IS NOT NULL" can sometimes be evaluated more efficiently
            // as "x>NULL" if x is not an INTEGER PRIMARY KEY.  So construct a virtual term of that form.
            //
            // Note that the virtual term must be tagged with TERM_VNULL.  This TERM_VNULL tag will suppress the not-null check at the beginning
            // of the loop.  Without the TERM_VNULL flag, the not-null check at the start of the loop will prevent any results from being returned.
            if (expr.OP == TK.NOTNULL && expr.Left.OP == TK.COLUMN && expr.Left.ColumnIdx >= 0)
            {
                Expr left = expr.Left;
                Expr newExpr = Expr.PExpr_(parse, TK.GT, Expr.Dup(ctx, left, 0), Expr.PExpr_(parse, TK.NULL, 0, 0, 0), 0);
                int idxNew = WhereClauseInsert(wc, newExpr, TERM.VIRTUAL | TERM.DYNAMIC | TERM.VNULL);
                if (idxNew != 0)
                {
                    WhereTerm newTerm = wc.Slots[idxNew];
                    newTerm.PrereqRight = 0;
                    newTerm.LeftCursor = left.TableIdx;
                    newTerm.u.LeftColumn = left.ColumnIdx;
                    newTerm.EOperator = WO.GT;
                    newTerm.Parent = idxTerm;
                    term = wc.Slots[idxTerm];
                    term.Childs = 1;
                    term.WtFlags |= TERM.COPIED;
                    newTerm.PrereqAll = term.PrereqAll;
                }
            }
#endif

            // Prevent ON clause terms of a LEFT JOIN from being used to drive an index for tables to the left of the join.
            term.PrereqRight |= extraRight;
        }

        static int FindIndexCol(Parse parse, ExprList list, int baseId, Index index, int column)
        {
            string collName = index.CollNames[column];
            for (int i = 0; i < list.Exprs; i++)
            {
                Expr expr = list.Ids[i].Expr.SkipCollate();
                if (expr.OP == TK.COLUMN && expr.ColumnIdx == index.Columns[column] && expr.TableIdx == baseId)
                {
                    CollSeq coll = list.Ids[i].Expr.CollSeq(parse);
                    if (SysEx.ALWAYS(coll != null) && string.Equals(coll.Name, collName))
                        return i;
                }
            }
            return -1;
        }

        static bool IsDistinctIndex(Parse parse, WhereClause wc, Index index, int baseId, ExprList distinct, int eqCols)
        {
            Debug.Assert(distinct != null);
            if (index.Name == null || distinct.Exprs >= BMS) return false;
            SysEx.ASSERTCOVERAGE(distinct.Exprs == BMS - 1);

            // Loop through all the expressions in the distinct list. If any of them are not simple column references, return early. Otherwise, test if the
            // WHERE clause contains a "col=X" clause. If it does, the expression can be ignored. If it does not, and the column does not belong to the
            // same table as index pIdx, return early. Finally, if there is no matching "col=X" expression and the column is on the same table as pIdx,
            // set the corresponding bit in variable mask.
            int i;
            Bitmask mask = 0; // Mask of unaccounted for pDistinct exprs
            for (i = 0; i < distinct.Exprs; i++)
            {
                Expr expr = distinct.Ids[i].Expr.SkipCollate();
                if (expr.OP != TK.COLUMN) return false;
                WhereTerm term = FindTerm(wc, expr.TableIdx, expr.ColumnIdx, ~(Bitmask)0, WO.EQ, 0);
                if (term != null)
                {
                    Expr x = term.Expr;
                    CollSeq p1 = Expr.BinaryCompareCollSeq(parse, x.Left, x.Right);
                    CollSeq p2 = expr.CollSeq(parse);
                    if (p1 == p2) continue;
                }
                if (expr.TableIdx != baseId) return false;
                mask |= (((Bitmask)1) << i);
            }
            for (i = eqCols; mask != null && i < index.Columns.length; i++)
            {
                int exprId = FindIndexCol(parse, distinct, baseId, index, i);
                if (exprId < 0) break;
                mask &= ~(((Bitmask)1) << exprId);
            }
            return (mask == 0);
        }

        static bool IsDistinctRedundant(Parse parse, SrcList list, WhereClause wc, ExprList distinct)
        {
            // If there is more than one table or sub-select in the FROM clause of this query, then it will not be possible to show that the DISTINCT 
            // clause is redundant.
            if (list.Srcs != 1) return false;
            int baseId = list.Ids[0].Cursor;
            Table table = list.Ids[0].Table;

            // If any of the expressions is an IPK column on table iBase, then return true. Note: The (p->iTable==iBase) part of this test may be false if the
            // current SELECT is a correlated sub-query.
            int i;
            for (i = 0; i < distinct.Exprs; i++)
            {
                Expr expr = distinct.Ids[i].Expr.SkipCollate();
                if (expr.OP == TK.COLUMN && expr.TableIdx == baseId && expr.ColumnIdx < 0) return true;
            }

            // Loop through all indices on the table, checking each to see if it makes the DISTINCT qualifier redundant. It does so if:
            //   1. The index is itself UNIQUE, and
            //   2. All of the columns in the index are either part of the pDistinct list, or else the WHERE clause contains a term of the form "col=X",
            //      where X is a constant value. The collation sequences of the comparison and select-list expressions must match those of the index.
            //   3. All of those index columns for which the WHERE clause does not contain a "col=X" term are subject to a NOT NULL constraint.
            for (Index index = table.Index; index != null; index = index.Next)
            {
                if (index.OnError == OE.None) continue;
                for (i = 0; i < index.Columns.length; i++)
                {
                    int column = index.Columns[i];
                    if (!FindTerm(wc, baseId, column, ~(Bitmask)0, WO.EQ, index))
                    {
                        int indexColumn = FindIndexCol(parse, distinct, baseId, index, i);
                        if (indexColumn < 0 || table.Cols[index.Columns[i]].NotNull == 0)
                            break;
                    }
                }
                if (i == index.Columns.length) // This index implies that the DISTINCT qualifier is redundant.
                    return true;
            }
            return false;
        }

        static double EstLog(double n)
        {
            double logN = 1;
            double x = 10;
            while (n > x)
            {
                logN += 1;
                x *= 10;
            }
            return logN;
        }

#if false && !OMIT_VIRTUALTABLE && DEBUG
        static void TRACE_IDX_INPUTS(sqlite3_index_info p)
        {
            if (!sqlite3WhereTrace) return;
            int i;
            for (i = 0; i < p.nConstraint; i++)
            {
                sqlite3DebugPrintf("  constraint[%d]: col=%d termid=%d op=%d usabled=%d\n",
                i,
                p.Constraints[i].ColumnIdx,
                p.Constraints[i].TermOffset,
                p.Constraints[i].OP,
                p.Constraints[i].Usable);
            }
            for (i = 0; i < p.OrderBys.length; i++)
            {
                sqlite3DebugPrintf("  orderby[%d]: col=%d desc=%d\n",
                i,
                p.OrderBys[i].ColumnIdx,
                p.OrderBys[i].Desc);
            }
        }
        static void TRACE_IDX_OUTPUTS(sqlite3_index_info p)
        {
            if (!sqlite3WhereTrace) return;
            int i;
            for (i = 0; i < p.Constraints.length; i++)
            {
                sqlite3DebugPrintf("  usage[%d]: argvIdx=%d omit=%d\n",
                i,
                p.ConstraintUsages[i].ArgvIndex,
                p.ConstraintUsages[i].Omit);
            }
            sqlite3DebugPrintf("  idxNum=%d\n", p.idxNum);
            sqlite3DebugPrintf("  idxStr=%s\n", p.idxStr);
            sqlite3DebugPrintf("  orderByConsumed=%d\n", p.orderByConsumed);
            sqlite3DebugPrintf("  estimatedCost=%g\n", p.estimatedCost);
        }
#else
        static void TRACE_IDX_INPUTS(sqlite3_index_info p) { }
        static void TRACE_IDX_OUTPUTS(sqlite3_index_info p) { }
#endif

        //static void BestOrClauseIndex(Parse parse, , Bitmask notReady, Bitmask notValid, ExprList orderBy, WhereCost cost)
        static void BestOrClauseIndex(WhereBestIdx p)
        {
#if !OMIT_OR_OPTIMIZATION
            WhereClause wc = p.WC; // The WHERE clause
            SrcList.SrcListItem src = p.Src; // The FROM clause term to search
            int cursor = src.Cursor; // The cursor of the table
            Bitmask maskSrc = GetMask(wc.MaskSet, cursor); // Bitmask for pSrc

            // No OR-clause optimization allowed if the INDEXED BY or NOT INDEXED clauses are used
            if (src.NotIndexed || src.Index != null)
                return;
            if ((wc.WctrlFlags & WHERE.AND_ONLY) != 0)
                return;

            // Search the WHERE clause terms for a usable WO_OR term.
            WhereTerm term;
            int _i0; //: WhereTerm wcEnd = wc.Slots[wc.Terms]; // End of pWC.a[]
            for (_i0 = 0, term = wc.Slots[_i0]; _i0 < wc.Terms; _i0++, term = wc.Slots[_i0])
            {
                if ((term.EOperator & WO.OR) != 0 && ((term.PrereqAll & ~maskSrc) & p.NotReady) == 0 && (term.u.OrInfo.Indexable & maskSrc) != 0)
                {
                    uint flags = WHERE_MULTI_OR;
                    double total = 0;
                    double rows = 0;
                    Bitmask used = 0;

                    WhereBestIdx sBOI;
                    sBOI = p;
                    sBOI.OrderBy = null;
                    sBOI.Distinct = null;
                    sBOI.IdxInfo = null;

                    WhereClause orWC = term.u.OrInfo.WC;
                    WhereTerm orTerm;
                    int _i1; //: WhereTerm orWCEnd = orWC.Slots[orWC.Terms];
                    for (_i1 = 0, orTerm = orWC.Slots[_i1]; _i1 < orWC.Terms; _i1++, orTerm = orWC.Slots[_i1])
                    {
                        WhereCost sTermCost = null;
                        WHERETRACE("... Multi-index OR testing for term %d of %d....\n", _i1, _i0);
                        if ((orTerm.EOperator & WO.AND) != 0)
                        {
                            sBOI.WC = orTerm.u.AndInfo.WC;
                            BestIndex(sBOI);
                        }
                        else if (orTerm.LeftCursor == cursor)
                        {
                            WhereClause tempWC = new WhereClause();
                            tempWC.Parse = wc.Parse;
                            tempWC.MaskSet = wc.MaskSet;
                            tempWC.OP = TK.AND;
                            tempWC.Slots.data = new WhereTerm[2];
                            tempWC.Slots[0] = orTerm;
                            tempWC.WctrlFlags = 0;
                            tempWC.Terms = 1;
                            sBOI.WC = tempWC;
                            BestIndex(sBOI);
                        }
                        else
                            continue;

                        total += sBOI.Cost.Cost;
                        rows += sBOI.Cost.Plan.Rows;
                        used |= sBOI.Cost.Used;
                        if (total >= p.Cost.Cost) break;
                    }

                    // If there is an ORDER BY clause, increase the scan cost to account for the cost of the sort.
                    if (p.OrderBy != null)
                    {
                        WHERETRACE("... sorting increases OR cost %.9g to %.9g\n", total, total + rows * EstLog(rows));
                        total += rows * EstLog(rows);
                    }

                    // If the cost of scanning using this OR term for optimization is less than the current cost stored in pCost, replace the contents of pCost.
                    WHERETRACE("... multi-index OR cost=%.9g nrow=%.9g\n", total, rows);
                    if (total < p.Cost.Cost)
                    {
                        p.Cost.Cost = total;
                        p.Cost.Used = used;
                        p.Cost.Plan.Rows = rows;
                        p.Cost.Plan.OBSats = (p.i ? p.Levels[p.i - 1].Plan.OBSats : 0);
                        p.Cost.Plan.WsFlags = flags;
                        p.Cost.Plan.u.Term = term;
                    }
                }
            }
#endif
        }

#if !OMIT_AUTOMATIC_INDEX
        static bool TermCanDriveIndex(WhereTerm term, SrcList.SrcListItem src, Bitmask notReady)
        {
            if (term.LeftCursor != src.Cursor) return false;
            if ((term.EOperator & WO.EQ) != 0) return false;
            if ((term.PrereqRight & notReady) != 0) return false;
            AFF aff = src.Table.Cols[term.u.LeftColumn].Affinity;
            if (!term.Expr.ValidIndexAffinity(aff)) return false;
            return true;
        }
#endif

#if !OMIT_AUTOMATIC_INDEX
        static void BestAutomaticIndex(WhereBestIdx p)
        {
            Parse parse = p.Parse; // The parsing context
            WhereClause wc = p.WC; // The WHERE clause
            SrcList.SrcListItem src = p.Src; // The FROM clause term to search

            if (parse.QueryLoops <= (double)1) return; // There is no point in building an automatic index for a single scan
            if ((parse.Ctx.Flags & Context.FLAG.AutoIndex) == 0) return; // Automatic indices are disabled at run-time
            if ((p.Cost.Plan.WsFlags & WHERE_NOT_FULLSCAN) != 0 && (p.Cost.Plan.WsFlags & WHERE_COVER_SCAN) == 0) return; // We already have some kind of index in use for this query.
            if (src.ViaCoroutine) return; // Cannot index a co-routine
            if (src.NotIndexed) return; // The NOT INDEXED clause appears in the SQL.
            if (src.IsCorrelated) return; // The source is a correlated sub-query. No point in indexing it.

            Debug.Assert(parse.QueryLoops >= (double)1);
            Table table = src.Table; // Table that might be indexed
            double tableRows = table.RowEst; // Rows in the input table
            double logN = EstLog(tableRows); // log(nTableRow)
            double costTempIdx = 2 * logN * (tableRows / parse.QueryLoops + 1); // per-query cost of the transient index
            if (costTempIdx >= p.Cost.Cost) return; // The cost of creating the transient table would be greater than doing the full table scan

            // Search for any equality comparison term
            WhereTerm term;
            int _i0; //: WhereTerm pWCEnd = pWC.a[pWC.nTerm]; // End of pWC.a[]
            for (_i0 = 0, term = wc.Slots[_i0]; _i0 < wc.Terms; _i0++, term = wc.Slots[_i0])
                if (TermCanDriveIndex(term, src, p.NotReady))
                {
                    WHERETRACE("auto-index reduces cost from %.2f to %.2f\n", p.Cost.Cost, costTempIdx);
                    p.Cost.Cost = costTempIdx;
                    p.Cost.Plan.Rows = logN + 1;
                    p.Cost.Plan.WsFlags = WHERE_TEMP_INDEX;
                    p.Cost.Used = term.PrereqRight;
                    break;
                }
        }
#else
        static void BestAutomaticIndex(WhereBestIdx p) { } // no-op
#endif

#if !OMIT_AUTOMATIC_INDEX
        static void ConstructAutomaticIndex(Parse parse, WhereClause wc, SrcList.SrcListItem src, Bitmask notReady, WhereLevel level)
        {
            int regIsInit;             // Register set by initialization

            int addrTop;               // Top of the index fill loop
            int regRecord;             // Register holding an index record

            // Generate code to skip over the creation and initialization of the transient index on 2nd and subsequent iterations of the loop.
            Vdbe v = parse.V; // Prepared statement under construction
            Debug.Assert(v != null);
            int addrInit = Expr.CodeOnce(parse); // Address of the initialization bypass jump

            // Count the number of columns that will be added to the index and used to match WHERE clause constraints
            int columns = 0; // Number of columns in the constructed index
            Table table = src.Table; // The table being indexed
            Bitmask idxCols = 0; // Bitmap of columns used for indexing
            WhereTerm term; // A single term of the WHERE clause
            int _i0; //: WhereTerm wcEnd = wc.Slots[wc.Terms]; // End of pWC.a[]
            for (_i0 = 0, term = wc.Slots[0]; _i0 < wc.Terms; _i0++, term = wc.Slots[_i0])
            {
                if (TermCanDriveIndex(term, src, notReady))
                {
                    int column = term.u.LeftColumn;
                    Bitmask mask = (column >= BMS ? ((Bitmask)1) << (BMS - 1) : ((Bitmask)1) << column);
                    SysEx.ASSERTCOVERAGE(column == BMS);
                    SysEx.ASSERTCOVERAGE(column == BMS - 1);
                    if ((idxCols & mask) == 0)
                    {
                        columns++;
                        idxCols |= mask;
                    }
                }
            }
            Debug.Assert(columns > 0);
            level.Plan.Eqs = (ushort)columns;

            // Count the number of additional columns needed to create a covering index.  A "covering index" is an index that contains all
            // columns that are needed by the query.  With a covering index, the original table never needs to be accessed.  Automatic indices must
            // be a covering index because the index will not be updated if the original table changes and the index and table cannot both be used
            // if they go out of sync.
            Bitmask extraCols = src.ColUsed & (~idxCols | (((Bitmask)1) << (BMS - 1))); // Bitmap of additional columns
            int maxBitCol = (table.Cols.length >= BMS - 1 ? BMS - 1 : table.Cols.length); // Maximum column in pSrc.colUsed
            SysEx.ASSERTCOVERAGE(table.Cols.length == BMS - 1);
            SysEx.ASSERTCOVERAGE(table.Cols.length == BMS - 2);
            int i;
            for (i = 0; i < maxBitCol; i++)
                if ((extraCols & (((Bitmask)1) << i)) != 0) columns++;
            if ((src.ColUsed & (((Bitmask)1) << (BMS - 1))) != 0)
                columns += table.Cols.length - BMS + 1;
            level.Plan.WsFlags |= WHERE_COLUMN_EQ | WHERE_IDX_ONLY | WO.EQ;

            // Construct the Index object to describe this index
            //: int bytes = sizeof(Index); // Byte of memory needed for pIdx
            //: bytes += columns * sizeof(int); // Index.aiColumn
            //: bytes += columns * sizeof(char*); // Index.azColl
            //: bytes += columns; // Index.aSortOrder
            Index index = new Index(); // Object describing the transient index
            if (index == null) return;
            index = new Index();
            level.Plan.u.Index = index;
            index.CollNames = new string[columns + 1];
            index.Columns.data = new int[columns + 1];
            index.SortOrders = new byte[columns + 1];
            index.Name = "auto-index";
            index.Columns.length = (ushort)columns;
            index.Table = table;
            int n = 0; // Column counter
            idxCols = 0;
            for (_i0 = 0, term = wc.Slots[0]; _i0 < wc.Terms; _i0++, term = wc.Slots[_i0])
            {
                if (TermCanDriveIndex(term, src, notReady))
                {
                    int column = term.u.LeftColumn;
                    Bitmask mask = (column >= BMS ? ((Bitmask)1) << (BMS - 1) : ((Bitmask)1) << column);
                    if ((idxCols & mask) == 0)
                    {
                        Expr x = term.Expr;
                        idxCols |= mask;
                        index.Columns[n] = term.u.LeftColumn;
                        CollSeq coll = Expr.BinaryCompareCollSeq(parse, x.Left, x.Right); // Collating sequence to on a column
                        index.CollNames[n] = (SysEx.ALWAYS(coll != null) ? coll.Name : "BINARY");
                        n++;
                    }
                }
            }
            Debug.Assert(level.Plan.Eqs == (uint)n);

            // Add additional columns needed to make the automatic index into a covering index
            for (i = 0; i < maxBitCol; i++)
            {
                if ((extraCols & (((Bitmask)1) << i)) != 0)
                {
                    index.Columns[n] = i;
                    index.CollNames[n] = "BINARY";
                    n++;
                }
            }
            if ((src.ColUsed & (((Bitmask)1) << (BMS - 1))) != 0)
            {
                for (i = BMS - 1; i < table.Cols.length; i++)
                {
                    index.Columns[n] = i;
                    index.CollNames[n] = "BINARY";
                    n++;
                }
            }
            Debug.Assert(n == columns);

            // Create the automatic index
            KeyInfo keyinfo = sqlite3IndexKeyinfo(parse, index); // Key information for the index
            Debug.Assert(level.IdxCur >= 0);
            v.AddOp4(OP.OpenAutoindex, level.IdxCur, columns + 1, 0, keyinfo, Vdbe.P4T.KEYINFO_HANDOFF);
            VdbeComment(v, "for %s", table.Name);

            // Fill the automatic index with content
            int addrTop = v.AddOp1(OP.Rewind, level.TabCur); // Top of the index fill loop
            int regRecord = parse.GetTempReg(); // Register holding an index record
            sqlite3GenerateIndexKey(parse, index, level.TabCur, regRecord, true);
            v.AddOp2(OP.IdxInsert, level.IdxCur, regRecord);
            v.ChangeP5(OPFLAG.USESEEKRESULT);
            v.AddOp2(OP.Next, level.TabCur, addrTop + 1);
            v.ChangeP5(STMTSTATUS_AUTOINDEX);
            v.JumpHere(addrTop);
            parse.ReleaseTempReg(regRecord);

            // Jump here when skipping the initialization
            v.JumpHere(addrInit);
        }
#endif

        #region NOTDONE

#if !OMIT_VIRTUALTABLE
        static IIndexInfo AllocateIndexInfo(WhereBestIdx p)
        {
            Parse parse = p.Parse;
            WhereClause wc = p.WC;
            SrcList.SrcListItem src = p.Src;
            ExprList orderBy = p.OrderBy;
            WHERETRACE("Recomputing index info for %s...\n", src.Table.Name);

            // following Debug.Asserts verify this fact.
            Debug.Assert((int)WO.EQ == (int)INDEX_CONSTRAINT.EQ);
            Debug.Assert((int)WO.LT == (int)INDEX_CONSTRAINT.LT);
            Debug.Assert((int)WO.LE == (int)INDEX_CONSTRAINT.LE);
            Debug.Assert((int)WO.GT == (int)INDEX_CONSTRAINT.GT);
            Debug.Assert((int)WO.GE == (int)INDEX_CONSTRAINT.GE);
            Debug.Assert((int)WO.MATCH == (int)INDEX_CONSTRAINT.MATCH);

            // Count the number of possible WHERE clause constraints referring to this virtual table
            int i;
            int terms;
            WhereTerm term;
            for (i = terms = 0, term = wc.Slots[0]; i < wc.Terms; i++, term = wc.Slots[i])
            {
                if (term.LeftCursor != src.Cursor) continue;
                Debug.Assert(IsPowerOfTwo(term.EOperator & ~WO.EQUIV));
                SysEx.ASSERTCOVERAGE((term.EOperator & WO.IN) != 0);
                SysEx.ASSERTCOVERAGE((term.EOperator & WO.ISNULL) != 0);
                if ((term.EOperator & WO.ISNULL) != 0) continue;
                if ((term.WtFlags & TERM.VNULL) != 0) continue;
                terms++;
            }

            // If the ORDER BY clause contains only columns in the current virtual table then allocate space for the aOrderBy part of
            // the sqlite3_index_info structure.
            int orderBys = 0;
            if (orderBy != null)
            {
                int n = orderBy.Exprs;
                for (i = 0; i < n; i++)
                {
                    Expr expr = orderBy.Ids[i].Expr;
                    if (expr.OP != TK.COLUMN || expr.TableIdx != src.Cursor) break;
                }
                if (i == n)
                    orderBys = n;
            }

            // Allocate the sqlite3_index_info structure
            IIndexInfo idxInfo = new IIndexInfo();
            if (idxInfo == null)
            {
                parse.ErrorMsg("out of memory");
                return null; // (double)0 In case of SQLITE_OMIT_FLOATING_POINT...
            }

            // Initialize the structure.  The sqlite3_index_info structure contains many fields that are declared "const" to prevent xBestIndex from
            // changing them.  We have to do some funky casting in order to initialize those fields.
            IIndexInfo.Constraint[] idxCons = new IIndexInfo.Constraint[terms];
            IIndexInfo.Orderby[] idxOrderBy = new IIndexInfo.Orderby[orderBys];
            IIndexInfo.ConstraintUsage[] usage = new IIndexInfo.ConstraintUsage[terms];
            idxInfo.Constraints.length = terms;
            idxInfo.OrderBys.length = orderBys;
            idxInfo.Constraints.data = idxCons;
            idxInfo.OrderBys.data = idxOrderBy;
            idxInfo.ConstraintUsages.data = usage;
            int j;
            for (i = j = 0, term = wc.Slots[0]; i < wc.Terms; i++, term = wc.Slots[i])
            {
                if (term.LeftCursor != src.Cursor) continue;
                Debug.Assert((term.EOperator & (term.EOperator - 1)) == 0);
                SysEx.ASSERTCOVERAGE((term.EOperator & WO.IN) != 0);
                SysEx.ASSERTCOVERAGE((term.EOperator & WO.ISNULL) != 0);
                if ((term.EOperator & WO.ISNULL) != 0) continue;
                if ((term.WtFlags & TERM.VNULL) != 0) continue;
                if (idxCons[j] == null)
                    idxCons[j] = new IIndexInfo.Constraint();
                idxCons[j].Column = term.u.LeftColumn;
                idxCons[j].TermOffset = i;
                WO op = (WO)term.EOperator & WO.ALL;
                if (op == WO.IN) op = WO.EQ;
                idxCons[j].OP = (INDEX_CONSTRAINT)op;
                // The direct Debug.Assignment in the previous line is possible only because the WO_ and SQLITE_INDEX_CONSTRAINT_ codes are identical.
                Debug.Assert((op & (WO.EQ | WO.LT | WO.LE | WO.GT | WO.GE | WO.MATCH)) != 0);
                j++;
            }
            for (i = 0; i < orderBys; i++)
            {
                Expr expr = orderBy.Ids[i].Expr;
                if (idxOrderBy[i] == null)
                    idxOrderBy[i] = new IIndexInfo.Orderby();
                idxOrderBy[i].Column = expr.ColumnIdx;
                idxOrderBy[i].Desc = (orderBy.Ids[i].SortOrder != 0);
            }
            return idxInfo;
        }

        static int VTableBestIndex(Parse parse, Table table, IIndexInfo p)
        {
            IVTable vtable = VTable.GetTable(parse.Ctx, table).IVTable;
            WHERETRACE("xBestIndex for %s\n", pTab.zName);
            TRACE_IDX_INPUTS(p);
            RC rc = vtable.IModule.BestIndex(vtable, p);
            TRACE_IDX_OUTPUTS(p);
            if (rc != RC.OK)
            {
                if (rc == RC.NOMEM) parse.Ctx.MallocFailed = true;
                else if (vtable.ErrMsg != null) parse.ErrorMsg("%s", ErrStr(rc));
                else parse.ErrorMsg("%s", vtable.ErrMsg);
            }
            SysEx.Free(vtable.ErrMsg);
            vtable.ErrMsg = null;
            for (int i = 0; i < p.Constraints.length; i++)
                if (!p.Constraints[i].Usable && p.ConstraintUsages[i].ArgvIndex > 0)
                    parse.ErrorMsg("table %s: xBestIndex returned an invalid plan", table.Name);
            return parse.Errs;
        }

        static void BestVirtualIndex(WhereBestIdx p)
        {
            Parse parse = p.Parse; // The parsing context
            WhereClause wc = p.WC; // The WHERE clause
            SrcList.SrcListItem src = p.Src; // The FROM clause term to search
            Table table = src.Table;

            // Make sure wsFlags is initialized to some sane value. Otherwise, if the malloc in allocateIndexInfo() fails and this function returns leaving
            // wsFlags in an uninitialized state, the caller may behave unpredictably.
            p.Cost = new WhereCost();
            p.Cost.Plan.WsFlags = WHERE_VIRTUALTABLE;

            // If the sqlite3_index_info structure has not been previously allocated and initialized, then allocate and initialize it now.
            IIndexInfo idxInfo = p.IdxInfo[0];
            if (idxInfo == null)
                p.IdxInfo[0] = idxInfo = AllocateIndexInfo(p);
            // At this point, the sqlite3_index_info structure that pIdxInfo points to will have been initialized, either during the current invocation or
            // during some prior invocation.  Now we just have to customize the details of pIdxInfo for the current invocation and pass it to xBestIndex.
            if (idxInfo == null) return;


            // The module name must be defined. Also, by this point there must be a pointer to an sqlite3_vtab structure. Otherwise
            // sqlite3ViewGetColumnNames() would have picked up the error. 
            Debug.Assert(table.ModuleArgs.data != null && table.ModuleArgs[0] != null);
            Debug.Assert(VTable.GetVTable(parse.Ctx, table) != null);

            // Try once or twice.  On the first attempt, allow IN optimizations. If an IN optimization is accepted by the virtual table xBestIndex
            // method, but the  pInfo->aConstrainUsage.omit flag is not set, then the query will not work because it might allow duplicate rows in
            // output.  In that case, run the xBestIndex method a second time without the IN constraints.  Usually this loop only runs once.
            // The loop will exit using a "break" statement.
            SO sortOrder;
            int orderBys;
            int allowIN; // Allow IN optimizations
            for (allowIN = 1; 1; allowIN--)
            {
                Debug.Assert(allowIN == 0 || allowIN == 1);
                // Set the aConstraint[].usable fields and initialize all output variables to zero.
                //
                // aConstraint[].usable is true for constraints where the right-hand side contains only references to tables to the left of the current
                // table.  In other words, if the constraint is of the form:
                //           column = expr
                // and we are evaluating a join, then the constraint on column is only valid if all tables referenced in expr occur to the left
                // of the table containing column.
                //
                // The aConstraints[] array contains entries for all constraints on the current table.  That way we only have to compute it once
                // even though we might try to pick the best index multiple times. For each attempt at picking an index, the order of tables in the
                // join might be different so we have to recompute the usable flag each time.
                int i, j;
                IIndexInfo.Constraint idxCons;
                IIndexInfo.ConstraintUsage[] usage = idxInfo.ConstraintUsages;
                for (i = 0, idxCons = idxInfo.Constraints[0]; i < idxInfo.Constraints.length; i++, idxCons = idxInfo.Constraints[i])
                {
                    j = idxCons.TermOffset;
                    WhereTerm term = wc.Slots[j];
                    idxCons.Usable = ((term.PrereqRight & p.NotReady) == 0 && (allowIN || (term.EOperator & WO.IN) == 0));
                    usage[i] = new IIndexInfo.ConstraintUsage();
                }
                if (idxInfo.NeedToFreeIdxStr)
                    SysEx.Free(ref idxInfo.IdxStr);
                idxInfo.IdxStr = null;
                idxInfo.IdxNum = 0;
                idxInfo.NeedToFreeIdxStr = false;
                idxInfo.OrderByConsumed = false;
                // ((double)2) In case of SQLITE_OMIT_FLOATING_POINT...
                idxInfo.EstimatedCost = cs.BIG_DOUBLE / ((double)2);
                orderBys = idxInfo.OrderBys.length;
                if (p.OrderBy == null)
                    idxInfo.OrderBys.length = 0;
                if (VTableBestIndex(parse, table, idxInfo) != 0)
                    return;

                sortOrder = SO.ASC; // Sort order for IN clauses
                for (i = 0, idxCons = idxInfo.Constraints[0]; i < idxInfo.Constraints.length; i++, idxCons = idxInfo.Constraints[i])
                {
                    if (usage[i].ArgvIndex > 0)
                    {
                        j = idxCons.TermOffset;
                        WhereTerm term = wc.Slots[j];
                        p.Cost.Used |= term.PrereqRight;
                        if ((term.EOperator & WO.IN) != 0)
                        {
                            // Do not attempt to use an IN constraint if the virtual table says that the equivalent EQ constraint cannot be safely omitted.
                            // If we do attempt to use such a constraint, some rows might be repeated in the output.
                            if (usage[i].Omit == 0) break;
                            for (int k = 0; k < idxInfo.OrderBys.length; k++)
                                if (idxInfo.OrderBys[k].Column == idxCons.Column)
                                {
                                    sortOrder = (idxInfo.OrderBys[k].Desc ? SO.DESC : SO.ASC);
                                    break;
                                }
                        }
                    }
                    if (i >= idxInfo.Constraints.length) break;
                }
            }

            // If there is an ORDER BY clause, and the selected virtual table index does not satisfy it, increase the cost of the scan accordingly. This
            // matches the processing for non-virtual tables in bestBtreeIndex().
            double cost = idxInfo.EstimatedCost;
            if (p.OrderBy != null && !idxInfo.OrderByConsumed)
                cost += EstLog(cost) * cost;
            // The cost is not allowed to be larger than SQLITE_BIG_DBL (the inital value of lowestCost in this loop. If it is, then the (cost<lowestCost) test below will never be true.
            // Use "(double)2" instead of "2.0" in case OMIT_FLOATING_POINT is defined.
            p.Cost.Cost = ((cs.BIG_DOUBLE / ((double)2)) < cost ? (cs.BIG_DOUBLE / ((double)2)) : cost);
            p.Cost.Plan.u.VTableIndex = idxInfo;
            if (idxInfo.OrderByConsumed)
            {
                Debug.Assert(sortOrder == (SO)0 || sortOrder == (SO)1);
                p.Cost.Plan.WsFlags |= WHERE_ORDERED + sortOrder * WHERE_REVERSE;
                p.Cost.Plan.OBSats = (ushort)orderBys;
            }
            else
                p.Cost.Plan.OBSats = (p.i != 0 ? p.Levels[p.i - 1].Plan.OBSats : (ushort)0);
            p.Cost.Plan.Eq s = 0;
            idxInfo.OrderBys.length = orderBys;

            // Try to find a more efficient access pattern by using multiple indexes to optimize an OR expression within the WHERE clause. 
            BestOrClauseIndex(p);
        }
#endif

#if ENABLE_STAT3
        static RC WhereKeyStats(Parse parse, Index index, Mem val, bool roundUp, tRowcnt[] stats)
        {
            Debug.Assert(index.Samples.length > 0);
            if (val == null) return RC.ERROR;
            tRowcnt n = index.RowEsts[0];
            IndexSample[] samples = index.Samples;
            TYPE type = sqlite3_value_type(val);

            int i;
            long v;
            double r;
            bool isEq = false;
            if (type == TYPE.INTEGER)
            {
                v = sqlite3_value_int64(pVal);
                r = v;
                for (i = 0; i < index.Samples.length; i++)
                {
                    if (samples[i].Type == TYPE.NULL) continue;
                    if (samples[i].Type >= TYPE.TEXT) break;
                    if (samples[i].Type == TYPE.INTEGER)
                    {
                        if (samples[i].u.I >= v)
                        {
                            isEq = (samples[i].u.I == v);
                            break;
                        }
                    }
                    else
                    {
                        Debug.Assert(samples[i].Type == TYPE.FLOAT);
                        if (samples[i].u.R >= r)
                        {
                            isEq = (samples[i].u.R == r);
                            break;
                        }
                    }
                }
            }
            else if (type == TYPE.FLOAT)
            {
                r = sqlite3_value_double(val);
                double rS;
                for (i = 0; i < index.Samples.length; i++)
                {
                    if (samples[i].Type == TYPE.NULL) continue;
                    if (samples[i].Type >= TYPE.TEXT) break;
                    if (samples[i].Type == TYPE.FLOAT)
                        rS = samples[i].u.R;
                    else
                        rS = samples[i].u.I;
                    if (rS >= r)
                    {
                        isEq = rS == r;
                        break;
                    }
                }
            }
            else if (type == TYPE.NULL)
            {
                i = 0;
                if (samples[0].Type == TYPE.NULL) isEq = true;
            }
            else
            {
                Debug.Assert(type == TYPE.TEXT || type == TYPE.BLOB);
                for (i = 0; i < index.Samples.length; i++)
                    if (samples[i].Type == TYPE.TEXT || samples[i].Type == TYPE.BLOB)
                        break;
                if (i < index.Samples.length)
                {
                    Context ctx = parse.Ctx;
                    CollSeq coll;
                    string z;
                    if (type == TYPE.BLOB)
                    {
                        byte[] blob = sqlite3_value_blob(val);
                        z = Encoding.UTF8.GetString(blob, 0, blob.Length);
                        coll = ctx.DefaultColl;
                        Debug.Assert(coll.Encode == TEXTENCODE.UTF8);
                    }
                    else
                    {
                        coll = sqlite3GetCollSeq(parse, TEXTENCODE.UTF8, 0, index.CollNames);
                        if (coll == null)
                            return RC.ERROR;
                        z = sqlite3ValueText(val, coll.Encode);
                        if (z == null)
                            return RC.NOMEM;
                        Debug.Assert(z != null && coll != null && coll.Cmp != null);
                    }
                    n = sqlite3ValueBytes(val, coll.Encode);

                    for (; i < index.Samples.length; i++)
                    {
                        int c;
                        TYPE sampleType = samples[i].Type;
                        if (sampleType < type) continue;
                        if (sampleType != type) break;
#if !OMIT_UTF16
                        if (coll.Encode != TEXTENCODE.UTF8)
                        {
                            int sampleBytes;
                            string sampleZ = sqlite3Utf8to16(ctx, coll.Encode, samples[i].u.Z, samples[i].Bytes, ref sampleBytes);
                            if (sampleZ == null)
                            {
                                Debug.Assert(ctx.MallocFailed);
                                return RC.NOMEM;
                            }
                            c = coll.Cmp(coll.User, sampleBytes, sampleZ, (int)n, z);
                            SysEx.TagFree(ctx, ref sampleZ);
                        }
                        else
#endif
                        {
                            c = coll.Cmp(coll.User, samples[i].Bytes, samples[i].u.Z, (int)n, z);
                        }
                        if (c >= 0)
                        {
                            if (c == 0) isEq = true;
                            break;
                        }
                    }
                }
            }

            // At this point, aSample[i] is the first sample that is greater than or equal to pVal.  Or if i==pIdx->nSample, then all samples are less
            // than pVal.  If aSample[i]==pVal, then isEq==1.
            if (isEq)
            {
                Debug.Assert(i < index.Samples.length);
                stats[0] = samples[i].Lts;
                stats[1] = samples[i].Eqs;
            }
            else
            {
                tRowcnt lowerId, upperId;
                if (i == 0)
                {
                    lowerId = 0;
                    upperId = samples[0].Lts;
                }
                else
                {
                    upperId = (i >= index.Samples.length ? n : samples[i].Lts);
                    lowerId = samples[i - 1].Eqs + samples[i - 1].Lts;
                }
                stats[1] = index.AvgEq;
                tRowcnt gapId = (lowerId >= upperId ? 0 : upperId - lowerId);
                gapId = (roundUp ? (gapId * 2) / 3 : gapId / 3);
                stats[0] = lowerId + gapId;
            }
            return RC.OK;
        }

        static RC ValueFromExpr(Parse parse, Expr expr, AFF aff, ref Mem val)
        {
            if (expr.OP == TK.VARIABLE || (expr.OP == TK.REGISTER && expr.OP2 == TK.VARIABLE))
            {
                int varId = expr.ColumnIdx;
                parse.V.SetVarmask(varId);
                val = parse.Reprepare.GetValue(varId, aff);
                return RC.OK;
            }
            return sqlite3ValueFromExpr(parse.Ctx, expr, TEXTENCODE.UTF8, aff, ref val);
        }

#endif
        static RC WhereRangeScanEst(Parse parse, Index p, int eqs, WhereTerm lower, WhereTerm upper, out double rangeDiv)
        {
            RC rc = RC.OK;
#if ENABLE_STAT3
            if (eqs == 0 && p.Samples.length != 0)
            {
                int iEst;
                int roundUpUpper = 0;
                int roundUpLower = 0;

                Mem rangeVal;
                tRowcnt lowerId = 0;
                tRowcnt upperId = p.RowEsts[0];
                tRowcnt[] a = new tRowcnt[2];
                AFF aff = p.Table.Cols[p.Columns[0]].Affinity;
                if (lower != null)
                {
                    Expr expr = lower.Expr.Right;
                    rc = ValueFromExpr(parse, expr, aff, ref rangeVal);
                    Debug.Assert((lower.EOperator & (WO.GT | WO.GE)) != 0);
                    if (rc == RC.OK && WhereKeyStats(parse, p, rangeVal, false, a) == RC.OK)
                    {
                        lowerId = a[0];
                        if ((lower.EOperator & WO.GT) != 0) lowerId += a[1];
                    }
                    sqlite3ValueFree(ref rangeVal);
                }
                if (rc == RC.OK && upper != null)
                {
                    Expr expr = upper.Expr.Right;
                    rc = ValueFromExpr(parse, expr, aff, ref rangeVal);
                    Debug.Assert((lower.EOperator & (WO.LT | WO.LE)) != 0);
                    if (rc == RC.OK && WhereKeyStats(parse, p, rangeVal, true, a) == RC.OK)
                    {
                        upperId = a[0];
                        if ((upper.EOperator & WO.LE) != 0) upperId += a[1];
                    }
                    sqlite3ValueFree(ref rangeVal);
                }
                if (rc == RC.OK)
                {
                    rangeDiv = (upperId <= lowerId ? (double)p.RowEsts[0] : (double)p.RowEsts[0] / (double)(upperId - lowerId));
                    WHERETRACE("range scan regions: %u..%u  div=%g\n", (uint)lowerId, (uint)upperId, rangeDiv);
                    return RC.OK;
                }
            }
#endif
            Debug.Assert(lower != null || upper != null);
            rangeDiv = (double)1;
            if (lower != null && (lower.WtFlags & TERM.VNULL) == 0) rangeDiv *= (double)4;
            if (upper != null) rangeDiv *= (double)4;
            return rc;
        }

#if ENABLE_STAT3
        static RC WhereEqualScanEst(Parse parse, Index p, Expr expr, ref double rows)
        {
            Mem rhs = null; // VALUE on right-hand side of pTerm
            RC rc; // Subfunction return code
            tRowcnt[] a = new tRowcnt[2]; // Statistics
            Debug.Assert(p.Samples.data != null);
            Debug.Assert(p.Samples.length > 0);
            AFF aff = p.Table.Cols[p.Columns[0]].Affinity; // Column affinity
            if (expr != null)
            {
                rc = ValueFromExpr(parse, expr, aff, ref rhs);
                if (rc != 0) goto cancel;
            }
            else
                rhs = sqlite3ValueNew(parse.Ctx);
            if (rhs == null) return RC.NOTFOUND;
            rc = WhereKeyStats(parse, p, rhs, false, a);
            if (rc == RC.OK)
            {
                WHERETRACE("equality scan regions: %d\n", (int)a[1]);
                rows = a[1];
            }

        cancel:
            sqlite3ValueFree(ref rhs);
            return rc;
        }

        static RC WhereInScanEst(Parse parse, Index p, ExprList list, ref double rows)
        {
            RC rc = RC.OK; // Subfunction return code
            double ests; // Number of rows for a single term
            double rowEsts = (double)0; // New estimate of the number of rows
            int i;
            Debug.Assert(p.Samples.data != null);
            for (i = 0; rc == RC.OK && i < list.Exprs; i++)
            {
                ests = p.RowEsts[0];
                rc = WhereEqualScanEst(parse, p, list.Ids[i].Expr, ref ests);
                rowEsts += ests;
            }
            if (rc == RC.OK)
            {
                if (rowEsts > p.RowEsts[0]) rowEsts = p.RowEsts[0];
                rows = rowEsts;
                WHERETRACE("IN row estimate: est=%g\n", rowEsts);
            }
            return rc;
        }
#endif

        static int IsOrderedColumn(WhereBestIdx p, int table, int column)
        {
            WhereLevel level = p.Levels[p.i - 1];
            Index index;
            SO sortOrder;
            int i, j;
            for (i = p.i - 1, level = p.Levels[i]; i >= 0; i--, level = p.Levels[i])
            {
                if (level.TabCur != table) continue;
                if ((level.Plan.WsFlags & WHERE_ALL_UNIQUE) != 0)
                    return 1;
                Debug.Assert((level.Plan.WsFlags & WHERE_ORDERED) != 0);
                if ((index = level.Plan.u.Index) != null)
                {
                    if (column < 0)
                    {
                        sortOrder = (int)SO.ASC;
                        SysEx.ASSERTCOVERAGE((level.Plan.WsFlags & WHERE_REVERSE) != 0);
                    }
                    else
                    {
                        int n = index.Columns.length;
                        for (j = 0; j < n; j++)
                            if (column == index.Columns[j]) break;
                        if (j >= n) return 0;
                        sortOrder = index.SortOrders[j];
                        SysEx.ASSERTCOVERAGE((level.Plan.WsFlags & WHERE_REVERSE) != 0);
                    }
                }
                else
                {
                    if (column != -1) return 0;
                    sortOrder = SO.ASC;
                    SysEx.ASSERTCOVERAGE((level.Plan.WsFlags & WHERE_REVERSE) != 0);
                }
                if ((level.Plan.WsFlags & WHERE_REVERSE) != 0)
                {
                    Debug.Assert(sortOrder == SO.ASC || sortOrder == SO.DESC);
                    SysEx.ASSERTCOVERAGE(sortOrder == SO.DESC);
                    sortOrder = 1 - sortOrder;
                }
                return (int)sortOrder + 2;
            }
            return 0;
        }

        static int IsSortingIndex(WhereBestIdx p, Index index, int baseId, ref int rev, ref bool unique)
        {
            int i; // Number of pIdx terms used
            int sortOrder = 2; // XOR of index and ORDER BY sort direction

            Parse parse = p.Parse; // Parser context
            Context ctx = parse.Ctx; // Database connection
            int priorSats; // ORDER BY terms satisfied by outer loops
            bool outerObUnique; // Outer loops generate different values in every row for the ORDER BY columns
            if (p.i == 0)
            {
                priorSats = 0;
                outerObUnique = true;
            }
            else
            {
                uint wsFlags = p.Levels[p.i - 1].Plan.WsFlags;
                priorSats = p.Levels[p.i - 1].Plan.OBSats;
                if ((wsFlags & WHERE_ORDERED) == 0) // This loop cannot be ordered unless the next outer loop is also ordered
                    return priorSats;
                if (E.CtxOptimizationDisabled(ctx, OPTFLAG.OrderByIdxJoin)) // Only look at the outer-most loop if the OrderByIdxJoin optimization is disabled
                    return priorSats;
                SysEx.ASSERTCOVERAGE((wsFlags & WHERE_OB_UNIQUE) != 0);
                SysEx.ASSERTCOVERAGE((wsFlags & WHERE_ALL_UNIQUE) != 0);
                outerObUnique = ((wsFlags & (WHERE_OB_UNIQUE | WHERE_ALL_UNIQUE)) != 0);
            }
            ExprList orderBy = p.OrderBy; // The ORDER BY clause
            Debug.Assert(orderBy != null);
            if (index.Unordered) // Hash indices (indicated by the "unordered" tag on sqlite_stat1) cannot be used for sorting
                return priorSats;
            int terms = orderBy.Exprs; // Number of ORDER BY terms
            bool uniqueNotNull = (index.OnError != OE.None); // pIdx is UNIQUE with all terms are NOT NULL
            Debug.Assert(terms > 0);

            // Argument pIdx must either point to a 'real' named index structure, or an index structure allocated on the stack by bestBtreeIndex() to
            // represent the rowid index that is part of every table.
            Debug.Assert(index.Name != null || (index.Columns.length == 1 && index.Columns[0] == -1));

            // Match terms of the ORDER BY clause against columns of the index.
            //
            // Note that indices have pIdx->nColumn regular columns plus one additional column containing the rowid.  The rowid column
            // of the index is also allowed to match against the ORDER BY clause.
            int j = priorSats; // Number of ORDER BY terms satisfied
            ExprList.ExprListItem obItem;// A term of the ORDER BY clause
            Table table = index.Table; // Table that owns index pIdx
            bool seenRowid = false; // True if an ORDER BY rowid term is seen
            for (i = 0, obItem = orderBy.Ids[j]; j < terms && i <= index.Columns.length; i++)
            {
                // If the next term of the ORDER BY clause refers to anything other than a column in the "base" table, then this index will not be of any
                // further use in handling the ORDER BY.
                Expr obExpr = obItem.Expr.SkipCollate(); // The expression of the ORDER BY pOBItem
                if (obExpr.OP != TK.COLUMN || obExpr.TableIdx != baseId)
                    break;

                // Find column number and collating sequence for the next entry in the index
                int column; // The i-th column of the index.  -1 for rowid
                SO sortOrder2; // 1 for DESC, 0 for ASC on the i-th index term
                string collName; // Name of collating sequence for i-th index term
                if (index.Name != null && i < index.Columns.length)
                {
                    column = index.Columns[i];
                    if (column == index.Table.PKey)
                        column = -1;
                    sortOrder2 = index.SortOrders[i];
                    collName = index.CollNames[i];
                    Debug.Assert(collName != null);
                }
                else
                {
                    column = -1;
                    sortOrder2 = 0;
                    collName = null;
                }

                // Check to see if the column number and collating sequence of the index match the column number and collating sequence of the ORDER BY
                // clause entry.  Set isMatch to 1 if they both match.
                bool isMatch; // ORDER BY term matches the index term
                if (obExpr.ColumnIdx != column)
                {
                    if (collName != null)
                    {
                        CollSeq coll = obItem.Expr.CollSeq(parse); // The collating sequence of pOBExpr
                        if (coll == null) coll = ctx.DefaultColl;
                        isMatch = string.Equals(coll.Name, collName, StringComparison.OrdinalIgnoreCase);
                    }
                    else isMatch = true;
                }
                else isMatch = false;

                // termSortOrder is 0 or 1 for whether or not the access loop should run forward or backwards (respectively) in order to satisfy this 
                // term of the ORDER BY clause.
                Debug.Assert(obItem.SortOrder == 0 || obItem.SortOrder == (SO)1);
                Debug.Assert(sortOrder2 == 0 || sortOrder2 == (SO)1);
                SO termSortOrder = sortOrder2 ^ obItem.SortOrder; // Sort order for this term

                // If X is the column in the index and ORDER BY clause, check to see if there are any X= or X IS NULL constraints in the WHERE clause.
                int isEq; // Subject to an == or IS NULL constraint
                WhereTerm constraint = FindTerm(p.WC, baseId, column, p.NotReady, WO.EQ | WO.ISNULL | WO.IN, index); // A constraint in the WHERE clause
                if (constraint == null) isEq = 0;
                else if ((constraint.EOperator & WO.IN) != 0) isEq = 0;
                else if ((constraint.EOperator & WO.ISNULL) != 0) { uniqueNotNull = false; isEq = 1; } // "X IS NULL" means X has only a single value
                else if (constraint.PrereqRight == 0) isEq = 1;  // Constraint "X=constant" means X has only a single value
                else
                {
                    Expr right = constraint.Expr.Right;
                    if (right.OP == TK.COLUMN)
                    {
                        WHERETRACE("       .. isOrderedColumn(tab=%d,col=%d)", right.TableIdx, right.ColumnIdx);
                        isEq = IsOrderedColumn(p, right.TableIdx, right.ColumnIdx);
                        WHERETRACE(" -> isEq=%d\n", isEq);
                        // If the constraint is of the form X=Y where Y is an ordered value in an outer loop, then make sure the sort order of Y matches the
                        // sort order required for X.
                        if (isMatch && isEq >= 2 && isEq != obItem.SortOrder + 2)
                        {
                            SysEx.ASSERTCOVERAGE(isEq == 2);
                            SysEx.ASSERTCOVERAGE(isEq == 3);
                            break;
                        }
                    }
                    else isEq = 0; // "X=expr" places no ordering constraints on X
                }
                if (!isMatch)
                {
                    if (isEq == 0) break;
                    else continue;
                }
                else if (isEq != 1)
                {
                    if (sortOrder == 2) sortOrder = (int)termSortOrder;
                    else if ((int)termSortOrder != sortOrder) break;
                }
                j++;
                obItem = orderBy.Ids[j];
                if (column < 0)
                {
                    seenRowid = true;
                    break;
                }
                else if (table.Cols[column].NotNull == 0 && isEq != 1)
                {
                    SysEx.ASSERTCOVERAGE(isEq == 0);
                    SysEx.ASSERTCOVERAGE(isEq == 2);
                    SysEx.ASSERTCOVERAGE(isEq == 3);
                    uniqueNotNull = false;
                }
            }
            if (seenRowid)
                uniqueNotNull = true;
            else if (!uniqueNotNull || i < index.Columns.length)
                uniqueNotNull = false;

            // If we have not found at least one ORDER BY term that matches the index, then show no progress.
            if (obItem == orderBy.Ids[priorSats]) return priorSats;

            // Either the outer queries must generate rows where there are no two rows with the same values in all ORDER BY columns, or else this
            // loop must generate just a single row of output.  Example:  Suppose the outer loops generate A=1 and A=1, and this loop generates B=3
            // and B=4.  Then without the following test, ORDER BY A,B would generate the wrong order output: 1,3 1,4 1,3 1,4
            if (!outerObUnique && !uniqueNotNull) return priorSats;
            unique = uniqueNotNull;

            // Return the necessary scan order back to the caller
            rev = sortOrder & 1;

            // If there was an "ORDER BY rowid" term that matched, or it is only possible for a single row from this table to match, then skip over
            // any additional ORDER BY terms dealing with this table.
            if (uniqueNotNull)
            {
                // Advance j over additional ORDER BY terms associated with base
                WhereMaskSet ms = p.WC.MaskSet;
                Bitmask m = ~GetMask(ms, baseId);
                while (j < terms && (ExprTableUsage(ms, orderBy.Ids[j].Expr) & m) == 0) j++;
            }
            return j;
        }

        const bool _UseCis = true;
        static void BestBtreeIndex(WhereBestIdx p)
        {
            Parse parse = p.Parse;  // The parsing context
            WhereClause wc = p.WC;  // The WHERE clause
            SrcList.SrcListItem src = p.Src; // The FROM clause term to search
            int cursor = src.Cursor;   // The cursor of the table to be accessed

            // Initialize the cost to a worst-case value
            if (p.Cost == null)
                p.Cost = new WhereCost();
            else
                p.Cost.memset();
            p.Cost.Cost = BIG_DOUBLE;

            // If the pSrc table is the right table of a LEFT JOIN then we may not use an index to satisfy IS NULL constraints on that table.  This is
            // because columns might end up being NULL if the table does not match - a circumstance which the index cannot help us discover.  Ticket #2177.
            Index probe; // An index we are evaluating
            Index index; // Copy of pProbe, or zero for IPK index
            int wsFlagMask; // Allowed flags in pCost.plan.wsFlag
            WO eqTermMask; // Current mask of valid equality operators
            WO idxEqTermMask; // Index mask of valid equality operators


            if (src.Index != null)
            {
                // An INDEXED BY clause specifies a particular index to use
                index = probe = src.Index;
                wsFlagMask = ~(WHERE.ROWID_EQ | WHERE.ROWID_RANGE);
                eqTermMask = idxEqTermMask;
            }
            else
            {
                Index sPk; // A fake index object for the primary key
                tRowcnt[] rowEstPks = new tRowcnt[2]; // The aiRowEst[] value for the sPk index
                int columnPks = -1; // The aColumn[] value for the sPk index
                // There is no INDEXED BY clause.  Create a fake Index object in local variable sPk to represent the rowid primary key index.  Make this
                // fake index the first in a chain of Index objects with all of the real indices to follow */
                sPk = new Index();
                sPk.SortOrders = new SO[1];
                sPk.CollNames = new string[1];
                sPk.CollNames[0] = string.Empty;
                sPk.Columns.length = 1;
                sPk.Columns.data = new int[1];
                sPk.Columns[0] = columnPks;
                sPk.RowEsts = rowEstPks;
                sPk.OnError = OE.Replace;
                sPk.Table = src.Table;
                rowEstPks[0] = src.Table.RowEst;
                rowEstPks[1] = 1;
                Index first = src.Table.Index;
                if (!src.NotIndexed)
                    sPk.Next = first; // The real indices of the table are only considered if the NOT INDEXED qualifier is omitted from the FROM clause
                probe = sPk;
                wsFlagMask = ~(WHERE_COLUMN_IN | WHERE_COLUMN_EQ | WHERE_COLUMN_NULL | WHERE_COLUMN_RANGE);
                eqTermMask = WO.EQ | WO.IN;
                index = null;
            }

            int orderBys = (p.OrderBy != null ? p.OrderBy.Exprs : 0); // Number of ORDER BY terms
            int priorSats; // ORDER BY terms satisfied by outer loops
            bool sortInit; // Initializer for bSort in inner loop
            bool distInit; // Initializer for bDist in inner loop
            if (p.i != 0)
            {
                priorSats = p.Levels[p.i - 1].Plan.OBSats;
                sortInit = (priorSats < orderBys);
                distInit = false;
            }
            else
            {
                priorSats = 0;
                sortInit = (orderBys > 0);
                distInit = (p.Distinct != null);
            }

            // Loop over all indices looking for the best one to use
            for (; probe != null; index = probe = probe.Next)
            {
                tRowcnt[] rowEsts = probe.RowEsts;
                WhereCost pc = new WhereCost(); // Cost of using pProbe
                double log10N = 1; // base-10 logarithm of nRow (inexact)

                // The following variables are populated based on the properties of index being evaluated. They are then used to determine the expected
                // cost and number of rows returned.
                //
                //  pc.plan.nEq: 
                //    Number of equality terms that can be implemented using the index. In other words, the number of initial fields in the index that
                //    are used in == or IN or NOT NULL constraints of the WHERE clause.
                //
                //  nInMul:  
                //    The "in-multiplier". This is an estimate of how many seek operations SQLite must perform on the index in question. For example, if the 
                //    WHERE clause is:
                //      WHERE a IN (1, 2, 3) AND b IN (4, 5, 6)
                //
                //    SQLite must perform 9 lookups on an index on (a, b), so nInMul is set to 9. Given the same schema and either of the following WHERE 
                //    clauses:
                //      WHERE a =  1
                //      WHERE a >= 2
                //
                //    nInMul is set to 1.
                //
                //    If there exists a WHERE term of the form "x IN (SELECT ...)", then the sub-select is assumed to return 25 rows for the purposes of 
                //    determining nInMul.
                //
                //  bInEst:  
                //    Set to true if there was at least one "x IN (SELECT ...)" term used in determining the value of nInMul.  Note that the RHS of the
                //    IN operator must be a SELECT, not a value list, for this variable to be true.
                //
                //  rangeDiv:
                //    An estimate of a divisor by which to reduce the search space due to inequality constraints.  In the absence of sqlite_stat3 ANALYZE
                //    data, a single inequality reduces the search space to 1/4rd its original size (rangeDiv==4).  Two inequalities reduce the search
                //    space to 1/16th of its original size (rangeDiv==16).
                //
                //  bSort:   
                //    Boolean. True if there is an ORDER BY clause that will require an external sort (i.e. scanning the index being evaluated will not 
                //    correctly order records).
                //
                //  bDist:
                //    Boolean. True if there is a DISTINCT clause that will require an external btree.
                //
                //  bLookup: 
                //    Boolean. True if a table lookup is required for each index entry visited.  In other words, true if this is not a covering index.
                //    This is always false for the rowid primary key index of a table. For other indexes, it is true unless all the columns of the table
                //    used by the SELECT statement are present in the index (such an index is sometimes described as a covering index).
                //    For example, given the index on (a, b), the second of the following two queries requires table b-tree lookups in order to find the value
                //    of column c, but the first does not because columns a and b are both available in the index.
                //             SELECT a, b    FROM tbl WHERE a = 1;
                //             SELECT a, b, c FROM tbl WHERE a = 1;
                bool inEst = false;				// True if "x IN (SELECT...)" seen
                int inMul = 1;					// Number of distinct equalities to lookup
                double rangeDiv = (double)1;	// Estimated reduction in search space
                int bounds = 0;					// Number of range constraints seen
                bool sort = sortInit;			// True if external sort required
                bool dist = distInit;			// True if index cannot help with DISTINCT
                bool lookup = false;			// True if not a covering index
#if ENABLE_STAT3
                WhereTerm firstTerm = null; // First term matching the index
#endif

                WHERETRACE("   %s(%s):\n", src.Table.Name, (index != null ? index.Name : "ipk"));
                pc.Plan.OBSats = (ushort)priorSats;

                // Determine the values of pc.plan.nEq and nInMul
                WhereTerm term; // A single term of the WHERE clause
                for (pc.Plan.Eqs = 0; pc.Plan.Eqs < probe.Columns.length; pc.Plan.Eqs++)
                {
                    int j = probe.Columns[pc.Plan.Eqs];
                    term = FindTerm(wc, cursor, j, p.NotReady, eqTermMask, index);
                    if (term == null) break;
                    pc.Plan.WsFlags |= (WHERE_COLUMN_EQ | WHERE_ROWID_EQ);
                    SysEx.ASSERTCOVERAGE(term.WC != wc);
                    if ((term.EOperator & WO.IN) != 0)
                    {
                        Expr expr = term.Expr;
                        pc.Plan.WsFlags |= WHERE_COLUMN_IN;
                        if (E.ExprHasProperty(expr, EP.xIsSelect))
                        {
                            // "x IN (SELECT ...)":  Assume the SELECT returns 25 rows
                            inMul *= 25;
                            inEst = true;
                        }
                        else if (SysEx.ALWAYS(expr.x.List != null) && expr.x.List.Exprs != 0)
                            inMul *= expr.x.List.Exprs; // "x IN (value, value, ...)"
                    }
                    else if ((term.EOperator & WO.ISNULL) != 0)
                        pc.Plan.WsFlags |= WHERE_COLUMN_NULL;
#if ENABLE_STAT3
                    if (pc.Plan.Eqs == 0 && probe.Samples.data != null) firstTerm = term;
#endif
                    pc.Used |= term.PrereqRight;
                }

                // If the index being considered is UNIQUE, and there is an equality constraint for all columns in the index, then this search will find
                // at most a single row. In this case set the WHERE_UNIQUE flag to indicate this to the caller.
                //
                // Otherwise, if the search may find more than one row, test to see if there is a range constraint on indexed column (pc.plan.nEq+1) that
                // can be optimized using the index.
                if (pc.Plan.Eqs == probe.Columns.length && probe.OnError != OE.None)
                {
                    SysEx.ASSERTCOVERAGE((pc.Plan.WsFlags & WHERE_COLUMN_IN) != 0);
                    SysEx.ASSERTCOVERAGE((pc.Plan.WsFlags & WHERE_COLUMN_NULL) != 0);
                    if ((pc.Plan.WsFlags & (WHERE_COLUMN_IN | WHERE_COLUMN_NULL)) == 0)
                    {
                        pc.Plan.WsFlags |= WHERE_UNIQUE;
                        if (p.i == 0 || (p.Levels[p.i - 1].Plan.WsFlags & WHERE_ALL_UNIQUE) != 0)
                            pc.Plan.WsFlags |= WHERE_ALL_UNIQUE;
                    }
                }
                else if (!probe.Unordered)
                {
                    int j = (pc.Plan.Eqs == probe.Columns.length ? -1 : probe.Columns[pc.Plan.Eqs]);
                    if (FindTerm(wc, cursor, j, p.NotReady, WO.LT | WO.LE | WO.GT | WO.GE, index) != null)
                    {
                        WhereTerm top = FindTerm(wc, cursor, j, p.NotReady, WO.LT | WO.LE, index);
                        WhereTerm btm = FindTerm(wc, cursor, j, p.NotReady, WO.GT | WO.GE, index);
                        WhereRangeScanEst(parse, probe, pc.Plan.Eqs, btm, top, out rangeDiv);
                        if (top != null)
                        {
                            bounds = 1;
                            pc.Plan.WsFlags |= WHERE_TOP_LIMIT;
                            pc.Used |= top.PrereqRight;
                            SysEx.ASSERTCOVERAGE(top.WC != wc);
                        }
                        if (btm != null)
                        {
                            bounds++;
                            pc.Plan.WsFlags |= WHERE_BTM_LIMIT;
                            pc.Used |= btm.PrereqRight;
                            SysEx.ASSERTCOVERAGE(btm.WC != wc);
                        }
                        pc.Plan.WsFlags |= (WHERE_COLUMN_RANGE | WHERE_ROWID_RANGE);
                    }
                }

                // If there is an ORDER BY clause and the index being considered will naturally scan rows in the required order, set the appropriate flags
                // in pc.plan.wsFlags. Otherwise, if there is an ORDER BY clause but the index will scan rows in a different order, set the bSort
                // variable.
                if (sort && (src.Jointype & JT.LEFT) == 0)
                {
                    int rev = 2;
                    bool obUnique = false;
                    WHERETRACE("      --> before isSortIndex: nPriorSat=%d\n", priorSats);
                    pc.Plan.OBSats = (ushort)IsSortingIndex(p, probe, cursor, ref rev, ref obUnique);
                    WHERETRACE("      --> after  isSortIndex: bRev=%d bObU=%d nOBSat=%d\n", rev, obUnique, pc.Plan.OBSats);
                    if (priorSats < pc.Plan.OBSats || (pc.Plan.WsFlags & WHERE_ALL_UNIQUE) != 0)
                    {
                        pc.Plan.WsFlags |= WHERE_ORDERED;
                        if (obUnique) pc.Plan.WsFlags |= WHERE_OB_UNIQUE;
                    }
                    if (orderBys == pc.Plan.OBSats)
                    {
                        sort = false;
                        pc.Plan.WsFlags |= WHERE_ROWID_RANGE | WHERE_COLUMN_RANGE;
                    }
                    if ((rev & 1) != 0) pc.Plan.WsFlags |= WHERE_REVERSE;
                }

                // If there is a DISTINCT qualifier and this index will scan rows in order of the DISTINCT expressions, clear bDist and set the appropriate
                // flags in pc.plan.wsFlags.
                if (dist && IsDistinctIndex(parse, wc, probe, cursor, p.Distinct, pc.Plan.Eqs) && (pc.Plan.WsFlags & WHERE_COLUMN_IN) == 0)
                {
                    dist = false;
                    pc.Plan.WsFlags |= WHERE_ROWID_RANGE | WHERE_COLUMN_RANGE | WHERE_DISTINCT;
                }

                // If currently calculating the cost of using an index (not the IPK index), determine if all required column data may be obtained without 
                // using the main table (i.e. if the index is a covering index for this query). If it is, set the WHERE_IDX_ONLY flag in
                // pc.plan.wsFlags. Otherwise, set the bLookup variable to true.
                if (index != null)
                {
                    Bitmask m = src.ColUsed;
                    for (int j = 0; j < index.Columns.length; j++)
                    {
                        int x = index.Columns[j];
                        if (x < BMS - 1)
                            m &= ~(((Bitmask)1) << x);
                    }
                    if (m == 0)
                        pc.Plan.WsFlags |= WHERE_IDX_ONLY;
                    else
                        lookup = true;
                }

                // Estimate the number of rows of output.  For an "x IN (SELECT...)" constraint, do not let the estimate exceed half the rows in the table.
                pc.Plan.Rows = (double)(rowEsts[pc.Plan.Eqs] * inMul);
                if (inEst && pc.Plan.Rows * 2 > rowEsts[0])
                {
                    pc.Plan.Rows = rowEsts[0] / 2;
                    inMul = (int)(pc.Plan.Rows / rowEsts[pc.Plan.Eqs]);
                }

#if ENABLE_STAT3
                // If the constraint is of the form x=VALUE or x IN (E1,E2,...) and we do not think that values of x are unique and if histogram
                // data is available for column x, then it might be possible to get a better estimate on the number of rows based on
                // VALUE and how common that value is according to the histogram.
                if (pc.Plan.Rows > (double)1 && pc.Plan.Eqs == 1 && firstTerm != null && rowEsts[1] > 1)
                {
                    Debug.Assert((firstTerm.EOperator & (WO.EQ | WO.ISNULL | WO.IN)) != 0);
                    if ((firstTerm.EOperator & (WO.EQ | WO.ISNULL)) != 0)
                    {
                        SysEx.ASSERTCOVERAGE((firstTerm.EOperator & WO.EQ) != 0);
                        SysEx.ASSERTCOVERAGE((firstTerm.EOperator & WO.EQUIV) != 0);
                        SysEx.ASSERTCOVERAGE((firstTerm.EOperator & WO.ISNULL) != 0);
                        WhereEqualScanEst(parse, probe, firstTerm.Expr.Right, ref pc.Plan.Rows);
                    }
                    else if (!inEst)
                    {
                        Debug.Assert((firstTerm.EOperator & WO.IN) != 0);
                        WhereInScanEst(parse, probe, firstTerm.Expr.x.List, ref pc.Plan.Rows);
                    }
                }
#endif

                // Adjust the number of output rows and downward to reflect rows that are excluded by range constraints.
                pc.Plan.Rows = pc.Plan.Rows / rangeDiv;
                if (pc.Plan.Rows < 1) pc.Plan.Rows = 1;

                // Experiments run on real SQLite databases show that the time needed to do a binary search to locate a row in a table or index is roughly
                // log10(N) times the time to move from one row to the next row within a table or index.  The actual times can vary, with the size of
                // records being an important factor.  Both moves and searches are slower with larger records, presumably because fewer records fit
                // on one page and hence more pages have to be fetched.
                //
                // The ANALYZE command and the sqlite_stat1 and sqlite_stat3 tables do not give us data on the relative sizes of table and index records.
                // So this computation assumes table records are about twice as big as index records
                if ((pc.Plan.WsFlags & ~(WHERE_REVERSE | WHERE_ORDERED | WHERE_OB_UNIQUE)) == WHERE_IDX_ONLY &&
                    (wc.WctrlFlags & (ushort)WHERE.ONEPASS_DESIRED) == 0 && _UseCis && E.CtxOptimizationEnabled(parse.Ctx, OPTFLAG.CoverIdxScan))
                {
                    // This index is not useful for indexing, but it is a covering index. A full-scan of the index might be a little faster than a full-scan
                    // of the table, so give this case a cost slightly less than a table scan.
                    pc.Cost = rowEsts[0] * 3 + probe.Columns.length;
                    pc.Plan.WsFlags |= WHERE_COVER_SCAN | WHERE_COLUMN_RANGE;
                }
                else if ((pc.Plan.WsFlags & WHERE_NOT_FULLSCAN) == 0)
                {
                    // The cost of a full table scan is a number of move operations equal to the number of rows in the table.
                    //
                    // We add an additional 4x penalty to full table scans.  This causes the cost function to err on the side of choosing an index over
                    // choosing a full scan.  This 4x full-scan penalty is an arguable decision and one which we expect to revisit in the future.  But
                    // it seems to be working well enough at the moment.
                    pc.Cost = rowEsts[0] * 4;
                    pc.Plan.WsFlags &= ~WHERE_IDX_ONLY;
                    if (index != null)
                    {
                        pc.Plan.WsFlags &= ~WHERE_ORDERED;
                        pc.Plan.OBSats = (ushort)priorSats;
                    }
                }
                else
                {
                    log10N = EstLog(rowEsts[0]);
                    pc.Cost = pc.Plan.Rows;
                    if (index != null)
                    {
                        if (lookup)
                        {
                            // For an index lookup followed by a table lookup:
                            //    nInMul index searches to find the start of each index range
                            //  + nRow steps through the index
                            //  + nRow table searches to lookup the table entry using the rowid
                            pc.Cost += (inMul + pc.Plan.Rows) * log10N;
                        }
                        else
                        {
                            // For a covering index:
                            //     nInMul index searches to find the initial entry 
                            //   + nRow steps through the index
                            pc.Cost += inMul * log10N;
                        }
                    }
                    else
                    {
                        // For a rowid primary key lookup:
                        //    nInMult table searches to find the initial entry for each range
                        //  + nRow steps through the table
                        pc.Cost += inMul * log10N;
                    }
                }

                // Add in the estimated cost of sorting the result.  Actual experimental measurements of sorting performance in SQLite show that sorting time
                // adds C*N*log10(N) to the cost, where N is the number of rows to be sorted and C is a factor between 1.95 and 4.3.  We will split the
                // difference and select C of 3.0.
                if (sort)
                {
                    double m = EstLog(pc.Plan.Rows * (orderBys - pc.Plan.OBSats) / orderBys);
                    m *= (double)(pc.Plan.OBSats != 0 ? 2 : 3);
                    pc.Cost += pc.Plan.Rows * m;
                }
                if (dist)
                    pc.Cost += pc.Plan.Rows * EstLog(pc.Plan.Rows) * 3;

                //// Cost of using this index has now been computed ////

                // If there are additional constraints on this table that cannot be used with the current index, but which might lower the number
                // of output rows, adjust the nRow value accordingly.  This only matters if the current index is the least costly, so do not bother
                // with this step if we already know this index will not be chosen. Also, never reduce the output row count below 2 using this step.
                //
                // It is critical that the notValid mask be used here instead of the notReady mask.  When computing an "optimal" index, the notReady
                // mask will only have one bit set - the bit for the current table. The notValid mask, on the other hand, always has all bits set for
                // tables that are not in outer loops.  If notReady is used here instead of notValid, then a optimal index that depends on inner joins loops
                // might be selected even when there exists an optimal index that has no such dependency.
                if (pc.Plan.Rows > 2 && pc.Cost <= p.Cost.Cost)
                {
                    int k;
                    int skipEqs = pc.Plan.Eqs; // Number of == constraints to skip
                    int skipRanges = bounds; // Number of < constraints to skip
                    Bitmask thisTab = GetMask(wc.MaskSet, cursor); // Bitmap for pSrc
                    for (term = wc.Slots[0], k = wc.Terms; pc.Plan.Rows > 2 && k != 0; k--, term = wc.Slots[wc.Terms - k])
                    {
                        if ((term.WtFlags & TERM.VIRTUAL) != 0) continue;
                        if ((term.PrereqAll & p.NotValid) != thisTab) continue;
                        if ((term.EOperator & (WO.EQ | WO.IN | WO.ISNULL)) != 0)
                        {
                            if (skipEqs != 0) skipEqs--; // Ignore the first pc.plan.nEq equality matches since the index has already accounted for these
                            else pc.Plan.Rows /= 10; // Assume each additional equality match reduces the result set size by a factor of 10
                        }
                        else if ((term.EOperator & (WO.LT | WO.LE | WO.GT | WO.GE)) != 0)
                        {
                            if (skipRanges != 0) skipRanges--; // Ignore the first nSkipRange range constraints since the index has already accounted for these
                            // Assume each additional range constraint reduces the result set size by a factor of 3.  Indexed range constraints reduce
                            // the search space by a larger factor: 4.  We make indexed range more selective intentionally because of the subjective 
                            // observation that indexed range constraints really are more selective in practice, on average.
                            else pc.Plan.Rows /= 3;
                        }
                        else if ((term.EOperator & WO.NOOP) == 0)
                            pc.Plan.Rows /= 2; // Any other expression lowers the output row count by half
                    }
                    if (pc.Plan.Rows < 2) pc.Plan.Rows = 2;
                }

                WHERETRACE(
                    "      nEq=%d nInMul=%d rangeDiv=%d bSort=%d bLookup=%d wsFlags=0x%08x\n" +
                    "      notReady=0x%llx log10N=%.1f nRow=%.1f cost=%.1f\n" +
                    "      used=0x%llx nOBSat=%d\n",
                    pc.Plan.Eqs, inMul, (int)rangeDiv, sort, lookup, pc.Plan.WsFlags,
                    p.NotReady, log10N, pc.Plan.Rows, pc.Cost, pc.Used,
                    pc.Plan.OBSats);


                // If this index is the best we have seen so far, then record this index and its cost in the p->cost structure.
                if ((index == null || pc.Plan.WsFlags != 0) && CompareCost(pc, p.Cost))
                {
                    p.Cost = pc;
                    p.Cost.Plan.WsFlags &= wsFlagMask;
                    p.Cost.Plan.u.Index = index;
                }

                // If there was an INDEXED BY clause, then only that one index is considered.
                if (src.Index != null) break;

                // Reset masks for the next index in the loop
                wsFlagMask = ~(WHERE_ROWID_EQ | WHERE_ROWID_RANGE);
                eqTermMask = idxEqTermMask;
            }

            // If there is no ORDER BY clause and the SQLITE_ReverseOrder flag is set, then reverse the order that the index will be scanned
            // in. This is used for application testing, to help find cases where application behavior depends on the (undefined) order that
            // SQLite outputs rows in in the absence of an ORDER BY clause.
            if (p.OrderBy == null && (parse.Ctx.Flags & Context.FLAG.ReverseOrder) != 0)
                p.Cost.Plan.WsFlags |= WHERE_REVERSE;

            Debug.Assert(p.OrderBy != null || (p.Cost.Plan.WsFlags & WHERE_ORDERED) == 0);
            Debug.Assert(p.Cost.Plan.u.Index == null || (p.Cost.Plan.WsFlags & WHERE_ROWID_EQ) == 0);
            Debug.Assert(src.Index == null || p.Cost.Plan.u.Index == null || p.Cost.Plan.u.Index == src.Index);

            WHERETRACE("   best index is %s cost=%.1f\n", (p.Cost.Plan.u.Index != null ? p.Cost.Plan.u.Index.Name : "ipk"), p.Cost.Cost);

            BestOrClauseIndex(p);
            BestAutomaticIndex(p);
            p.Cost.Plan.WsFlags |= eqTermMask;
        }

        static void BestIndex(WhereBestIdx p)
        {
#if !OMIT_VIRTUALTABLE
            if (E.IsVirtual(p.Src.Table))
            {
                p.IdxInfo = null;
                BestVirtualIndex(p);
                IIndexInfo indexInfo = p.IdxInfo[0];
                Debug.Assert(indexInfo != null || p.Parse.Ctx.MallocFailed);
                if (indexInfo != null && indexInfo.NeedToFreeIdxStr)
                    SysEx.Free(ref indexInfo.IdxStr);
                SysEx.TagFree(p.Parse.Ctx, ref indexInfo);
            }
            else
#endif
            {
                BestBtreeIndex(p);
            }
        }

        static void DisableTerm(WhereLevel level, WhereTerm term)
        {
            if (term != null && (term.WtFlags & TERM.CODED) == 0 && (level.LeftJoin == 0 || E.ExprHasProperty(term.Expr, EP.FromJoin)))
            {
                term.WtFlags |= TERM.CODED;
                if (term.Parent >= 0)
                {
                    WhereTerm other = term.WC.Slots[term.Parent];
                    if (--other.Childs == 0)
                        DisableTerm(level, other);
                }
            }
        }

        static void CodeApplyAffinity(Parse parse, int baseId, int n, string affs)
        {
            Vdbe v = parse.V;
            if (affs == null)
            {
                Debug.Assert(parse.Ctx.MallocFailed);
                return;
            }
            Debug.Assert(v != null);
            // Adjust base and n to skip over SQLITE_AFF_NONE entries at the beginning and end of the affinity string.
            while (n > 0 && (AFF)affs[0] == AFF.NONE)
            {
                n--;
                baseId++;
                affs = affs.Substring(1); //: affs++;
            }
            while (n > 1 && (AFF)affs[n - 1] == AFF.NONE) n--;
            // Code the OP_Affinity opcode if there is anything left to do.
            if (n > 0)
            {
                v.AddOp2(OP.Affinity, baseId, n);
                v.ChangeP4(-1, affs, n);
                Expr.CacheAffinityChange(parse, baseId, n);
            }
        }

        static int CodeEqualityTerm(Parse parse, WhereTerm term, WhereLevel level, int targetId)
        {
            Expr x = term.Expr;
            Vdbe v = parse.V;
            int regId; // Register holding results
            Debug.Assert(targetId > 0);
            if (x.OP == TK.EQ)
                regId = Expr.CodeTarget(parse, x.Right, targetId);
            else if (x.OP == TK.ISNULL)
            {
                regId = targetId;
                v.AddOp2(OP.Null, 0, regId);
#if !OMIT_SUBQUERY
            }
            else
            {
                int eType;
                int iTab;
                InLoop pIn;

                Debug.Assert(x.op == TK_IN);
                regId = targetId;
                int iDummy = -1;
                eType = sqlite3FindInIndex(parse, x, ref iDummy);
                iTab = x.iTable;
                sqlite3VdbeAddOp2(v, OP_Rewind, iTab, 0);
                Debug.Assert((level.plan.wsFlags & WHERE_IN_ABLE) != 0);
                if (level.u._in.nIn == 0)
                {
                    level.addrNxt = sqlite3VdbeMakeLabel(v);
                }
                level.u._in.nIn++;
                if (level.u._in.aInLoop == null)
                    level.u._in.aInLoop = new InLoop[level.u._in.nIn];
                else
                    Array.Resize(ref level.u._in.aInLoop, level.u._in.nIn);
                //sqlite3DbReallocOrFree(pParse.db, pLevel.u._in.aInLoop,
                //                       sizeof(pLevel.u._in.aInLoop[0])*pLevel.u._in.nIn);
                //pIn = pLevel.u._in.aInLoop;
                if (level.u._in.aInLoop != null)//(pIn )
                {
                    level.u._in.aInLoop[level.u._in.nIn - 1] = new InLoop();
                    pIn = level.u._in.aInLoop[level.u._in.nIn - 1];//pIn++
                    pIn.iCur = iTab;
                    if (eType == IN_INDEX_ROWID)
                    {
                        pIn.addrInTop = sqlite3VdbeAddOp2(v, OP_Rowid, iTab, regId);
                    }
                    else
                    {
                        pIn.addrInTop = sqlite3VdbeAddOp3(v, OP_Column, iTab, 0, regId);
                    }
                    sqlite3VdbeAddOp1(v, OP_IsNull, regId);
                }
                else
                {
                    level.u._in.nIn = 0;
                }
#endif
            }
            DisableTerm(level, term);
            return regId;
        }

        static int CodeAllEqualityTerms(Parse parse, WhereLevel level, WhereClause wc, Bitmask notReady, int extraRegs, out StringBuilder affOut)
        {
            int eqs = (int)level.Plan.Eqs; // The number of == or IN constraints to code
            Vdbe v = parse.V;       // The vm under construction
            int cursor = level.TabCur; // The cursor of the table

            // This module is only called on query plans that use an index.
            Debug.Assert((level.Plan.WsFlags & WHERE_INDEXED) != 0);
            Index index = level.Plan.u.Index; // The index being used for this loop

            // Figure out how many memory cells we will need then allocate them.
            int regBase = parse.Mems + 1; // Base register
            int regs = (int)(level.Plan.Eqs + extraRegs); // Number of registers to allocate
            parse.Mems += regs;

            StringBuilder aff = new StringBuilder(Index.AffinityStr(v, index)); // Affinity string to return
            if (aff == null)
                parse.Ctx.MallocFailed = true;

            // Evaluate the equality constraints
            Debug.Assert(index.Columns.length >= eqs);
            for (int j = 0; j < eqs; j++)
            {
                int k = index.Columns[j];
                WhereTerm term = FindTerm(wc, cursor, k, notReady, level.Plan.WsFlags, index); // A single constraint term
                if (term == null) break;
                // The following true for indices with redundant columns. Ex: CREATE INDEX i1 ON t1(a,b,a); SELECT * FROM t1 WHERE a=0 AND b=0;
                SysEx.ASSERTCOVERAGE((term.wtFlags & TERM.CODED) != 0);
                SysEx.ASSERTCOVERAGE((term.wtFlags & TERM.VIRTUAL) != 0); // EV: R-30575-11662
                int r1 = CodeEqualityTerm(parse, term, level, regBase + j);
                if (r1 != regBase + j)
                {
                    if (regs == 1)
                    {
                        parse.ReleaseTempReg(regBase);
                        regBase = r1;
                    }
                    else
                        v.AddOp2(OP.SCopy, r1, regBase + j);
                }
                SysEx.ASSERTCOVERAGE((term.EOperator & WO.ISNULL) != 0);
                SysEx.ASSERTCOVERAGE((term.EOperator & WO.IN) != 0);
                if ((term.EOperator & (WO.ISNULL | WO.IN)) == 0)
                {
                    Expr right = term.Expr.Right;
                    Expr.CodeIsNullJump(v, right, regBase + j, level.AddrBrk);
                    if (aff.Length > 0)
                        if (right.CompareAffinity(aff[j]) == AFF.NONE || right.NeedsNoAffinityChange(aff[j]))
                            aff[j] = AFF.NONE;
                }
            }
            affOut = aff;
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
            pLevel = pWInfo.Data[iLevel];
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
                        DisableTerm(pLevel, pWC.a[iTerm]);
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
                    iRowidReg = CodeEqualityTerm(pParse, pTerm, pLevel, iReleaseReg);
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
                        DisableTerm(pLevel, pStart);
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
                        DisableTerm(pLevel, pEnd);
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
                    regBase = CodeAllEqualityTerms(
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
                    CodeApplyAffinity(pParse, regBase, nConstraint, zStartAff.ToString());
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
                        CodeApplyAffinity(pParse, regBase, nEq + 1, zEndAff.ToString());
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
                    DisableTerm(pLevel, pRangeStart);
                    DisableTerm(pLevel, pRangeEnd);
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
                                pOrTab.a[k] = origSrc[pWInfo.Data[iLevel + k].iFrom];// memcpy(&pOrTab.a[k], &origSrc[pLevel[k].iFrom], sizeof(pOrTab.a[k]));
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
                                    pParse, pOrTab, pSubWInfo.Data[0], iLevel, pLevel.iFrom, 0
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
                            DisableTerm(pLevel, pTerm);
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
                    sqlite3_index_info pInfo = pWInfo.Data[i] != null ? pWInfo.Data[i].pIdxInfo : null;
                    if (pInfo != null)
                    {
                        /* Debug.Assert( pInfo.needToFreeIdxStr==0 || db.mallocFailed ); */
                        if (pInfo.needToFreeIdxStr != 0)
                        {
                            //sqlite3_free( ref pInfo.idxStr );
                        }
                        sqlite3DbFree(db, ref pInfo);
                    }
                    if (pWInfo.Data[i] != null && (pWInfo.Data[i].plan.wsFlags & WHERE_TEMP_INDEX) != 0)
                    {
                        Index pIdx = pWInfo.Data[i].plan.u.pIdx;
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
            pWInfo.Data = new WhereLevel[pTabList.nSrc];
            for (int ai = 0; ai < pWInfo.Data.Length; ai++)
            {
                pWInfo.Data[ai] = new WhereLevel();
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
            pWInfo.BreakId = sqlite3VdbeMakeLabel(v);
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
                sqlite3ExprIfFalse(pParse, pWhere, pWInfo.BreakId, SQLITE_JUMPIFNULL);
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
                pLevel = pWInfo.Data[i];
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
                            sqlite3_index_info pp = pWInfo.Data[j].pIdxInfo;
                            bestVirtualIndex(pParse, pWC, pTabItem, mask, notReady, pOrderBy,
                             ref sCost, ref pp);
                        }
                        else
#endif
                        {
                            BestBtreeIndex(pParse, pWC, pTabItem, mask, notReady, pOrderBy,
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
                pWInfo.Data[0].plan.wsFlags = (u32)(pWInfo.Data[0].plan.wsFlags & ~WHERE_IDX_ONLY);
            }

            /* Open all tables in the pTabList and any indices selected for
            ** searching those tables.
            */
            sqlite3CodeVerifySchema(pParse, -1); /* Insert the cookie verifier Goto */
            notReady = ~(Bitmask)0;
            pWInfo.nRowOut = (double)1;
            for (i = 0; i < nTabList; i++)//, pLevel++ )
            {
                pLevel = pWInfo.Data[i];
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
            pWInfo.TopId = sqlite3VdbeCurrentAddr(v);
            //if( db.mallocFailed ) goto whereBeginError;

            /* Generate the code to do the search.  Each iteration of the for
            ** loop below generates code for a single nested loop of the VM
            ** program.
            */
            notReady = ~(Bitmask)0;
            for (i = 0; i < nTabList; i++)
            {
                pLevel = pWInfo.Data[i];
                explainOneScan(pParse, pTabList, pLevel, i, pLevel.iFrom, wctrlFlags);
                notReady = codeOneLoopStart(pWInfo, i, wctrlFlags, notReady);
                pWInfo.ContinueId = pLevel.addrCont;
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
                pLevel = pWInfo.Data[i];
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
            sqlite3VdbeResolveLabel(v, pWInfo.BreakId);

            /* Close all of the cursors that were opened by sqlite3WhereBegin.
            */
            Debug.Assert(pWInfo.nLevel == 1 || pWInfo.nLevel == pTabList.nSrc);
            for (i = 0; i < pWInfo.nLevel; i++)//  for(i=0, pLevel=pWInfo.a; i<pWInfo.nLevel; i++, pLevel++){
            {
                pLevel = pWInfo.Data[i];
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
                    for (k = pWInfo.TopId; k < last; k++)//, pOp++ )
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
        #endregion

}
