using System;
using System.Diagnostics;
using System.Text;
using Bitmask = System.UInt64;

namespace Core
{
    public partial class Walker
    {
        public int WalkExpr(ref Expr expr)
        {
            if (expr == null) return WRC.Continue;
            ASSERTCOVERAGE(ExprHasProperty(expr, EP.TokenOnly));
            ASSERTCOVERAGE(ExprHasProperty(expr, EP.Reduced));
            int rc = ExprCallback(this, ref expr);
            if (rc == WRC.Continue && !ExprHasAnyProperty(expr, EP.TokenOnly))
            {
                if (WalkExpr(pWalker, ref expr.Left) != 0) return WRC.Abort;
                if (WalkExpr(pWalker, ref expr.Right) != 0) return WRC.Abort;
                if (ExprHasProperty(expr, EP_xIsSelect))
                {
                    if (WalkSelect(expr.x.Select) != 0) return WRC.Abort;
                }
                else
                {
                    if (WalkExprList(expr.x.List) != 0) return WRC.Abort;
                }
            }
            return rc & WRC.Abort;
        }

        public int WalkExprList(ExprList p)
        {
            int i;
            ExprList_item item;
            if (p != null)
                for (i = p.Exprs; i > 0; i--)
                {
                    item = p.Ids[p.Exprs - i];
                    if (WalkExpr(ref item.Expr) != 0) return WRC.Abort;
                }
            return WRC.Continue;
        }

        public static int WalkSelectExpr(Select p)
        {
            if (WalkExprList(p.EList) != 0) return WRC.Abort;
            if (WalkExpr(ref p.Where) != 0) return WRC.Abort;
            if (WalkExprList(p.GroupBy) != 0) return WRC.Abort;
            if (WalkExpr(ref p.Having) != 0) return WRC.Abort;
            if (WalkExprList(p.OrderBy) != 0) return WRC.Abort;
            if (WalkExpr(ref p.Limit) != 0) return WRC.Abort;
            if (WalkExpr(ref p.Offset) != 0) return WRC.Abort;
            return WRC.Continue;
        }

        public int WalkSelectFrom(Select p)
        {
            SrcList src = p.Src;
            int i;
            SrcList.SrcListItem item;
            if (SysEx.ALWAYS(src))
                for (i = src.Srcs; i > 0; i--)
                {
                    item = src.Ids[src.Srcs - i];
                    if (WalkSelect(item.Select) != 0) return WRC.Abort;
                }
            return WRC.Continue;
        }

        public static int WalkSelect(Select p)
        {
            if (p == null || SelectCallback == null) return WRC.Continue;
            int rc = WRC.Continue;
            while (p != null)
            {
                rc = SelectCallback(this, p);
                if (rc != 0) break;
                if (WalkSelectExpr(p) != 0 || SelectFrom(p) != 0)
                {
                    WalkerDepth--;
                    return WRC_Abort;
                }
                p = p.Prior;
            }
            return rc & WRC.Abort;
        }
    }
}
