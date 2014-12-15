using System;
using System.Diagnostics;
using System.Text;
using Bitmask = System.UInt64;
#if !MAX_VARIABLE_NUMBER
using yVars = System.Int16;
#else
using yVars = System.Int32; 
#endif

namespace Core
{
    public partial class Walker
    {
        static WRC IncrAggDepth(Walker walker, Expr expr)
        {
            if (expr.OP == TK.AGG_FUNCTION) expr.OP2 += (byte)walker.u.I;
            return WRC.Continue;
        }

        static void IncrAggFunctionDepth(Expr expr, int n)
        {
            if (n > 0)
            {
                Walker w = new Walker();
                w.ExprCallback = IncrAggDepth;
                w.u.I = n;
                w.WalkExpr(expr);
            }
        }

        static void ResolveAlias(Parse parse, ExprList list, int colId, Expr expr, string type, int subqueries)
        {
            Debug.Assert(colId >= 0 && colId < list.Exprs);
            Expr orig = list.Ids[colId].Expr; // The iCol-th column of the result set
            Debug.Assert(orig != null);
            Debug.Assert((orig.Flags & EP.Resolved) != 0);
            Context ctx = parse.Ctx; // The database connection
            Expr dup = Expr.Dup(ctx, orig, 0); // Copy of pOrig
            if (orig.OP != TK.COLUMN && (type.Length == 0 || type[0] != 'G'))
            {
                IncrAggFunctionDepth(dup, subqueries);
                dup = Expr.PExpr_(parse, TK.AS, dup, null, null);
                if (dup == null) return;
                if (list.Ids[colId].Alias == 0)
                    list.Ids[colId].Alias = (ushort)(++parse.Alias.length);
                dup.TableIdx = list.Ids[colId].Alias;
            }
            if (expr.OP == TK.COLLATE)
                dup = Expr.AddCollateString(parse, dup, expr.u.Token);

            // Before calling sqlite3ExprDelete(), set the EP_Static flag. This prevents ExprDelete() from deleting the Expr structure itself,
            // allowing it to be repopulated by the memcpy() on the following line.
            E.ExprSetProperty(expr, EP.Static);
            Expr.Delete(ctx, ref expr);
            expr.memcpy(dup);
            if (!E.ExprHasProperty(expr, EP.IntValue) && expr.u.Token != null)
            {
                Debug.Assert((dup.Flags & (EP.Reduced | EP.TokenOnly)) == 0);
                dup.u.Token = expr.u.Token;
                dup.Flags2 |= EP2.MallocedToken;
            }
            C._tagfree(ctx, ref dup);
        }

        static bool NameInUsingClause(IdList using_, string colName)
        {
            if (using_ != null)
                for (int k = 0; k < using_.Ids.length; k++)
                    if (string.Compare(using_.Ids[k].Name, colName, StringComparison.OrdinalIgnoreCase) == 0) return true;
            return false;
        }

        public static bool MatchSpanName(string span, string colName, string table, string dbName)
        {
            int n;
            for (n = 0; C._ALWAYS(span[n] != null) && span[n] != '.'; n++) { }
            if (dbName != null && (string.Compare(span, 0, dbName, 0, n, StringComparison.OrdinalIgnoreCase) != 0) || dbName[n] != 0)
                return false;
            span += n + 1;
            for (n = 0; C._ALWAYS(span[n] != null) && span[n] != '.'; n++) { }
            if (table != null && (string.Compare(span, 0, table, 0, n, StringComparison.OrdinalIgnoreCase) != 0 || table[n] != 0))
                return false;
            span += n + 1;
            if (colName != null && string.Compare(span, colName) != 0)
                return false;
            return true;
        }

        static int LookupName(Parse parse, string dbName, string tableName, string colName, NameContext nc, Expr expr)
        {
            int cnt = 0; // Number of matching column names
            int cntTab = 0; // Number of matching table names
            int subquerys = 0; // How many levels of subquery
            Context ctx = parse.Ctx; // The database connection
            SrcList.SrcListItem item; // Use for looping over pSrcList items
            SrcList.SrcListItem match = null; // The matching pSrcList item
            NameContext topNC = nc; // First namecontext in the list
            Schema schema = null;  // Schema of the expression
            bool isTrigger = false;
            int i, j;

            Debug.Assert(nc != null); // the name context cannot be NULL.
            Debug.Assert(colName != null);    // The Z in X.Y.Z cannot be NULL
            Debug.Assert(!E.ExprHasAnyProperty(expr, EP.TokenOnly | EP.Reduced));

            // Initialize the node to no-match
            expr.TableIdx = -1;
            expr.Table = null;
            E.ExprSetIrreducible(expr);

            // Translate the schema name in zDb into a pointer to the corresponding schema.  If not found, pSchema will remain NULL and nothing will match
            // resulting in an appropriate error message toward the end of this routine
            if (dbName != null)
            {
                for (i = 0; i < ctx.DBs.length; i++)
                {
                    Debug.Assert(ctx.DBs[i].Name != null);
                    if (string.Compare(ctx.DBs[i].Name, dbName) == 0)
                    {
                        schema = ctx.DBs[i].Schema;
                        break;
                    }
                }
            }

            // Start at the inner-most context and move outward until a match is found
            while (nc != null && cnt == 0)
            {
                ExprList list;
                SrcList srcList = nc.SrcList;
                if (srcList != null)
                {
                    for (i = 0; i < srcList.Srcs; i++)
                    {
                        item = srcList.Ids[i];
                        Table table = item.Table;
                        Debug.Assert(table != null && table.Name != null);
                        Debug.Assert(table.Cols.length > 0);
                        if (item.Select != null && (item.Select.SelFlags & SF.NestedFrom) != 0)
                        {
                            bool hit = false;
                            list = item.Select.EList;
                            for (j = 0; j < list.Exprs; j++)
                            {
                                if (sqlite3MatchSpanName(list.Ids[j].Span, colName, tableName, dbName))
                                {
                                    cnt++;
                                    cntTab = 2;
                                    match = item;
                                    expr.ColumnIdx = j;
                                    hit = true;
                                }
                            }
                            if (hit || table == null) continue;
                        }
                        if (dbName != null && table.Schema != schema)
                            continue;
                        if (tableName != null)
                        {
                            string tableName2 = (item.Alias != null ? item.Alias : table.Name);
                            Debug.Assert(tableName2 != null);
                            if (!string.Equals(tableName2, tableName, StringComparison.OrdinalIgnoreCase))
                                continue;
                        }
                        if (cntTab++ == 0)
                            match = item;
                        Column col;
                        for (j = 0; j < table.Cols.length; j++)
                        {
                            col = table.Cols[j];
                            if (string.Equals(col.Name, colName, StringComparison.InvariantCultureIgnoreCase))
                            {
                                // If there has been exactly one prior match and this match is for the right-hand table of a NATURAL JOIN or is in a 
                                // USING clause, then skip this match.
                                if (cnt == 1)
                                {
                                    if ((item.Jointype & JT.NATURAL) != 0) continue;
                                    if (NameInUsingClause(item.Using, colName)) continue;
                                }
                                cnt++;
                                match = item;
                                // Substitute the rowid (column -1) for the INTEGER PRIMARY KEY
                                expr.ColumnIdx = (j == table.PKey ? -1 : (short)j);
                                break;
                            }
                        }
                    }
                    if (match != null)
                    {
                        expr.TableIdx = match.Cursor;
                        expr.Table = match.Table;
                        schema = expr.Table.Schema;
                    }
                }

#if !OMIT_TRIGGER
                // If we have not already resolved the name, then maybe it is a new.* or old.* trigger argument reference
                if (dbName == null && tableName != null && cnt == 0 && parse.TriggerTab != null)
                {
                    TK op = parse.TriggerOp;
                    Table table = null;
                    Debug.Assert(op == TK.DELETE || op == TK.UPDATE || op == TK.INSERT);
                    if (op != TK.DELETE && string.Equals("new", tableName, StringComparison.InvariantCultureIgnoreCase))
                    {
                        expr.TableIdx = 1;
                        table = parse.TriggerTab;
                    }
                    else if (op != TK.INSERT && string.Equals("old", tableName, StringComparison.InvariantCultureIgnoreCase))
                    {
                        expr.TableIdx = 0;
                        table = parse.TriggerTab;
                    }
                    if (table != null)
                    {
                        int colId;
                        schema = table.Schema;
                        cntTab++;
                        for (colId = 0; colId < table.Cols.length; colId++)
                        {
                            Column col = table.Cols[colId];
                            if (string.Equals(col.Name, colName, StringComparison.InvariantCultureIgnoreCase))
                            {
                                if (colId == table.PKey)
                                    colId = -1;
                                break;
                            }
                        }
                        if (colId >= table.Cols.length && sqlite3IsRowid(colName))
                            colId = -1; // IMP: R-44911-55124
                        if (colId < table.Cols.length)
                        {
                            cnt++;
                            if (colId < 0)
                                expr.Aff = AFF.INTEGER;
                            else if (expr.TableIdx == 0)
                            {
                                C.ASSERTCOVERAGE(colId == 31);
                                C.ASSERTCOVERAGE(colId == 32);
                                parse.Oldmask |= (colId >= 32 ? 0xffffffff : (((uint)1) << colId));
                            }
                            else
                            {
                                C.ASSERTCOVERAGE(colId == 31);
                                C.ASSERTCOVERAGE(colId == 32);
                                parse.Newmask |= (colId >= 32 ? 0xffffffff : (((uint)1) << colId));
                            }
                            expr.ColumnIdx = (short)colId;
                            expr.Table = table;
                            isTrigger = true;
                        }
                    }
                }
#endif

                // Perhaps the name is a reference to the ROWID
                if (cnt == 0 && cntTab == 1 && sqlite3IsRowid(colName))
                {
                    cnt = 1;
                    expr.ColumnIdx = -1; // IMP: R-44911-55124
                    expr.Aff = AFF.INTEGER;
                }

                // If the input is of the form Z (not Y.Z or X.Y.Z) then the name Z might refer to an result-set alias.  This happens, for example, when
                // we are resolving names in the WHERE clause of the following command:
                //
                //     SELECT a+b AS x FROM table WHERE x<10;
                //
                // In cases like this, replace pExpr with a copy of the expression that forms the result set entry ("a+b" in the example) and return immediately.
                // Note that the expression in the result set should have already been resolved by the time the WHERE clause is resolved.
                if (cnt == 0 && (list = nc.EList) != null && tableName == null)
                {
                    for (j = 0; j < list.Exprs; j++)
                    {
                        string asName = list.Ids[j].Name;
                        if (asName != null && string.Equals(asName, colName, StringComparison.InvariantCultureIgnoreCase))
                        {
                            Debug.Assert(expr.Left == null && expr.Right == null);
                            Debug.Assert(expr.x.List == null);
                            Debug.Assert(expr.x.Select == null);
                            Expr orig = list.Ids[j].Expr;
                            if ((nc.NCFlags & NC.AllowAgg) == 0 && E.ExprHasProperty(orig, EP.Agg))
                            {
                                parse.ErrorMsg("misuse of aliased aggregate %s", asName);
                                return WRC.Abort;
                            }
                            ResolveAlias(parse, list, j, expr, "", subquerys);
                            cnt = 1;
                            match = null;
                            Debug.Assert(tableName == null && dbName == null);
                            goto lookupname_end;
                        }
                    }
                }

                // Advance to the next name context.  The loop will exit when either we have a match (cnt>0) or when we run out of name contexts.
                if (cnt == 0)
                {
                    nc = nc.Next;
                    subquerys++;
                }
            }

            // If X and Y are NULL (in other words if only the column name Z is supplied) and the value of Z is enclosed in double-quotes, then
            // Z is a string literal if it doesn't match any column names.  In that case, we need to return right away and not make any changes to
            // pExpr.
            //
            // Because no reference was made to outer contexts, the pNC->nRef fields are not changed in any context.
            if (cnt == 0 && tableName == null && E.ExprHasProperty(expr, EP.DblQuoted))
            {
                expr.OP = TK.STRING;
                expr.Table = null;
                return WRC.Prune;
            }

            // cnt==0 means there was not match.  cnt>1 means there were two or more matches.  Either way, we have an error.
            if (cnt != 1)
            {
                string err = (cnt == 0 ? "no such column" : "ambiguous column name");
                if (dbName != null)
                    parse.ErrorMsg("%s: %s.%s.%s", err, dbName, tableName, colName);
                else if (tableName != null)
                    parse.ErrorMsg("%s: %s.%s", err, tableName, colName);
                else
                    parse.ErrorMsg("%s: %s", err, colName);
                parse.CheckSchema = 1;
                topNC.Errs++;
            }

            // If a column from a table in pSrcList is referenced, then record this fact in the pSrcList.a[].colUsed bitmask.  Column 0 causes
            // bit 0 to be set.  Column 1 sets bit 1.  And so forth.  If the column number is greater than the number of bits in the bitmask
            // then set the high-order bit of the bitmask.
            if (expr.ColumnIdx >= 0 && match != null)
            {
                int n = expr.ColumnIdx;
                C.ASSERTCOVERAGE(n == BMS - 1);
                if (n >= BMS)
                    n = BMS - 1;
                Debug.Assert(match.Cursor == expr.TableIdx);
                match.ColUsed |= ((Bitmask)1) << n;
            }

            // Clean up and return
            Expr.Delete(ctx, ref expr.Left);
            expr.Left = null;
            Expr.Delete(ctx, ref expr.Right);
            expr.Right = null;
            expr.OP = (isTrigger ? TK.TRIGGER : TK.COLUMN);
        lookupname_end:
            if (cnt == 1)
            {
                Debug.Assert(nc != null);
                Auth.Read(parse, expr, schema, nc.SrcList);
                // Increment the nRef value on all name contexts from TopNC up to the point where the name matched.
                for (; ; )
                {
                    Debug.Assert(topNC != null);
                    topNC.Refs++;
                    if (topNC == nc) break;
                    topNC = topNC.Next;
                }
                return WRC.Prune;
            }
            return WRC.Abort;
        }

        public static Expr CreateColumnExpr(Context ctx, SrcList src, int srcId, int colId)
        {
            Expr p = sqlite3ExprAlloc(ctx, TK.COLUMN, null, 0);
            if (p != null)
            {
                SrcList.SrcListItem item = src.Ids[srcId];
                p.Table = item.Table;
                p.TableIdx = item.Cursor;
                if (p.Table.PKey == colId)
                    p.ColumnIdx = -1;
                else
                {
                    p.ColumnIdx = (yVars)colId;
                    C.ASSERTCOVERAGE(colId == BMS);
                    C.ASSERTCOVERAGE(colId == BMS - 1);
                    item.ColUsed |= ((Bitmask)1) << (colId >= BMS ? BMS - 1 : colId);
                }
                E.ExprSetProperty(p, EP.Resolved);
            }
            return p;
        }

        public static WRC ResolveExprStep(Walker walker, ref Expr expr)
        {
            NameContext nc = walker.u.NC;
            Debug.Assert(nc != null);
            Parse parse = nc.Parse;
            Debug.Assert(parse == walker.Parse);

            if (E.ExprHasAnyProperty(expr, EP.Resolved)) return WRC.Prune;
            E.ExprSetProperty(expr, EP.Resolved);
#if !NDEBUG
            if (nc.SrcList != null && nc.SrcList.Allocs > 0)
            {
                SrcList srcList = nc.SrcList;
                for (int i = 0; i < nc.SrcList.Srcs; i++)
                    Debug.Assert(srcList.Ids[i].Cursor >= 0 && srcList.Ids[i].Cursor < parse.Tabs);
            }
#endif
            switch (expr.OP)
            {

#if ENABLE_UPDATE_DELETE_LIMIT && !OMIT_SUBQUERY
                // The special operator TK_ROW means use the rowid for the first column in the FROM clause.  This is used by the LIMIT and ORDER BY
                // clause processing on UPDATE and DELETE statements.
                case TK.ROW:
                    {
                        SrcList srcList = nc.SrcList;
                        Debug.Assert(srcList != null && srcList.Srcs == 1);
                        SrcList.SrcListItem item = srcList.Ids[0];
                        expr.OP = TK.COLUMN;
                        expr.Table = item.Table;
                        expr.TableIdx = item.Cursor;
                        expr.ColumnIdx = -1;
                        expr.Aff = AFF.INTEGER;
                        break;
                    }
#endif

                case TK.ID:  // A lone identifier is the name of a column.
                    {
                        return LookupName(parse, null, null, expr.u.Token, nc, expr);
                    }

                case TK.DOT: // A table name and column name: ID.ID Or a database, table and column: ID.ID.ID
                    {
                        string columnName;
                        string tableName;
                        string dbName;
                        // if (srcList == nullptr) break;
                        Expr right = expr.Right;
                        if (right.OP == TK.ID)
                        {
                            dbName = null;
                            tableName = expr.Left.u.Token;
                            columnName = right.u.Token;
                        }
                        else
                        {
                            Debug.Assert(right.OP == TK.DOT);
                            dbName = expr.Left.u.Token;
                            tableName = right.Left.u.Token;
                            columnName = right.Right.u.Token;
                        }
                        return LookupName(parse, dbName, tableName, columnName, nc, expr);
                    }

                case TK.CONST_FUNC:
                case TK.FUNCTION: // Resolve function names
                    {
                        ExprList pList = expr.x.pList;    /* The argument list */
                        int n = pList != null ? pList.nExpr : 0;  /* Number of arguments */
                        bool no_such_func = false;       /* True if no such function exists */
                        bool wrong_num_args = false;     /* True if wrong number of arguments */
                        bool is_agg = false;             /* True if is an aggregate function */
                        int auth;                   /* Authorization to use the function */
                        int nId;                    /* Number of characters in function name */
                        string zId;                 /* The function name. */
                        FuncDef pDef;              /* Information about the function */
                        u8 enc = (u8)parse.db.aDbStatic[0].pSchema.enc;// ENC( pParse.db );   /* The database encoding */

                        testcase(expr.op == TK_CONST_FUNC);
                        Debug.Assert(!ExprHasProperty(expr, EP_xIsSelect));
                        zId = expr.u.zToken;
                        nId = sqlite3Strlen30(zId);
                        pDef = sqlite3FindFunction(parse.db, zId, nId, n, enc, 0);
                        if (pDef == null)
                        {
                            pDef = sqlite3FindFunction(parse.db, zId, nId, -1, enc, 0);
                            if (pDef == null)
                            {
                                no_such_func = true;
                            }
                            else
                            {
                                wrong_num_args = true;
                            }
                        }
                        else
                        {
                            is_agg = pDef.xFunc == null;
                        }
#if !SQLITE_OMIT_AUTHORIZATION
                        if (pDef)
                        {
                            auth = sqlite3AuthCheck(parse, SQLITE_FUNCTION, 0, pDef.zName, 0);
                            if (auth != SQLITE_OK)
                            {
                                if (auth == SQLITE_DENY)
                                {
                                    sqlite3ErrorMsg(parse, "not authorized to use function: %s",
                                    pDef.zName);
                                    nc.nErr++;
                                }
                                expr.op = TK_NULL;
                                return WRC_Prune;
                            }
                        }
#endif
                        if (is_agg && 0 == nc.allowAgg)
                        {
                            sqlite3ErrorMsg(parse, "misuse of aggregate function %.*s()", nId, zId);
                            nc.nErr++;
                            is_agg = false;
                        }
                        else if (no_such_func)
                        {
                            sqlite3ErrorMsg(parse, "no such function: %.*s", nId, zId);
                            nc.nErr++;
                        }
                        else if (wrong_num_args)
                        {
                            sqlite3ErrorMsg(parse, "wrong number of arguments to function %.*s()",
                            nId, zId);
                            nc.nErr++;
                        }
                        if (is_agg)
                        {
                            expr.op = TK_AGG_FUNCTION;
                            nc.hasAgg = 1;
                        }
                        if (is_agg)
                            nc.allowAgg = 0;
                        sqlite3WalkExprList(walker, pList);
                        if (is_agg)
                            nc.allowAgg = 1;
                        /* FIX ME:  Compute pExpr.affinity based on the expected return
                        ** type of the function
                        */
                        return WRC_Prune;
                    }
#if !SQLITE_OMIT_SUBQUERY
                case TK_SELECT:
                case TK_EXISTS:
                    {
                        testcase(expr.op == TK_EXISTS);
                        goto case TK_IN;
                    }
#endif
                case TK_IN:
                    {
                        testcase(expr.op == TK_IN);
                        if (ExprHasProperty(expr, EP_xIsSelect))
                        {
                            int nRef = nc.nRef;
#if !SQLITE_OMIT_CHECK
                            if (nc.isCheck != 0)
                            {
                                sqlite3ErrorMsg(parse, "subqueries prohibited in CHECK constraints");
                            }
#endif
                            sqlite3WalkSelect(walker, expr.x.pSelect);
                            Debug.Assert(nc.nRef >= nRef);
                            if (nRef != nc.nRef)
                            {
                                ExprSetProperty(expr, EP_VarSelect);
                            }
                        }
                        break;
                    }
#if !SQLITE_OMIT_CHECK
                case TK_VARIABLE:
                    {
                        if (nc.isCheck != 0)
                        {
                            sqlite3ErrorMsg(parse, "parameters prohibited in CHECK constraints");
                        }
                        break;
                    }
#endif
            }
            return (parse.nErr != 0 /* || pParse.db.mallocFailed != 0 */ ) ? WRC_Abort : WRC_Continue;
        }

        /*
        ** pEList is a list of expressions which are really the result set of the
        ** a SELECT statement.  pE is a term in an ORDER BY or GROUP BY clause.
        ** This routine checks to see if pE is a simple identifier which corresponds
        ** to the AS-name of one of the terms of the expression list.  If it is,
        ** this routine return an integer between 1 and N where N is the number of
        ** elements in pEList, corresponding to the matching entry.  If there is
        ** no match, or if pE is not a simple identifier, then this routine
        ** return 0.
        **
        ** pEList has been resolved.  pE has not.
        */
        static int resolveAsName(
        Parse pParse,     /* Parsing context for error messages */
        ExprList pEList,  /* List of expressions to scan */
        Expr pE           /* Expression we are trying to match */
        )
        {
            int i;             /* Loop counter */

            UNUSED_PARAMETER(pParse);

            if (pE.op == TK_ID)
            {
                string zCol = pE.u.zToken;

                for (i = 0; i < pEList.nExpr; i++)
                {
                    string zAs = pEList.a[i].zName;
                    if (zAs != null && zAs.Equals(zCol, StringComparison.InvariantCultureIgnoreCase))
                    {
                        return i + 1;
                    }
                }
            }
            return 0;
        }

        /*
        ** pE is a pointer to an expression which is a single term in the
        ** ORDER BY of a compound SELECT.  The expression has not been
        ** name resolved.
        **
        ** At the point this routine is called, we already know that the
        ** ORDER BY term is not an integer index into the result set.  That
        ** case is handled by the calling routine.
        **
        ** Attempt to match pE against result set columns in the left-most
        ** SELECT statement.  Return the index i of the matching column,
        ** as an indication to the caller that it should sort by the i-th column.
        ** The left-most column is 1.  In other words, the value returned is the
        ** same integer value that would be used in the SQL statement to indicate
        ** the column.
        **
        ** If there is no match, return 0.  Return -1 if an error occurs.
        */
        static int resolveOrderByTermToExprList(
        Parse pParse,     /* Parsing context for error messages */
        Select pSelect,   /* The SELECT statement with the ORDER BY clause */
        Expr pE           /* The specific ORDER BY term */
        )
        {
            int i = 0;         /* Loop counter */
            ExprList pEList;   /* The columns of the result set */
            NameContext nc;    /* Name context for resolving pE */
            sqlite3 db;        /* Database connection */
            int rc;            /* Return code from subprocedures */
            u8 savedSuppErr;   /* Saved value of db->suppressErr */

            Debug.Assert(sqlite3ExprIsInteger(pE, ref i) == 0);
            pEList = pSelect.pEList;

            /* Resolve all names in the ORDER BY term expression
            */
            nc = new NameContext();// memset( &nc, 0, sizeof( nc ) );
            nc.pParse = pParse;
            nc.pSrcList = pSelect.pSrc;
            nc.pEList = pEList;
            nc.allowAgg = 1;
            nc.nErr = 0;
            db = pParse.db;
            savedSuppErr = db.suppressErr;
            db.suppressErr = 1;
            rc = sqlite3ResolveExprNames(nc, ref pE);
            db.suppressErr = savedSuppErr;
            if (rc != 0)
                return 0;

            /* Try to match the ORDER BY expression against an expression
            ** in the result set.  Return an 1-based index of the matching
            ** result-set entry.
            */
            for (i = 0; i < pEList.nExpr; i++)
            {
                if (sqlite3ExprCompare(pEList.a[i].pExpr, pE) < 2)
                {
                    return i + 1;
                }
            }

            /* If no match, return 0. */
            return 0;
        }

        /*
        ** Generate an ORDER BY or GROUP BY term out-of-range error.
        */
        static void resolveOutOfRangeError(
        Parse pParse,         /* The error context into which to write the error */
        string zType,     /* "ORDER" or "GROUP" */
        int i,                 /* The index (1-based) of the term out of range */
        int mx                 /* Largest permissible value of i */
        )
        {
            sqlite3ErrorMsg(pParse,
            "%r %s BY term out of range - should be " +
            "between 1 and %d", i, zType, mx);
        }

        /*
        ** Analyze the ORDER BY clause in a compound SELECT statement.   Modify
        ** each term of the ORDER BY clause is a constant integer between 1
        ** and N where N is the number of columns in the compound SELECT.
        **
        ** ORDER BY terms that are already an integer between 1 and N are
        ** unmodified.  ORDER BY terms that are integers outside the range of
        ** 1 through N generate an error.  ORDER BY terms that are expressions
        ** are matched against result set expressions of compound SELECT
        ** beginning with the left-most SELECT and working toward the right.
        ** At the first match, the ORDER BY expression is transformed into
        ** the integer column number.
        **
        ** Return the number of errors seen.
        */
        static int resolveCompoundOrderBy(
        Parse pParse,        /* Parsing context.  Leave error messages here */
        Select pSelect       /* The SELECT statement containing the ORDER BY */
        )
        {
            int i;
            ExprList pOrderBy;
            ExprList pEList;
            sqlite3 db;
            int moreToDo = 1;

            pOrderBy = pSelect.pOrderBy;
            if (pOrderBy == null)
                return 0;
            db = pParse.db;
            //#if SQLITE_MAX_COLUMN
            if (pOrderBy.nExpr > db.aLimit[SQLITE_LIMIT_COLUMN])
            {
                sqlite3ErrorMsg(pParse, "too many terms in ORDER BY clause");
                return 1;
            }
            //#endif
            for (i = 0; i < pOrderBy.nExpr; i++)
            {
                pOrderBy.a[i].done = 0;
            }
            pSelect.pNext = null;
            while (pSelect.pPrior != null)
            {
                pSelect.pPrior.pNext = pSelect;
                pSelect = pSelect.pPrior;
            }
            while (pSelect != null && moreToDo != 0)
            {
                ExprList_item pItem;
                moreToDo = 0;
                pEList = pSelect.pEList;
                Debug.Assert(pEList != null);
                for (i = 0; i < pOrderBy.nExpr; i++)//, pItem++)
                {
                    pItem = pOrderBy.a[i];
                    int iCol = -1;
                    Expr pE, pDup;
                    if (pItem.done != 0)
                        continue;
                    pE = pItem.pExpr;
                    if (sqlite3ExprIsInteger(pE, ref iCol) != 0)
                    {
                        if (iCol <= 0 || iCol > pEList.nExpr)
                        {
                            resolveOutOfRangeError(pParse, "ORDER", i + 1, pEList.nExpr);
                            return 1;
                        }
                    }
                    else
                    {
                        iCol = resolveAsName(pParse, pEList, pE);
                        if (iCol == 0)
                        {
                            pDup = sqlite3ExprDup(db, pE, 0);
                            ////if ( 0 == db.mallocFailed )
                            {
                                Debug.Assert(pDup != null);
                                iCol = resolveOrderByTermToExprList(pParse, pSelect, pDup);
                            }
                            sqlite3ExprDelete(db, ref pDup);
                        }
                    }
                    if (iCol > 0)
                    {
                        CollSeq pColl = pE.pColl;
                        int flags = pE.flags & EP_ExpCollate;
                        sqlite3ExprDelete(db, ref pE);
                        pItem.pExpr = pE = sqlite3Expr(db, TK_INTEGER, null);
                        if (pE == null)
                            return 1;
                        pE.pColl = pColl;
                        pE.flags = (u16)(pE.flags | EP_IntValue | flags);
                        pE.u.iValue = iCol;
                        pItem.iCol = (u16)iCol;
                        pItem.done = 1;
                    }
                    else
                    {
                        moreToDo = 1;
                    }
                }
                pSelect = pSelect.pNext;
            }
            for (i = 0; i < pOrderBy.nExpr; i++)
            {
                if (pOrderBy.a[i].done == 0)
                {
                    sqlite3ErrorMsg(pParse, "%r ORDER BY term does not match any " +
                    "column in the result set", i + 1);
                    return 1;
                }
            }
            return 0;
        }

        /*
        ** Check every term in the ORDER BY or GROUP BY clause pOrderBy of
        ** the SELECT statement pSelect.  If any term is reference to a
        ** result set expression (as determined by the ExprList.a.iCol field)
        ** then convert that term into a copy of the corresponding result set
        ** column.
        **
        ** If any errors are detected, add an error message to pParse and
        ** return non-zero.  Return zero if no errors are seen.
        */
        static int sqlite3ResolveOrderGroupBy(
        Parse pParse,        /* Parsing context.  Leave error messages here */
        Select pSelect,      /* The SELECT statement containing the clause */
        ExprList pOrderBy,   /* The ORDER BY or GROUP BY clause to be processed */
        string zType         /* "ORDER" or "GROUP" */
        )
        {
            int i;
            sqlite3 db = pParse.db;
            ExprList pEList;
            ExprList_item pItem;

            if (pOrderBy == null /* || pParse.db.mallocFailed != 0 */ )
                return 0;
            //#if SQLITE_MAX_COLUMN
            if (pOrderBy.nExpr > db.aLimit[SQLITE_LIMIT_COLUMN])
            {
                sqlite3ErrorMsg(pParse, "too many terms in %s BY clause", zType);
                return 1;
            }
            //#endif
            pEList = pSelect.pEList;
            Debug.Assert(pEList != null);  /* sqlite3SelectNew() guarantees this */
            for (i = 0; i < pOrderBy.nExpr; i++)//, pItem++)
            {
                pItem = pOrderBy.a[i];
                if (pItem.iCol != 0)
                {
                    if (pItem.iCol > pEList.nExpr)
                    {
                        resolveOutOfRangeError(pParse, zType, i + 1, pEList.nExpr);
                        return 1;
                    }
                    resolveAlias(pParse, pEList, pItem.iCol - 1, pItem.pExpr, zType);
                }
            }
            return 0;
        }

        /*
        ** pOrderBy is an ORDER BY or GROUP BY clause in SELECT statement pSelect.
        ** The Name context of the SELECT statement is pNC.  zType is either
        ** "ORDER" or "GROUP" depending on which type of clause pOrderBy is.
        **
        ** This routine resolves each term of the clause into an expression.
        ** If the order-by term is an integer I between 1 and N (where N is the
        ** number of columns in the result set of the SELECT) then the expression
        ** in the resolution is a copy of the I-th result-set expression.  If
        ** the order-by term is an identify that corresponds to the AS-name of
        ** a result-set expression, then the term resolves to a copy of the
        ** result-set expression.  Otherwise, the expression is resolved in
        ** the usual way - using sqlite3ResolveExprNames().
        **
        ** This routine returns the number of errors.  If errors occur, then
        ** an appropriate error message might be left in pParse.  (OOM errors
        ** excepted.)
        */
        static int resolveOrderGroupBy(
        NameContext pNC,     /* The name context of the SELECT statement */
        Select pSelect,      /* The SELECT statement holding pOrderBy */
        ExprList pOrderBy,   /* An ORDER BY or GROUP BY clause to resolve */
        string zType         /* Either "ORDER" or "GROUP", as appropriate */
        )
        {
            int i;                         /* Loop counter */
            int iCol;                      /* Column number */
            ExprList_item pItem;   /* A term of the ORDER BY clause */
            Parse pParse;                 /* Parsing context */
            int nResult;                   /* Number of terms in the result set */

            if (pOrderBy == null)
                return 0;
            nResult = pSelect.pEList.nExpr;
            pParse = pNC.pParse;
            for (i = 0; i < pOrderBy.nExpr; i++)//, pItem++ )
            {
                pItem = pOrderBy.a[i];
                Expr pE = pItem.pExpr;
                iCol = resolveAsName(pParse, pSelect.pEList, pE);
                if (iCol > 0)
                {
                    /* If an AS-name match is found, mark this ORDER BY column as being
                    ** a copy of the iCol-th result-set column.  The subsequent call to
                    ** sqlite3ResolveOrderGroupBy() will convert the expression to a
                    ** copy of the iCol-th result-set expression. */
                    pItem.iCol = (u16)iCol;
                    continue;
                }
                if (sqlite3ExprIsInteger(pE, ref iCol) != 0)
                {
                    /* The ORDER BY term is an integer constant.  Again, set the column
                    ** number so that sqlite3ResolveOrderGroupBy() will convert the
                    ** order-by term to a copy of the result-set expression */
                    if (iCol < 1)
                    {
                        resolveOutOfRangeError(pParse, zType, i + 1, nResult);
                        return 1;
                    }
                    pItem.iCol = (u16)iCol;
                    continue;
                }

                /* Otherwise, treat the ORDER BY term as an ordinary expression */
                pItem.iCol = 0;
                if (sqlite3ResolveExprNames(pNC, ref pE) != 0)
                {
                    return 1;
                }
            }
            return sqlite3ResolveOrderGroupBy(pParse, pSelect, pOrderBy, zType);
        }

        /*
        ** Resolve names in the SELECT statement p and all of its descendents.
        */
        static int resolveSelectStep(Walker pWalker, Select p)
        {
            NameContext pOuterNC;  /* Context that contains this SELECT */
            NameContext sNC;       /* Name context of this SELECT */
            bool isCompound;       /* True if p is a compound select */
            int nCompound;         /* Number of compound terms processed so far */
            Parse pParse;          /* Parsing context */
            ExprList pEList;       /* Result set expression list */
            int i;                 /* Loop counter */
            ExprList pGroupBy;     /* The GROUP BY clause */
            Select pLeftmost;      /* Left-most of SELECT of a compound */
            sqlite3 db;            /* Database connection */


            Debug.Assert(p != null);
            if ((p.selFlags & SF_Resolved) != 0)
            {
                return WRC_Prune;
            }
            pOuterNC = pWalker.u.pNC;
            pParse = pWalker.pParse;
            db = pParse.db;

            /* Normally sqlite3SelectExpand() will be called first and will have
            ** already expanded this SELECT.  However, if this is a subquery within
            ** an expression, sqlite3ResolveExprNames() will be called without a
            ** prior call to sqlite3SelectExpand().  When that happens, let
            ** sqlite3SelectPrep() do all of the processing for this SELECT.
            ** sqlite3SelectPrep() will invoke both sqlite3SelectExpand() and
            ** this routine in the correct order.
            */
            if ((p.selFlags & SF_Expanded) == 0)
            {
                sqlite3SelectPrep(pParse, p, pOuterNC);
                return (pParse.nErr != 0 /*|| db.mallocFailed != 0 */ ) ? WRC_Abort : WRC_Prune;
            }

            isCompound = p.pPrior != null;
            nCompound = 0;
            pLeftmost = p;
            while (p != null)
            {
                Debug.Assert((p.selFlags & SF_Expanded) != 0);
                Debug.Assert((p.selFlags & SF_Resolved) == 0);
                p.selFlags |= SF_Resolved;

                /* Resolve the expressions in the LIMIT and OFFSET clauses. These
                ** are not allowed to refer to any names, so pass an empty NameContext.
                */
                sNC = new NameContext();// memset( &sNC, 0, sizeof( sNC ) );
                sNC.pParse = pParse;
                if (sqlite3ResolveExprNames(sNC, ref p.pLimit) != 0 ||
                sqlite3ResolveExprNames(sNC, ref p.pOffset) != 0)
                {
                    return WRC_Abort;
                }

                /* Set up the local name-context to pass to sqlite3ResolveExprNames() to
                ** resolve the result-set expression list.
                */
                sNC.allowAgg = 1;
                sNC.pSrcList = p.pSrc;
                sNC.pNext = pOuterNC;

                /* Resolve names in the result set. */
                pEList = p.pEList;
                Debug.Assert(pEList != null);
                for (i = 0; i < pEList.nExpr; i++)
                {
                    Expr pX = pEList.a[i].pExpr;
                    if (sqlite3ResolveExprNames(sNC, ref pX) != 0)
                    {
                        return WRC_Abort;
                    }
                }

                /* Recursively resolve names in all subqueries
                */
                for (i = 0; i < p.pSrc.nSrc; i++)
                {
                    SrcList_item pItem = p.pSrc.a[i];
                    if (pItem.pSelect != null)
                    {
                        string zSavedContext = pParse.zAuthContext;
                        if (pItem.zName != null)
                            pParse.zAuthContext = pItem.zName;
                        sqlite3ResolveSelectNames(pParse, pItem.pSelect, pOuterNC);
                        pParse.zAuthContext = zSavedContext;
                        if (pParse.nErr != 0 /*|| db.mallocFailed != 0 */ )
                            return WRC_Abort;
                    }
                }

                /* If there are no aggregate functions in the result-set, and no GROUP BY
                ** expression, do not allow aggregates in any of the other expressions.
                */
                Debug.Assert((p.selFlags & SF_Aggregate) == 0);
                pGroupBy = p.pGroupBy;
                if (pGroupBy != null || sNC.hasAgg != 0)
                {
                    p.selFlags |= SF_Aggregate;
                }
                else
                {
                    sNC.allowAgg = 0;
                }

                /* If a HAVING clause is present, then there must be a GROUP BY clause.
                */
                if (p.pHaving != null && pGroupBy == null)
                {
                    sqlite3ErrorMsg(pParse, "a GROUP BY clause is required before HAVING");
                    return WRC_Abort;
                }

                /* Add the expression list to the name-context before parsing the
                ** other expressions in the SELECT statement. This is so that
                ** expressions in the WHERE clause (etc.) can refer to expressions by
                ** aliases in the result set.
                **
                ** Minor point: If this is the case, then the expression will be
                ** re-evaluated for each reference to it.
                */
                sNC.pEList = p.pEList;
                if (sqlite3ResolveExprNames(sNC, ref p.pWhere) != 0 ||
                sqlite3ResolveExprNames(sNC, ref p.pHaving) != 0
                )
                {
                    return WRC_Abort;
                }

                /* The ORDER BY and GROUP BY clauses may not refer to terms in
                ** outer queries
                */
                sNC.pNext = null;
                sNC.allowAgg = 1;

                /* Process the ORDER BY clause for singleton SELECT statements.
                ** The ORDER BY clause for compounds SELECT statements is handled
                ** below, after all of the result-sets for all of the elements of
                ** the compound have been resolved.
                */
                if (!isCompound && resolveOrderGroupBy(sNC, p, p.pOrderBy, "ORDER") != 0)
                {
                    return WRC_Abort;
                }
                //if ( db.mallocFailed != 0 )
                //{
                //  return WRC_Abort;
                //}

                /* Resolve the GROUP BY clause.  At the same time, make sure
                ** the GROUP BY clause does not contain aggregate functions.
                */
                if (pGroupBy != null)
                {
                    ExprList_item pItem;

                    if (resolveOrderGroupBy(sNC, p, pGroupBy, "GROUP") != 0 /*|| db.mallocFailed != 0 */ )
                    {
                        return WRC_Abort;
                    }
                    for (i = 0; i < pGroupBy.nExpr; i++)//, pItem++)
                    {
                        pItem = pGroupBy.a[i];
                        if ((pItem.pExpr.flags & EP_Agg) != 0)//HasProperty(pItem.pExpr, EP_Agg) )
                        {
                            sqlite3ErrorMsg(pParse, "aggregate functions are not allowed in " +
                            "the GROUP BY clause");
                            return WRC_Abort;
                        }
                    }
                }

                /* Advance to the next term of the compound
                */
                p = p.pPrior;
                nCompound++;
            }

            /* Resolve the ORDER BY on a compound SELECT after all terms of
            ** the compound have been resolved.
            */
            if (isCompound && resolveCompoundOrderBy(pParse, pLeftmost) != 0)
            {
                return WRC_Abort;
            }

            return WRC_Prune;
        }

        /*
        ** This routine walks an expression tree and resolves references to
        ** table columns and result-set columns.  At the same time, do error
        ** checking on function usage and set a flag if any aggregate functions
        ** are seen.
        **
        ** To resolve table columns references we look for nodes (or subtrees) of the
        ** form X.Y.Z or Y.Z or just Z where
        **
        **      X:   The name of a database.  Ex:  "main" or "temp" or
        **           the symbolic name assigned to an ATTACH-ed database.
        **
        **      Y:   The name of a table in a FROM clause.  Or in a trigger
        **           one of the special names "old" or "new".
        **
        **      Z:   The name of a column in table Y.
        **
        ** The node at the root of the subtree is modified as follows:
        **
        **    Expr.op        Changed to TK_COLUMN
        **    Expr.pTab      Points to the Table object for X.Y
        **    Expr.iColumn   The column index in X.Y.  -1 for the rowid.
        **    Expr.iTable    The VDBE cursor number for X.Y
        **
        **
        ** To resolve result-set references, look for expression nodes of the
        ** form Z (with no X and Y prefix) where the Z matches the right-hand
        ** size of an AS clause in the result-set of a SELECT.  The Z expression
        ** is replaced by a copy of the left-hand side of the result-set expression.
        ** Table-name and function resolution occurs on the substituted expression
        ** tree.  For example, in:
        **
        **      SELECT a+b AS x, c+d AS y FROM t1 ORDER BY x;
        **
        ** The "x" term of the order by is replaced by "a+b" to render:
        **
        **      SELECT a+b AS x, c+d AS y FROM t1 ORDER BY a+b;
        **
        ** Function calls are checked to make sure that the function is
        ** defined and that the correct number of arguments are specified.
        ** If the function is an aggregate function, then the pNC.hasAgg is
        ** set and the opcode is changed from TK_FUNCTION to TK_AGG_FUNCTION.
        ** If an expression contains aggregate functions then the EP_Agg
        ** property on the expression is set.
        **
        ** An error message is left in pParse if anything is amiss.  The number
        ** if errors is returned.
        */
        static int sqlite3ResolveExprNames(
        NameContext pNC,       /* Namespace to resolve expressions in. */
        ref Expr pExpr         /* The expression to be analyzed. */
        )
        {
            u8 savedHasAgg;
            Walker w = new Walker();

            if (pExpr == null)
                return 0;
#if SQLITE_MAX_EXPR_DEPTH//>0
{
Parse pParse = pNC.pParse;
if( sqlite3ExprCheckHeight(pParse, pExpr.nHeight+pNC.pParse.nHeight) ){
return 1;
}
pParse.nHeight += pExpr.nHeight;
}
#endif
            savedHasAgg = pNC.hasAgg;
            pNC.hasAgg = 0;
            w.xExprCallback = resolveExprStep;
            w.xSelectCallback = resolveSelectStep;
            w.pParse = pNC.pParse;
            w.u.pNC = pNC;
            sqlite3WalkExpr(w, ref pExpr);
#if SQLITE_MAX_EXPR_DEPTH//>0
pNC.pParse.nHeight -= pExpr.nHeight;
#endif
            if (pNC.nErr > 0 || w.pParse.nErr > 0)
            {
                ExprSetProperty(pExpr, EP_Error);
            }
            if (pNC.hasAgg != 0)
            {
                ExprSetProperty(pExpr, EP_Agg);
            }
            else if (savedHasAgg != 0)
            {
                pNC.hasAgg = 1;
            }
            return ExprHasProperty(pExpr, EP_Error) ? 1 : 0;
        }


        /*
        ** Resolve all names in all expressions of a SELECT and in all
        ** decendents of the SELECT, including compounds off of p.pPrior,
        ** subqueries in expressions, and subqueries used as FROM clause
        ** terms.
        **
        ** See sqlite3ResolveExprNames() for a description of the kinds of
        ** transformations that occur.
        **
        ** All SELECT statements should have been expanded using
        ** sqlite3SelectExpand() prior to invoking this routine.
        */
        static void sqlite3ResolveSelectNames(
        Parse pParse,         /* The parser context */
        Select p,             /* The SELECT statement being coded. */
        NameContext pOuterNC  /* Name context for parent SELECT statement */
        )
        {
            Walker w = new Walker();

            Debug.Assert(p != null);
            w.xExprCallback = resolveExprStep;
            w.xSelectCallback = resolveSelectStep;
            w.pParse = pParse;
            w.u.pNC = pOuterNC;
            sqlite3WalkSelect(w, p);
        }
    }
}
