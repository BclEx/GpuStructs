#region OMIT_AUTHORIZATION
#if !OMIT_AUTHORIZATION
using System;
using System.Diagnostics;
using System.Text;
namespace Core
{
    public partial class Auth
    {
        public RC SetAauthorizer(Context ctx, Func<object, int, string, string, string, byte, ARC> auth, object arg)
        {
            MutexEx.Enter(ctx.Mutex);
            ctx.Auth = auth;
            ctx.AuthArg = arg;
            sqlite3ExpirePreparedStatements(ctx);
            MutexEx.Leave(ctx.Mutex);
            return RC.OK;
        }

        public static void BadReturnCode(Parse parse)
        {
            parse.ErrorMsg("authorizer malfunction");
            parse.RC = RC.ERROR;
        }

        public RC ReadColumn(Parse parse, string table, string column, int db)
        {
            Context ctx = parse.Ctx; // Database handle
            string dbName = ctx.DBs[db].Name; // Name of attached database
            RC rc = ctx.Auth(ctx.AuthArg, AUTH.READ, table, column, dbName, parse.AuthContext); // Auth callback return code
            if (rc == RC.DENY)
            {
                if (ctx.DBs.length > 2 || db != 0)
                    parse.ErrorMsg("access to %s.%s.%s is prohibited", dbName, table, column);
                else
                    parse.ErrorMsg("access to %s.%s is prohibited", table, column);
                parse.RC = RC.AUTH;
            }
            else if (rc != RC.IGNORE && rc != RC.OK)
                BadReturnCode(parse);
            return rc;
        }

        public void Read(Parse parse, Expr expr, Schema schema, SrcList tableList)
        {

            Context ctx = parse.Ctx;
            if (ctx.Auth == null) return;
            int db = sqlite3SchemaToIndex(ctx, schema);// The index of the database the expression refers to
            if (db < 0)
                return; // An attempt to read a column out of a subquery or other temporary table.

            Debug.Assert(expr.OP == TK.COLUMN || expr.OP == TK.TRIGGER);
            Table table = null; // The table being read
            if (expr.OP == TK.TRIGGER)
                table = parse.TriggerTab;
            else
            {
                Debug.Assert(tableList != null);
                for (int src = 0; SysEx.ALWAYS(src < tableList.Srcs); src++)
                    if (expr.TableIdx == tableList.Ids[src].Cursor)
                    {
                        table = tableList.Ids[src].pTab;
                        break;
                    }
            }
            int col = expr.ColumnIdx; // Index of column in table
            if (SysEx.NEVER(table == null)) return;

            string colName;     // Name of the column of the table
            if (col >= 0)
            {
                Debug.Assert(col < table.Cols.length);
                colName = table.Cols[col].Name;
            }
            else if (table.PKey >= 0)
            {
                Debug.Assert(table.PKey < table.Cols.length);
                colName = table.Cols[table.PKey].Name;
            }
            else
                colName = "ROWID";
            Debug.Assert(db >= 0 && db < ctx.DBs.length);
            if (ReadColumn(parse, table.Name, colName, db) == RC.IGNORE)
                expr.OP = TK.NULL;
        }

        public RC Check(Parse parse, int code, string arg1, string arg2, string arg3)
        {
            Context ctx = parse.Ctx;
            // Don't do any authorization checks if the database is initialising or if the parser is being invoked from within sqlite3_declare_vtab.
            if (ctx.Init.Busy || INDECLARE_VTABLE)
                return RC.OK;

            if (ctx.Auth == null)
                return RC.OK;
            RC rc = ctx.Auth(ctx.AuthArg, code, arg1, arg2, arg3, parse.AuthContext);
            if (rc == RC.DENY)
            {
                parse.ErrorMsg("not authorized");
                parse.RC = RC.AUTH;
            }
            else if (rc != RC.OK && rc != RC.IGNORE)
            {
                rc = RC.DENY;
                BadReturnCode(parse);
            }
            return rc;
        }

        public void ContextPush(Parse parse, AuthContext actx, string context)
        {
            Debug.Assert(parse != null);
            actx.Parse = parse;
            actx.AuthCtx = parse.AuthContext;
            parse.AuthContext = context;
        }

        public void ContextPop(AuthContext actx)
        {
            if (actx.Parse != null)
            {
                actx.Parse.AuthContext = actx.AuthCtx;
                actx.Parse = null;
            }
        }
    }
}
#endif
#endregion
