using System;
namespace Core
{
    public partial class Sql
    {
        public static RC sqlite3_exec(Context ctx, string sql, Func<object, int, string[], string[], bool> callback, object x, ref string errmsg)
        {
            return RC.OK;
        }
    }
}