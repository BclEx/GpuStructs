using System;
namespace Core
{
    #region Table

    public interface ITableModule
    {
        //int iVersion;
        int Create(Context a, object aux, int argc, string[] argv, VTable[] vtabs, string[] b);
        int Connect(Context a, object aux, int argc, string[] argv, VTable[] vtabs, string[] b);
        int BestIndex(VTable vtab, IndexInfo a);
        int Disconnect(VTable vtab);
        int Destroy(VTable vtab);
        int Open(VTable vtab, VTableCursor[] cursors);
        int Close(VTableCursor a);
        int Filter(VTableCursor a, int idxNum, string idxStr, int argc, Mem[] argv);
        int Next(VTableCursor a);
        int Eof(VTableCursor a);
        int Column(VTableCursor a, FuncContext b, int c);
        int Rowid(VTableCursor a, long[] rowid);
        int Update(VTable a, int b, Mem[] c, long[] d);
        int Begin(VTable vtab);
        int Sync(VTable vtab);
        int Commit(VTable vtab);
        int Rollback(VTable vtab);
        int FindFunction(VTable vtab, int argsLength, string name, Action<FuncContext, int, Mem[]> func, object[] args);
        int Rename(VTable vtab, string new_);
        int Savepoint(VTable vtab, int a);
        int Release(VTable vtab, int a);
        int RollbackTo(VTable vtab, int a);
    }

    public enum INDEX_CONSTRAINT : byte
    {
        INDEX_CONSTRAINT_EQ = 2,
        INDEX_CONSTRAINT_GT = 4,
        INDEX_CONSTRAINT_LE = 8,
        INDEX_CONSTRAINT_LT = 16,
        INDEX_CONSTRAINT_GE = 32,
        INDEX_CONSTRAINT_MATCH = 64,
    };

    public class IndexInfo
    {
        public struct IndexInfo_Constraint
        {
            int Column;         // Column on left-hand side of constraint
            INDEX_CONSTRAINT OP;// Constraint operator
            bool Usable;		// True if this constraint is usable
            int TermOffset;     // Used internally - xBestIndex should ignore
        }
        public struct IndexInfo_Orderby
        {
            int Column;			// Column number
            bool Desc;			// True for DESC.  False for ASC.
        }
        public struct IndexInfo_Constraintusage
        {
            int argvIndex;		// if >0, constraint is part of argv to xFilter
            byte Omit; // Do not code a test for this constraint
        }
        // INPUTS
        public IndexInfo_Constraint[] Constraints; // Table of WHERE clause constraints
        public int ConstraintsLength;
        public IndexInfo_Orderby[] OrderBys; // The ORDER BY clause
        public int OrderBysLength;
        // OUTPUTS
        public IndexInfo_Constraintusage[] ConstraintUsage;
        public int ConstraintUsageLength;
        public int IdxNum;                // Number used to identify the index
        public string IdxStr;              // String, possibly obtained from sqlite3_malloc
        public int NeedToFreeIdxStr;      // Free idxStr using sqlite3_free() if true
        public int OrderByConsumed;       // True if output is already ordered
        public double EstimatedCost;      // Estimated cost of using this index
    }

    //int sqlite3_create_module(Context db, string name, ITableModule p, byte[] clientData, Action<object> destroy);

    public struct VTable
    {
        public ITableModule Module;		// The module for this virtual table
        public string ErrMsg;					// Error message from sqlite3_mprintf()
        // Virtual table implementations will typically add additional fields
    }

    public struct VTableCursor
    {
        VTable VTable;		// Virtual table of this cursor
        // Virtual table implementations will typically add additional fields
    }

    //int sqlite3_declare_vtab(Context *, const char *sql);
    //int sqlite3_overload_function(Context *, const char *funcName, int args);

    #endregion

    public partial class Sql
    {
        public static RC sqlite3_exec(Context ctx, string sql, Func<object, int, string[], string[], bool> callback, object x, ref string errmsg)
        {
            return RC.OK;
        }
    }
}