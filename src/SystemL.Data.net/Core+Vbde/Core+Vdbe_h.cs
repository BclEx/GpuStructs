using System;
namespace Core
{
    #region IVdbe

    public enum FUNC : byte
    {
        FUNC_LIKE = 0x01,		// Candidate for the LIKE optimization
        FUNC_CASE = 0x02,		// Case-sensitive LIKE-type function
        FUNC_EPHEM = 0x04,		// Ephemeral.  Delete with VDBE
        FUNC_NEEDCOLL = 0x08,	// sqlite3GetFuncCollSeq() might be called
        FUNC_COUNT = 0x10,		// Built-in count(*) aggregate
        FUNC_COALESCE = 0x20,	// Built-in coalesce() or ifnull() function
        FUNC_LENGTH = 0x40,		// Built-in length() function
        FUNC_TYPEOF = 0x80,		// Built-in typeof() function
    }

    public class FuncDef
    {
        short Args;				// Number of arguments.  -1 means unlimited
        byte PrefEnc;			// Preferred text encoding (SQLITE_UTF8, 16LE, 16BE)
        FUNC Flags;			    // Some combination of SQLITE_FUNC_*
        object UserData;			// User data parameter
        FuncDef Next;	// Next function with same name */
        Action<FuncContext, int, Mem[]> Func; // Regular function
        Action<FuncContext, int, Mem[]> Step; // Aggregate step
        Action<FuncContext> Finalize; // Aggregate finalizer
        string Name;				// SQL name of the function.
        FuncDef Hash;			// Next with a different name but the same hash
        FuncDestructor Destructor; // Reference counted destructor function
    }

    public class FuncDestructor
    {
        int Refs;
        Action<object> Destroy;
        object UserData;
    }

    #endregion
}