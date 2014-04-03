using System;
namespace Core
{
    public enum TYPE : byte
    {
        INTEGER = 1,
        FLOAT = 2,
        BLOB = 4,
        NULL = 5,
        TEXT = 3,
    }

    #region Func

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
        public short Args;				// Number of arguments.  -1 means unlimited
        public TEXTENCODE PrefEncode;	// Preferred text encoding (SQLITE_UTF8, 16LE, 16BE)
        public FUNC Flags;			    // Some combination of SQLITE_FUNC_*
        public object UserData;			// User data parameter
        public FuncDef Next;	// Next function with same name */
        public Action<FuncContext, int, Mem[]> Func; // Regular function
        public Action<FuncContext, int, Mem[]> Step; // Aggregate step
        public Action<FuncContext> Finalize; // Aggregate finalizer
        public string Name;				// SQL name of the function.
        public FuncDef Hash;			// Next with a different name but the same hash
        public FuncDestructor Destructor; // Reference counted destructor function
    }

    public class FuncDestructor
    {
        public int Refs;
        public Action<object> Destroy;
        public object UserData;
    }

    //   FUNCTION(zName, nArg, iArg, bNC, xFunc)
    //     Used to create a scalar function definition of a function zName implemented by C function xFunc that accepts nArg arguments. The
    //     value passed as iArg is cast to a (void*) and made available as the user-data (sqlite3_user_data()) for the function. If 
    //     argument bNC is true, then the SQLITE_FUNC_NEEDCOLL flag is set.
    //
    //   AGGREGATE(zName, nArg, iArg, bNC, xStep, xFinal)
    //     Used to create an aggregate function definition implemented by the C functions xStep and xFinal. The first four parameters
    //     are interpreted in the same way as the first 4 parameters to FUNCTION().
    //
    //   LIKEFUNC(zName, nArg, pArg, flags)
    //     Used to create a scalar function definition of a function zName that accepts nArg arguments and is implemented by a call to C 
    //     function likeFunc. Argument pArg is cast to a (void *) and made available as the function user-data (sqlite3_user_data()). The
    //     FuncDef.flags variable is set to the value passed as the flags parameter.
    //#define FUNCTION(zName, nArg, iArg, bNC, xFunc) {nArg, SQLITE_UTF8, (bNC*FUNC_NEEDCOLL), \
    //    INT_TO_PTR(iArg), 0, xFunc, 0, 0, #zName, 0, 0}
    //#define FUNCTION2(zName, nArg, iArg, bNC, xFunc, extraFlags) {nArg, SQLITE_UTF8, (bNC*FUNC_NEEDCOLL)|extraFlags, \
    //    INT_TO_PTR(iArg), 0, xFunc, 0, 0, #zName, 0, 0}
    //#define STR_FUNCTION(zName, nArg, pArg, bNC, xFunc) {nArg, SQLITE_UTF8, bNC*FUNC_NEEDCOLL, \
    //    pArg, 0, xFunc, 0, 0, #zName, 0, 0}
    //#define LIKEFUNC(zName, nArg, arg, flags) {nArg, SQLITE_UTF8, flags, (void *)arg, 0, likeFunc, 0, 0, #zName, 0, 0}
    //#define AGGREGATE(zName, nArg, arg, nc, xStep, xFinal) {nArg, SQLITE_UTF8, nc*SQLITE_FUNC_NEEDCOLL, \
    //    INT_TO_PTR(arg), 0, 0, xStep,xFinal,#zName,0,0}

    #endregion

    #region Savepoint

    public class Savepoint
    {
        public string Name;			// Savepoint name (nul-terminated)
        public long DeferredCons;	// Number of deferred fk violations
        public Savepoint Next;		// Parent savepoint (if any)
    }

    public enum SAVEPOINT
    {
        SAVEPOINT_BEGIN = 0,
        SAVEPOINT_RELEASE = 1,
        SAVEPOINT_ROLLBACK = 2,
    }

    #endregion

    #region Column Affinity

    public enum AFF
    {
        TEXT = 'a',
        NONE = 'b',
        NUMERIC = 'c',
        INTEGER = 'd',
        REAL = 'e',
        MASK = 0x67, // The SQLITE_AFF_MASK values masks off the significant bits of an affinity value. 
        // Additional bit values that can be ORed with an affinity without changing the affinity.
        BIT_JUMPIFNULL = 0x08, // jumps if either operand is NULL
        BIT_STOREP2 = 0x10,	// Store result in reg[P2] rather than jump
        BIT_NULLEQ = 0x80,  // NULL=NULL
    }

    //#define sqlite3IsNumericAffinity(X) ((X) >= AFF_NUMERIC)

    #endregion
}