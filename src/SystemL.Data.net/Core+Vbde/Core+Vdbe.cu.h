#include "../Core+Table/Core+Table.cu.h"
namespace Core
{
	struct Parse;
	struct VTable;

	enum TYPE : uint8
	{
		TYPE_INTEGER = 1,
		TYPE_FLOAT = 2,
		TYPE_BLOB = 4,
		TYPE_NULL = 5,
		TYPE_TEXT = 3,
	};

#pragma region Internal

	struct VdbeCursor;
	struct VdbeFrame;
	struct Mem;
	struct VdbeFunc;
	struct FuncContext;
	struct Explain;

#pragma endregion 

#pragma region Func

	enum FUNC : uint8
	{
		FUNC_LIKE = 0x01,		// Candidate for the LIKE optimization
		FUNC_CASE = 0x02,		// Case-sensitive LIKE-type function
		FUNC_EPHEM = 0x04,		// Ephemeral.  Delete with VDBE
		FUNC_NEEDCOLL = 0x08,	// sqlite3GetFuncCollSeq() might be called
		FUNC_COUNT = 0x10,		// Built-in count(*) aggregate
		FUNC_COALESCE = 0x20,	// Built-in coalesce() or ifnull() function
		FUNC_LENGTH = 0x40,		// Built-in length() function
		FUNC_TYPEOF = 0x80,		// Built-in typeof() function
	};

	struct FuncDestructor
	{
		int Refs;
		void (*Destroy)(void *);
		void *UserData;
	};

	struct FuncDef
	{
		int16 Args;				// Number of arguments.  -1 means unlimited
		uint8 PrefEnc;			// Preferred text encoding (SQLITE_UTF8, 16LE, 16BE)
		FUNC Flags;				// Some combination of SQLITE_FUNC_*
		void *UserData;			// User data parameter
		FuncDef *Next;	// Next function with same name */
		void (*Func)(FuncContext *, int, Mem**); // Regular function
		void (*Step)(FuncContext *, int, Mem**); // Aggregate step
		void (*Finalize)(FuncContext *); // Aggregate finalizer
		char *Name;				// SQL name of the function.
		FuncDef *Hash;			// Next with a different name but the same hash
		FuncDestructor *Destructor; // Reference counted destructor function
	};

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
#define FUNCTION(zName, nArg, iArg, bNC, xFunc) {nArg, SQLITE_UTF8, (bNC*FUNC_NEEDCOLL), \
	INT_TO_PTR(iArg), 0, xFunc, 0, 0, #zName, 0, 0}
#define FUNCTION2(zName, nArg, iArg, bNC, xFunc, extraFlags) {nArg, SQLITE_UTF8, (bNC*FUNC_NEEDCOLL)|extraFlags, \
	INT_TO_PTR(iArg), 0, xFunc, 0, 0, #zName, 0, 0}
#define STR_FUNCTION(zName, nArg, pArg, bNC, xFunc) {nArg, SQLITE_UTF8, bNC*FUNC_NEEDCOLL, \
	pArg, 0, xFunc, 0, 0, #zName, 0, 0}
#define LIKEFUNC(zName, nArg, arg, flags) {nArg, SQLITE_UTF8, flags, (void *)arg, 0, likeFunc, 0, 0, #zName, 0, 0}
#define AGGREGATE(zName, nArg, arg, nc, xStep, xFinal) {nArg, SQLITE_UTF8, nc*SQLITE_FUNC_NEEDCOLL, \
	INT_TO_PTR(arg), 0, 0, xStep,xFinal,#zName,0,0}

#pragma endregion

#pragma region Limits

#if MAX_ATTACHED > 30
	typedef uint64 yDbMask;
#else
	typedef unsigned int yDbMask;
#endif

#if MAX_VARIABLE_NUMBER <= 32767
	typedef int16 yVars;
#else
	typedef int yVars;
#endif

#pragma endregion 

#pragma region Savepoint

	struct Savepoint
	{
		char *Name;				// Savepoint name (nul-terminated)
		int64 DeferredCons;		// Number of deferred fk violations
		Savepoint *Next;		// Parent savepoint (if any)
	};

	enum SAVEPOINT
	{
		SAVEPOINT_BEGIN = 0,
		SAVEPOINT_RELEASE = 1,
		SAVEPOINT_ROLLBACK = 2,
	};

#pragma endregion

#pragma region Column Affinity

	enum AFF
	{
		AFF_TEXT = 'a',
		AFF_NONE = 'b',
		AFF_NUMERIC = 'c',
		AFF_INTEGER = 'd',
		AFF_REAL = 'e',
		AFF_MASK = 0x67, // The SQLITE_AFF_MASK values masks off the significant bits of an affinity value. 
		// Additional bit values that can be ORed with an affinity without changing the affinity.
		AFF_BIT_JUMPIFNULL = 0x08, // jumps if either operand is NULL
		AFF_BIT_STOREP2 = 0x10,	// Store result in reg[P2] rather than jump
		AFF_BIT_NULLEQ = 0x80,  // NULL=NULL
	};

#define sqlite3IsNumericAffinity(X) ((X) >= AFF_NUMERIC)

#pragma endregion

}
#include "Vdbe.cu.h"

