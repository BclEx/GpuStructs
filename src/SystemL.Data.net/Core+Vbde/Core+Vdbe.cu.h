#include "../Core+Btree/Core+Btree.cu.h"

typedef struct Mem Mem;
typedef struct FuncContext FuncContext;

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

typedef struct FuncDef
{
	int16 Args;				// Number of arguments.  -1 means unlimited
	uint8 PrefEnc;			// Preferred text encoding (SQLITE_UTF8, 16LE, 16BE)
	FUNC Flags;				// Some combination of SQLITE_FUNC_*
	void *UserData;			// User data parameter
	struct FuncDef *Next;	// Next function with same name */
	void (*Func)(FuncContext *, int, Mem**); // Regular function
	void (*Step)(FuncContext *, int, Mem**); // Aggregate step
	void (*Finalize)(FuncContext *); // Aggregate finalizer
	char *Name;				// SQL name of the function.
	FuncDef *Hash;			// Next with a different name but the same hash
	FuncDestructor *Destructor; // Reference counted destructor function
} FuncDef;

struct FuncDestructor
{
	int Refs;
	void (*Destroy)(void *);
	void *UserData;
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

#include "Vdbe.cu.h"
