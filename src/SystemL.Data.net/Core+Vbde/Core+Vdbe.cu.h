#include "../Core+Syntax/Core+Syntax.cu.h"
namespace Core
{
	struct Parse;
	struct VTable;

#pragma region Internal

	struct VdbeCursor;
	struct VdbeFrame;
	struct Mem;
	struct VdbeFunc;
	struct Explain;

#pragma endregion 

	typedef void (*Destructor_t)(void *);
#define DESTRUCTOR_STATIC ((Destructor_t)0)
#define DESTRUCTOR_TRANSIENT ((Destructor_t)-1)
#define DESTRUCTOR_DYNAMIC ((Destructor_t)SysEx::AllocSize)

#pragma region Savepoint

	struct Savepoint
	{
		char *Name;				// Savepoint name (nul-terminated)
		int64 DeferredCons;		// Number of deferred fk violations
		Savepoint *Next;		// Parent savepoint (if any)
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

#pragma region Mem

	__device__ void Mem_ApplyAffinity(Mem *mem, uint8 affinity, TEXTENCODE encode);
	__device__ const void *Mem_Text(Mem *mem, TEXTENCODE encode);
	__device__ int Mem_Bytes(Mem *mem, TEXTENCODE encode);
	__device__ void Mem_SetStr(Mem *mem, int n, const void *z, TEXTENCODE encode, void (*del)(void *));
	__device__ void Mem_Free(Mem *mem);
	__device__ Mem *Mem_New(Context *db);
	__device__ RC Mem_FromExpr(Context *db, Expr *expr, TEXTENCODE encode, AFF affinity, Mem **value);

#pragma endregion

}
#include "Vdbe.cu.h"

