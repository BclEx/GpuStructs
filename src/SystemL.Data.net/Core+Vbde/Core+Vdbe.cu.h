#include "../Core+Syntax/Core+Syntax.cu.h"
namespace Core
{

#pragma region Internal

	struct VdbeCursor;
	struct VdbeFrame;
	struct Mem;
	struct VdbeFunc;
	struct Explain;

	typedef void (*Destructor_t)(void *);
#define DESTRUCTOR_STATIC ((Destructor_t)0)
#define DESTRUCTOR_TRANSIENT ((Destructor_t)-1)
#define DESTRUCTOR_DYNAMIC ((Destructor_t)SysEx::AllocSize)

#pragma endregion 

#pragma region Savepoint

	struct Savepoint
	{
		char *Name;				// Savepoint name (nul-terminated)
		int64 DeferredCons;		// Number of deferred fk violations
		Savepoint *Next;		// Parent savepoint (if any)
	};

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

#ifdef OMIT_TEMPDB
#define OMIT_TEMPDB 1
#else
#define OMIT_TEMPDB 0
#endif