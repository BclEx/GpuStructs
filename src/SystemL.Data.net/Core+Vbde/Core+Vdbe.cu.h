#include "../Core+Syntax/Core+Syntax.cu.h"
namespace Core
{

#pragma region Internal

	struct VdbeCursor;
	struct VdbeFrame;
	struct VdbeFunc;
	struct Explain;

#pragma endregion 

#pragma region Savepoint

	struct Savepoint
	{
		char *Name;				// Savepoint name (nul-terminated)
		int64 DeferredCons;		// Number of deferred fk violations
		Savepoint *Next;		// Parent savepoint (if any)
	};

#pragma endregion

}
#include "Vdbe.cu.h"
