#include <stdarg.h>
#include "Core.cu.h"

namespace Core
{
	bool OSTrace;
	bool IOTrace;

	__device__ RC SysEx::Initialize()
	{
		// mutex
		RC rc = RC_OK; // = 3MutexEx::Initialize();
		if (rc) return rc;
		//rc = Alloc::Initialize();
		//rc = PCache::Initialize();
		//if (rc) return rc;
		rc = VSystem::Initialize();
		if (rc) return rc;
		//PCache::PageBufferSetup(_config.Page, _config.SizePage, _config.Pages);
		return RC_OK;
	}

	__device__ void SysEx::Shutdown()
	{
		VSystem::Shutdown();
		//PCache::Shutdown();
		//Alloc::Shutdown();
		//MutexEx::Shutdown();
	}

#ifdef ENABLE_8_3_NAMES
	__device__ void SysEx::FileSuffix3(const char *baseFilename, char *z)
	{
#if ENABLE_8_3_NAMES<2
		if (!sqlite3_uri_boolean(baseFilename, "8_3_names", 0)) return;
#endif
		int sz = _strlen30(z);
		int i;
		for (i = sz-1; i > 0 && z[i] != '/' && z[i] !='.'; i--) { }
		if (z[i] == '.' && _ALWAYS(sz > i+4)) _memmove(&z[i+1], &z[sz-3], 4);
	}
#endif

}
