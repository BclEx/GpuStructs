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
}
