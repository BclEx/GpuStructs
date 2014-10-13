#include "Core.cu.h"
#include <stdarg.h>

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

#ifndef OMIT_BLOB_LITERAL
	__device__ void *SysEx::HexToBlob(void *tag, const char *z, size_t size)
	{
		char *b = (char *)TagAlloc(tag, size / 2 + 1);
		size--;
		if (b)
		{
			int bIdx = 0;
			for (int i = 0; i < size; i += 2, bIdx++)
				b[bIdx] = (_hextobyte(z[i]) << 4) | _hextobyte(z[i + 1]);
			b[bIdx] = 0;
		}
		return b;
	}
#endif
}
