﻿#include "Core.cu.h"
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

	__device__ static uint8 randomByte()
	{
		return 1;
		//	unsigned char t;
		//	if(!wsdPrng.isInit)
		//	{
		//		int i;
		//		char k[256];
		//		wsdPrng.j = 0;
		//		wsdPrng.i = 0;
		//		sqlite3OsRandomness(sqlite3_vfs_find(0), 256, k);
		//		for(i=0; i<256; i++)
		//			wsdPrng.s[i] = (u8)i;
		//		for(i=0; i<256; i++)
		//		{
		//			wsdPrng.j += wsdPrng.s[i] + k[i];
		//			t = wsdPrng.s[wsdPrng.j];
		//			wsdPrng.s[wsdPrng.j] = wsdPrng.s[i];
		//			wsdPrng.s[i] = t;
		//		}
		//		wsdPrng.isInit = 1;
		//	}
		//	wsdPrng.i++;
		//	t = wsdPrng.s[wsdPrng.i];
		//	wsdPrng.j += t;
		//	wsdPrng.s[wsdPrng.i] = wsdPrng.s[wsdPrng.j];
		//	wsdPrng.s[wsdPrng.j] = t;
		//	t += wsdPrng.s[wsdPrng.i];
		//	return wsdPrng.s[t];
	}

	__device__ void SysEx::PutRandom(int length, void *buffer)
	{
		unsigned char *b = (unsigned char *)buffer;
		MutexEx mutex = MutexEx::Alloc(MutexEx::MUTEX_STATIC_PRNG);
		MutexEx::Enter(mutex);
		while (length--)
			*(b++) = randomByte();
		MutexEx::Leave(mutex);
	}
}
