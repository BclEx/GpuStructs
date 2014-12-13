// random.c
#include "Core.cu.h"

namespace Core
{
	__device__ static struct PrngType
	{
		unsigned char IsInit;
		unsigned char I;
		unsigned char J;
		unsigned char S[256];
	} _prng;

	__device__ static uint8 RandomByte()
	{
		// The "wsdPrng" macro will resolve to the pseudo-random number generator state vector.  If writable static data is unsupported on the target,
		// we have to locate the state vector at run-time.  In the more common case where writable static data is supported, wsdPrng can refer directly
		// to the "sqlite3Prng" state vector declared above.
		PrngType prng = _prng;

		// Initialize the state of the random number generator once, the first time this routine is called.  The seed value does
		// not need to contain a lot of randomness since we are not trying to do secure encryption or anything like that...
		//
		// Nothing in this file or anywhere else in SQLite does any kind of encryption.  The RC4 algorithm is being used as a PRNG (pseudo-random
		// number generator) not as an encryption device.
		unsigned char t;
		if (!prng.IsInit)
		{
			char k[256];
			prng.J = 0;
			prng.I = 0;
			VSystem::FindVfs(nullptr)->Randomness(256, k);
			int i;
			for (i = 0; i < 256; i++)
				prng.S[i] = (uint8)i;
			for (i = 0; i < 256; i++)
			{
				prng.J += prng.S[i] + k[i];
				t = prng.S[prng.J];
				prng.S[prng.J] = prng.S[i];
				prng.S[i] = t;
			}
			prng.IsInit = true;
		}
		// Generate and return single random u8
		prng.I++;
		t = prng.S[prng.I];
		prng.J += t;
		prng.S[prng.I] = prng.S[prng.J];
		prng.S[prng.J] = t;
		t += prng.S[prng.I];
		return prng.S[t];
	}

	__device__ void SysEx::PutRandom(int length, void *buffer) //: sqlite3_randomness
	{
		unsigned char *b = (unsigned char *)buffer;
#if THREADSAFE
		MutexEx mutex = MutexEx::Alloc(MutexEx::MUTEX_STATIC_PRNG);
		MutexEx::Enter(mutex);
#endif
		while (length--)
			*(b++) = RandomByte();
#if THREADSAFE
		MutexEx::Leave(mutex);
#endif
	}

//#if !OMIT_BUILTIN_TEST
//	__device__ static PrngType *_savedPrng = nullptr;
//	__device__ inline static void PrngSaveState() { _memcpy(_savedPrng, &_prng, sizeof(PrngType)); }
//	__device__ inline static void PrngRestoreState() { _memcpy(&_prng, _savedPrng, sizeof(PrngType)); }
//	__device__ inline static void PrngResetState() { _prng.IsInit = false; }
//#endif

}