// os.h
#include <malloc.h>
namespace Core
{
#pragma region Log & Trace

#ifdef _DEBUG
	extern bool OSTrace;
	extern bool IOTrace;
#ifndef __CUDACC__
	__device__ inline static void SysEx_LOG(RC rc, const char *fmt, ...) { }
	__device__ inline static void SysEx_OSTRACE(const char *fmt, ...) { }
	__device__ inline static void SysEx_IOTRACE(const char *fmt, ...) { }
#else
	__device__ inline static void SysEx_LOG(RC rc, const char *fmt) { }
	template <typename T1> __device__ inline static void SysEx_LOG(RC rc, const char *fmt, T1 arg1) { }
	template <typename T1, typename T2> __device__ inline static void SysEx_LOG(RC rc, const char *fmt, T1 arg1, T2 arg2) { }
	template <typename T1, typename T2, typename T3> __device__ inline static void SysEx_LOG(RC rc, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { }
	template <typename T1, typename T2, typename T3, typename T4> __device__ inline static void SysEx_LOG(RC rc, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { }
	__device__ inline static void SysEx_OSTRACE(const char *fmt) { }
	template <typename T1> __device__ inline static void SysEx_OSTRACE(const char *fmt, T1 arg1) { }
	template <typename T1, typename T2> __device__ inline static void SysEx_OSTRACE(const char *fmt, T1 arg1, T2 arg2) { }
	template <typename T1, typename T2, typename T3> __device__ inline static void SysEx_OSTRACE(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { }
	template <typename T1, typename T2, typename T3, typename T4> __device__ inline static void SysEx_OSTRACE(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { }
	__device__ inline static void SysEx_IOTRACE(const char *fmt) { }
	template <typename T1> __device__ inline static void SysEx_IOTRACE(const char *fmt, T1 arg1) { }
	template <typename T1, typename T2> __device__ inline static void SysEx_IOTRACE(const char *fmt, T1 arg1, T2 arg2) { }
	template <typename T1, typename T2, typename T3> __device__ inline static void SysEx_IOTRACE(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { }
	template <typename T1, typename T2, typename T3, typename T4> __device__ inline static void SysEx_IOTRACE(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { }
#endif
#else
#define SysEx_LOG(X, ...) ((void)0)
#define SysEx_OSTRACE(X, ...) ((void)0)
#define SysEx_IOTRACE(X, ...) ((void)0)
#endif

#pragma endregion

#define SysEx_VERSION_NUMBER 3007016

	class TagBase
	{
	public:
		MutexEx Mutex;
		bool MallocFailed; // True if we have seen a malloc failure
		RC ErrCode; // Most recent error code (RC_*)
		int ErrMask; // & result codes with this before returning
	};

	class SysEx
	{
	public:
#pragma region Memory Allocation

		typedef void (*Destructor_t)(void *);
#define DESTRUCTOR_STATIC ((Destructor_t)0)
#define DESTRUCTOR_TRANSIENT ((Destructor_t)-1)
#define DESTRUCTOR_DYNAMIC ((Destructor_t)SysEx::AllocSize)

		enum MEMTYPE : uint8
		{
			MEMTYPE_HEAP = 0x01,         // General heap allocations
			MEMTYPE_LOOKASIDE = 0x02,    // Might have been lookaside memory
			MEMTYPE_SCRATCH = 0x04,      // Scratch allocations
			MEMTYPE_PCACHE = 0x08,       // Page cache allocations
			MEMTYPE_DB = 0x10,           // Uses sqlite3DbMalloc, not sqlite_malloc
		};
		__device__ inline static void BeginBenignAlloc() { }
		__device__ inline static void EndBenignAlloc() { }
		__device__ inline static void *Alloc(size_t size) { return (char *)malloc(size); }
		__device__ inline static void *Alloc(size_t size, bool clear) { char *b = (char *)malloc(size); if (clear) _memset(b, 0, size); return b; }
		__device__ inline static void *TagAlloc(void *tag, size_t size) { return (char *)malloc(size); }
		__device__ inline static void *TagAlloc(void *tag, size_t size, bool clear) { char *b = (char *)malloc(size); if (clear) _memset(b, 0, size); return b; }
		__device__ inline static int AllocSize(void *p)
		{
			_assert(MemdebugHasType(p, MEMTYPE_HEAP));
			_assert(MemdebugNoType(p, MEMTYPE_DB));
			return 0; 
		}
		__device__ inline static int TagAllocSize(void *tag, void *p)
		{
			_assert(MemdebugHasType(p, MEMTYPE_HEAP));
			_assert(MemdebugNoType(p, MEMTYPE_DB));
			return 0; 
		}
		__device__ inline static void Free(void *p) { free(p); }
		__device__ inline static void TagFree(void *tag, void *p) { free(p); }
#ifndef __CUDACC__
		__device__ inline static void *ScratchAlloc(size_t size) { return alloca(size); }
		__device__ inline static void ScratchFree(void *p) { }
#else
		__device__ inline static void *ScratchAlloc(size_t size) { return malloc(size); }
		__device__ inline static void ScratchFree(void *p) { free(p); }
#endif
		__device__ inline static bool HeapNearlyFull() { return false; }
		__device__ inline static void *Realloc(void *old, size_t newSize) { return nullptr; }
		__device__ inline static void *TagRealloc(void *tag, void *old, size_t newSize) { return nullptr; }
		//
#if MEMDEBUG
#else
		__device__ inline static void MemdebugSetType(void *p, MEMTYPE memType) { }
		__device__ inline static bool MemdebugHasType(void *p, MEMTYPE memType) { return true; }
		__device__ inline static bool MemdebugNoType(void *p, MEMTYPE memType) { return true; }
#endif

		__device__ inline static void *TagRellocOrFree(void *tag, void *old, size_t newSize)
		{
			void *p = TagRealloc(tag, old, newSize);
			if (!p) TagFree(tag, old);
			return p;
		}

		__device__ inline static char *TagStrDup(void *tag, const char *z)
		{
			if (z == nullptr) return nullptr;
			size_t n = _strlen30(z) + 1;
			_assert((n & 0x7fffffff) == n);
			char *newZ = (char *)TagAlloc(tag, (int)n);
			if (newZ)
				_memcpy(newZ, z, n);
			return newZ;
		}
		__device__ inline static char *TagStrNDup(void *tag, const char *z, int n)
		{
			if (z == nullptr) return nullptr;
			_assert((n & 0x7fffffff) == n);
			char *newZ = (char *)TagAlloc(tag, n + 1);
			if (newZ)
			{
				_memcpy(newZ, z, n);
				newZ[n] = 0;
			}
			return newZ;
		}

#if __CUDACC__
//__device__ static char *_mprintf(const char *fmt) { char *z = _vmtagprintf(tag, fmt, nullptr); }
//template <typename T1> __device__ static char *_mprintf(const char *fmt, T1 arg1) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1)); }
//template <typename T1, typename T2> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2)); }
//template <typename T1, typename T2, typename T3> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3)); }
//template <typename T1, typename T2, typename T3, typename T4> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4)); }
//template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5)); }
//template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6)); }
//template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7)); }
//template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)); }
//template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)); }
//template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA)); }
#else
		__device__ inline static void TagStrSet(void *tag, char **source, const char *fmt, ...)
		{
			va_list args;
			va_start(args, fmt);
			char *z = _vmtagprintf(tag, fmt, args);
			va_end(args);
			TagFree(tag, *source);
			*source = z;
		}
#endif

#pragma endregion
		__device__ static RC Initialize();
		__device__ static void Shutdown();
		__device__ static void PutRandom(int length, void *buffer);
#ifndef OMIT_BLOB_LITERAL
		__device__ static void *HexToBlob(void *tag, const char *z, size_t size);
#endif
		// UTF
		__device__ static uint32 Utf8Read(const unsigned char **z);
		__device__ static int Utf8CharLen(const char *z, int bytes);
#if defined(TEST) && defined(_DEBUG)
		__device__ static int Utf8To8(unsigned char *z);
#endif
		__device__ static int Utf16ByteLen(const void *z, int chars);
#if defined(TEST)
		__device__ static void UtfSelfTest();
#endif

		//////////////////////
		// MPRINTF
#pragma region MPRINTF

		inline __device__ static char *Mprintf(void *tag, const char *fmt) { __snprintf(nullptr, 0, fmt); return nullptr; }
		template <typename T1> inline __device__ static char *Mprintf(void *tag, const char *fmt, T1 arg1) { __snprintf(nullptr, 0, fmt, arg1); return nullptr; }
		template <typename T1, typename T2> inline __device__ static char *Mprintf(void *tag, const char *fmt, T1 arg1, T2 arg2) { __snprintf(nullptr, 0, fmt, arg1, arg2); return nullptr; }
		template <typename T1, typename T2, typename T3> inline __device__ static char *Mprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { __snprintf(nullptr, 0, fmt, arg1, arg2, arg3); return nullptr; }
		template <typename T1, typename T2, typename T3, typename T4> inline __device__ static char *Mprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { __snprintf(nullptr, 0, fmt, arg1, arg2, arg3, arg4); return nullptr; }
		template <typename T1, typename T2, typename T3, typename T4, typename T5> inline __device__ static char *Mprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { __snprintf(nullptr, 0, fmt, arg1, arg2, arg3, arg4, arg5); return nullptr; }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> inline __device__ static char *Mprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { __snprintf(nullptr, 0, fmt, arg1, arg2, arg3, arg4, arg5, arg6); return nullptr; }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> inline __device__ static char *Mprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { __snprintf(nullptr, 0, fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7); return nullptr; }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> inline __device__ static char *Mprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { __snprintf(nullptr, 0, fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); return nullptr; }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> inline __device__ static char *Mprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { __snprintf(nullptr, 0, fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); return nullptr; }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> inline __device__ static char *Mprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { __snprintf(nullptr, 0, fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); return nullptr; }

#pragma endregion

		__device__ inline static RC ApiExit(TagBase *tag, RC rc)
		{
			// If the ctx handle is not NULL, then we must hold the connection handle mutex here. Otherwise the read (and possible write) of db->mallocFailed 
			// is unsafe, as is the call to sqlite3Error().
			_assert(!tag || MutexEx::Held(tag->Mutex));
			if (tag && (tag->MallocFailed || rc == RC_IOERR_NOMEM))
			{
				Error(tag, RC_NOMEM, nullptr);
				tag->MallocFailed = false;
				rc = RC_NOMEM;
			}
			return (RC)(rc & (tag ? tag->ErrMask : 0xff));
		}

		//////////////////////
		// ERROR
#pragma region ERROR

		inline __device__ static void Error(void *tag, RC errorCode, const char *fmt) { }
		template <typename T1> inline __device__ static void Error(void *tag, RC errorCode, const char *fmt, T1 arg1) { }
		template <typename T1, typename T2> inline __device__ static void Error(void *tag, RC errorCode, const char *fmt, T1 arg1, T2 arg2) { }
		template <typename T1, typename T2, typename T3> inline __device__ static void Error(void *tag, RC errorCode, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { }
		template <typename T1, typename T2, typename T3, typename T4> inline __device__ static void Error(void *tag, RC errorCode, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { }
		template <typename T1, typename T2, typename T3, typename T4, typename T5> inline __device__ static void Error(void *tag, RC errorCode, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { }

#pragma endregion

	};

#define SKIP_UTF8(z) { if ((*(z++)) >= 0xc0) while ((*z & 0xc0) == 0x80) { z++; } }

#define SysEx_ALWAYS(X) (X)
#define SysEx_NEVER(X) (X)

#define SysEx_ROUND8(x)     (((x)+7)&~7)
#define SysEx_ROUNDDOWN8(x) ((x)&~7)
#ifdef BYTEALIGNED4
#define SysEx_HASALIGNMENT8(X) ((((char *)(X) - (char *)0)&3) == 0)
#else
#define SysEx_HASALIGNMENT8(X) ((((char *)(X) - (char *)0)&7) == 0)
#endif

#if _DEBUG
	__device__ inline static RC CORRUPT_BKPT_(int line)
	{
		SysEx_LOG(RC_CORRUPT, "database corruption at line %d of [%.10s]", line, "src");
		return RC_CORRUPT;
	}
	__device__ inline static RC MISUSE_BKPT_(int line)
	{
		SysEx_LOG(RC_MISUSE, "misuse at line %d of [%.10s]", line, "src");
		return RC_MISUSE;
	}
	__device__ inline static RC CANTOPEN_BKPT_(int line)
	{
		SysEx_LOG(RC_CANTOPEN, "cannot open file at line %d of [%.10s]", line, "src");
		return RC_CANTOPEN;
	}
#define SysEx_CORRUPT_BKPT CORRUPT_BKPT_(__LINE__)
#define SysEx_MISUSE_BKPT MISUSE_BKPT_(__LINE__)
#define SysEx_CANTOPEN_BKPT CANTOPEN_BKPT_(__LINE__)
#else
#define SysEx_CORRUPT_BKPT RC_CORRUPT
#define SysEx_MISUSE_BKPT RC_MISUSE
#define SysEx_CANTOPEN_BKPT RC_CANTOPEN
#endif

}
