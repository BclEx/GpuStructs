#if __CUDACC__
#include "Runtime.cu.h"
#else
#include "Runtime.cpu.h"
#endif

#pragma region Limits

// The maximum length of a TEXT or BLOB in bytes.   This also limits the size of a row in a table or index.
#ifndef CORE_MAX_LENGTH
#define CORE_MAX_LENGTH 1000000000 // The hard limit is the ability of a 32-bit signed integer to count the size: 2^31-1 or 2147483647.
#endif

#pragma endregion

///////////////////////////////////////////////////////////////////////////////
// RUNTIME
__device__ extern void *(*cudaRuntimeAlloc)(size_t size);
__device__ extern void *(*cudaRuntimeTagAlloc)(void *tag, size_t size);
__device__ extern void *(*cudaRuntimeRealloc)(void *old, size_t newSize);
__device__ extern void *(*cudaRuntimeTagRealloc)(void *tag, void *old, size_t newSize);
__device__ extern void (*cudaRuntimeFree)(void *p);
__device__ extern void (*cudaRuntimeTagFree)(void *tag, void *p);

#ifndef OMIT_AUTOINIT
__device__ extern int cudaRuntimeInitialized;
__device__ extern int (*cudaRuntimeInitialize)();
__device__ bool RuntimeInitialize()
{
	if (!cudaRuntimeInitialize) return true;
	if (cudaRuntimeInitialized == -1) cudaRuntimeInitialized = cudaRuntimeInitialize();
	return (cudaRuntimeInitialized == 1);
}
#endif


//////////////////////
// STRINGBUILDER
#pragma region STRINGBUILDER
class StringBuilder
{
public:
	void *Tag;			// Optional database for lookaside.  Can be NULL
	char *Base;			// A base allocation.  Not from malloc.
	char *Text;			// The string collected so far
	int Index;			// Length of the string so far
	int Size;			// Amount of space allocated in zText
	int MaxSize;		// Maximum allowed string length
	bool MallocFailed;  // Becomes true if any memory allocation fails
	unsigned char AllocType; // 0: none,  1: sqlite3DbMalloc,  2: sqlite3_malloc
	bool Overflowed;    // Becomes true if string size exceeds limits

	__device__ void Printf(bool useExtended, const char *fmt, void *args);
	__device__ void Append(const char *z, int length);
	__device__ char *ToString();
	__device__ void Reset();
	__device__ static void Init(StringBuilder *b, char *text, int capacity, int maxAlloc);
};
#pragma endregion

//////////////////////
// SNPRINTF
#pragma region SNPRINTF
__device__ char *__vsnprintf(const char *buf, size_t bufLen, const char *fmt, va_list args);
#if __CUDACC__
__device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt) { return __vsnprintf(buf, bufLen, fmt, nullptr); }
template <typename T1> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1) { return __vsnprintf(buf, bufLen, fmt, __argSet(arg1)); }
template <typename T1, typename T2> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2) { return __vsnprintf(buf, bufLen, fmt, __argSet(arg1, arg2)); }
template <typename T1, typename T2, typename T3> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { return __vsnprintf(buf, bufLen, fmt, __argSet(arg1, arg2, arg3)); }
template <typename T1, typename T2, typename T3, typename T4> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { return __vsnprintf(buf, bufLen, fmt, __argSet(arg1, arg2, arg3, arg4)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { return __vsnprintf(buf, bufLen, fmt, __argSet(arg1, arg2, arg3, arg4, arg5)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { return __vsnprintf(buf, bufLen, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { return __vsnprintf(buf, bufLen, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { return __vsnprintf(buf, bufLen, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { return __vsnprintf(buf, bufLen, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { return __vsnprintf(buf, bufLen, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA)); }
#else
__device__ char *__snprintf(const char *buf, size_t bufLen, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	char *z = __vsnprintf(buf, bufLen, fmt, args);
	va_end(args);
	return z;
}
#endif

#pragma endregion

//////////////////////
// MPRINTF
#pragma region MPRINTF
__device__ char *_vmtagprintf(void *tag, const char *fmt, va_list args);
__device__ char *_vmprintf(const char *fmt, va_list args);
#if __CUDACC__
__device__ static char *_mprintf(const char *fmt) { return _vmprintf(fmt, nullptr); }
template <typename T1> __device__ static char *_mprintf(const char *fmt, T1 arg1) { return _vmprintf(fmt, __argSet(arg1)); }
template <typename T1, typename T2> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2) { return _vmprintf(fmt, __argSet(arg1, arg2)); }
template <typename T1, typename T2, typename T3> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { return _vmprintf(fmt, __argSet(arg1, arg2, arg3)); }
template <typename T1, typename T2, typename T3, typename T4> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { return _vmprintf(fmt, __argSet(arg1, arg2, arg3, arg4)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { return _vmprintf(fmt, __argSet(arg1, arg2, arg3, arg4, arg5)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { return _vmprintf(fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { return _vmprintf(fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { return _vmprintf(fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { return _vmprintf(fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { return _vmprintf(fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA)); }
//
__device__ static char *_mtagprintf(void *tag, const char *fmt) { return _vmtagprintf(tag, fmt, nullptr); }
template <typename T1> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1) { return _vmtagprintf(tag, fmt, __argSet(arg1)); }
template <typename T1, typename T2> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2) { return _vmtagprintf(tag, fmt, __argSet(arg1, arg2)); }
template <typename T1, typename T2, typename T3> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { return _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3)); }
template <typename T1, typename T2, typename T3, typename T4> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { return _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { return _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { return _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { return _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { return _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { return _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { return _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA)); }
//
__device__ static char *_mtagappendf(void *tag, char *str, const char *fmt) { char *z = _vmtagprintf(tag, fmt, nullptr); cudaRuntimeTagFree(tag, str); return z; }
template <typename T1> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1)); cudaRuntimeTagFree(tag, str); return z; }
template <typename T1, typename T2> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2)); cudaRuntimeTagFree(tag, str); return z; }
template <typename T1, typename T2, typename T3> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3)); cudaRuntimeTagFree(tag, str); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4)); cudaRuntimeTagFree(tag, str); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5)); cudaRuntimeTagFree(tag, str); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6)); cudaRuntimeTagFree(tag, str); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7)); cudaRuntimeTagFree(tag, str); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)); cudaRuntimeTagFree(tag, str); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)); cudaRuntimeTagFree(tag, str); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { char *z = _vmtagprintf(tag, fmt, __argSet(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA)); cudaRuntimeTagFree(tag, str); return z; }
#else
__device__ char *_mprintf(const char *fmt, ...)
{
	if (!RuntimeInitialize()) return nullptr;
	va_list args;
	va_start(args, fmt);
	char *z = _vmprintf(fmt, args);
	va_end(args);
	return z;
}
//
__device__ char *_mtagprintf(void *tag, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	char *z = _vmtagprintf(tag, fmt, args);
	va_end(args);
	return z;
}
//
__device__ char *_mtagappendf(void *tag, char *str, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	char *z = _vmtagprintf(tag, fmt, args);
	va_end(args);
	cudaRuntimeTagFree(tag, str);
	return z;
}
#endif

#pragma endregion