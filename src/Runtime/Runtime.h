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
struct cudaRuntime
{
	int Initialized;
	void *(*Alloc)(size_t size);
	void *(*TagAlloc)(void *tag, size_t size);
	void *(*Realloc)(void *old, size_t newSize);
	void *(*TagRealloc)(void *tag, void *old, size_t newSize);
	void (*Free)(void *p);
	void (*TagFree)(void *tag, void *p);
	void (*TagAllocFailed)(void *tag);
};

__device__ extern cudaRuntime __curt;
__device__ extern int (*cudaRuntimeInitialize)(cudaRuntime *curt);
__device__ bool RuntimeInitialize()
{
	if (!cudaRuntimeInitialize) return true;
	if (__curt.Initialized == -1) __curt.Initialized = cudaRuntimeInitialize(&__curt);
	return (__curt.Initialized == 1);
}

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
__device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt) { va_list args; va_start(args, nullptr); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1) { va_list args; va_start(args, arg1); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2) { va_list args; va_start(args, arg1, arg2); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list args; va_start(args, arg1, arg2, arg3); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list args; va_start(args, arg1, arg2, arg3, arg4); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
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
__device__ static char *_mprintf(const char *fmt) { va_list args; va_start(args, nullptr); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1> __device__ static char *_mprintf(const char *fmt, T1 arg1) { va_list args; va_start(args, arg1); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2) { va_list args; va_start(args, arg1, arg2); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list args; va_start(args, arg1, arg2, arg3); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list args; va_start(args, arg1, arg2, arg3, arg4); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ static char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmprintf(fmt, args); va_end(args); return z; }
//
__device__ static char *_mtagprintf(void *tag, const char *fmt) { va_list args; va_start(args, nullptr); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1) { va_list args; va_start(args, arg1); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2) { va_list args; va_start(args, arg1, arg2); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list args; va_start(args, arg1, arg2, arg3); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list args; va_start(args, arg1, arg2, arg3, arg4); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ static char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
//
__device__ static char *_mtagappendf(void *tag, char *str, const char *fmt) { va_list args; va_start(args, nullptr); char *z = _vmtagprintf(tag, fmt, args); va_end(args); __curt.TagFree(tag, str); return z; }
template <typename T1> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1) { va_list args; va_start(args, arg1); char *z = _vmtagprintf(tag, fmt, args); va_end(args); __curt.TagFree(tag, str); return z; }
template <typename T1, typename T2> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2) { va_list args; va_start(args, arg1, arg2); char *z = _vmtagprintf(tag, fmt, args); va_end(args); __curt.TagFree(tag, str); return z; }
template <typename T1, typename T2, typename T3> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list args; va_start(args, arg1, arg2, arg3); char *z = _vmtagprintf(tag, fmt, args); va_end(args); __curt.TagFree(tag, str); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list args; va_start(args, arg1, arg2, arg3, arg4); char *z = _vmtagprintf(tag, fmt, args); va_end(args); __curt.TagFree(tag, str); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmtagprintf(tag, fmt, args); va_end(args); __curt.TagFree(tag, str); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmtagprintf(tag, fmt, args); va_end(args); __curt.TagFree(tag, str); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmtagprintf(tag, fmt, args); va_end(args); __curt.TagFree(tag, str); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmtagprintf(tag, fmt, args); va_end(args); __curt.TagFree(tag, str); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmtagprintf(tag, fmt, args); va_end(args); __curt.TagFree(tag, str); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ static char *_mtagappendf(void *tag, char *str, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmtagprintf(tag, fmt, args); va_end(args); __curt.TagFree(tag, str); return z; }
#else
__device__ char *_mprintf(const char *fmt, ...)
{
	//if (!RuntimeInitialize()) return nullptr;
	va_list args;
	va_start(args, fmt);
	char *z = _vmprintf(fmt, args);
	va_end(args);
	return z;
}
//
__device__ char *_mtagprintf(void *tag, const char *fmt, ...)
{
	//if (!RuntimeInitialize()) return nullptr;
	va_list args;
	va_start(args, fmt);
	char *z = _vmtagprintf(tag, fmt, args);
	va_end(args);
	return z;
}
//
__device__ char *_mtagappendf(void *tag, char *str, const char *fmt, ...)
{
	//if (!RuntimeInitialize()) return nullptr;
	va_list args;
	va_start(args, fmt);
	char *z = _vmtagprintf(tag, fmt, args);
	va_end(args);
	__curt.TagFree(tag, str);
	return z;
}
#endif

#pragma endregion