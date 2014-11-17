#if __CUDACC__
#include "Runtime.cu.h"
#else
#include <malloc.h>
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
//struct cudaRuntime
//{
//	int Initialized;
//	void *(*Alloc)(size_t size);
//	void *(*TagAlloc)(void *tag, size_t size);
//	void *(*Realloc)(void *old, size_t newSize);
//	void *(*TagRealloc)(void *tag, void *old, size_t newSize);
//	void (*Free)(void *p);
//	void (*TagFree)(void *tag, void *p);
//	void (*TagAllocFailed)(void *tag);
//};
//
//__device__ extern cudaRuntime __curt;
//__device__ extern int (*cudaRuntimeInitialize)(cudaRuntime *curt);
//__device__ bool RuntimeInitialize()
//{
//	if (!cudaRuntimeInitialize) return true;
//	if (__curt.Initialized == -1) __curt.Initialized = cudaRuntimeInitialize(&__curt);
//	return (__curt.Initialized == 1);
//}

//////////////////////
// FUNC
#pragma region FUNC

#undef _toupper
#undef _tolower
#define _toupper(x) ((x)&~(__curtCtypeMap[(unsigned char)(x)]&0x20))
#define _isspace(x) (__curtCtypeMap[(unsigned char)(x)]&0x01)
#define _isalnum(x) (__curtCtypeMap[(unsigned char)(x)]&0x06)
#define _isalpha(x) (__curtCtypeMap[(unsigned char)(x)]&0x02)
#define _isdigit(x) (__curtCtypeMap[(unsigned char)(x)]&0x04)
#define _isxdigit(x) (__curtCtypeMap[(unsigned char)(x)]&0x08)
#define _isidchar(x) (__curtCtypeMap[(unsigned char)(x)]&0x46)
#define _tolower(x) (__curtUpperToLower[(unsigned char)(x)])

// array
template <typename T> struct array_t { int length; T *data; __device__ inline array_t() { data = nullptr; length = 0; } __device__ inline array_t(T *a) { data = a; length = 0; } __device__ inline array_t(T *a, int b) { data = a; length = b; } __device__ inline void operator=(T *a) { data = a; } __device__ inline operator T *() { return data; } };
template <typename TLength, typename T> struct array_t2 { TLength length; T *data; __device__ inline array_t2() { data = nullptr; length = 0; } __device__ inline array_t2(T *a) { data = a; length = 0; } __device__ inline array_t2(T *a, size_t b) { data = a; length = b; } __device__ inline void operator=(T *a) { data = a; } __device__ inline operator T *() { return data; } };
template <typename TLength, typename T, size_t size> struct array_t3 { TLength length; T data[size]; inline array_t3() { length = 0; } __device__ inline void operator=(T *a) { data = a; } __device__ inline operator T *() { return data; } };
#define _lengthof(symbol) (sizeof(symbol) / sizeof(symbol[0]))

// strskiputf8
//#define _strskiputf8(z) { if ((*(z++)) >= 0xc0) while ((*z & 0xc0) == 0x80) { z++; } }
template <typename T> __device__ inline void _strskiputf8(const T *z)
{
	if (*(z++) >= 0xc0) while ((*z & 0xc0) == 0x80) { z++; }
}

// strcmp
template <typename T> __device__ inline int _strcmp(const T *left, const T *right)
{
	register unsigned char *a, *b;
	a = (unsigned char *)left;
	b = (unsigned char *)right;
	while (*a != 0 && __curtUpperToLower[*a] == __curtUpperToLower[*b]) { a++; b++; }
	return __curtUpperToLower[*a] - __curtUpperToLower[*b];
}

// strncmp
#undef _fstrncmp
template <typename T> __device__ inline int _strncmp(const T *left, const T *right, int n)
{
	register unsigned char *a, *b;
	a = (unsigned char *)left;
	b = (unsigned char *)right;
	while (n-- > 0 && *a != 0 && __curtUpperToLower[*a] == __curtUpperToLower[*b]) { a++; b++; }
	return (n < 0 ? 0 : __curtUpperToLower[*a] - __curtUpperToLower[*b]);
}
#define _fstrncmp(x, y) (_tolower(*(unsigned char *)(x))==_tolower(*(unsigned char *)(y))&&!_strcmp((x)+1,(y)+1))

// memcpy
template <typename T> __device__ inline void _memcpy(T *dest, const T *src, size_t length)
{
	register unsigned char *a, *b;
	a = (unsigned char *)dest;
	b = (unsigned char *)src;
	for (size_t i = 0; i < length; ++i, ++a, ++b)
		*a = *b;
}

// memset
template <typename T> __device__ inline void _memset(T *dest, const char value, size_t length)
{
	register unsigned char *a;
	a = (unsigned char *)dest;
	for (size_t i = 0; i < length; ++i, ++a)
		*a = value;
}

// memcmp
template <typename T, typename Y> __device__ inline int _memcmp(T *left, Y *right, size_t length)
{
	register unsigned char *a, *b;
	a = (unsigned char *)left;
	b = (unsigned char *)right;
	while (*a != 0 && *a == *b) { a++; b++; }
	return *a - *b;
}

// strlen30
__device__ inline int _strlen30(const char *z)
{
	register const char *z2 = z;
	if (z == nullptr) return 0;
	while (*z2) { z2++; }
	return 0x3fffffff & (int)(z2 - z);
}

// hextobyte
__device__ inline unsigned char _hextobyte(char h)
{
	_assert((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F'));
	return (unsigned char)((h + 9*(1&(h>>6))) & 0xf);
}
#ifndef OMIT_BLOB_LITERAL
__device__ void *_taghextoblob(void *tag, const char *z, size_t size);
#endif

#ifndef OMIT_FLOATING_POINT
__device__ inline bool _isnan(double x)
{
#if !defined(HAVE_ISNAN)
	// Systems that support the isnan() library function should probably make use of it by compiling with -DHAVE_ISNAN.  But we have
	// found that many systems do not have a working isnan() function so this implementation is provided as an alternative.
	//
	// This NaN test sometimes fails if compiled on GCC with -ffast-math. On the other hand, the use of -ffast-math comes with the following
	// warning:
	//
	//      This option [-ffast-math] should never be turned on by any -O option since it can result in incorrect output for programs
	//      which depend on an exact implementation of IEEE or ISO rules/specifications for math functions.
	//
	// Under MSVC, this NaN test may fail if compiled with a floating-point precision mode other than /fp:precise.  From the MSDN 
	// documentation:
	//
	//      The compiler [with /fp:precise] will properly handle comparisons involving NaN. For example, x != x evaluates to true if x is NaN 
#ifdef __FAST_MATH__
#error Runtime will not work correctly with the -ffast-math option of GCC.
#endif
	volatile double y = x;
	volatile double z = y;
	return (y != z);
#else
	return isnan(x);
#endif
}
#endif

#pragma endregion

//////////////////////
// MEMORY ALLOCATION
#pragma region MEMORY ALLOCATION

enum MEMTYPE : unsigned char
{
	MEMTYPE_HEAP = 0x01,         // General heap allocations
	MEMTYPE_LOOKASIDE = 0x02,    // Might have been lookaside memory
	MEMTYPE_SCRATCH = 0x04,      // Scratch allocations
	MEMTYPE_PCACHE = 0x08,       // Page cache allocations
	MEMTYPE_DB = 0x10,           // Uses sqlite3DbMalloc, not sqlite_malloc
};
#if MEMDEBUG
#else
__device__ inline static void _memdbg_settype(void *p, MEMTYPE memType) { }
__device__ inline static bool _memdbg_hastype(void *p, MEMTYPE memType) { return true; }
__device__ inline static bool _memdbg_nottype(void *p, MEMTYPE memType) { return true; }
#endif

__device__ inline static void _benignalloc_begin() { }
__device__ inline static void _benignalloc_end() { }
__device__ inline static void *_alloc(size_t size) { return (char *)malloc(size); }
__device__ inline static void *_alloc2(size_t size, bool clear) { char *b = (char *)malloc(size); if (clear) _memset(b, 0, size); return b; }
__device__ inline static void *_tagalloc(void *tag, size_t size) { return (char *)malloc(size); }
__device__ inline static void *_tagalloc2(void *tag, size_t size, bool clear) { char *b = (char *)malloc(size); if (clear) _memset(b, 0, size); return b; }
__device__ inline static int _allocsize(void *p)
{
	_assert(_memdbg_hastype(p, MEMTYPE_HEAP));
	_assert(_memdbg_nottype(p, MEMTYPE_DB));
	return 0; 
}
__device__ inline static int _tagallocsize(void *tag, void *p)
{
	_assert(_memdbg_hastype(p, MEMTYPE_HEAP));
	_assert(_memdbg_nottype(p, MEMTYPE_DB));
	return 0; 
}
__device__ inline static void _free(void *p) { free(p); }
__device__ inline static void _tagfree(void *tag, void *p) { free(p); }
#if __CUDACC__
__device__ inline static void *_stackalloc(size_t size) { return malloc(size); }
__device__ inline static void _stackfree(void *p) { free(p); }
#else
__device__ inline static void *_stackalloc(size_t size) { return alloca(size); }
__device__ inline static void _stackfree(void *p) { }
#endif
__device__ inline static void *_realloc(void *old, size_t newSize) { return nullptr; }
__device__ inline static void *_tagrealloc(void *tag, void *old, size_t newSize) { return nullptr; }
__device__ inline static bool _heapnearlyfull() { return false; }

__device__ inline static void *_tagrelloc_or_free(void *tag, void *old, size_t newSize)
{
	void *p = _tagrealloc(tag, old, newSize);
	if (!p) _tagfree(tag, old);
	return p;
}

__device__ inline static char *_tagstrdup(void *tag, const char *z)
{
	if (z == nullptr) return nullptr;
	size_t n = _strlen30(z) + 1;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_tagalloc(tag, (int)n);
	if (newZ)
		_memcpy(newZ, z, n);
	return newZ;
}
__device__ inline static char *_tagstrndup(void *tag, const char *z, int n)
{
	if (z == nullptr) return nullptr;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_tagalloc(tag, n + 1);
	if (newZ)
	{
		_memcpy(newZ, z, n);
		newZ[n] = 0;
	}
	return newZ;
}

#pragma endregion

//////////////////////
// PRINT
#pragma region PRINT
class TextBuilder
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
	__device__ static void Init(TextBuilder *b, char *text, int capacity, int maxAlloc);
};
#pragma endregion

//////////////////////
// SNPRINTF
#pragma region SNPRINTF
__device__ char *__vsnprintf(const char *buf, size_t bufLen, const char *fmt, va_list args);
#if __CUDACC__
__device__ inline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt) { va_list args; va_start(args, nullptr); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1> __device__ inline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1) { va_list args; va_start(args, arg1); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2> __device__ inline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2) { va_list args; va_start(args, arg1, arg2); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ inline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list args; va_start(args, arg1, arg2, arg3); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ inline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list args; va_start(args, arg1, arg2, arg3, arg4); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ inline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ inline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ inline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ inline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ inline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ inline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = __vsnprintf(buf, bufLen, fmt, args); va_end(args); return z; }
#else
__device__ inline static char *__snprintf(const char *buf, size_t bufLen, const char *fmt, ...)
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
__device__ inline char *_mprintf(const char *fmt) { va_list args; va_start(args, nullptr); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1> __device__ inline char *_mprintf(const char *fmt, T1 arg1) { va_list args; va_start(args, arg1); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2> __device__ inline char *_mprintf(const char *fmt, T1 arg1, T2 arg2) { va_list args; va_start(args, arg1, arg2); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ inline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list args; va_start(args, arg1, arg2, arg3); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ inline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list args; va_start(args, arg1, arg2, arg3, arg4); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ inline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ inline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ inline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ inline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ inline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmprintf(fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ inline char *_mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmprintf(fmt, args); va_end(args); return z; }
//
__device__ inline char *_mtagprintf(void *tag, const char *fmt) { va_list args; va_start(args, nullptr); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1> __device__ inline char *_mtagprintf(void *tag, const char *fmt, T1 arg1) { va_list args; va_start(args, arg1); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2> __device__ inline char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2) { va_list args; va_start(args, arg1, arg2); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3> __device__ inline char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list args; va_start(args, arg1, arg2, arg3); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ inline char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list args; va_start(args, arg1, arg2, arg3, arg4); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ inline char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ inline char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ inline char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ inline char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ inline char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ inline char *_mtagprintf(void *tag, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmtagprintf(tag, fmt, args); va_end(args); return z; }
//
__device__ inline static char *_mtagappendf(void *tag, char *src, const char *fmt) { va_list args; va_start(args, nullptr); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1> __device__ inline static char *_mtagappendf(void *tag, char *src, const char *fmt, T1 arg1) { va_list args; va_start(args, arg1); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2> __device__ inline static char *_mtagappendf(void *tag, char *src, const char *fmt, T1 arg1, T2 arg2) { va_list args; va_start(args, arg1, arg2); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3> __device__ inline static char *_mtagappendf(void *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list args; va_start(args, arg1, arg2, arg3); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ inline static char *_mtagappendf(void *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list args; va_start(args, arg1, arg2, arg3, arg4); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ inline static char *_mtagappendf(void *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ inline static char *_mtagappendf(void *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ inline static char *_mtagappendf(void *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ inline static char *_mtagappendf(void *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ inline static char *_mtagappendf(void *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); return z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ inline static char *_mtagappendf(void *tag, char *src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); return z; }

__device__ inline static void _mtagassignf(void *tag, char **src, const char *fmt) { va_list args; va_start(args, nullptr); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); *src = z; }
template <typename T1> __device__ inline static void _mtagassignf(void *tag, char **src, const char *fmt, T1 arg1) { va_list args; va_start(args, arg1); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); *src = z; }
template <typename T1, typename T2> __device__ inline static void _mtagassignf(void *tag, char **src, const char *fmt, T1 arg1, T2 arg2) { va_list args; va_start(args, arg1, arg2); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); *src = z; }
template <typename T1, typename T2, typename T3> __device__ inline static void _mtagassignf(void *tag, char **src, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list args; va_start(args, arg1, arg2, arg3); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4> __device__ inline static void _mtagassignf(void *tag, char **src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list args; va_start(args, arg1, arg2, arg3, arg4); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ inline static void _mtagassignf(void *tag, char **src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ inline static void _mtagassignf(void *tag, char **src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ inline static void _mtagassignf(void *tag, char **src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ inline static void _mtagassignf(void *tag, char **src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ inline static void _mtagassignf(void *tag, char **src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); *src = z; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ inline static void _mtagassignf(void *tag, char **src, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); char *z = _vmtagprintf(tag, fmt, args); va_end(args); _tagfree(tag, src); *src = z; }
#else
__device__ inline static char *_mprintf(const char *fmt, ...)
{
	//if (!RuntimeInitialize()) return nullptr;
	va_list args;
	va_start(args, fmt);
	char *z = _vmprintf(fmt, args);
	va_end(args);
	return z;
}
//
__device__ inline static char *_mtagprintf(void *tag, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	char *z = _vmtagprintf(tag, fmt, args);
	va_end(args);
	return z;
}
//
__device__ inline static char *_mtagappendf(void *tag, char *src, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	char *z = _vmtagprintf(tag, fmt, args);
	va_end(args);
	_tagfree(tag, src);
	return z;
}
__device__ inline static void _mtagassignf(void *tag, char **src, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	char *z = _vmtagprintf(tag, fmt, args);
	va_end(args);
	_tagfree(tag, *src);
	*src = z;
}
#endif
#pragma endregion