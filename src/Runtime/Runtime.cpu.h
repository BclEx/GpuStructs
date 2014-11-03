#ifndef __RUNTIME_CPU_H__
#define __RUNTIME_CPU_H__
#include <stdarg.h> // Needed for the definition of va_list
#include <stdio.h>
//#ifndef __device__
//#define __device__
//#endif
//#ifndef __constant__
//#define __constant__ const
//#endif
//#ifndef __shared__
//#define __shared__
//#endif
#include <assert.h>
//#include <string.h>
#ifdef HAVE_ISNAN
#include <math.h>
#endif
#pragma warning(disable:4996)
#include "RuntimeHost.h"

///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code

// Assert
#undef _assert
#ifndef NDEBUG
#define _assert(X) assert(X)
//extern "C" _CRTIMP void __cdecl _wassert(_In_z_ const wchar_t * _Message, _In_z_ const wchar_t *_File, _In_ unsigned _Line);
//#define _assert(X) (void)((!!(X))||(_wassert(#X, __FILE__, __LINE__), 0))
#define ASSERTONLY(X) X
__device__ inline void Coverage(int line) { }
#define ASSERTCOVERAGE(X) if (X) { Coverage(__LINE__); }
#else
#define _assert(X) ((void)0)
#define ASSERTONLY(X)
#define ASSERTCOVERAGE(X)
#endif

///////////////////////////////////////////////////////////////////////////////
// HEAP
#pragma region HEAP

#define RUNTIME_UNRESTRICTED -1

#pragma endregion

// Heap
extern "C" __device__ inline void _runtimeSetHeap(void *heap) { }
extern "C" inline cudaError_t cudaRuntimeSetHeap(void *heap) { return cudaSuccess; }
extern "C" __device__ inline void runtimeRestrict(int threadid, int blockid) { }

///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code

//////////////////////
// PRINTF
#pragma region PRINTF

inline __device__ int _printf(const char *fmt) { return printf(fmt); }
template <typename T1> inline __device__ int _printf(const char *fmt, T1 arg1) { return printf(fmt, arg1); }
template <typename T1, typename T2> inline __device__ int _printf(const char *fmt, T1 arg1, T2 arg2) { return printf(fmt, arg1, arg2); }
template <typename T1, typename T2, typename T3> inline __device__ int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { return printf(fmt, arg1, arg2, arg3); }
template <typename T1, typename T2, typename T3, typename T4> inline __device__ int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { return printf(fmt, arg1, arg2, arg3, arg4); }
template <typename T1, typename T2, typename T3, typename T4, typename T5> inline __device__ int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { return printf(fmt, arg1, arg2, arg3, arg4, arg5); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> inline __device__ int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { return printf(fmt, arg1, arg2, arg3, arg4, arg5, arg6); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> inline __device__ int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { return printf(fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> inline __device__ int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { return printf(fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> inline __device__ int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { return printf(fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> inline __device__ int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { return printf(fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); }

#pragma endregion

//////////////////////
// TRANSFER
#pragma region TRANSFER

inline __device__ int _transfer(const char *fmt) { return 0; }
template <typename T1> inline __device__ int _transfer(const char *fmt, T1 arg1) { return 0; }
template <typename T1, typename T2> inline __device__ int _transfer(const char *fmt, T1 arg1, T2 arg2) { return 0; }
template <typename T1, typename T2, typename T3> inline __device__ int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { return 0; }
template <typename T1, typename T2, typename T3, typename T4> inline __device__ int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { return 0; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> inline __device__ int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { return 0; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> inline __device__ int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { return 0; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> inline __device__ int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { return 0; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> inline __device__ int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { return 0; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> inline __device__ int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { return 0; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> inline __device__ int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { return 0; }

#pragma endregion

//////////////////////
// ARGSET
#pragma region ARGSET

//typedef char *va_list;
//#define _INTSIZEOF(n) ((sizeof(n) + sizeof(int) - 1) & ~(sizeof(int) - 1))
//#define va_arg(args, t) (*(t *)((args += _INTSIZEOF(t)) - _INTSIZEOF(t)))

#pragma endregion

//////////////////////
// THROW
#pragma region THROW

__device__ inline void _throw(const char *fmt) { printf(fmt); }
template <typename T1> __device__ inline void _throw(const char *fmt, T1 arg1) { printf(fmt, arg1); }
template <typename T1, typename T2> __device__ inline void _throw(const char *fmt, T1 arg1, T2 arg2) { printf(fmt, arg1, arg2); }
template <typename T1, typename T2, typename T3> __device__ inline void _throw(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { printf(fmt, arg1, arg2, arg3); }
template <typename T1, typename T2, typename T3, typename T4> __device__ inline void _throw(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { printf(fmt, arg1, arg2, arg3, arg4); }

#pragma endregion

extern "C" const unsigned char _runtimeUpperToLower[256];
extern "C" const unsigned char _runtimeCtypeMap[256];

#define __toupper(x) ((x)&~(_runtimeCtypeMap[(unsigned char)(x)]&0x20))
#define _isspace(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x01)
#define _isalnum(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x06)
#define _isalpha(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x02)
#define _isdigit(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x04)
#define _isxdigit(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x08)
#define _isidchar(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x46)
#define __tolower(x) (_runtimeUpperToLower[(unsigned char)(x)])

// array
template <typename T> struct array_t { int length; T *data; inline array_t() { data = nullptr; length = 0; } inline array_t(T *a) { data = a; length = 0; } inline array_t(T *a, int b) { data = a; length = b; } inline void operator=(T *a) { data = a; } inline operator T *() { return data; } };
template <typename TLength, typename T> struct array_t2 { TLength length; T *data; inline array_t2() { data = nullptr; length = 0; } inline array_t2(T *a) { data = a; length = 0; } inline array_t2(T *a, TLength b) { data = a; length = b; } inline void operator=(T *a) { data = a; } inline operator T *() { return data; } };
template <typename TLength, typename T, int size> struct array_t3 { TLength length; T data[size]; inline array_t3() { length = 0; } inline void operator=(T *a) { data = a; } inline operator T *() { return data; } };
#define __arrayStaticLength(symbol) (sizeof(symbol) / sizeof(symbol[0]))

// skiputf8
template <typename T> __device__ inline void _skiputf8(const T *z)
{
	if (*(z++) >= 0xc0) while ((*z & 0xc0) == 0x80) { z++; }
}

// strcmp
template <typename T> __device__ inline int _strcmp(const T *left, const T *right)
{
	register unsigned char *a, *b;
	a = (unsigned char *)left;
	b = (unsigned char *)right;
	while (*a != 0 && _runtimeUpperToLower[*a] == _runtimeUpperToLower[*b]) { a++; b++; }
	return _runtimeUpperToLower[*a] - _runtimeUpperToLower[*b];
}

// strncmp
template <typename T> __device__ inline int _strncmp(const T *left, const T *right, int n)
{
	register unsigned char *a, *b;
	a = (unsigned char *)left;
	b = (unsigned char *)right;
	while (n-- > 0 && *a != 0 && _runtimeUpperToLower[*a] == _runtimeUpperToLower[*b]) { a++; b++; }
	return (n < 0 ? 0 : _runtimeUpperToLower[*a] - _runtimeUpperToLower[*b]);
}

// memcpy
template <typename T> __device__ inline void _memcpy(T *dest, const T *src, size_t length)
{
	//memcpy(dest, src, length);
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

#ifndef OMIT_FLOATING_POINT
__device__ inline int _isNaN(double x)
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

#endif // __RUNTIME_CPU_H__
