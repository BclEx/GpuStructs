#ifndef __RUNTIME_CPU_H__
#define __RUNTIME_CPU_H__
#include <stdio.h>
#ifndef __device__
#define __device__
#endif
#ifndef __constant__
#define __constant__ const
#endif
#ifndef __shared__
#define __shared__
#endif
#include <assert.h>
//#include <string.h>
#pragma warning(disable:4996)
#include "Runtime.h"

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
// SNPRINTF
#pragma region SNPRINTF

inline __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt) { return _snprintf(buf, bufLen, fmt); }
template <typename T1> inline __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1) { return _snprintf(buf, bufLen, fmt, arg1); }
template <typename T1, typename T2> inline __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2) { return _snprintf(buf, bufLen, fmt, arg1, arg2); }
template <typename T1, typename T2, typename T3> inline __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { return _snprintf(buf, bufLen, fmt, arg1, arg2, arg3); }
template <typename T1, typename T2, typename T3, typename T4> inline __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { return _snprintf(buf, bufLen, fmt, arg1, arg2, arg3, arg4); }
template <typename T1, typename T2, typename T3, typename T4, typename T5> inline __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { return _snprintf(buf, bufLen, fmt, arg1, arg2, arg3, arg4, arg5); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> inline __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { return _snprintf(buf, bufLen, fmt, arg1, arg2, arg3, arg4, arg5, arg6); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> inline __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { return _snprintf(buf, bufLen, fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> inline __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { return _snprintf(buf, bufLen, fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> inline __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { return _snprintf(buf, bufLen, fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> inline __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { return _snprintf(buf, bufLen, fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); }

#pragma endregion

//////////////////////
// MPRINTF
#pragma region MPRINTF

inline __device__ char *__mprintf(const char *fmt) { _snprintf(nullptr, 0, fmt); return nullptr; }
template <typename T1> inline __device__ char *__mprintf(const char *fmt, T1 arg1) { _snprintf(nullptr, 0, fmt, arg1); return nullptr; }
template <typename T1, typename T2> inline __device__ char *__mprintf(const char *fmt, T1 arg1, T2 arg2) { _snprintf(nullptr, 0, fmt, arg1, arg2); return nullptr; }
template <typename T1, typename T2, typename T3> inline __device__ char *__mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { _snprintf(nullptr, 0, fmt, arg1, arg2, arg3); return nullptr; }
template <typename T1, typename T2, typename T3, typename T4> inline __device__ char *__mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { _snprintf(nullptr, 0, fmt, arg1, arg2, arg3, arg4); return nullptr; }
template <typename T1, typename T2, typename T3, typename T4, typename T5> inline __device__ char *__mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { _snprintf(nullptr, 0, fmt, arg1, arg2, arg3, arg4, arg5); return nullptr; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> inline __device__ char *__mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { _snprintf(nullptr, 0, fmt, arg1, arg2, arg3, arg4, arg5, arg6); return nullptr; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> inline __device__ char *__mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { _snprintf(nullptr, 0, fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7); return nullptr; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> inline __device__ char *__mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { _snprintf(nullptr, 0, fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); return nullptr; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> inline __device__ char *__mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { _snprintf(nullptr, 0, fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); return nullptr; }
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> inline __device__ char *__mprintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { _snprintf(nullptr, 0, fmt, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); return nullptr; }

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
#define __tolower(x) (_runtimeUpperToLower[(unsigned char)(x)])

// array
template <typename T> struct array_t { size_t length; T *data; inline array_t() { data = nullptr; length = 0; } inline array_t(T *a) { data = a; length = 0; } inline array_t(T *a, size_t b) { data = a; length = b; } inline void operator=(T *a) { data = a; } inline operator T *() { return data; } };
template <typename TLength, typename T> struct array_t2 { TLength length; T *data; inline array_t2() { data = nullptr; length = 0; } inline array_t2(T *a) { data = a; length = 0; } inline array_t2(T *a, TLength b) { data = a; length = b; } inline void operator=(T *a) { data = a; } inline operator T *() { return data; } };
#define __arrayStaticLength(symbol) (sizeof(symbol) / sizeof(symbol[0]))

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

#endif // __RUNTIME_CPU_H__
