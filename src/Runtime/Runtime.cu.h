#ifndef __RUNTIME_CU_H__
#define __RUNTIME_CU_H__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 100
#error Atomics only used with > sm_10 architecture
#endif
#include "Runtime.h"

///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code

// Assert
#undef  _assert
#ifndef NDEBUG
extern "C" __device__ void __assert(const char *message, const char *file, unsigned int line);
#define _assert(X) (void)((!!(X))||(__assert(#X, __FILE__, __LINE__), 0))
#define ASSERTONLY(X) X
__device__ __forceinline__ void Coverage(int line) { }
#define ASSERTCOVERAGE(X) if (X) { Coverage(__LINE__); }
#else
#define _assert(X) ((void)0)
#define ASSERTONLY(X)
#define ASSERTCOVERAGE(X)
#endif

#define RUNTIME_UNRESTRICTED -1
#if defined(__EMBED__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 200)
#ifndef __EMBED__
#define __static__ static
#endif
#include "Runtime.cu.native.h"
#else

// Heap
extern "C" __device__ void _runtimeSetHeap(void *heap);
extern "C" __device__ void runtimeRestrict(int threadid, int blockid);

// Abuse of templates to simulate varargs
extern __device__ int _printf(const char *fmt);
template <typename T1> extern __device__ int _printf(const char *fmt, T1 arg1);
template <typename T1, typename T2> extern __device__ int _printf(const char *fmt, T1 arg1, T2 arg2);
template <typename T1, typename T2, typename T3> extern __device__ int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3);
template <typename T1, typename T2, typename T3, typename T4> extern __device__ int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4);
template <typename T1, typename T2, typename T3, typename T4, typename T5> extern __device__ int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> extern __device__ int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> extern __device__ int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> extern __device__ int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> extern __device__ int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> extern __device__ int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA);
// Abuse of templates to simulate varargs
extern __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt);
template <typename T1> extern __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1);
template <typename T1, typename T2> extern __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2);
template <typename T1, typename T2, typename T3> extern __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3);
template <typename T1, typename T2, typename T3, typename T4> extern __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4);
template <typename T1, typename T2, typename T3, typename T4, typename T5> extern __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> extern __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> extern __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> extern __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> extern __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> extern __device__ int __snprintf(char *buf, size_t bufLen, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA);

// Abuse of templates to simulate varargs
extern __device__ void _throw(const char *fmt);
template <typename T1> extern __device__ void _throw(const char *fmt, T1 arg1);
template <typename T1, typename T2> extern __device__ void _throw(const char *fmt, T1 arg1, T2 arg2);
template <typename T1, typename T2, typename T3> extern __device__ void _throw(const char *fmt, T1 arg1, T2 arg2, T3 arg3);
template <typename T1, typename T2, typename T3, typename T4> extern __device__ void _throw(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4);

#endif // __CUDA_ARCH__

///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code

extern __constant__ unsigned char _runtimeUpperToLower[];
extern __constant__ unsigned char _runtimeCtypeMap[256];

#define _toupperA(x) ((x)&~(_runtimeCtypeMap[(unsigned char)(x)]&0x20))
#define _isspaceA(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x01)
#define _isalnumA(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x06)
#define _isalphaA(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x02)
#define _isdigitA(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x04)
#define _isxdigitA(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x08)
#define _tolowerA(x) (_runtimeUpperToLower[(unsigned char)(x)])

// array
template <typename T> struct array_t { size_t length; T *data; __device__ inline void operator=(T *rhs) { data = rhs; } __device__ inline operator T *() { return data; } };
#define __arrayAlloc(t,Ti,length) (Ti*)((int*)malloc(sizeof(Ti)*length+4)+1);*((int*)t&-1)=length
#define __arrayLength(t) t.length
#define __arrayStaticLength(symbol) (sizeof(symbol) / sizeof(symbol[0]))

// strcmp
template <typename T>
__device__ inline int _strcmp(const T *dest, const T *src)
{
	register unsigned char *a, *b;
	a = (unsigned char *)dest;
	b = (unsigned char *)src;
	while (*a != 0 && _runtimeUpperToLower[*a] == _runtimeUpperToLower[*b]) { a++; b++; }
	return _runtimeUpperToLower[*a] - _runtimeUpperToLower[*b];
}

// strncmp
template <typename T>
__device__ inline int _strncmp(const T *dest, const T *src, int n)
{
	register unsigned char *a, *b;
	a = (unsigned char *)dest;
	b = (unsigned char *)src;
	while (n-- > 0 && *a != 0 && _runtimeUpperToLower[*a] == _runtimeUpperToLower[*b]) { a++; b++; }
	return (n < 0 ? 0 : _runtimeUpperToLower[*a] - _runtimeUpperToLower[*b]);
}

// memcpy
template <typename T>
__device__ inline void _memcpy(T *dest, const T *src, size_t length)
{
	register unsigned char *a, *b;
	a = (unsigned char *)dest;
	b = (unsigned char *)src;
	for (size_t i = 0; i < length; ++i, ++a, ++b)
		*a = *b;
}

// memset
template <typename T>
__device__ inline void _memset(T *dest, const char value, size_t length)
{
	register unsigned char *a;
	a = (unsigned char *)dest;
	for (size_t i = 0; i < length; ++i, ++a)
		*a = value;
}

// memcmp
template <typename T, typename Y>
__device__ inline int _memcmp(T *a, Y *b, size_t length)
{
	return 0;
}

// strlen30
__device__ inline int _strlen30(const char *z)
{
	register const char *z2 = z;
	if (z == nullptr) return 0;
	while (*z2) { z2++; }
	return 0x3fffffff & (int)(z2 - z);
}

#endif // __RUNTIME_CU_H__
