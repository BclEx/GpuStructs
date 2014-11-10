#ifndef __RUNTIME_CPU_H__
#define __RUNTIME_CPU_H__
#include <stdarg.h> // Needed for the definition of va_list
#include <stdio.h>
#include <process.h>
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

//////////////////////
// ASSERT
#pragma region ASSERT
#undef _assert
#ifndef NDEBUG
#define _assert(X) assert(X)
#define ASSERTONLY(X) X
__device__ inline void Coverage(int line) { }
#define ASSERTCOVERAGE(X) if (X) { Coverage(__LINE__); }
#else
#define _assert(X) ((void)0)
#define ASSERTONLY(X)
#define ASSERTCOVERAGE(X)
#endif
#define _ALWAYS(X) (X)
#define _NEVER(X) (X)

#pragma endregion

///////////////////////////////////////////////////////////////////////////////
// HEAP
#pragma region HEAP

#define CURT_UNRESTRICTED -1

#pragma endregion

// Heap
extern "C" __device__ inline static void _runtimeSetHeap(void *heap) { }
extern "C" inline cudaError_t cudaDeviceHeapSelect(cudaDeviceHeap &host) { return cudaSuccess; }
extern "C" __device__ inline static void runtimeRestrict(int threadid, int blockid) { }

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
// STDARG
#pragma region STDARG
// included in stdargs.h
#pragma endregion

//////////////////////
// THROW
#pragma region THROW

__device__ inline void _throw(const char *fmt) { printf(fmt); exit(0); }
template <typename T1> __device__ inline void _throw(const char *fmt, T1 arg1) { printf(fmt, arg1); exit(0); }
template <typename T1, typename T2> __device__ inline void _throw(const char *fmt, T1 arg1, T2 arg2) { printf(fmt, arg1, arg2); exit(0); }
template <typename T1, typename T2, typename T3> __device__ inline void _throw(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { printf(fmt, arg1, arg2, arg3); exit(0); }
template <typename T1, typename T2, typename T3, typename T4> __device__ inline void _throw(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { printf(fmt, arg1, arg2, arg3, arg4); exit(0); }

#pragma endregion

extern "C" const unsigned char __curtUpperToLower[256];
extern "C" const unsigned char __curtCtypeMap[256];

#endif // __RUNTIME_CPU_H__
