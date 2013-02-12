#ifndef __RUNTIME_CU_H__
#define __RUNTIME_CU_H__

#if __CUDA_ARCH__ == 100
#error Atomics only used with > sm_10 architecture
#elif __CUDA_ARCH__ < 200
#include "Runtime.cu"
#else

///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code

// Abuse of templates to simulate varargs
__device__ int __printf(const char *fmt);
template <typename T1> __device__ int __printf(const char *fmt, T1 arg1);
template <typename T1, typename T2> __device__ int __printf(const char *fmt, T1 arg1, T2 arg2);
template <typename T1, typename T2, typename T3> __device__ int __printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3);
template <typename T1, typename T2, typename T3, typename T4> __device__ int __printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4);
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ int __printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ int __printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ int __printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ int __printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ int __printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10> __device__ int __printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10);

//
//	cuRuntimeRestrict
//
//	Called to restrict output to a given thread/block. Pass the constant RUNTIME_UNRESTRICTED to unrestrict output
//	for thread/block IDs. Note you can therefore allow "all printfs from block 3" or "printfs from thread 2
//	on all blocks", or "printfs only from block 1, thread 5".
//
//	Arguments:
//		threadid - Thread ID to allow printfs from
//		blockid - Block ID to allow printfs from
//
//	NOTE: Restrictions last between invocations of kernels unless cudaRuntimeInit() is called again.
//
#define RUNTIME_UNRESTRICTED -1
__device__ void runtimeRestrict(int threadid, int blockid);

#endif

#endif // __RUNTIME_CU_H__