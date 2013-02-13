#ifndef __FALLOC_CU_H__
#define __FALLOC_CU_H__

#if __CUDA_ARCH__ == 100
#error Atomics only used with > sm_10 architecture
#elif __CUDA_ARCH__ < 200
#include "Falloc.cu"
#else

///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE

// External function definitions for device-side code
typedef struct _cuFallocHeap fallocHeap;
__device__ void *fallocGetBlock(fallocHeap *heap);
__device__ void fallocFreeBlock(fallocHeap *heap, void *obj);
#if MULTIBLOCK
__device__ void *fallocGetBlocks(fallocHeap *heap, size_t length, size_t *allocLength = nullptr);
__device__ void fallocFreeBlocks(fallocHeap *heap, void *obj);
#endif

// CONTEXT
typedef struct _cuFallocContext fallocCtx;
__device__ fallocCtx *fallocCreateCtx(fallocHeap *heap);
__device__ void fallocDisposeCtx(fallocCtx *ctx);
__device__ void *falloc(fallocCtx *ctx, unsigned short bytes, bool alloc = true);
__device__ void *fallocRetract(fallocCtx *ctx, unsigned short bytes);
__inline__ __device__ void fallocMark(fallocCtx *ctx, void *&mark, unsigned short &mark2);
__inline__ __device__ bool fallocAtMark(fallocCtx *ctx, void *mark, unsigned short mark2);
template <typename T> __device__ T* falloc(fallocCtx *ctx) { return (T *)falloc(ctx, sizeof(T), true); }
template <typename T> __device__ void fallocPush(fallocCtx *ctx, T t) { *((T *)falloc(ctx, sizeof(T), false)) = t; }
template <typename T> __device__ T fallocPop(fallocCtx *ctx) { return *((T *)fallocRetract(ctx, sizeof(T))); }

#endif

#endif // __FALLOC_CU_H__