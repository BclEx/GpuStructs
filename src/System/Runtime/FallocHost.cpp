#if __CUDA_ARCH__ == 100
#error Atomics only used with > sm_10 architecture
#endif
#include "cuda_runtime_api.h"
#include "Falloc.h"
#include <string.h>

typedef struct __align__(8) _cuFallocBlock
{
	unsigned short magic;
	unsigned short count;
	struct _cuFallocBlock *next;
	void *reserved;
} fallocBlock;

typedef struct _cuFallocBlockRef
{
	fallocBlock *b;
} fallocBlockRef;

typedef struct __align__(8) _cuFallocHeap
{
	void *reserved;
	size_t blockSize;
	size_t blocks;
	size_t blockRefSize; // Size of circular buffer (set up by host)
	fallocBlockRef *blockRefs; // Start of circular buffer (set up by host)
	volatile fallocBlock **freeBlockPtr; // Current atomically-incremented non-wrapped offset
	volatile fallocBlock **retnBlockPtr; // Current atomically-incremented non-wrapped offset
} fallocHeap;


///////////////////////////////////////////////////////////////////////////////
// HOST SIDE

//
//  cudaFallocInit
//
//  Takes a buffer length to allocate, creates the memory on the device and
//  returns a pointer to it for when a kernel is called. It's up to the caller
//  to free it.
//
extern "C" cudaFallocHost cudaFallocInit(size_t blockSize, size_t length, cudaError_t *error, void *reserved)
{
	cudaError_t localError; if (error == nullptr) error = &localError;
	cudaFallocHost host; memset(&host, 0, sizeof(cudaFallocHost));
	// fix up blockSize to include fallocBlock
	blockSize = (blockSize + sizeof(fallocBlock) + 15) & ~15;
	// fix up length to be a multiple of blockSize
	if (!length || length % blockSize)
		length += blockSize - (length % blockSize);
	size_t blocks = (size_t)(length / blockSize);
	if (!blocks)
		return host;
	// fix up length to include fallocHeap + freeblocks
	unsigned int blockRefSize = blocks * sizeof(fallocBlockRef);
	length = (length + blockRefSize + sizeof(fallocHeap) + 15) & ~15;
	// allocate a heap on the device and zero it
	fallocHeap *heap;
	if ((*error = cudaMalloc((void **)&heap, length)) != cudaSuccess || (*error = cudaMemset(heap, 0, length)) != cudaSuccess)
		return host;
	// transfer to heap
	fallocHeap hostHeap;
	hostHeap.reserved = reserved;
	hostHeap.blockSize = blockSize;
	hostHeap.blocks = blocks;
	hostHeap.blockRefSize = blockRefSize;
	hostHeap.blockRefs = (fallocBlockRef *)heap + sizeof(fallocHeap);
	hostHeap.freeBlockPtr = hostHeap.retnBlockPtr = (volatile fallocBlock **)hostHeap.blockRefs;
	if ((*error = cudaMemcpy(heap, &hostHeap, sizeof(fallocHeap), cudaMemcpyHostToDevice)) != cudaSuccess)
		return host;
	// return the heap
	host.reserved = reserved;
	host.heap = heap;
	host.blocks = blocks;
	host.length = length;
	return host;
}

//
//  cudaFallocEnd
//
//  Frees up the memory which we allocated
//
extern "C" void cudaFallocEnd(cudaFallocHost &host) {
	if (!host.heap)
		return;
	cudaFree(host.heap); host.heap = nullptr;
}
