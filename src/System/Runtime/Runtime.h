#ifndef __FALLOC_H__
#define __FALLOC_H__

/*
*	This is the header file supporting cuFalloc.cu and defining both the host and device-side interfaces. See that file for some more
*	explanation and sample use code. See also below for details of the host-side interfaces.
*
*  Quick sample code:
*
#include "cuFalloc.cu"

__global__ void TestFalloc(fallocHeap* heap)
{
fallocInit(heap);

// create/free heap
void* obj = fallocGetChunk(heap);
fallocFreeChunk(heap, obj);

// create/free alloc
fallocContext* ctx = fallocCreateCtx(heap);
char* testString = (char*)falloc(ctx, 10);
int* testInteger = falloc<int>(ctx);
fallocDisposeCtx(ctx);
}

int main()
{
cudaFallocHost fallocHost = cudaFallocInit(1);

// test
TestFalloc<<<1, 1>>>(fallocHost.heap);

// free and exit
cudaFallocEnd(fallocHost);
printf("\ndone.\n"); // char c; scanf("%c", &c);
return 0;
}
*/

///////////////////////////////////////////////////////////////////////////////
// HOST SIDE
// External function definitions for host-side code

typedef struct _cuFallocHeap fallocHeap;
typedef struct _cudaFallocHost
{
	void* reserved;
	fallocHeap* heap;
	size_t blocks;
	size_t length;
} cudaFallocHost;

//
//	cudaFallocInit
//
//	Call this to initialise a falloc heap. If the buffer size needs to be changed, call cudaFallocEnd()
//	before re-calling cudaFallocInit().
//
//	The default size for the buffer is 1 megabyte. For CUDA
//	architecture 1.1 and above, the buffer is filled linearly and
//	is completely used.
//
//	Arguments:
//		length - Length, in bytes, of total space to reserve (in device global memory) for output.
//
//	Returns:
//		cudaSuccess if all is well.
//
// default 256b chucks, 1-meg heap
extern "C" cudaFallocHost cudaFallocInit(size_t blockSize = 256, size_t length = 1048576, cudaError_t* error = nullptr, void* reserved = nullptr);

//
//	cudaFallocEnd
//
//	Cleans up all memories allocated by cudaFallocInit() for a heap.
//	Call this at exit, or before calling cudaFallocInit() again.
//
extern "C" void cudaFallocEnd(cudaFallocHost &host);

#endif // __FALLOC_H__