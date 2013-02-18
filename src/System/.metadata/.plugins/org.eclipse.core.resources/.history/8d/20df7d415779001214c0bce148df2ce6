#include "cuda_runtime_api.h"
#include "Falloc.cu.h"

__device__ void *_testObj;
__device__ fallocCtx *_testCtx;

__global__ static void Keypress(fallocHeap *heap, unsigned char key)
{
	switch (key)
	{
	case 'a':
		_testObj = fallocGetBlock(heap);
		break;
	case 'b':
		fallocFreeBlock(heap, _testObj);
		break;
	case 'x':
		_testCtx = fallocCreateCtx(heap);
		break;
	case 'y':
		char *testString = (char *)falloc(_testCtx, 10);
		int *testInteger = falloc<int>(_testCtx);
		break;
	case 'z':
		fallocDisposeCtx(_testCtx);
		break;
	}
}

__host__ void LaunchFallocKeypress(fallocHeap *heap, unsigned char key)
{
	dim3 heapBlock(1, 1, 1);
	dim3 heapGrid(1, 1, 1);
	Keypress<<<heapGrid, heapBlock>>>(heap, key);
}
