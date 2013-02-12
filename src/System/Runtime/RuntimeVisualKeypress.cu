#include "cuda_runtime_api.h"
#include "Runtime.h"
#include "Runtime.cu.h"

__global__ void Keypress(runtimeHeap *heap, unsigned char key)
{
	_heap = heap;
	switch (key)
	{
	case 'a':
		__printf("Test");
		break;
	}
}

__host__ void LaunchKeypress(cudaRuntimeHost &host, unsigned char key)
{
	//cudaMemcpyToSymbol(_heap, heap, sizeof(runtimeHeap *));
	dim3 heapBlock(1, 1, 1);
	dim3 heapGrid(1, 1, 1);
	Keypress<<<heapGrid, heapBlock>>>((runtimeHeap *)host.heap, key);
	cudaRuntimeExecute(host);
}
