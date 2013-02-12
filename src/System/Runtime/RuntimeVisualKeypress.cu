#include "cuda_runtime_api.h"
#include "Runtime.h"
#include "Runtime.cu.h"

__global__ static void Keypress(runtimeHeap *heap, unsigned char key)
{
	_heap = heap;
	switch (key)
	{
	case 'a':
		__printf("Test\n");
		break;
	}
}

__host__ void LaunchRuntimeKeypress(cudaRuntimeHost &host, unsigned char key)
{
	if (key == 'b')
	{
		cudaRuntimeExecute(host);
		return;
	}
	//cudaMemcpyToSymbol(_heap, heap, sizeof(runtimeHeap *));
	dim3 heapBlock(1, 1, 1);
	dim3 heapGrid(1, 1, 1);
	Keypress<<<heapGrid, heapBlock>>>((runtimeHeap *)host.heap, key);
}
