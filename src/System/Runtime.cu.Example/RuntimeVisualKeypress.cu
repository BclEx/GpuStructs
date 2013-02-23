#include "cuda_runtime_api.h"
#include "../Runtime/Runtime.h"
#include "../Runtime/Runtime.cu.h"

__global__ static void Keypress(runtimeHeap *heap, unsigned char key)
{
	setRuntimeHeap(heap);
	switch (key)
	{
	case 'a':
		_printf("Test\n");
		break;
	case 'b':
		_printf("Test %d\n", threadIdx.x);
		break;
	case 'c':
		_assert(true, "Error");
		break;
	case 'd':
		_assert(false, "Error");
		break;
	}
}

__host__ void LaunchRuntimeKeypress(cudaRuntimeHost &host, unsigned char key)
{
	if (key == 'z')
	{
		cudaRuntimeExecute(host);
		return;
	}
	//cudaMemcpyToSymbol(_heap, heap, sizeof(runtimeHeap *));
	dim3 heapBlock(1, 1, 1);
	dim3 heapGrid(1, 1, 1);
	Keypress<<<heapGrid, heapBlock>>>((runtimeHeap *)host.heap, key);
}
