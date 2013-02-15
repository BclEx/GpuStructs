#include "cuda_runtime_api.h"
#include "Runtime.h"
#include "Runtime.cu.h"

__global__ static void Keypress(runtimeHeap *heap, unsigned char key)
{
	setRuntimeHeap(heap);
	switch (key)
	{
	case 'a':
		__printf("Test\n");
		break;
	case 'b':
		__printf("Test %d\n", threadIdx.x);
		break;
	case 'c':
		__assert(true, "Error");
		break;
	case 'd':
		__assert(false, "Error");
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
