#include "..\Runtime\Runtime.cu.h"

__global__ static void runtimeExample(void *r)
{
	_runtimeSetHeap(r);
	_printf("test");
}

void __runtimeExample(cudaRuntimeHost &r)
{
	cudaRuntimeSetHeap(r.heap);
	runtimeExample<<<1, 1>>>(r.heap);
	cudaRuntimeExecute(r);
}