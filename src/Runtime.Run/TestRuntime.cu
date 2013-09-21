#include "..\Runtime\Runtime.cu.h"

__global__ static void runtimeExample(void *r)
{
	_runtimeSetHeap(r);
	_printf("test\n");
	_printf("test %s\n", "one");
	_printf("test %s %d\n", "one", 3);
}

void __runtimeExample(cudaRuntimeHost &r)
{
	cudaRuntimeSetHeap(r.heap);
	runtimeExample<<<1, 1>>>(r.heap);
	cudaRuntimeExecute(r);
}