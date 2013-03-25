#include "..\Runtime.src\Runtime.h"
#include "..\Runtime.src\Runtime.cu.h"
__global__ static void runtimeExample(void *r)
{
	runtimeSetHeap(r);
	_printf("test");
}

void __runtimeExample(cudaRuntimeHost &r)
{
	cudaRuntimeSetHeap(r.heap);
	runtimeExample<<<1, 1>>>(r.heap);
	cudaRuntimeExecute(r);
}