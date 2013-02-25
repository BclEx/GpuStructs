#include "..\..\System\Runtime\Runtime.h"
#include "..\..\System\Runtime\Runtime.cu.h"
__global__ static void runtimeExample(void *r)
{
	//setRuntimeHeap(r);
	_printf("test");
}

void __runtimeExample(cudaRuntimeHost &r)
{
	setRuntimeHeap(r.heap);
	runtimeExample<<<1, 1>>>(r.heap);
	cudaRuntimeExecute(r);
}