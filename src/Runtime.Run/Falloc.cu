#include "..\Runtime.src\Falloc.h"
#include "..\Runtime.src\Falloc.cu.h"
__global__ static void fallocExample(void *f)
{
	fallocGetBlock((fallocHeap *)f);
}

void __fallocExample(cudaFallocHost &f)
{
	fallocExample<<<1, 1>>>(f.heap);
}