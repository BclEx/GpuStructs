#include "..\Runtime.src\Runtime.h"
#include "..\Runtime.src\Runtime.cu.h"

#define TEST(id) \
	__global__ void runtimeTest##id(runtimeHeap *r); \
	void runtimeTest##id##_host(cudaRuntimeHost &r) { runtimeTest##id<<<1, 1>>>((runtimeHeap *)r.heap); cudaRuntimeExecute(r); } \
	__global__ void runtimeTest##id(runtimeHeap *r) \
{ \
	setRuntimeHeap(r);

//////////////////////////////////////////////////

// printf outputs
TEST(0) {
	__printf("test");
}}
