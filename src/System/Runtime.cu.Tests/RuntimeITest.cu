#include "..\..\System\Runtime\Runtime.h"
#include "..\..\System\Runtime\Runtime.cu.h"
//#include <cassert>

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
