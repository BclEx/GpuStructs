#include <Runtime.cu.h>

#define TEST(id) \
	__global__ void runtimeTest##id(void *r); \
	void runtimeTest##id##_host(cudaRuntimeHost &r) { runtimeTest##id<<<1, 1>>>(r.heap); cudaRuntimeExecute(r); } \
	__global__ void runtimeTest##id(void *r) \
{ \
	_runtimeSetHeap(r);

//////////////////////////////////////////////////

// printf outputs
TEST(0) {
	_printf("test");
}}
