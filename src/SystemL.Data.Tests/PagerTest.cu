#include <Core\Core.cu.h>

#define TEST(id) \
	__global__ void pagerTest##id(void *r); \
	void pagerTest##id##_host(cudaRuntimeHost &r) { pagerTest##id<<<1, 1>>>(r.heap); cudaRuntimeExecute(r); } \
	__global__ void pagerTest##id(void *r) \
{ \
	_runtimeSetHeap(r); \
	SysEx::Initialize();

//////////////////////////////////////////////////

// printf outputs
TEST(0) {
	_printf("test");
}}
