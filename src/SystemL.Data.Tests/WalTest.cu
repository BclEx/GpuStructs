#include <Core\Core.cu.h>

#define TEST(id) \
	__global__ void walTest##id(void *r); \
	void walTest##id##_host(cudaRuntimeHost &r) { walTest##id<<<1, 1>>>(r.heap); cudaRuntimeExecute(r); } \
	__global__ void walTest##id(void *r) \
{ \
	_runtimeSetHeap(r); \
	SysEx::Initialize();

//////////////////////////////////////////////////

// printf outputs
TEST(0) {
	_printf("test");
}}
