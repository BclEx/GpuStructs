#include <Core\Core.cu.h>

#define TEST(id) \
	__global__ void textTest##id(void *r); \
	void textTest##id##_host(cudaRuntimeHost &r) { textTest##id<<<1, 1>>>(r.heap); cudaRuntimeExecute(r); } \
	__global__ void textTest##id(void *r) \
{ \
	_runtimeSetHeap(r); \
	SysEx::Initialize();

//////////////////////////////////////////////////

// printf outputs
TEST(0) {
	_printf("test");
}}
