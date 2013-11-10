#include <Core\Core.cu.h>

#define TEST(id) \
	__global__ void pcacheTest##id(void *r); \
	void pcacheTest##id##_host(cudaRuntimeHost &r) { pcacheTest##id<<<1, 1>>>(r.heap); cudaRuntimeExecute(r); } \
	__global__ void pcacheTest##id(void *r) \
{ \
	_runtimeSetHeap(r); \
	SysEx::Initialize();

//////////////////////////////////////////////////

// printf outputs
TEST(0) {
	_printf("test");
}}