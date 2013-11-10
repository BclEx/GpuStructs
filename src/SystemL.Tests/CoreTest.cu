#include <Core\Core.cu.h>

#define TEST(id) \
	__global__ void coreTest##id(void *r); \
	void coreTest##id##_host(cudaRuntimeHost &r) { coreTest##id<<<1, 1>>>(r.heap); cudaRuntimeExecute(r); } \
	__global__ void coreTest##id(void *r) \
{ \
	_runtimeSetHeap(r); \
	SysEx::Initialize();

//////////////////////////////////////////////////
