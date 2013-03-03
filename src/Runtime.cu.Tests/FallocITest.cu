#include "..\..\System\Runtime\Runtime.h"
#include "..\..\System\Runtime\Runtime.cu.h"
#include "..\..\System\Runtime\Falloc.cu.h"

__device__ static fallocHeap *_heap;
#define TEST(id) \
	__global__ void fallocTest##id(runtimeHeap *r, fallocHeap *f); \
	void fallocTest##id##_host(cudaRuntimeHost &r, void *f) { fallocTest##id<<<1, 1>>>((runtimeHeap *)r.heap, (fallocHeap *)f); cudaRuntimeExecute(r); } \
	__global__ void fallocTest##id(runtimeHeap *r, fallocHeap *f) \
{ \
	setRuntimeHeap(r); \
	_heap = f;

//////////////////////////////////////////////////

// launches cuda kernel
TEST(A) {
	int gtid = blockIdx.x*blockDim.x + threadIdx.x;
	__assert(gtid < 1);
}}

// alloc with get block
TEST(0) {
	void* obj = fallocGetBlock(_heap);
	__assert(obj != nullptr);
	fallocFreeBlock(_heap, obj);
}}

// alloc with getblocks
TEST(1) {
	//void* obj = fallocGetBlocks(_heap, 144 * 2);
	//__assert(obj != nullptr);
	//fallocFreeBlocks(_heap, obj);
	//
	//void* obj2 = fallocGetBlocks(_heap, 144 * 2);
	//__assert(obj2 != nullptr);
	//fallocFreeBlocks(_heap, obj2);
}}

// alloc with context
TEST(2) {
	fallocCtx *ctx = fallocCreateCtx(_heap);
	__assert(ctx != nullptr);
	char *testString = (char*)falloc(ctx, 10);
	__assert(testString != nullptr);
	int *testInteger = falloc<int>(ctx);
	__assert(testInteger != nullptr);
	fallocDisposeCtx(ctx);
}}

// alloc with context as stack
TEST(3) {
	fallocCtx *ctx = fallocCreateCtx(_heap);
	__assert(ctx != nullptr);
	fallocPush<int>(ctx, 1);
	fallocPush<int>(ctx, 2);
	int b = fallocPop<int>(ctx);
	int a = fallocPop<int>(ctx);
	__assert(b == 2 && a == 1);
	fallocDisposeCtx(ctx);
}}
