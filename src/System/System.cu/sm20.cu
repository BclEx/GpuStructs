#include "..\System\System.h"
using namespace Sys;

class FakeNode
{
public:
	int Key;
	int Value;
};

__device__ void TestHeap_TempArray()
{
	TempArray<FakeNode> a(10);
	TempArray<FakeNode> *a2 = new TempArray<FakeNode>(a); delete a2;
	a[0];
	a.Ptr();
	a.Size();
	a.Zero();
}

__device__ void TestHeap_BlockAlloc()
{
	BlockAlloc<FakeNode, 10> a;
	a.Allocated();
	a.Size();
	a.SetFixedBlocks(10);
	a.FreeEmptyBlocks();
	FakeNode *node = a.Alloc();
	a.Free(node);
	a.GetTotalCount();
	a.GetAllocCount();
	a.GetFreeCount();
	a.Shutdown();
}

__device__ void TestHeap_DynamicAlloc()
{
	DynamicAlloc<FakeNode, 10, 2> a;
	a.Init();
	a.SetFixedBlocks(10);
	a.SetLockMemory(false);
	a.FreeEmptyBaseBlocks();
	FakeNode *node = a.Alloc(2);
	node = a.Resize(node, 4);
	a.Free(node);
	a.CheckMemory(node);
	a.GetNumBaseBlocks();
	a.GetBaseBlockMemory();
	a.GetNumUsedBlocks();
	a.GetUsedBlockMemory();
	a.GetNumFreeBlocks();
	a.GetFreeBlockMemory();
	a.GetNumEmptyBaseBlocks();
	a.Shutdown();
}

__global__ void TestHeap()
{
  //NetworkLoadException *ex = new NetworkLoadException();
  TestHeap_TempArray();
  TestHeap_BlockAlloc();
  TestHeap_DynamicAlloc();
}
