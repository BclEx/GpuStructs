#include "..\..\System\System.h"
using namespace Sys::Collections;

__device__ void TestList_ListArray()
{
	void* a = ListArrayNew<byte, TAG_IDLIB_LIST>(2, true);
	ListArrayResize<byte, TAG_IDLIB_LIST>(a, 2, 4, true);
	ListArrayDelete<byte>(a, 4);
}

__device__ void TestList_ListNewElement()
{
	void* a = ListNewElement<byte>();
}

__device__ void TestList_List()
{
	List<byte> a(16);
	List<byte> *a2 = new List<byte>(a); delete a2;
}

__device__ void TestList_Clear()
{
	List<byte*> a(2);
	a.Clear();
	a.DeleteContents(true);
	a.DeleteContents(false);
}

__device__ void TestList_Properties()
{
	List<byte> a(2);
	List<byte> b(2);
	a.Allocated();
	a.Size();
	a.MemoryUsed();
	a.Num();
	a.NumAllocated();
	a.SetNum(1);
	a.SetNum(4);
	a.SetGranularity(3);
	a.GetGranularity();
	a.Ptr();
	a = b;
	a[0];
}

__device__ void TestList_Resize()
{
	List<byte> a(2);
	a.Condense();
	a.Resize(3);
	a.Resize(4, 2);
	a.AssureSize(4);
	a.AssureSize(4, 4);
	//a.AssureSize(4, nullptr);
}

__device__ void TestList_Append()
{
	List<byte> a(2);
	List<byte> b(2);
	a.Alloc();
	a.Append(1);
	a.Insert(1, 1);
	a.Append(b);
	a.AddUnique(3);
}

__device__ void TestList_Find()
{
	byte v;
	List<byte*> a(2);
	a.FindIndex(&v);
	a.Find(&v);
	a.FindNull();
	a.IndexOf(nullptr);
}

__device__ void TestList_Remove()
{
	byte v;
	List<byte*> a(2);
	a.RemoveIndex(1);
	a.RemoveIndexFast(1);
	a.Remove(&v);
}

__device__ void TestList_Sort()
{
	Sort_InsertionDefault<byte> sorter;
	List<byte> a(2);
	a.SortWithTemplate(sorter);
}

__device__ void TestList_Generic()
{
	//FindFromGeneric();
	//FindFromGenericPtr();
}

__global__ void TestList_Kernel()
{
	TestList_ListArray();
	TestList_ListNewElement();
	TestList_Clear();
	TestList_Properties();
	TestList_Resize();
	TestList_Append();
	TestList_Find();
	TestList_Remove();
	TestList_Sort();
	TestList_Generic();
}

void TestList()
{
#ifndef __CUDACC__
	TestList_Kernel();
#else
	TestList_Kernel<<<1, 1>>>();
#endif
}