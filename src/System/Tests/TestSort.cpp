#include "..\..\System\System.h"
using namespace Sys::Collections;

__device__ void TestSort_QuickDefault()
{
	Sort_QuickDefault<byte> sorter;
	byte data[] = { 4, 2, 33, 1, 3, 5, 34 };
	sorter.Sort(data, 7);
	//
	Sort_QuickDefault<float> sorterFloat;
	float dataFloat[] = { 4, 2, 33, 1, 3, 5, 34 };
	sorterFloat.Sort(dataFloat, 7);
}

__device__ void TestSort_HeapDefault()
{
	Sort_HeapDefault<byte> sorter;
	byte data[] = { 4, 2, 33, 1, 3, 5, 34 };
	sorter.Sort(data, 7);
}

__device__ void TestSort_InsertionDefault()
{
	Sort_InsertionDefault<byte> sorter;
	byte data[] = { 4, 2, 33, 1, 3, 5, 34 };
	sorter.Sort(data, 7);
}

__global__ void TestSort_Kernel()
{
	TestSort_QuickDefault();
	TestSort_HeapDefault();
	TestSort_InsertionDefault();
}

void TestSort()
{
#ifndef __CUDACC__
	TestSort_Kernel();
#else
	TestSort_Kernel<<<1, 1>>>();
#endif
}