#include "..\System.net\Core\Core.cu.h"
using namespace Core;
using namespace Core::IO;

__device__ static void TestVFS();

__global__ void MainTest(void *heap)
{
	SysEx::Initialize();
	//TestVFS();
}

void __main(cudaRuntimeHost &r)
{	
	cudaRuntimeSetHeap(r.heap);
	MainTest<<<1, 1>>>(r.heap);
}

__device__ static void TestVFS()
{
	auto vfs = VSystem::Find("gpu");
	auto file = (VFile *)SysEx::Alloc(vfs->SizeOsFile);
	auto rc = vfs->Open("C:\\T_\\Test.db", file, (VSystem::OPEN)((int)VSystem::OPEN_CREATE | (int)VSystem::OPEN_READWRITE | (int)VSystem::OPEN_MAIN_DB), nullptr);
	_printf("%d\n", rc);
	file->Write4(0, 123145);
	file->Close();
	SysEx::Free(file);
}
