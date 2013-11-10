#include <Core\Core.cu.h>

#define TEST(id) \
	__global__ void ioTest##id(void *r); \
	void ioTest##id##_host(cudaRuntimeHost &r) { ioTest##id<<<1, 1>>>(r.heap); cudaRuntimeExecute(r); } \
	__global__ void ioTest##id(void *r) \
{ \
	_runtimeSetHeap(r); \
	SysEx::Initialize();

//////////////////////////////////////////////////

// printf outputs
TEST(0) {
	_printf("test");
}}

// printf outputs
TEST(1) {
	auto vfs = VSystem::Find("gpu");
	auto file = (VFile *)SysEx::Alloc(vfs->SizeOsFile);
	auto rc = vfs->Open("C:\\T_\\Test.db", file, (VSystem::OPEN)((int)VSystem::OPEN_CREATE | (int)VSystem::OPEN_READWRITE | (int)VSystem::OPEN_MAIN_DB), nullptr);
	_printf("%d\n", rc);
	file->Write4(0, 123145);
	file->Close();
	SysEx::Free(file);
}}
