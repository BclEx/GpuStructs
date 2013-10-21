#ifndef _LIB
#include "..\SystemL.net\Core\Core.cu.h"
#include <stdio.h>
#include <string.h>
using namespace Core;
using namespace Core::IO;

static void TestVFS();

void main()
{
	SysEx::Initialize();
	TestVFS();
}

static void TestVFS()
{
	auto vfs = VSystem::Find("win32");
	auto file = (VFile *)SysEx::Alloc(vfs->SizeOsFile);
	auto rc = vfs->Open("C:\\T_\\Test.db", file, (VSystem::OPEN)((int)VSystem::OPEN_CREATE | (int)VSystem::OPEN_READWRITE | (int)VSystem::OPEN_MAIN_DB), nullptr);
	file->Write4(0, 123145);
	file->Close();
}

//namespace Core { int Bitvec_BuiltinTest(int size, int *ops); }
//void TestBitvec()
//{
//	int ops[] = { 5, 1, 1, 1, 0 };
//	Core::Bitvec_BuiltinTest(400, ops);
//}
#endif