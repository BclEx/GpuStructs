#include <stdio.h>
#include <string.h>
#include "..\Runtime.src\Runtime.h"
#include "..\Runtime.src\Falloc.h"
using namespace System;
using namespace Xunit;

extern void fallocTestA_host(cudaRuntimeHost &r, void *h);
extern void fallocTest0_host(cudaRuntimeHost &r, void *h);
extern void fallocTest1_host(cudaRuntimeHost &r, void *h);
extern void fallocTest2_host(cudaRuntimeHost &r, void *h);
extern void fallocTest3_host(cudaRuntimeHost &r, void *h);

namespace Tests
{
	static cudaRuntimeHost _runtimeHost;
	static cudaFallocHost _fallocHost;

	public ref class FallocITest
	{
	public:
		void AssertHandler()
		{
			throw gcnew Exception();
		}

		FallocITest()
		{
			// Choose which GPU to run on, change this on a multi-GPU system.
			//checkCudaErrors(cudaSetDevice(0), throw gcnew Exception("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"));
			_runtimeHost = cudaRuntimeInit(256, 4096);
			//cudaRuntimeSetHandler(&Tests::FallocITest::AssertHandler);
			TestInitialize();
		}
		~FallocITest()
		{
			TestCleanup();
			//static int id = 0; char path[50]; memcpy(path, _path, strlen(_path) + 1); path[strlen(_path)-5] += id++;
			//FILE *f = fopen(path, "w");
			//cudaRuntimeExecute(f, true);
			cudaRuntimeEnd(_runtimeHost);
			//fclose(f);
			//checkCudaErrors(cudaDeviceReset(), throw gcnew Exception("cudaDeviceReset failed!"));
		}

		void TestInitialize() { _fallocHost = cudaFallocInit(1024, 4098); }
		void TestCleanup() { cudaFallocEnd(_fallocHost); }

		[Fact]
		void Initialize_Returns_FallocHost()
		{
			Assert::True(_fallocHost.heap != nullptr);
		}

		[Fact]
		void lauched_cuda_kernel()
		{
			fallocTestA_host(_runtimeHost, _fallocHost.heap);
			Assert::Equal(1, 1);
		}

		[Fact]
		void alloc_with_getblock() { fallocTest0_host(_runtimeHost, _fallocHost.heap); }

		[Fact]
		void alloc_with_getblocks() { fallocTest1_host(_runtimeHost, _fallocHost.heap); }

		[Fact]
		void alloc_with_context() { fallocTest2_host(_runtimeHost, _fallocHost.heap); }

		[Fact]
		void alloc_with_context_as_stack() { fallocTest3_host(_runtimeHost, _fallocHost.heap); }
	};
}
