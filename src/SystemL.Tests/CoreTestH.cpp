#include <stdio.h>
#include <string.h>
#include <Runtime.h>
using namespace System;
using namespace Xunit;

extern void coreTest0_host(cudaRuntimeHost &r);
extern void coreTest1_host(cudaRuntimeHost &r);

namespace Tests
{
	static cudaRuntimeHost _runtimeHost;

	public ref class CoreTest
	{
	public:
		CoreTest()
		{
			// Choose which GPU to run on, change this on a multi-GPU system.
			//checkCudaErrors(cudaSetDevice(0), throw gcnew Exception("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"));
			_runtimeHost = cudaRuntimeInit(256, 4096); cudaRuntimeSetHeap(_runtimeHost.heap);
		}
		~CoreTest()
		{
			//static int id = 0; char path[50]; memcpy(path, _path, strlen(_path) + 1); path[strlen(_path)-5] += id++;
			//FILE *f = fopen(path, "w");
			//cudaRuntimeExecute(f, true);
			cudaRuntimeEnd(_runtimeHost);
			//fclose(f);
			//checkCudaErrors(cudaDeviceReset(), throw gcnew Exception("cudaDeviceReset failed!"));
		}

		[Fact]
		void bitvec() { coreTest0_host(_runtimeHost); }

		[Fact]
		void bitvec_failures() { coreTest1_host(_runtimeHost); }
	};
}
