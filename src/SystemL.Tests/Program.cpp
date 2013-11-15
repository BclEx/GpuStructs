#include <RuntimeEx.h>

extern void coreTest0_host(cudaRuntimeHost &r);
static cudaRuntimeHost _runtimeHost;

void main(int argc, char **argv)
{
	_runtimeHost = cudaRuntimeInit(256, 4096); cudaRuntimeSetHeap(_runtimeHost.heap);

	coreTest0_host(_runtimeHost);

	cudaRuntimeEnd(_runtimeHost); cudaDeviceReset();
	printf("\nEnd\n"); char c; scanf("%c", &c);
}
