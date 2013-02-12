#include <cstdio>
#include "../Runtime/Cuda.h"
#include "../Runtime/Visual.h"
#include "../Runtime/Runtime.h"
#include "../Runtime/Falloc.h"

cudaError _lastError;
void main(int argc, char **argv)
{
	cudaRuntimeHost runtimeHost = cudaRuntimeInit(256, 4096);
	cudaFallocHost fallocHost = cudaFallocInit(100, 1024);

	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	//IVisualRender* render = new RuntimeVisualRender(runtimeHost);
	IVisualRender* render = new FallocVisualRender(fallocHost);
	if (!Visual::InitGL(render, &argc, argv))
		return;
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
	Visual::Main();
	atexit(Visual::Dispose);
	// free
	cudaRuntimeEnd(runtimeHost);
	cudaFallocEnd(fallocHost);
	cudaDeviceReset();
	printf("End.");
	char c; scanf_s("%c", &c);
}
