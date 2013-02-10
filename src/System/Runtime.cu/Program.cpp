#include <cstdio>
#include "../Runtime/Cuda.h"
#include "../Runtime/Visual.h"
#include "../Runtime/Falloc.h"

cudaError _lastError;
void main(int argc, char **argv)
{
	cudaFallocHost fallocHost = cudaFallocInit(100, 1024);

	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	IVisualRender* render = new FallocVisualRender(fallocHost);
	if (!Visual::InitGL(render, &argc, argv))
		return;
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
	Visual::Main();
	atexit(Visual::Dispose);
	// free
	cudaFallocEnd(fallocHost);
	cudaDeviceReset();
	printf("End.");
	char c; scanf_s("%c", &c);
}
