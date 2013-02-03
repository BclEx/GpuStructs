#include <cstdio>
#include "../Runtime/Cuda.h"
#include "../Runtime/Visual.h"

cudaError _lastError;
void main(int argc, char **argv)
{
	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	IVisualRender* render = new FallocVisualRender();
	if (!Visual::InitGL(render, &argc, argv))
		return;
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
	Visual::Main();
	atexit(Visual::Dispose);
	cudaDeviceReset();
	printf("End.");
	char c; scanf_s("%c", &c);
}
