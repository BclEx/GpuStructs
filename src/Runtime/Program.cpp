#pragma warning(disable: 4996)
#ifndef _LIB
#include "Cuda.h"
#include "RuntimeHost.h"
#include "FallocHost.h"
#include <stdio.h>

int main(int argc, char **argv)
{
	cudaRuntimeHost runtimeHost = cudaRuntimeInit(256, 4096);
	cudaFallocHost fallocHost = cudaFallocInit(100, 1024);

#if VISUAL
	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	IVisualRender* render = new RuntimeVisualRender(runtimeHost);
	//IVisualRender* render = new FallocVisualRender(fallocHost);
	if (!Visual::InitGL(render, &argc, argv))
		return 0;
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
	Visual::Main();
	atexit(Visual::Dispose);
#endif

	cudaRuntimeEnd(runtimeHost);
	cudaFallocEnd(fallocHost);

	printf("End.");
	char c; scanf("%c", &c);
    return 1;
}

#endif