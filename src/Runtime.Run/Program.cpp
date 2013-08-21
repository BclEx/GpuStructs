#define VISUAL
#include "../Runtime.src/Cuda.h"
#include "../Runtime.src/Runtime.h"
#include "../Runtime.src/Falloc.h"

cudaError _lastError;
int main(int argc, char **argv)
{
	cudaRuntimeHost runtimeHost = cudaRuntimeInit(256, 4096);
	//cudaFallocHost fallocHost = cudaFallocInit(100, 1024);

	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	IVisualRender* render = new RuntimeVisualRender(runtimeHost);
	//IVisualRender* render = new FallocVisualRender(fallocHost);
	if (!Visual::InitGL(render, &argc, argv))
		return 0;
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
	Visual::Main();
	atexit(Visual::Dispose);
	
	cudaRuntimeEnd(runtimeHost);
	//cudaFallocEnd(fallocHost);

	cudaDeviceReset();
	printf("End.");
	char c; scanf("%c", &c);
	return 1;
}
