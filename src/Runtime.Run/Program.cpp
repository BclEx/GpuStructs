#include "../Runtime/RuntimeEx.h"
#include "../Runtime/Falloc.h"

void __fallocExample(cudaFallocHost &f);
void __runtimeExample(cudaRuntimeHost &r);

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

	// test
	//__fallocExample(fallocHost);
	__runtimeExample(runtimeHost);

	// run
	Visual::Main();
	atexit(Visual::Dispose);
	
	cudaRuntimeEnd(runtimeHost);
	//cudaFallocEnd(fallocHost);

	cudaDeviceReset();
	printf("End.");
	char c; scanf("%c", &c);
	return 1;
}
