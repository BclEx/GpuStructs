#ifdef _CONSOLE
#if __CUDA_ARCH__ == 100
#error Atomics only used with > sm_10 architecture
#elif __CUDA_ARCH__ >= 200
#include "../Runtime.src/Runtime.cu.h"
#include "../Runtime.src/Falloc.cu.h"
#endif
#endif