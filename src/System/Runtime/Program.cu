#if __CUDA_ARCH__ == 100
#error Atomics only used with > sm_10 architecture
#elif __CUDA_ARCH__ >= 200

#include "Runtime.cu.h"
#include "Falloc.cu.h"

#endif
