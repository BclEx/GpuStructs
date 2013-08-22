//#ifdef _LIB
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
#include "Runtime.cu.h"
#include "Falloc.cu.h"
#endif
//#endif