#ifndef __SYSTEM_SYSTEM_CUDA_H__
#define __SYSTEM_SYSTEM_CUDA_H__
namespace Sys {

#ifndef __CUDACC__
#define __constant__
#define __host__
#define __global__
#define __device__
#define __shared__
#include <string.h>
#else

	__device__ char *strncpy(char *dest, const char *src, size_t len)
	{
		int pad = 0;
		char *new_dest = dest;
		for (size_t i = 0; i < len; ++i, ++src, ++new_dest) {
			if (pad != 0)
				*new_dest = '\0';
			else
			{
				*new_dest = *src;
				if ('\0' == *src)
					pad = 1;
			}
		}
		return dest;
	}

#endif

}
#endif /* __SYSTEM_SYSTEM_CUDA_H__ */
