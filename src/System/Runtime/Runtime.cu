#if __CUDA_ARCH__ == 100 
#error Atomics only used with > sm_10 architecture
#endif
#include "cuda_runtime_api.h"
#include <malloc.h>
#include <string.h>

#define RUNTIME_UNRESTRICTED -1

typedef struct __align__(8)
{
	int threadid; // RUNTIME_UNRESTRICTED for unrestricted
	int blockid;  // RUNTIME_UNRESTRICTED for unrestricted
} runtimeRestriction;

typedef struct __align__(8)
{
	volatile char *blockPtr; // current atomically-incremented non-wrapped offset
	void *reserved;
	runtimeRestriction restriction;
	size_t blockSize;
	size_t blocksLength; // size of circular buffer (set up by host)
	char *blocks; // start of circular buffer (set up by host)
} runtimeHeap;

typedef struct __align__(8)
{
	unsigned short magic;		// magic number says we're valid
	unsigned short fmtoffset;	// offset of fmt string into buffer
	unsigned short blockid;		// block ID of author
	unsigned short threadid;	// thread ID of author
} runtimeBlockHeader;

__device__ static runtimeHeap *_heap = nullptr;

///////////////////////////////////////////////////////////////////////////////
// HEAP
#pragma region HEAP

#define RUNTIME_MAGIC (unsigned short)0xC811
#define RUNTIME_ALIGNSIZE sizeof(long long)

__device__ static char *moveNextPtr()
{
	if (!_heap) __THROW;
	// thread/block restriction check
	runtimeRestriction restriction = _heap->restriction;
	if (restriction.blockid != RUNTIME_UNRESTRICTED && restriction.blockid != (blockIdx.x + gridDim.x*blockIdx.y))
		return nullptr;
	if (restriction.threadid != RUNTIME_UNRESTRICTED && restriction.threadid != (threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z))
		return nullptr;
	// advance pointer
	char *blocks = _heap->blocks;
	size_t offset = atomicAdd((unsigned int *)&_heap->blockPtr, _heap->blockSize) - (size_t)blocks;
	offset %= _heap->blocksLength;
	return blocks + offset;
}

__device__ static void runtimeRestrict(int threadid, int blockid)
{
	int threadMax = blockDim.x * blockDim.y * blockDim.z;
	if ((threadid < threadMax && threadid >= 0) || threadid == RUNTIME_UNRESTRICTED)
		_heap->restriction.threadid = threadid;
	int blockMax = gridDim.x * gridDim.y;
	if ((blockid < blockMax && blockid >= 0) || blockid == RUNTIME_UNRESTRICTED)
		_heap->restriction.blockid = blockid;
}

#pragma endregion


//////////////////////
// PRINTF
#pragma region PRINTF

__device__ static void writePrintfHeader(char *ptr, char *fmtptr)
{
	if (ptr)
	{
		runtimeBlockHeader header;
		header.magic = RUNTIME_MAGIC;
		header.fmtoffset = (unsigned short)(fmtptr - ptr);
		header.blockid = blockIdx.x + gridDim.x*blockIdx.y;
		header.threadid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
		*(runtimeBlockHeader *)(void *)ptr = header;
	}
}

__device__ static char *printfStrncpy(char *dest, const char *src, int n, char *end)
{
	// initialization and overflow check
	if (!dest || !src || dest >= end)
		return nullptr;
	// prepare to write the length specifier. We're guaranteed to have at least "RUNTIME_ALIGNSIZE" bytes left because we only write out in
	// chunks that size, and blockSize is aligned with RUNTIME_ALIGNSIZE.
	int *lenptr = (int *)(void *)dest;
	int len = 0;
	dest += RUNTIME_ALIGNSIZE;
	// now copy the string
	while (n--)
	{
		if (dest >= end) // overflow check
			break;
		len++;
		*dest++ = *src;
		if (*src++ == '\0')
			break;
	}
	// now write out the padding bytes, and we have our length.
	while (dest < end && ((long)dest & (RUNTIME_ALIGNSIZE - 1)) != 0)
	{
		len++;
		*dest++ = 0;
	}
	*lenptr = len;
	return (dest < end ? dest : nullptr); // overflow means return nullptr
}

__device__ static char *copyArg(char *ptr, const char *arg, char *end)
{
	// initialisation check
	if (!ptr || !arg)
		return nullptr;
	// strncpy does all our work. We just terminate.
	if ((ptr = printfStrncpy(ptr, arg, _heap->blockSize, end)) != nullptr)
		*ptr = 0;
	return ptr;
}

template <typename T>
__device__ static char *copyArg(char *ptr, T &arg, char *end)
{
	// initisalisation and overflow check. Alignment rules mean that we're at least CUPRINTF_ALIGN_SIZE away from "end", so we only need to check that one offset.
	if (!ptr || (ptr + RUNTIME_ALIGNSIZE) >= end)
		return nullptr;
	// write the length and argument
	*(int *)(void *)ptr = sizeof(arg);
	ptr += RUNTIME_ALIGNSIZE;
	*(T *)(void *)ptr = arg;
	ptr += RUNTIME_ALIGNSIZE;
	*ptr = 0;
	return ptr;
}

#define PRINTF_PREAMBLE \
	char *start, *end, *bufptr, *fmtstart; \
	if ((start = moveNextPtr()) == nullptr) return 0; \
	end = start + _heap->blockSize; \
	bufptr = start + sizeof(runtimeBlockHeader);
#define PRINTF_ARG(argname) \
	bufptr = copyArg(bufptr, argname, end);
#define PRINTF_POSTAMBLE \
	fmtstart = bufptr; \
	end = printfStrncpy(bufptr, fmt, _heap->blockSize, end); \
	writePrintfHeader(start, (end ? fmtstart : nullptr)); \
	return (end ? (int)(end - start) : 0);

__device__ static int __printf(const char *fmt)
{
	PRINTF_PREAMBLE;
	PRINTF_POSTAMBLE;
}
template <typename T1> __device__ static int __printf(const char *fmt, T1 arg1)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2> __device__ static int __printf(const char *fmt, T1 arg1, T2 arg2)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1);
	PRINTF_ARG(arg2);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3> __device__ static int __printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1);
	PRINTF_ARG(arg2);
	PRINTF_ARG(arg3);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4> __device__ static int __printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1);
	PRINTF_ARG(arg2);
	PRINTF_ARG(arg3);
	PRINTF_ARG(arg4);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ static int __printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1);
	PRINTF_ARG(arg2);
	PRINTF_ARG(arg3);
	PRINTF_ARG(arg4);
	PRINTF_ARG(arg5);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ static int __printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1);
	PRINTF_ARG(arg2);
	PRINTF_ARG(arg3);
	PRINTF_ARG(arg4);
	PRINTF_ARG(arg5);
	PRINTF_ARG(arg6);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ static int __printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1);
	PRINTF_ARG(arg2);
	PRINTF_ARG(arg3);
	PRINTF_ARG(arg4);
	PRINTF_ARG(arg5);
	PRINTF_ARG(arg6);
	PRINTF_ARG(arg7);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ static int __printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1);
	PRINTF_ARG(arg2);
	PRINTF_ARG(arg3);
	PRINTF_ARG(arg4);
	PRINTF_ARG(arg5);
	PRINTF_ARG(arg6);
	PRINTF_ARG(arg7);
	PRINTF_ARG(arg8);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ static int __printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1);
	PRINTF_ARG(arg2);
	PRINTF_ARG(arg3);
	PRINTF_ARG(arg4);
	PRINTF_ARG(arg5);
	PRINTF_ARG(arg6);
	PRINTF_ARG(arg7);
	PRINTF_ARG(arg8);
	PRINTF_ARG(arg9);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10> __device__ static int __printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1);
	PRINTF_ARG(arg2);
	PRINTF_ARG(arg3);
	PRINTF_ARG(arg4);
	PRINTF_ARG(arg5);
	PRINTF_ARG(arg6);
	PRINTF_ARG(arg7);
	PRINTF_ARG(arg8);
	PRINTF_ARG(arg9);
	PRINTF_ARG(arg10);
	PRINTF_POSTAMBLE;
}
#undef PRINTF_PREAMBLE
#undef PRINTF_ARG
#undef PRINTF_POSTAMBLE

#pragma endregion
