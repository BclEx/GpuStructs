#pragma hdrstop
#include <malloc.h>
#include "System.h"
#undef new

/// <summary>
/// Mem_Alloc16
/// </summary>
void * Mem_Alloc16(const int size, const memTag_t tag)
{
	if (!size)
		return nullptr;
	const int paddedSize = (size + 15) & ~15;
	return _aligned_malloc(paddedSize, 16);
}

/// <summary>
/// Mem_Free16
/// </summary>
void Mem_Free16(void *ptr)
{
	if (ptr == nullptr)
		return;
	_aligned_free(ptr);
}

/// <summary>
/// Mem_ClearedAlloc
/// </summary>
void *Mem_ClearedAlloc(const int size, const memTag_t tag)
{
	void *mem = Mem_Alloc(size, tag);
	//SIMDProcessor->Memset(mem, 0, size);
	return mem;
}

/// <summary>
/// Mem_CopyString
/// </summary>
char *Mem_CopyString(const char *in)
{
	char *out = (char *)Mem_Alloc(strlen(in) + 1, TAG_STRING);
	strcpy(out, in);
	return out;
}

