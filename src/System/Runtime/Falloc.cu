#if __CUDA_ARCH__ == 100 
#error Atomics only used with > sm_10 architecture
#endif
#include "cuda_runtime_api.h"
#include <malloc.h>
#include <string.h>

typedef struct __align__(8) _cuFallocBlock
{
	unsigned short magic;
	unsigned short count;
	struct _cuFallocBlock* next;
	void* reserved;
} fallocBlock;

typedef struct __align__(8) _cuFallocHeap
{
	size_t blockSize;
	size_t blocks;
	size_t offset;
	size_t freeBlocksSize; // Size of circular buffer (set up by host)
	fallocBlock** freeBlocks; // Start of circular buffer (set up by host)
	volatile fallocBlock** freeBlocksPtr; // Current atomically-incremented non-wrapped offset
	void* reserved;
} fallocHeap;


///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE

const static int FALLOCNODE_SLACK = 0x10;
#define FALLOC_MAGIC (unsigned short)0x3412 // All our headers are prefixed with a magic number so we know they're ours
#define FALLOCNODE_MAGIC (unsigned short)0x7856 // All our headers are prefixed with a magic number so we know they're ours

typedef struct _cuFallocNode
{
	struct _cuFallocNode* next;
	struct _cuFallocNode* nextAvailable;
	unsigned short freeOffset;
	unsigned short magic;
} fallocNode;

typedef struct _cuFallocContext
{
	fallocNode node;
	fallocNode* nodes;
	fallocNode* availableNodes;
	fallocHeap* heap;
} fallocCtx;

__device__ void fallocInit(fallocHeap* heap)
{
	if (threadIdx.x || threadIdx.y || threadIdx.z) return;
	size_t blocks = heap->blocks;
	if (!blocks)
		__THROW;
	fallocBlock** freeBlocks = heap->freeBlocks;
	size_t blockSize = heap->blockSize;
	// preset all blocks
	fallocBlock* block = (fallocBlock*)((__int8*)heap + heap->offset);
	block->magic = FALLOC_MAGIC;
	block->count = 1;
	block->reserved = nullptr;
	while (blocks-- > 1)
	{
		block = *freeBlocks++ = block->next = (fallocBlock*)((__int8 *)block + blockSize);
		block->magic = FALLOC_MAGIC;
		block->count = 1;
		block->reserved = nullptr;
	}
	block->next = nullptr;
	heap->freeBlocksPtr = freeBlocks;
}

__device__ inline void* fallocGetBlock(fallocHeap* heap)
{
	if (threadIdx.x || threadIdx.y || threadIdx.z) __THROW;
	volatile fallocBlock* block = heap->freeBlocks;
	if (!block)
		return nullptr;
	{ // critical
		heap->freeBlocks = block->next;
		block->next = nullptr;
	}
	return (void*)((__int8*)block + sizeof(fallocBlock));
}

__device__ inline void* fallocGetBlocks(fallocHeap* heap, size_t length, size_t* allocLength = nullptr)
{
	if (threadIdx.x || threadIdx.y || threadIdx.z) __THROW;
	size_t blockSize = heap->blockSize;
	// fix up length to be a multiple of blockSize
	length = (length < blockSize ? blockSize : length);
	if (length % blockSize)
		length += blockSize - (length % blockSize);
	// set length, if requested
	if (allocLength)
		*allocLength = length - sizeof(fallocBlock);
	size_t blocks = (size_t)(length / blockSize);
	if (blocks > heap->blocks)
		__THROW;
	// single, equals: fallocGetBlock
	if (blocks == 1)
		return fallocGetBlock(heap);
	// multiple, find a contiguous chuck
	size_t index = blocks;
	volatile fallocBlock* block;
	volatile fallocBlock* endBlock = (fallocBlock*)((__int8*)heap + sizeof(fallocHeap) + (blockSize * heap->blocks));
	{ // critical
		for (block = (fallocBlock*)((__int8*)heap + sizeof(fallocHeap)); index && block < endBlock; block = (fallocBlock*)((__int8*)block + (blockSize * block->count)))
		{
			if (block->magic != FALLOC_MAGIC)
				__THROW;
			index = (block->next ? index - 1 : blocks);
		}
		if (index)
			return nullptr;
		// found chuck, remove from freeBlocks
		endBlock = block;
		block = (fallocBlock*)((__int8*)block - (blockSize * blocks));
		for (volatile fallocBlock* chunk2 = heap->freeBlocks; chunk2; chunk2 = chunk2->next)
			if (chunk2 >= block && chunk2 <= endBlock)
				chunk2->next = (chunk2->next ? chunk2->next->next : nullptr);
		block->count = blocks;
		block->next = nullptr;
	}
	return (void*)((__int8*)block + sizeof(fallocBlock));
}

__device__ inline void fallocFreeBlock(fallocHeap* heap, void* obj)
{
	if (threadIdx.x || threadIdx.y || threadIdx.z) __THROW;
	volatile fallocBlock* block = (fallocBlock *)((__int8 *)obj - (int)sizeof(fallocBlock));
	if (block->magic != FALLOC_MAGIC || block->count > 1)
		__THROW;
	{ // critical
		block->next = heap->freeBlocks;
		heap->freeBlocks = block;
	}
}

__device__ inline void fallocFreeBlocks(fallocHeap* heap, void* obj)
{
	volatile fallocBlock* block = (fallocBlock*)((__int8*)obj - sizeof(fallocBlock));
	if (block->magic != FALLOC_MAGIC)
		__THROW;
	size_t blocks = block->count;
	// single, equals: fallocFreeChunk
	if (blocks == 1)
	{
		{ // critical
			block->next = heap->freeBlocks;
			heap->freeBlocks = block;
		}
		return;
	}
	// retag blocks
	size_t blockSize = heap->blockSize;
	block->count = 1;
	while (blocks-- > 1)
	{
		block = block->next = (fallocBlock*)((__int8*)block + sizeof(fallocBlock) + blockSize);
		block->magic = FALLOC_MAGIC;
		block->count = 1;
		block->reserved = nullptr;
	}
	{ // critical
		block->next = heap->freeBlocks;
		heap->freeBlocks = block;
	}
}


//////////////////////
// ALLOC

__device__ static fallocCtx* fallocCreateCtx(fallocHeap* heap)
{
	size_t blockSize = heap->blockSize;
	if (sizeof(fallocCtx) > blockSize)
		__THROW;
	fallocCtx* ctx = (fallocCtx*)fallocGetBlock(heap);
	if (!ctx)
		return nullptr;
	ctx->heap = heap;
	unsigned short freeOffset = ctx->node.freeOffset = sizeof(fallocCtx);
	ctx->node.magic = FALLOCNODE_MAGIC;
	ctx->node.next = nullptr; ctx->nodes = (fallocNode*)ctx;
	ctx->node.nextAvailable = nullptr; ctx->availableNodes = (fallocNode*)ctx;
	// close node
	if ((freeOffset + FALLOCNODE_SLACK) > blockSize)
		ctx->availableNodes = nullptr;
	return ctx;
}

__device__ static void fallocDisposeCtx(fallocCtx* ctx)
{
	fallocHeap* heap = ctx->heap;
	for (fallocNode* node = ctx->nodes; node; node = node->next)
		fallocFreeBlock(heap, node);
}

__device__ static void* falloc(fallocCtx* ctx, unsigned short bytes, bool alloc)
{
	if (bytes > (HEAPBLOCK_SIZE - sizeof(fallocCtx)))
		__THROW;
	// find or add available node
	fallocNode* node;
	unsigned short freeOffset;
	unsigned char hasFreeSpace;
	fallocNode* lastNode;
	for (lastNode = (fallocNode*)ctx, node = ctx->availableNodes; node; lastNode = node, node = (alloc ? node->nextAvailable : node->next))
		if (hasFreeSpace = ((freeOffset = (node->freeOffset + bytes)) <= HEAPBLOCK_SIZE))
			break;
	if (!node || !hasFreeSpace) {
		// add node
		node = (fallocNode*)fallocGetBlock(ctx->heap);
		if (!node)
			__THROW;
		freeOffset = node->freeOffset = sizeof(fallocNode); 
		freeOffset += bytes;
		node->magic = FALLOCNODE_MAGIC;
		node->next = ctx->nodes; ctx->nodes = node;
		node->nextAvailable = (alloc ? ctx->availableNodes : nullptr); ctx->availableNodes = node;
	}
	//
	void* obj = (__int8*)node + node->freeOffset;
	node->freeOffset = freeOffset;
	// close node
	if (alloc && ((freeOffset + FALLOCNODE_SLACK) > HEAPBLOCK_SIZE)) {
		if (lastNode == (fallocNode*)ctx)
			ctx->availableNodes = node->nextAvailable;
		else
			lastNode->nextAvailable = node->nextAvailable;
		node->nextAvailable = nullptr;
	}
	return obj;
}

__device__ static void* fallocRetract(fallocCtx* ctx, unsigned short bytes)
{
	fallocNode* node = ctx->availableNodes;
	int freeOffset = (int)node->freeOffset - bytes;
	// multi node, retract node
	if (node != &ctx->node && freeOffset < sizeof(fallocNode)) {
		node->freeOffset = sizeof(fallocNode);
		// search for previous node
		fallocNode* lastNode;
		for (lastNode = (fallocNode*)ctx, node = ctx->nodes; node; lastNode = node, node = node->next)
			if (node == ctx->availableNodes)
				break;
		node = ctx->availableNodes = lastNode;
		freeOffset = (int)node->freeOffset - bytes;
	}
	// first node && !overflow
	if (node == &ctx->node && freeOffset < sizeof(fallocCtx))
		__THROW;
	node->freeOffset = (unsigned short)freeOffset;
	return (__int8*)node + freeOffset;
}

__device__ static void fallocMark(fallocCtx* ctx, void* &mark, unsigned short &mark2) { mark = ctx->availableNodes; mark2 = ctx->availableNodes->freeOffset; }
__device__ static bool fallocAtMark(fallocCtx* ctx, void* mark, unsigned short mark2) { return (mark == ctx->availableNodes && mark2 == ctx->availableNodes->freeOffset); }
