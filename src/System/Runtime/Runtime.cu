//[GL VBO] http://3dgep.com/?p=2596
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
	void* reserved;
	size_t blockSize;
	size_t blocks;
	size_t offset;
	size_t freeBlocksSize; // Size of circular buffer (set up by host)
	fallocBlock** freeBlocks; // Start of circular buffer (set up by host)
	volatile fallocBlock** freeBlockPtr; // Current atomically-incremented non-wrapped offset
	volatile fallocBlock** retnBlockPtr; // Current atomically-incremented non-wrapped offset
} fallocHeap;


///////////////////////////////////////////////////////////////////////////////
// VISUAL

#define BLOCKPITCH 64
struct quad4
{ 
	float4 av, ac;
	float4 bv, bc;
	float4 cv, cc;
	float4 dv, dc;
};
static __inline__ __device__ quad4 make_quad4(
	float4 av, float4 ac,
	float4 bv, float4 bc,
	float4 cv, float4 cc,
	float4 dv, float4 dc)
{
	quad4 q; q.av = av; q.ac = ac; q.bv = bv; q.bc = bc; q.cv = cv; q.cc = cc; q.dv = dv; q.dc = dc; return q;
}

#define red make_float4(1, 0, 0, 1)
#define green make_float4(0, 1, 0, 1)
#define blue make_float4(0, 0, 1, 1)
#define blue2 make_float4(0, 0, .8, 1)
#define yellow make_float4(1, 0, 1, 1)

static __inline__ __device__ void OffsetBlockRef(unsigned int x, unsigned int y, float *x1, float *y1)
{
	*x1 = x * 1; *y1 = y * 1 + 1;
}

__global__ void RenderHeap(quad4 *b, fallocHeap* heap)
{
	// heap
	b[0] = make_quad4(
		make_float4(00, 1, 1, 1), green,
		make_float4(10, 1, 1, 1), green,
		make_float4(10, 0, 1, 1), green,
		make_float4(00, 0, 1, 1), green);
	// free
	float x1, y1;
	int freePtr = 1; //(int)(heap->freeBlockPtr - heap->freeBlocks);
	if (freePtr > 0)
	{
		OffsetBlockRef(1, 1, &x1, &y1);
		b[1] = make_quad4(
			make_float4(x1 + .0, y1 + .2, 1, 2), green,
			make_float4(x1 + .2, y1 + .2, 1, 2), green,
			make_float4(x1 + .2, y1 + .0, 1, 2), green,
			make_float4(x1 + .0, y1 + .0, 1, 2), green);
	}
	// retn
	int retnPtr = 2; //(int)(heap->retnBlockPtr - heap->freeBlocks);
	if (retnPtr > 0)
	{
		OffsetBlockRef(2, 1, &x1, &y1);
		b[2] = make_quad4(
			make_float4(x1 + .7, y1 + .9, 1, 2), yellow,
			make_float4(x1 + .9, y1 + .9, 1, 2), yellow,
			make_float4(x1 + .9, y1 + .7, 1, 2), yellow,
			make_float4(x1 + .7, y1 + .7, 1, 2), yellow);
	}
}

__global__ void RenderBlock(quad4 *b, size_t blocks, unsigned int blocksY, fallocHeap* heap, unsigned int offset)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	int blockIndex = y*BLOCKPITCH + x;
	if (blockIndex >= blocks)
		return;
	int index = blockIndex * 3 + offset;
	//
	float x1, y1; OffsetBlockRef(x, y, &x1, &y1);
	b[index] = make_quad4(
		make_float4(x1 + 0.0, y1 + 0.9, 1, 1), red,
		make_float4(x1 + 0.9, y1 + 0.9, 1, 1), red,
		make_float4(x1 + 0.9, y1 + 0.0, 1, 1), red,
		make_float4(x1 + 0.0, y1 + 0.0, 1, 1), red);

	// block
	float x2 = x * 10;
	float y2 = y * 20 + blocksY + 3;
	b[index + 1] = make_quad4(
		make_float4(x2 + 0, y2 + 19, 1, 1), blue,
		make_float4(x2 + 9, y2 + 19, 1, 1), blue,
		make_float4(x2 + 9, y2 + 00, 1, 1), blue,
		make_float4(x2 + 0, y2 + 00, 1, 1), blue);
	b[index + 2] = make_quad4(
		make_float4(x2 + 0, y2 + 1, 1, 1), green,
		make_float4(x2 + 3.9, y2 + 1, 1, 1), green,
		make_float4(x2 + 3.9, y2 + 0, 1, 1), green,
		make_float4(x2 + 0, y2 + 0, 1, 1), green);
}

#define MAX(a,b) (a > b ? a : b)
__host__ int GetRenderQuads(size_t blocks) { return 3 + (blocks * 3); }
__host__ void LaunchRender(float4 *b, size_t blocks, fallocHeap* heap)
{
	dim3 heapBlock(1, 1, 1);
	dim3 heapGrid(1, 1, 1);
	RenderHeap<<<heapGrid, heapBlock>>>((quad4 *)b, heap);
	//
	dim3 blockBlock(16, 16, 1);
	dim3 blockGrid(MAX(BLOCKPITCH / 16, 1), MAX(blocks / BLOCKPITCH / 16, 1), 1);
	RenderBlock<<<blockGrid, blockBlock>>>((quad4 *)b, blocks, blocks / BLOCKPITCH, heap, 3);
}


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
	size_t HEAPBLOCK_SIZE;
} fallocCtx;

__device__ void fallocInit(fallocHeap* heap)
{
	if (threadIdx.x || threadIdx.y || threadIdx.z) return;
	//
	size_t blocks = heap->blocks;
	if (!blocks)
		__THROW;
	size_t blockSize = heap->blockSize;
	fallocBlock** freeBlocks = heap->freeBlocks;
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
}

__device__ inline void* fallocGetBlock(fallocHeap* heap)
{
	if (threadIdx.x || threadIdx.y || threadIdx.z) __THROW;
	// advance circular buffer
	fallocBlock** freeBlocks = heap->freeBlocks;
	//volatile fallocBlock** freeBlockPtr = heap->freeBlockPtr;
	size_t offset = atomicAdd((unsigned int *)&heap->freeBlockPtr, sizeof(fallocBlock*)) - (size_t)freeBlocks;
	offset %= heap->freeBlocksSize;
	fallocBlock* block = (fallocBlock*)(freeBlocks + offset);
	//
	return (void*)((__int8*)block + sizeof(fallocBlock));
}

__device__ inline void fallocFreeBlock(fallocHeap* heap, void* obj)
{
	if (threadIdx.x || threadIdx.y || threadIdx.z) __THROW;
	//
	fallocBlock* block = (fallocBlock *)((__int8 *)obj - (int)sizeof(fallocBlock));
	if (block->magic != FALLOC_MAGIC || block->count > 1) // bad magic or not a singular block
		__THROW;
	// advance circular buffer
	fallocBlock** freeBlocks = heap->freeBlocks;
	//volatile fallocBlock** retnBlockPtr = heap->retnBlockPtr;
	size_t offset = atomicAdd((unsigned int *)&heap->retnBlockPtr, sizeof(fallocBlock*)) - (size_t)freeBlocks;
	offset %= heap->freeBlocksSize;
	*(freeBlocks + offset) = block;
}

#if MULTIBLOCK
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
#endif

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
	if (bytes > (ctx->HEAPBLOCK_SIZE - sizeof(fallocCtx)))
		__THROW;
	// find or add available node
	fallocNode* node;
	unsigned short freeOffset;
	unsigned char hasFreeSpace;
	fallocNode* lastNode;
	for (lastNode = (fallocNode*)ctx, node = ctx->availableNodes; node; lastNode = node, node = (alloc ? node->nextAvailable : node->next))
		if (hasFreeSpace = ((freeOffset = (node->freeOffset + bytes)) <= ctx->HEAPBLOCK_SIZE))
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
	if (alloc && ((freeOffset + FALLOCNODE_SLACK) > ctx->HEAPBLOCK_SIZE)) {
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