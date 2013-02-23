#ifndef nullptr
#define nullptr NULL
#endif
//#define __THROW *(int*)0=0;
#if __CUDA_ARCH__ == 100
#error Atomics only used with > sm_10 architecture
#elif __CUDA_ARCH__ < 200
#define STATIC static
#else
#define STATIC
#endif
#include <string.h>
#include "Cuda.h"

typedef struct __align__(8)
{
	unsigned short magic;		// magic number says we're valid
	unsigned short count;		// number of blocks in sequence
	unsigned short blockid;		// block ID of author
	unsigned short threadid;	// thread ID of author
} fallocBlockHeader;

typedef struct __align__(8)
{
	fallocBlockHeader *block;	// block reference
	unsigned short blockid;		// block ID of author
	unsigned short threadid;	// thread ID of author
} fallocBlockRef;

typedef struct __align__(8)
{
	void *reserved;
	size_t blockSize;
	size_t blocksLength;
	size_t blockRefsLength; // Size of circular buffer (set up by host)
	fallocBlockRef *blockRefs; // Start of circular buffer (set up by host)
	volatile fallocBlockRef *freeBlockPtr; // Current atomically-incremented non-wrapped offset
	volatile fallocBlockRef *retnBlockPtr; // Current atomically-incremented non-wrapped offset
	char *blocks;
} fallocHeap;

///////////////////////////////////////////////////////////////////////////////
// HEAP
#pragma region HEAP

#define FALLOC_MAGIC (unsigned short)0x3412 // All our headers are prefixed with a magic number so we know they're ours

__inline__ __device__ static void writeBlockRef(fallocBlockRef *ref, fallocBlockHeader *block)
{
	ref->block = block;
	ref->blockid = gridDim.x*blockIdx.y + blockIdx.x;
	ref->threadid = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
}

__inline__ __device__ static void writeBlockHeader(fallocBlockHeader *hdr, unsigned short count)
{
	fallocBlockHeader header;
	header.magic = FALLOC_MAGIC;
	header.count = count;
	header.blockid = gridDim.x*blockIdx.y + blockIdx.x;
	header.threadid = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
	*hdr = header;
}

static __inline__ __device__ void *fallocGetBlock(fallocHeap *heap)
{
	// advance circular buffer
	fallocBlockRef *blockRefs = heap->blockRefs;
	size_t offset = atomicAdd((unsigned int *)&heap->freeBlockPtr, sizeof(fallocBlockRef)) - (size_t)blockRefs;
	offset %= heap->blockRefsLength;
	fallocBlockRef *blockRef = (fallocBlockRef *)((char *)blockRefs + offset);
	fallocBlockHeader *block = blockRef->block;
	writeBlockHeader(block, 1);
	blockRef->block = nullptr;
	return (void *)((char *)block + sizeof(fallocBlockHeader));
}

static __inline__ __device__ void fallocFreeBlock(fallocHeap *heap, void *obj)
{
	fallocBlockHeader *block = (fallocBlockHeader *)((char *)obj - sizeof(fallocBlockHeader));
	if (block->magic != FALLOC_MAGIC || block->count > 1) __THROW;// bad magic or not a singular block
	// advance circular buffer
	fallocBlockRef *blockRefs = heap->blockRefs;
	size_t offset = atomicAdd((unsigned int *)&heap->retnBlockPtr, sizeof(fallocBlockRef)) - (size_t)blockRefs;
	offset %= heap->blockRefsLength;
	writeBlockRef((fallocBlockRef *)((char *)blockRefs + offset), block);
	block->magic = 0;
}

/*
__device__ inline void *fallocGetBlocks(fallocHeap *heap, size_t length, size_t *allocLength = nullptr)
{
	if (threadIdx.x || threadIdx.y || threadIdx.z) __THROW;
	size_t blockSize = heap->blockSize;
	// fix up length to be a multiple of blockSize
	if (length % blockSize)
		length += blockSize - (length % blockSize);
	// set length, if requested
	if (allocLength)
		*allocLength = length - sizeof(fallocBlockHeader);
	size_t blocks = (size_t)(length / blockSize);
	if (blocks > heap->blocks) __THROW;
	// single, equals: fallocGetBlock
	if (blocks == 1)
		return fallocGetBlock(heap);
	// multiple, find a contiguous chuck
	size_t index = blocks;
	volatile fallocBlockHeader* block;
	volatile fallocBlockHeader* endBlock = (fallocBlockHeader*)((__int8*)heap + sizeof(fallocHeap) + (blockSize * heap->blocks));
	{ // critical
		for (block = (fallocBlockHeader*)((__int8*)heap + sizeof(fallocHeap)); index && block < endBlock; block = (fallocBlockHeader*)((__int8*)block + (blockSize * block->count)))
		{
			if (block->magic != FALLOC_MAGIC)
				__THROW;
			index = (block->next ? index - 1 : blocks);
		}
		if (index)
			return nullptr;
		// found chuck, remove from blockRefs
		endBlock = block;
		block = (fallocBlockHeader*)((__int8*)block - (blockSize * blocks));
		for (volatile fallocBlockHeader* chunk2 = heap->blockRefs; chunk2; chunk2 = chunk2->next)
			if (chunk2 >= block && chunk2 <= endBlock)
				chunk2->next = (chunk2->next ? chunk2->next->next : nullptr);
		block->count = blocks;
		block->next = nullptr;
	}
	return (void*)((__int8*)block + sizeof(fallocBlockHeader));
}


__device__ inline void fallocFreeBlocks(fallocHeap *heap, void *obj)
{
	volatile fallocBlockHeader* block = (fallocBlockHeader*)((__int8*)obj - sizeof(fallocBlockHeader));
	if (block->magic != FALLOC_MAGIC)
		__THROW;
	size_t blocks = block->count;
	// single, equals: fallocFreeChunk
	if (blocks == 1)
	{
		{ // critical
			block->next = heap->blockRefs;
			heap->blockRefs = block;
		}
		return;
	}
	// retag blocks
	size_t blockSize = heap->blockSize;
	block->count = 1;
	while (blocks-- > 1)
	{
		block = block->next = (fallocBlockHeader*)((__int8*)block + sizeof(fallocBlockHeader) + blockSize);
		block->magic = FALLOC_MAGIC;
		block->count = 1;
		block->reserved = nullptr;
	}
	{ // critical
		block->next = heap->blockRefs;
		heap->blockRefs = block;
	}
}
*/
#pragma endregion


//////////////////////
// CONTEXT
#pragma region CONTEXT

const static int FALLOCNODE_SLACK = 0x10;
#define FALLOCNODE_MAGIC (unsigned short)0x7856 // All our headers are prefixed with a magic number so we know they're ours
#define FALLOCCTX_MAGIC (unsigned short)0xCC56 // All our headers are prefixed with a magic number so we know they're ours

typedef struct __align__(8) _cuFallocNode
{
	struct _cuFallocNode *next;
	struct _cuFallocNode *nextAvailable;
	unsigned short freeOffset;
	unsigned short magic;
} fallocNode;

typedef struct __align__(8)
{
	fallocNode node;
	fallocNode *nodes;
	fallocNode *availableNodes;
	fallocHeap *heap;
	size_t blockSize;
	unsigned short magic;
} fallocCtx;

STATIC __device__ fallocCtx *fallocCreateCtx(fallocHeap *heap)
{
	size_t blockSize = heap->blockSize;
	if (sizeof(fallocCtx) > blockSize) __THROW;
	fallocCtx *ctx = (fallocCtx *)fallocGetBlock(heap);
	if (!ctx)
		return nullptr;
	ctx->node.magic = FALLOCNODE_MAGIC;
	ctx->node.next = nullptr;
	ctx->node.nextAvailable = nullptr;
	unsigned short freeOffset = ctx->node.freeOffset = sizeof(fallocCtx);
	ctx->nodes = (fallocNode *)ctx;
	ctx->availableNodes = (fallocNode *)ctx;
	ctx->heap = heap;
	ctx->blockSize = heap->blockSize;
	ctx->magic = FALLOCCTX_MAGIC;
	// close node
	if (freeOffset + FALLOCNODE_SLACK > blockSize)
		ctx->availableNodes = nullptr;
	return ctx;
}

STATIC __device__ void fallocDisposeCtx(fallocCtx *ctx)
{
	fallocHeap *heap = ctx->heap;
	for (fallocNode *node = ctx->nodes; node; node = node->next)
		fallocFreeBlock(heap, node);
}

STATIC __device__ void *falloc(fallocCtx *ctx, unsigned short bytes, bool alloc = true)
{
	if (bytes > (ctx->blockSize - sizeof(fallocCtx))) __THROW;
	// find or add available node
	fallocNode *node;
	unsigned short freeOffset;
	unsigned char hasFreeSpace;
	fallocNode *lastNode;
	for (lastNode = (fallocNode *)ctx, node = ctx->availableNodes; node; lastNode = node, node = (alloc ? node->nextAvailable : node->next))
		if (hasFreeSpace = ((freeOffset = node->freeOffset + bytes) <= ctx->blockSize))
			break;
	if (!node || !hasFreeSpace) {
		// add node
		node = (fallocNode *)fallocGetBlock(ctx->heap);
		if (!node) __THROW;
		node->magic = FALLOCNODE_MAGIC;
		node->next = ctx->nodes; ctx->nodes = node;
		node->nextAvailable = (alloc ? ctx->availableNodes : nullptr); ctx->availableNodes = node;
		freeOffset = node->freeOffset = sizeof(fallocNode); 
		freeOffset += bytes;
	}
	//
	void *obj = (char *)node + node->freeOffset;
	node->freeOffset = freeOffset;
	// close node
	if (alloc && (freeOffset + FALLOCNODE_SLACK > ctx->blockSize)) {
		if (lastNode == (fallocNode *)ctx)
			ctx->availableNodes = node->nextAvailable;
		else
			lastNode->nextAvailable = node->nextAvailable;
		node->nextAvailable = nullptr;
	}
	return obj;
}

STATIC __device__ void *fallocRetract(fallocCtx *ctx, unsigned short bytes)
{
	fallocNode *node = ctx->availableNodes;
	int freeOffset = (int)node->freeOffset - bytes;
	// multi node, retract node
	if (node != &ctx->node && freeOffset < sizeof(fallocNode)) {
		node->freeOffset = sizeof(fallocNode);
		// search for previous node
		fallocNode *lastNode;
		for (lastNode = (fallocNode *)ctx, node = ctx->nodes; node; lastNode = node, node = node->next)
			if (node == ctx->availableNodes)
				break;
		node = ctx->availableNodes = lastNode;
		freeOffset = (int)node->freeOffset - bytes;
	}
	// first node && !overflow
	if (node == &ctx->node && freeOffset < sizeof(fallocCtx)) __THROW;
	node->freeOffset = (unsigned short)freeOffset;
	return (char *)node + freeOffset;
}

static __inline__ __device__ void fallocMark(fallocCtx *ctx, void *&mark, unsigned short &mark2)
{
	mark = ctx->availableNodes; mark2 = ctx->availableNodes->freeOffset;
}

static __inline__ __device__ bool fallocAtMark(fallocCtx *ctx, void *mark, unsigned short mark2)
{
	return (mark == ctx->availableNodes && mark2 == ctx->availableNodes->freeOffset);
}

template <typename T> __device__ T* falloc(fallocCtx *ctx) { return (T *)falloc(ctx, sizeof(T), true); }
template <typename T> __device__ void fallocPush(fallocCtx *ctx, T t) { *((T *)falloc(ctx, sizeof(T), false)) = t; }
template <typename T> __device__ T fallocPop(fallocCtx *ctx) { return *((T *)fallocRetract(ctx, sizeof(T))); }

#pragma endregion


///////////////////////////////////////////////////////////////////////////////
// VISUAL
#pragma region VISUAL
#ifdef VISUAL
#include "Falloc.h"

#define BLOCKPITCH 64
#define HEADERPITCH 4
#define BLOCKREFCOLOR make_float4(.7, 0, 0, 1)
#define BLOCKREF2COLOR make_float4(1, 0, 0, 1)
#define HEADERCOLOR make_float4(0, 1, 0, 1)
#define BLOCKCOLOR make_float4(0, 0, .7, 1)
#define BLOCK2COLOR make_float4(0, 0, 1, 1)
#define BLOCK3COLOR make_float4(0, .4, 1, 1)
#define BLOCK4COLOR make_float4(0, .7, 1, 1)
#define BLOCK5COLOR make_float4(.5, .7, 1, 1)
#define MARKERCOLOR make_float4(1, 1, 0, 1)

#define MAX(a,b) (a > b ? a : b)

static __inline__ __device__ void OffsetBlockRef(unsigned int x, unsigned int y, float *x1, float *y1)
{
	x += (y % HEADERPITCH) * BLOCKPITCH; 
	y /= HEADERPITCH;
	*x1 = x * 1; *y1 = y * 1 + 1;
}

static __global__ void RenderHeap(quad4 *b, fallocHeap *heap, unsigned int offset)
{
	int index = offset;
	// heap
	b[index] = make_quad4(
		make_float4(00, 1, 1, 1), HEADERCOLOR,
		make_float4(10, 1, 1, 1), HEADERCOLOR,
		make_float4(10, 0, 1, 1), HEADERCOLOR,
		make_float4(00, 0, 1, 1), HEADERCOLOR);
	// free
	float x1, y1;
	if (heap->freeBlockPtr)
	{
		size_t offset = ((char *)heap->freeBlockPtr - (char *)heap->blockRefs);
		offset %= heap->blockRefsLength;
		offset /= sizeof(fallocBlockRef);
		//
		OffsetBlockRef(offset % BLOCKPITCH, offset / BLOCKPITCH, &x1, &y1);
		b[index + 1] = make_quad4(
			make_float4(x1 + .0, y1 + .2, 1, 1), MARKERCOLOR,
			make_float4(x1 + .2, y1 + .2, 1, 1), MARKERCOLOR,
			make_float4(x1 + .2, y1 + .0, 1, 1), MARKERCOLOR,
			make_float4(x1 + .0, y1 + .0, 1, 1), MARKERCOLOR);
	}
	// retn
	if (heap->retnBlockPtr)
	{
		size_t offset = ((char *)heap->retnBlockPtr - (char *)heap->blockRefs);
		offset %= heap->blockRefsLength;
		offset /= sizeof(fallocBlockRef);
		//
		OffsetBlockRef(offset % BLOCKPITCH, offset / BLOCKPITCH, &x1, &y1);
		b[index + 2] = make_quad4(
			make_float4(x1 + .7, y1 + .9, 1, 1), MARKERCOLOR,
			make_float4(x1 + .9, y1 + .9, 1, 1), MARKERCOLOR,
			make_float4(x1 + .9, y1 + .7, 1, 1), MARKERCOLOR,
			make_float4(x1 + .7, y1 + .7, 1, 1), MARKERCOLOR);
	}
}

static __global__ void RenderBlock(quad4 *b, size_t blocks, unsigned int blocksY, fallocHeap *heap, unsigned int offset)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	int blockIndex = y * BLOCKPITCH + x;
	if (blockIndex >= blocks)
		return;
	fallocBlockRef *ref = (fallocBlockRef *)((char *)heap->blockRefs + blockIndex * sizeof(fallocBlockRef));
	fallocBlockHeader *hdr = (fallocBlockHeader *)(heap->blocks + blockIndex * heap->blockSize);
	int index = blockIndex * 4 + offset;
	//
	float x1, y1; OffsetBlockRef(x, y, &x1, &y1);
	if (ref->block == nullptr)
	{
		b[index] = make_quad4(
			make_float4(x1 + 0.0, y1 + 0.9, 1, 1), BLOCKREFCOLOR,
			make_float4(x1 + 0.9, y1 + 0.9, 1, 1), BLOCKREFCOLOR,
			make_float4(x1 + 0.9, y1 + 0.0, 1, 1), BLOCKREFCOLOR,
			make_float4(x1 + 0.0, y1 + 0.0, 1, 1), BLOCKREFCOLOR);
	}
	else
	{
		b[index] = make_quad4(
			make_float4(x1 + 0.0, y1 + 0.9, 1, 1), BLOCKREF2COLOR,
			make_float4(x1 + 0.9, y1 + 0.9, 1, 1), BLOCKREF2COLOR,
			make_float4(x1 + 0.9, y1 + 0.0, 1, 1), BLOCKREF2COLOR,
			make_float4(x1 + 0.0, y1 + 0.0, 1, 1), BLOCKREF2COLOR);
	}
	// block
	float x2 = x * 10; float y2 = y * 20 + (blocksY / HEADERPITCH) + 3;
	if (hdr->magic != FALLOC_MAGIC)
	{
		b[index + 1] = make_quad4(
			make_float4(x2 + 0, y2 + 19, 1, 1), BLOCKCOLOR,
			make_float4(x2 + 9, y2 + 19, 1, 1), BLOCKCOLOR,
			make_float4(x2 + 9, y2 + 00, 1, 1), BLOCKCOLOR,
			make_float4(x2 + 0, y2 + 00, 1, 1), BLOCKCOLOR);
	} 
	else 
	{
		b[index + 1] = make_quad4(
			make_float4(x2 + 0, y2 + 1, 1, 1), HEADERCOLOR,
			make_float4(x2 + 3.9, y2 + 1, 1, 1), HEADERCOLOR,
			make_float4(x2 + 3.9, y2 + 0, 1, 1), HEADERCOLOR,
			make_float4(x2 + 0, y2 + 0, 1, 1), HEADERCOLOR);
		// block or node
		fallocCtx *ctx = (fallocCtx *)((char *)hdr + sizeof(fallocBlockHeader));
		if (ctx->node.magic != FALLOCNODE_MAGIC)
		{
			b[index + 2] = make_quad4(
				make_float4(x2 + 0, y2 + 19, 1, 1), BLOCK2COLOR,
				make_float4(x2 + 9, y2 + 19, 1, 1), BLOCK2COLOR,
				make_float4(x2 + 9, y2 + 00, 1, 1), BLOCK2COLOR,
				make_float4(x2 + 0, y2 + 00, 1, 1), BLOCK2COLOR);
		}
		else 
		{
			float percent = .303; //ceilf((heap->blockSize / ctx->node.freeOffset) * 100) / 100;
			float split = 19 * percent;
			if (ctx->magic != FALLOCCTX_MAGIC)
			{
				b[index + 2] = make_quad4(
					make_float4(x2 + 0, y2 + split, 1, 1), BLOCK4COLOR,
					make_float4(x2 + 9, y2 + split, 1, 1), BLOCK4COLOR,
					make_float4(x2 + 9, y2 + 00, 1, 1), BLOCK4COLOR,
					make_float4(x2 + 0, y2 + 00, 1, 1), BLOCK4COLOR);
			}
			else
			{
				b[index + 2] = make_quad4(
					make_float4(x2 + 0, y2 + split, 1, 1), BLOCK5COLOR,
					make_float4(x2 + 9, y2 + split, 1, 1), BLOCK5COLOR,
					make_float4(x2 + 9, y2 + 00, 1, 1), BLOCK5COLOR,
					make_float4(x2 + 0, y2 + 00, 1, 1), BLOCK5COLOR);
			}
			b[index + 3] = make_quad4(
				make_float4(x2 + 0, y2 + 19, 1, 1), BLOCK3COLOR,
				make_float4(x2 + 9, y2 + 19, 1, 1), BLOCK3COLOR,
				make_float4(x2 + 9, y2 + split, 1, 1), BLOCK3COLOR,
				make_float4(x2 + 0, y2 + split, 1, 1), BLOCK3COLOR);
		}
	}
}

static int GetFallocRenderQuads(size_t blocks)
{ 
	return 3 + (blocks * 4);
}

static void LaunchFallocRender(float4 *b, size_t blocks, fallocHeap *heap)
{
	dim3 heapBlock(1, 1, 1);
	dim3 heapGrid(1, 1, 1);
	RenderHeap<<<heapGrid, heapBlock>>>((quad4 *)b, heap, 0);
	//
	dim3 blockBlock(16, 16, 1);
	dim3 blockGrid(MAX(BLOCKPITCH / 16, 1), MAX(blocks / BLOCKPITCH / 16, 1), 1);
	RenderBlock<<<blockGrid, blockBlock>>>((quad4 *)b, blocks, blocks / BLOCKPITCH, heap, 3);
}

// _vbo variables
static GLuint _vbo;
static GLsizei _vboSize;
static struct cudaGraphicsResource *_vboResource;

static void RunCuda(size_t blocks, fallocHeap *heap, struct cudaGraphicsResource **resource)
{
	// map OpenGL buffer object for writing from CUDA
	checkCudaErrors(cudaGraphicsMapResources(1, resource, nullptr), exit(0));
	float4 *b;
	size_t size;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&b, &size, *resource), exit(0));
	//printf("CUDA mapped VBO: May access %ld bytes\n", size);
	LaunchFallocRender(b, blocks, heap);
	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, resource, nullptr), exit(0));
}

static void CreateVBO(size_t blocks, GLuint *vbo, struct cudaGraphicsResource **resource, unsigned int vbo_res_flags)
{
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	// initialize buffer object
	_vboSize = GetFallocRenderQuads(blocks) * 4;
	unsigned int size = _vboSize * 2 * sizeof(float4);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(resource, *vbo, vbo_res_flags), exit(0));
	SDK_CHECK_ERROR_GL();
}

static void DeleteVBO(GLuint *vbo, struct cudaGraphicsResource *resource)
{
	// unregister this buffer object with CUDA
	cudaGraphicsUnregisterResource(resource);
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	*vbo = 0;
}

void FallocVisualRender::Dispose()
{
	if (_vbo)
		DeleteVBO(&_vbo, _vboResource);
}

extern void LaunchFallocKeypress(fallocHeap *heap, unsigned char key);
void FallocVisualRender::Keyboard(unsigned char key)
{
	switch (key)
	{
	case 'a':
	case 'b':
	case 'x':
	case 'y':
	case 'z':
		LaunchFallocKeypress((fallocHeap *)_fallocHost.heap, key);
		break;
	}
}

void FallocVisualRender::Display()
{
	size_t blocks = _fallocHost.blocksLength / _fallocHost.blockSize;
	// run CUDA kernel to generate vertex positions
	RunCuda(blocks, (fallocHeap *)_fallocHost.heap, &_vboResource);

	//gluLookAt(0, 0, 200, 0, 0, 0, 0, 1, 0);
	//glScalef(.02, .02, .02);
	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(Visual::TranslateX, Visual::TranslateY, Visual::TranslateZ);
	glRotatef(Visual::RotateX, 1.0, 0.0, 0.0);
	glRotatef(Visual::RotateY, 0.0, 1.0, 0.0);

	// render from the _vbo
	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glVertexPointer(4, GL_FLOAT, sizeof(float4) * 2, 0);
	glColorPointer(4, GL_FLOAT, sizeof(float4) * 2, (GLvoid*)sizeof(float4));

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_QUADS, 0, _vboSize);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void FallocVisualRender::Initialize()
{
	size_t blocks = _fallocHost.blocksLength / _fallocHost.blockSize;
	// create VBO
	CreateVBO(blocks, &_vbo, &_vboResource, cudaGraphicsMapFlagsWriteDiscard);
	// run the cuda part
	RunCuda(blocks, (fallocHeap *)_fallocHost.heap, &_vboResource);
}

#endif
#pragma endregion