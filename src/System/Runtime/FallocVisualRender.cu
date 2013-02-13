#include <cstdio>
#include <cassert>
#include "Visual.h"

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

#define FALLOC_MAGIC (unsigned short)0x3412 // All our headers are prefixed with a magic number so we know they're ours

//////////////////////
// CONTEXT

#define FALLOCNODE_MAGIC (unsigned short)0x7856 // All our headers are prefixed with a magic number so we know they're ours

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
} fallocCtx;


///////////////////////////////////////////////////////////////////////////////
// VISUAL
#pragma region VISUAL

#define BLOCKPITCH 64
#define HEADERPITCH 4
#define BLOCKREFCOLOR make_float4(.7, 0, 0, 1)
#define BLOCKREF2COLOR make_float4(1, 0, 0, 1)
#define HEADERCOLOR make_float4(0, 1, 0, 1)
#define BLOCKCOLOR make_float4(0, 0, .7, 1)
#define BLOCK2COLOR make_float4(0, 0, 1, 1)
#define BLOCK3COLOR make_float4(0, .5, 1, 1)
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
	int index = blockIndex * 3 + offset;
	//
	float x1, y1; OffsetBlockRef(x, y, &x1, &y1);
	if (!ref) { }
	else if (ref->block == nullptr)
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
	if (!hdr) { }
	else if (hdr->magic != FALLOC_MAGIC)
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
		if (!ctx || ctx->node.magic != FALLOCNODE_MAGIC)
		{
			b[index + 2] = make_quad4(
				make_float4(x2 + 0, y2 + 19, 1, 1), BLOCK2COLOR,
				make_float4(x2 + 9, y2 + 19, 1, 1), BLOCK2COLOR,
				make_float4(x2 + 9, y2 + 00, 1, 1), BLOCK2COLOR,
				make_float4(x2 + 0, y2 + 00, 1, 1), BLOCK2COLOR);
		}
		else 
		{
			b[index + 2] = make_quad4(
				make_float4(x2 + 0, y2 + 19, 1, 1), BLOCK3COLOR,
				make_float4(x2 + 9, y2 + 19, 1, 1), BLOCK3COLOR,
				make_float4(x2 + 9, y2 + 00, 1, 1), BLOCK3COLOR,
				make_float4(x2 + 0, y2 + 00, 1, 1), BLOCK3COLOR);
		}
	}
}

static int GetFallocRenderQuads(size_t blocks)
{ 
	return 3 + (blocks * 3);
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

#pragma endregion


///////////////////////////////////////////////////////////////////////////////
// VISUALRENDERER
#pragma region VISUALRENDERER

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
	assert(vbo);
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

#pragma endregion
