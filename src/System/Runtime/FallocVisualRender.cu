#include <cstdio>
#include <cassert>
#include "Visual.h"

typedef struct __align__(8)
{
	unsigned short magic;		// magic number says we're valid
	unsigned short count;
	unsigned short blockid;		// block ID of author
	unsigned short threadid;	// thread ID of author
} fallocBlockHeader;

typedef struct _cuFallocBlockRef
{
	fallocBlockHeader *b;
	struct _cuFallocBlockRef *next;
} fallocBlockRef;

typedef struct __align__(8)
{
	void *reserved;
	size_t blockSize;
	size_t blocks;
	size_t blockRefsLength; // Size of circular buffer (set up by host)
	fallocBlockRef *blockRefs; // Start of circular buffer (set up by host)
	volatile fallocBlockHeader **freeBlockPtr; // Current atomically-incremented non-wrapped offset
	volatile fallocBlockHeader **retnBlockPtr; // Current atomically-incremented non-wrapped offset
} fallocHeap;


///////////////////////////////////////////////////////////////////////////////
// VISUAL
#pragma region VISUAL

#define BLOCKPITCH 64
#define HEADERPITCH 4
#define BLOCKREFCOLOR make_float4(1, 0, 0, 1)
#define HEADERCOLOR make_float4(0, 1, 0, 1)
#define BLOCKCOLOR make_float4(0, 0, 1, 1)
#define BLOCK2COLOR make_float4(0, 0, .8, 1)
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
	int index = blockIndex * 3 + offset;
	//
	float x1, y1; OffsetBlockRef(x, y, &x1, &y1);
	b[index] = make_quad4(
		make_float4(x1 + 0.0, y1 + 0.9, 1, 1), BLOCKREFCOLOR,
		make_float4(x1 + 0.9, y1 + 0.9, 1, 1), BLOCKREFCOLOR,
		make_float4(x1 + 0.9, y1 + 0.0, 1, 1), BLOCKREFCOLOR,
		make_float4(x1 + 0.0, y1 + 0.0, 1, 1), BLOCKREFCOLOR);
	// block
	float x2 = x * 10;
	float y2 = y * 20 + (blocksY / HEADERPITCH) + 3;
	b[index + 1] = make_quad4(
		make_float4(x2 + 0, y2 + 1, 1, 1), HEADERCOLOR,
		make_float4(x2 + 3.9, y2 + 1, 1, 1), HEADERCOLOR,
		make_float4(x2 + 3.9, y2 + 0, 1, 1), HEADERCOLOR,
		make_float4(x2 + 0, y2 + 0, 1, 1), HEADERCOLOR);
	b[index + 2] = make_quad4(
		make_float4(x2 + 0, y2 + 19, 1, 1), BLOCKCOLOR,
		make_float4(x2 + 9, y2 + 19, 1, 1), BLOCKCOLOR,
		make_float4(x2 + 9, y2 + 00, 1, 1), BLOCKCOLOR,
		make_float4(x2 + 0, y2 + 00, 1, 1), BLOCKCOLOR);
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
	// run CUDA kernel to generate vertex positions
	RunCuda(_fallocHost.blocks, (fallocHeap *)_fallocHost.heap, &_vboResource);

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

extern void LaunchFallocInit(fallocHeap *heap);
void FallocVisualRender::Initialize()
{
	LaunchFallocInit((fallocHeap *)_fallocHost.heap);

	// create VBO
	CreateVBO(_fallocHost.blocks, &_vbo, &_vboResource, cudaGraphicsMapFlagsWriteDiscard);

	// run the cuda part
	RunCuda(_fallocHost.blocks, (fallocHeap *)_fallocHost.heap, &_vboResource);
}

#pragma endregion


#pragma region junk
/*
/////////////////////////////
///
//struct vertex4 { float4 v, c; };
#define MEMBER_OFFSET(s,m) ((char *)NULL + (offsetof(s,m)))
#define BUFFER_OFFSET(i) ((char *)NULL + (i))
void RenderCreate();
void Render();
void RenderDispose();


struct VertexXYZColor
{
float3 m_Pos;
float3 m_Color;
};

VertexXYZColor g_Vertices[8] = {
{ make_float3(  1,  1,  1 ), make_float3( 1, 1, 1 ) }, // 0
{ make_float3( -1,  1,  1 ), make_float3( 0, 1, 1 ) }, // 1
{ make_float3( -1, -1,  1 ), make_float3( 0, 0, 1 ) }, // 2
{ make_float3(  1, -1,  1 ), make_float3( 1, 0, 1 ) }, // 3
{ make_float3(  1, -1, -1 ), make_float3( 1, 0, 0 ) }, // 4
{ make_float3( -1, -1, -1 ), make_float3( 0, 0, 0 ) }, // 5
{ make_float3( -1,  1, -1 ), make_float3( 0, 1, 0 ) }, // 6
{ make_float3(  1,  1, -1 ), make_float3( 1, 1, 0 ) }, // 7
};

GLuint g_Indices[24] = {
0, 1, 2, 3,                 // Front face
7, 4, 5, 6,                 // Back face
6, 5, 2, 1,                 // Left face
7, 0, 3, 4,                 // Right face
7, 6, 1, 0,                 // Top face
3, 2, 5, 4,                 // Bottom face
};

GLuint g_uiVerticesVBO = 0;
GLuint g_uiIndicesVBO = 0;

void RenderCreate()
{
glGenBuffers(1, &g_uiVerticesVBO);
glGenBuffers(1, &g_uiIndicesVBO);

// Copy the vertex data to the VBO
glBindBuffer( GL_ARRAY_BUFFER, g_uiVerticesVBO );
glBufferData( GL_ARRAY_BUFFER, sizeof(g_Vertices), g_Vertices, GL_STATIC_DRAW);
glBindBuffer( GL_ARRAY_BUFFER, 0 );

// Copy the index data to the VBO
glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, g_uiIndicesVBO );
glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof(g_Indices), g_Indices, GL_STATIC_DRAW);
glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
}
void Render()
{

// We need to enable the client stats for the vertex attributes we want 
// to render even if we are not using client-side vertex arrays.
glEnableClientState(GL_VERTEX_ARRAY);
glEnableClientState(GL_COLOR_ARRAY);

// Bind the vertices's VBO
glBindBuffer( GL_ARRAY_BUFFER, g_uiVerticesVBO );
glVertexPointer( 3, GL_FLOAT, sizeof(VertexXYZColor), MEMBER_OFFSET(VertexXYZColor,m_Pos) );
glColorPointer( 3, GL_FLOAT, sizeof(VertexXYZColor), MEMBER_OFFSET(VertexXYZColor,m_Color) );

// Bind the indices's VBO
//glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, g_uiIndicesVBO );
//glDrawElements( GL_QUADS, 24, GL_UNSIGNED_INT,  BUFFER_OFFSET( 0 ) );

glDrawArrays(GL_QUADS, 0, 8);

// Unbind buffers so client-side vertex arrays still work.
glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
glBindBuffer( GL_ARRAY_BUFFER, 0 );

// Disable the client side arrays again.
glDisableClientState(GL_VERTEX_ARRAY);
glDisableClientState(GL_COLOR_ARRAY);
}

void RenderDispose()
{
if ( g_uiIndicesVBO != 0 )
{
glDeleteBuffersARB( 1, &g_uiIndicesVBO );
g_uiIndicesVBO = 0;
}
if ( g_uiVerticesVBO != 0 )
{
glDeleteBuffersARB( 1, &g_uiVerticesVBO );
g_uiVerticesVBO = 0;
}
}
*/
#pragma endregion