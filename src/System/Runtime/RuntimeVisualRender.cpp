#include <cstdio>
#include <cassert>
#include "Visual.h"

// _vbo variables
GLuint _vbo;
GLsizei _vboSize;
struct cudaGraphicsResource *_vboResource;

extern int GetRenderQuads(size_t blocks);
extern void LaunchRender(float4 *b, size_t blocks, fallocHeap* heap);

void RunCuda(size_t blocks, fallocHeap* heap, struct cudaGraphicsResource **resource)
{
	// map OpenGL buffer object for writing from CUDA
	checkCudaErrors(cudaGraphicsMapResources(1, resource, nullptr), exit(0));
	float4 *b;
	size_t size;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&b, &size, *resource), exit(0));
	//printf("CUDA mapped VBO: May access %ld bytes\n", size);
	LaunchRender(b, blocks, heap);
	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, resource, nullptr), exit(0));
}

void CreateVBO(size_t blocks, GLuint *vbo, struct cudaGraphicsResource **resource, unsigned int vbo_res_flags)
{
	assert(vbo);
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	// initialize buffer object
	_vboSize = GetRenderQuads(blocks) * 4;
	unsigned int size = _vboSize * 2 * sizeof(float4);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(resource, *vbo, vbo_res_flags), exit(0));
	SDK_CHECK_ERROR_GL();
}

void DeleteVBO(GLuint *vbo, struct cudaGraphicsResource *resource)
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

void FallocVisualRender::Keyboard(unsigned char key)
{
	switch (key)
	{
	case 'a':
		break;
	}
}

void FallocVisualRender::Display()
{
	// run CUDA kernel to generate vertex positions
	RunCuda(_fallocHost.blocks, _fallocHost.heap, &_vboResource);

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
	// create VBO
	CreateVBO(_fallocHost.blocks, &_vbo, &_vboResource, cudaGraphicsMapFlagsWriteDiscard);

	// run the cuda part
	RunCuda(_fallocHost.blocks, _fallocHost.heap, &_vboResource);
}

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