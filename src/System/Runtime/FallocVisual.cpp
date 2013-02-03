#include <cstdio>
#include <cassert>
#include "FallocVisual.h"

// _vbo variables
GLuint _vbo;
struct cudaGraphicsResource *_vboResource;
float _anim = 0.0;

extern void launch_kernel(float4 *pos, unsigned int MeshWidth, unsigned int MeshHeight, float time);
void RunCuda(struct cudaGraphicsResource **resource)
{
	// map OpenGL buffer object for writing from CUDA
	checkCudaErrors(cudaGraphicsMapResources(1, resource, 0), exit(0));
	float4 *devPtr;
	size_t size;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&devPtr, &size, *resource), exit(0));
	//printf("CUDA mapped VBO: May access %ld bytes\n", size);
	launch_kernel(devPtr, MeshWidth, MeshHeight, _anim);
	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, resource, 0), exit(0));
}

void CreateVBO(GLuint *vbo, struct cudaGraphicsResource **resource, unsigned int vbo_res_flags)
{
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = MeshWidth * MeshHeight * 4 * sizeof(float);
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

int Visual::_mouseLastX;
int Visual::_mouseLastY;
int Visual::_mouseState;
float Visual::_rotateX;
float Visual::_rotateY;
float Visual::_translateZ;
//StopWatchInterface *Visual::_timer;
int Visual::_fpsCount; // FPS count for averaging
int Visual::_fpsLimit; // FPS limit for sampling
float Visual::_avgFps;

void Visual::Dispose()
{
	//sdkDeleteTimer(&_timer);
	if (_vbo)
		DeleteVBO(&_vbo, _vboResource);
}

void Visual::Display()
{
	//sdkStartTimer(&_timer);

	// run CUDA kernel to generate vertex positions
	RunCuda(&_vboResource);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, _translateZ);
	glRotatef(_rotateX, 1.0, 0.0, 0.0);
	glRotatef(_rotateY, 0.0, 1.0, 0.0);

	// render from the _vbo
	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_POINTS, 0, MeshWidth * MeshHeight);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();

	_anim += 0.01f;

	//sdkStopTimer(&_timer);
	ComputeFPS();
}

void Visual::Keyboard(unsigned char key, int, int)
{
	switch (key)
	{
	case 27:
		exit(1);
		break;
	}
}

void Visual::Mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
		_mouseState |= 1 << button;
	else if (state == GLUT_UP)
		_mouseState = 0;
	_mouseLastX = x;
	_mouseLastY = y;
}

void Visual::Motion(int x, int y)
{
	float dx = (float)(x - _mouseLastX);
	float dy = (float)(y - _mouseLastY);
	if (_mouseState & 1)
	{
		_rotateX += dy * 0.2f;
		_rotateY += dx * 0.2f;
	}
	else if (_mouseState & 4)
		_translateZ += dy * 0.01f;
	_mouseLastX = x;
	_mouseLastY = y;
}

void Visual::TimerEvent(int value)
{
	glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, TimerEvent, 0);
}

void Visual::Main()
{
	// create VBO
	CreateVBO(&_vbo, &_vboResource, cudaGraphicsMapFlagsWriteDiscard);

	// run the cuda part
	RunCuda(&_vboResource);

	// start rendering mainloop
	glutMainLoop();
}