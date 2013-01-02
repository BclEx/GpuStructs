#include <cstdio>
#include <cassert>
#include "..\Runtime\Cuda.h"
#include "..\Runtime\CudaGL.h"

// constants
#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD         0.30f
#define REFRESH_DELAY     10 //ms
const unsigned int WindowWidth = 512;
const unsigned int WindowHeight = 512;
const unsigned int MeshWidth = 256;
const unsigned int MeshHeight = 256;

// _vbo variables
GLuint _vbo;
struct cudaGraphicsResource *_vboResource;
float _anim = 0.0;
// mouse controls
int _mouseLastX, _mouseLastY;
int _mouseState = 0;
float _rotateX = 0.0, _rotateY = 0.0;
float _translateZ = -3.0;
//StopWatchInterface *timer = NULL;
int _fpsCount = 0; // FPS count for averaging
int _fpsLimit = 1; // FPS limit for sampling
float _avgFps = 0.0f;

extern void launch_kernel(float4 *pos, unsigned int MeshWidth, unsigned int MeshHeight, float time);
void runCuda(struct cudaGraphicsResource **resource)
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

void createVBO(GLuint *vbo, struct cudaGraphicsResource **resource, unsigned int vbo_res_flags)
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

void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *resource)
{
	// unregister this buffer object with CUDA
	cudaGraphicsUnregisterResource(resource);

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

void computeFPS()
{
	_fpsCount++;
	if (_fpsCount == _fpsLimit)
	{
		_avgFps = 1.f; //1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		_fpsCount = 0;
		_fpsLimit = (int)(_avgFps > 1.f ? _avgFps : 1.f);
		//sdkResetTimer(&timer);
	}
	char fps[256];
	sprintf_s(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", _avgFps);
	glutSetWindowTitle(fps);
}

void display()
{
	//sdkStartTimer(&timer);

	// run CUDA kernel to generate vertex positions
	runCuda(&_vboResource);

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

	//sdkStopTimer(&timer);
	computeFPS();
}

void timerEvent(int value)
{
	glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void cleanup()
{
	//sdkDeleteTimer(&timer);
	if (_vbo)
		deleteVBO(&_vbo, _vboResource);
}

void keyboard(unsigned char key, int, int)
{
	switch (key)
	{
	case 27:
		exit(1);
		break;
	}
}

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
		_mouseState |= 1 << button;
	else if (state == GLUT_UP)
		_mouseState = 0;
	_mouseLastX = x;
	_mouseLastY = y;
}

void motion(int x, int y)
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

bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(WindowWidth, WindowHeight);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, WindowWidth, WindowHeight);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)WindowWidth / (GLfloat)WindowHeight, 0.1, 10.0);

    SDK_CHECK_ERROR_GL();
    return true;
}


void main(int argc, char **argv)
{
	// First initialize OpenGL context, so we can properly set the GL for CUDA. This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (!initGL(&argc, argv))
		return;
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	// create VBO
	createVBO(&_vbo, &_vboResource, cudaGraphicsMapFlagsWriteDiscard);

	// run the cuda part
	runCuda(&_vboResource);

	// start rendering mainloop
	glutMainLoop();

	//
	atexit(cleanup);
	cudaDeviceReset();
	printf("End.");
	char c; scanf_s("%c", &c);
}
