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

class IVisualRender
{
public:
	virtual void Dispose() = 0;
	virtual void Display() = 0;
	virtual void Initialize() = 0;
};

class FallocVisualRender : public IVisualRender
{
public:
	virtual void Dispose();
	virtual void Display();
	virtual void Initialize();
};

class Visual
{
public:
	static float RotateX;
	static float RotateY;
	static float TranslateZ;

protected:
	static IVisualRender* _render;
	// mouse controls
	static int _mouseLastX;
	static int _mouseLastY;
	static int _mouseState;
	//StopWatchInterface *_timer;
	static int _fpsCount; // FPS count for averaging
	static int _fpsLimit; // FPS limit for sampling
	static float _avgFps;

	static void Display();
	static void Keyboard(unsigned char key, int, int);
	static void Mouse(int button, int state, int x, int y);
	static void Motion(int x, int y);
	static void TimerEvent(int value);
	inline static void ComputeFPS()
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

public:
	static void Dispose();
	static void Main();
	inline static void InitState()
	{
		// mouse controls
		_mouseState = 0;
		RotateX = 0.0, RotateY = 0.0;
		TranslateZ = -3.0;
		//
		//_timer = nullptr;
		_fpsCount = 0; // FPS count for averaging
		_fpsLimit = 1; // FPS limit for sampling
		_avgFps = 0.0f;
		//
	}
	inline static bool InitGL(IVisualRender* render, int *argc, char **argv)
	{
		_render = render;
		InitState();
		glutInit(argc, argv);
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
		glutInitWindowSize(WindowWidth, WindowHeight);
		glutCreateWindow("Cuda GL Interop (VBO)");
		// register callbacks
		glutDisplayFunc(Display);
		glutKeyboardFunc(Keyboard);
		glutMouseFunc(Mouse);
		glutMotionFunc(Motion);
		glutTimerFunc(REFRESH_DELAY, TimerEvent, 0);

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
};
