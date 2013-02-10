#include "..\Runtime\Cuda.h"
#include "..\Runtime\CudaGL.h"
#include "..\Runtime\Falloc.h"

// constants
#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD         0.30f
#define REFRESH_DELAY     10 //ms
const unsigned int WindowWidth = 800;
const unsigned int WindowHeight = 600;

class IVisualRender
{
public:
	virtual void Dispose() = 0;
	virtual void Display() = 0;
	virtual void Keyboard(unsigned char key) = 0;
	virtual void Initialize() = 0;
};

class FallocVisualRender : public IVisualRender
{
private:
	cudaFallocHost _fallocHost;
public:
	FallocVisualRender(cudaFallocHost fallocHost)
		: _fallocHost(fallocHost) { }
	virtual void Dispose();
	virtual void Keyboard(unsigned char key);
	virtual void Display();
	virtual void Initialize();
};

class Visual
{
public:
	static float RotateX;
	static float RotateY;
	static float TranslateX;
	static float TranslateY;
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
	//inline ~Visual() { Dispose(); }
	inline static void Dispose()
	{
		//sdkDeleteTimer(&_timer);
		if (_render)
		{
			_render->Dispose();
			delete _render; _render = nullptr;
		}
	}
	static void Main();
	inline static void InitState()
	{
		// mouse controls
		_mouseState = 0;
		RotateX = 180.0, RotateY = 0.0;
		//TranslateX = -200, TranslateY = 150.0, TranslateZ = -200.0;
		TranslateX = -30, TranslateY = 20.0, TranslateZ = -50.0;
		//TranslateX = 0, TranslateY = 0.0, TranslateZ = -8.0;
		//
		//_timer = nullptr;
		_fpsCount = 0; // FPS count for averaging
		_fpsLimit = 1; // FPS limit for sampling
		_avgFps = 0.0f;
	}
	inline static bool InitGL(IVisualRender* render, int *argc, char **argv)
	{
		_render = render;
		InitState();
		glutInit(argc, argv);
		int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
		int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);
		glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
		glutInitWindowPosition((screenWidth - WindowWidth) / 2, (screenHeight - WindowHeight) / 2);
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
		if (!glewIsSupported("GL_VERSION_2_0"))
		{
			fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
			fflush(stderr);
			return false;
		}

		// default initialization
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glEnable(GL_DEPTH_TEST);
		glPointSize(5);

		// viewport
		glViewport(0, 0, WindowWidth, WindowHeight);

		// projection
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(80.0, (GLfloat)WindowWidth / (GLfloat)WindowHeight, 0.01, 500.0);

		SDK_CHECK_ERROR_GL();
		return true;
	}
};
