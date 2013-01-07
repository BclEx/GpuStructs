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

class Visual
{
private:
	// mouse controls
	static int _mouseLastX;
	static int _mouseLastY;
	static int _mouseState;
	static float _rotateX;
	static float _rotateY;
	static float _translateZ;
	//StopWatchInterface *_timer;
	static int _fpsCount; // FPS count for averaging
	static int _fpsLimit; // FPS limit for sampling
	static float _avgFps;

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
	static void Display();
	static void Keyboard(unsigned char key, int, int);
	static void Mouse(int button, int state, int x, int y);
	static void Motion(int x, int y);
	static void TimerEvent(int value);

public:
	static void Dispose();
	static void Main();
	inline static void InitState()
	{
		// mouse controls
		_mouseState = 0;
		_rotateX = 0.0, _rotateY = 0.0;
		_translateZ = -3.0;
		//
		//_timer = nullptr;
		_fpsCount = 0; // FPS count for averaging
		_fpsLimit = 1; // FPS limit for sampling
		_avgFps = 0.0f;
		//
	}
	inline static bool InitGL(int *argc, char **argv)
	{
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
