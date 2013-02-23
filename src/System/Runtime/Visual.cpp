//[GL VBO] http://3dgep.com/?p=2596
//#include <cstdio>
//#include <cassert>
#include "Visual.h"

float Visual::RotateX;
float Visual::RotateY;
float Visual::TranslateX;
float Visual::TranslateY;
float Visual::TranslateZ;
//
IVisualRender* Visual::_render;
int Visual::_mouseLastX;
int Visual::_mouseLastY;
int Visual::_mouseState;
//StopWatchInterface *Visual::_timer;
int Visual::_fpsCount; // FPS count for averaging
int Visual::_fpsLimit; // FPS limit for sampling
float Visual::_avgFps;

void Visual::Display()
{
	//sdkStartTimer(&_timer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	_render->Display();
	//sdkStopTimer(&_timer);
	glutSwapBuffers();
	glutPostRedisplay();
	ComputeFPS();
}

void Visual::Keyboard(unsigned char key, int, int)
{
	_render->Keyboard(key);
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
	if (_mouseState & 2)
	{
		RotateX += dy * 0.2f;
		RotateY += dx * 0.2f;
	}
	if (_mouseState & 1)
	{
		TranslateX += dx * 1.5f;
		TranslateY += dy * -1.5f;
	}
	if (_mouseState & 4)
		TranslateZ += dy * -0.2f;
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
	_render->Initialize();
	// start rendering mainloop
	glutMainLoop();
}