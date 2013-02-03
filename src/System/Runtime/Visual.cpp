#include <cstdio>
#include <cassert>
#include "Visual.h"

float Visual::RotateX;
float Visual::RotateY;
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

void Visual::Dispose()
{
	//sdkDeleteTimer(&_timer);
	_render->Dispose();
}

void Visual::Display()
{
	//sdkStartTimer(&_timer);
	_render->Display();
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
		RotateX += dy * 0.2f;
		RotateY += dx * 0.2f;
	}
	else if (_mouseState & 4)
		TranslateZ += dy * 0.01f;
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