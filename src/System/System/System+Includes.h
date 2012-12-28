#ifndef __SYSTEM_SYSTEM_INCLUDES_H__
#define __SYSTEM_SYSTEM_INCLUDES_H__
namespace Sys {

#define _ATL_CSTRING_EXPLICIT_CONSTRUCTORS	// prevent auto literal to string conversion
#pragma warning(disable : 4100)				// unreferenced formal parameter
#pragma warning(disable : 4127)				// conditional expression is constant
#pragma warning(disable : 4244)				// conversion to smaller type, possible loss of data
#pragma warning(disable : 4714)				// function marked as __forceinline not inlined
#pragma warning(disable : 4996)				// unsafe string operations
//
//#include <malloc.h>			// no malloc.h on mac or unix
//#include <windows.h>		// for qgl.h
#undef FindText				// fix namespace pollution
//
#if !defined(_DEBUG) && !defined(NDEBUG)
#define NDEBUG // don't generate asserts
#endif
//#include <stdio.h>
//#include <stdlib.h>
//#include <stdarg.h>
//#include <string.h>
//#include <assert.h>
//#include <time.h>
//#include <ctype.h>
////#include <typeinfo.h>
//#include <errno.h>
//#include <math.h>
//#include <limits.h>
//#include <memory.h>
//	//
//#include <stddef.h>

}
#endif /* __SYSTEM_SYSTEM_INCLUDES_H__ */
