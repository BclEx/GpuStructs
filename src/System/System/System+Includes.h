#ifndef __SYSTEM_SYSTEM_INCLUDES_H__
#define __SYSTEM_SYSTEM_INCLUDES_H__

#pragma warning(disable : 4100)				// unreferenced formal parameter
#pragma warning(disable : 4127)				// conditional expression is constant
#pragma warning(disable : 4244)				// conversion to smaller type, possible loss of data
#pragma warning(disable : 4714)				// function marked as __forceinline not inlined
#pragma warning(disable : 4996)				// unsafe string operations

#if !defined(_DEBUG) && !defined(NDEBUG)
#define NDEBUG // don't generate asserts
#endif
#ifdef __CUDACC__
//#include "System+Cuda.h"
#else
// cuda replacements
#define __constant__
#define __host__
#define __global__
#define __device__
#define __shared__
// system includes
#define _ATL_CSTRING_EXPLICIT_CONSTRUCTORS	// prevent auto literal to string conversion
//#include <malloc.h> // no malloc.h on mac or unix
//#include <windows.h>		// for qgl.h
#undef FindText				// fix namespace pollution
// common
//#include <cstdio>
//#include <cstdlib>
//#include <cstdarg>
#include <cstring>
//#include <casserth>
//#include <ctime>
//#include <ctype>
////#include <ctypeinfo>
//#include <cerrno>
//#include <cmath>
//#include <climits>
//#include <cmemory>
//	//
//#include <cstddef>

#endif

#endif /* __SYSTEM_SYSTEM_INCLUDES_H__ */
