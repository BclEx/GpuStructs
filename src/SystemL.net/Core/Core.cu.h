#if __CUDACC__
#include <Runtime.cu.h>
#else
#include <Runtime.cpu.h>
#endif

#pragma region Limits

#ifndef CORE_MAX_LENGTH
#define CORE_MAX_LENGTH 1000000000
#endif

#pragma endregion

#if defined(__GNUC__) && 0
#define likely(X)    __builtin_expect((X),1)
#define unlikely(X)  __builtin_expect((X),0)
#else
#define likely(X) !!(X)
#define unlikely(X) !!(X)
#endif

#if defined(__PTRDIFF_TYPE__)  // This case should work for GCC
# define INT_TO_PTR(X)  ((void*)(__PTRDIFF_TYPE__)(X))
# define PTR_TO_INT(X)  ((int)(__PTRDIFF_TYPE__)(X))
#elif !defined(__GNUC__)       // Works for compilers other than LLVM
# define INT_TO_PTR(X)  ((void*)&((char*)0)[X])
# define PTR_TO_INT(X)  ((int)(((char*)X)-(char*)0))
#elif defined(HAVE_STDINT_H)   // Use this case if we have ANSI headers
# define INT_TO_PTR(X)  ((void*)(intptr_t)(X))
# define PTR_TO_INT(X)  ((int)(intptr_t)(X))
#else                          // Generates a warning - but it always works
# define INT_TO_PTR(X)  ((void*)(X))
# define PTR_TO_INT(X)  ((int)(X))
#endif

#include "Core+Types.cu.h"
#include "10.ConvertEx.cu.h"
#include "30.RC.cu.h"
#include "45.VAlloc.cu.h"
#include "50.SysEx.cu.h"
#include "20.MutexEx.cu.h"
#include "00.Bitvec.cu.h"
#include "05.Hash.cu.h"
#include "40.StatusEx.cu.h"
#include "50.VSystem.cu.h"
#include "60.MathEx.cu.h"
//
#include "IO\30.VFile.cu.h"
#include "Text\00.StringBuilder.cu.h"
using namespace Core;
using namespace Core::IO;
