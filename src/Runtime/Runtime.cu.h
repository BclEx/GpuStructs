#ifndef __RUNTIME_CU_H__
#define __RUNTIME_CU_H__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 100
#error Atomics only used with > sm_10 architecture
#endif
#ifdef HAVE_ISNAN
#include <math.h>
#endif
#include <assert.h>
#include "RuntimeHost.h"

///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code

// Assert
#undef  _assert
#ifndef NDEBUG
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 200
extern "C" __device__ static void __assertWrite(const char *message, const char *file, unsigned int line);
#undef assert
#define assert(X) _assert(X)
#else
extern "C" __device__ void __assertWrite(const char *message, const char *file, unsigned int line);
#endif
//
#define _assert(X) (void)((!!(X))||(__assertWrite(#X, __FILE__, __LINE__), 0))
#define ASSERTONLY(X) X
__device__ __forceinline__ void Coverage(int line) { }
#define ASSERTCOVERAGE(X) if (X) { Coverage(__LINE__); }
#else
#define _assert(X) ((void)0)
#define ASSERTONLY(X)
#define ASSERTCOVERAGE(X)
#endif

///////////////////////////////////////////////////////////////////////////////
// HEAP
#pragma region HEAP

#define RUNTIME_UNRESTRICTED -1
#define RUNTIME_ALIGNSIZE sizeof(long long)
#define RUNTIMETYPE_RAW 0
#define RUNTIMETYPE_PRINTF 1
#define RUNTIMETYPE_TRANSFER 2
#define RUNTIMETYPE_ASSERT 4
#define RUNTIMETYPE_THROW 5

#pragma endregion

#if defined(__EMBED__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 200)
#ifndef __EMBED__
#define __static__ static
#endif
#include "Runtime.cu+native.h"
#else

// Heap
extern "C" __device__ char *__runtimeMoveNext(char *&end, char *&bufptr);
extern "C" __device__ void __runtimeWriteHeader(unsigned short type, char *ptr, char *fmtptr);
extern "C" __device__ char *__runtimeWrite(char *dest, const char *src, int maxLength, char *end);
extern "C" __device__ void _runtimeSetHeap(void *heap);
extern "C" __device__ void runtimeRestrict(int threadid, int blockid);

// Embed
extern __constant__ unsigned char _runtimeUpperToLower[256];
extern __constant__ unsigned char _runtimeCtypeMap[256];

#endif // __CUDA_ARCH__

///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code

///////////////////////////////////////////////////////////////////////////////
// HEAP
#pragma region HEAP

__device__ static char *__copyArg(char *ptr, const char *arg, char *end)
{
	// initialization check
	if (!ptr) // || !arg)
		return nullptr;
	// strncpy does all our work. We just terminate.
	if ((ptr = __runtimeWrite(ptr, arg, 0, end)) != nullptr)
		*ptr = 0;
	return ptr;
}

template <typename T> __device__ static char *__copyArg(char *ptr, T &arg, char *end)
{
	// initialization and overflow check. Alignment rules mean that we're at least RUNTIME_ALIGNSIZE away from "end", so we only need to check that one offset.
	if (!ptr || (ptr + RUNTIME_ALIGNSIZE) >= end)
		return nullptr;
	// write the length and argument
	*(int *)(void *)ptr = sizeof(arg);
	ptr += RUNTIME_ALIGNSIZE;
	*(T *)(void *)ptr = arg;
	ptr += RUNTIME_ALIGNSIZE;
	*ptr = 0;
	return ptr;
}

#pragma endregion

//////////////////////
// PRINTF
#pragma region PRINTF

#define PRINTF_PREAMBLE \
	char *start, *end, *bufptr, *fmtstart; \
	if ((start = __runtimeMoveNext(end, bufptr)) == nullptr) return 0;
#define PRINTF_ARG(argname) \
	bufptr = __copyArg(bufptr, argname, end);
#define PRINTF_POSTAMBLE \
	fmtstart = bufptr; end = __runtimeWrite(bufptr, fmt, 0, end); \
	__runtimeWriteHeader(RUNTIMETYPE_PRINTF, start, (end ? fmtstart : nullptr)); \
	return (end ? (int)(end - start) : 0);

__device__ static int _printf(const char *fmt)
{
	PRINTF_PREAMBLE;
	PRINTF_POSTAMBLE;
}
template <typename T1> __device__ static int _printf(const char *fmt, T1 arg1)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8); PRINTF_ARG(arg9);
	PRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ static int _printf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA)
{
	PRINTF_PREAMBLE;
	PRINTF_ARG(arg1); PRINTF_ARG(arg2); PRINTF_ARG(arg3); PRINTF_ARG(arg4); PRINTF_ARG(arg5); PRINTF_ARG(arg6); PRINTF_ARG(arg7); PRINTF_ARG(arg8); PRINTF_ARG(arg9); PRINTF_ARG(argA);
	PRINTF_POSTAMBLE;
}

#undef PRINTF_PREAMBLE
#undef PRINTF_ARG
#undef PRINTF_POSTAMBLE

#pragma endregion

//////////////////////
// TRANSFER
#pragma region TRANSFER

#define TRANSFER_PREAMBLE \
	char *start, *end, *bufptr, *fmtstart; \
	if ((start = __runtimeMoveNext(end, bufptr)) == nullptr) return 0;
#define TRANSFER_ARG(argname) \
	bufptr = __copyArg(bufptr, argname, end);
#define TRANSFER_POSTAMBLE \
	fmtstart = bufptr; end = __runtimeWrite(bufptr, fmt, 0, end); \
	__runtimeWriteHeader(RUNTIMETYPE_TRANSFER, start, (end ? fmtstart : nullptr)); \
	return (end ? (int)(end - start) : 0);

__device__ static int _transfer(const char *buf, size_t bufLen, const char *fmt)
{
	TRANSFER_PREAMBLE;
	TRANSFER_POSTAMBLE;
}
template <typename T1> __device__ static int _transfer(const char *fmt, T1 arg1)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5); TRANSFER_ARG(arg6);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5); TRANSFER_ARG(arg6); TRANSFER_ARG(arg7);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5); TRANSFER_ARG(arg6); TRANSFER_ARG(arg7); TRANSFER_ARG(arg8);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5); TRANSFER_ARG(arg6); TRANSFER_ARG(arg7); TRANSFER_ARG(arg8); TRANSFER_ARG(arg9);
	TRANSFER_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ static int _transfer(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA)
{
	TRANSFER_PREAMBLE;
	TRANSFER_ARG(arg1); TRANSFER_ARG(arg2); TRANSFER_ARG(arg3); TRANSFER_ARG(arg4); TRANSFER_ARG(arg5); TRANSFER_ARG(arg6); TRANSFER_ARG(arg7); TRANSFER_ARG(arg8); TRANSFER_ARG(arg9); TRANSFER_ARG(argA);
	TRANSFER_POSTAMBLE;
}

#undef TRANSFER_PREAMBLE
#undef TRANSFER_ARG
#undef TRANSFER_POSTAMBLE

#pragma endregion

//////////////////////
// STDARG
#pragma region STDARG

#undef va_start
#undef va_arg
#undef va_end

typedef char *va_list;
#define _INTSIZEOF(n) ((sizeof(n) + sizeof(int) - 1) & ~(sizeof(int) - 1))
//#define va_start(ap, v)  (ap = (va_list)_ADDRESSOF(v) + _INTSIZEOF(v))
#define va_arg(ap, t) (*(t *)((ap += _INTSIZEOF(t)) - _INTSIZEOF(t)))
#define va_end(ap) (ap = (va_list)0)

#define STDARG_PREAMBLE \
	char *start, *end, *bufptr, *fmtstart; \
	if ((start = __runtimeMoveNext(end, bufptr)) == nullptr) return;
#define STDARG_ARG(argname) \
	bufptr = __copyArg(bufptr, argname, end);
#define STDARG_POSTAMBLE \
	fmtstart = bufptr; \
	__runtimeWriteHeader(RUNTIMETYPE_RAW, start, (end ? fmtstart : nullptr)); \
	args = (end ? (va_list)(end - start) : 0);

__device__ static void va_start(va_list &args)
{
	STDARG_PREAMBLE;
	STDARG_POSTAMBLE;
}
template <typename T1> __device__ static void va_start(va_list &args, T1 arg1)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5); STDARG_ARG(arg6);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5); STDARG_ARG(arg6); STDARG_ARG(arg7);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5); STDARG_ARG(arg6); STDARG_ARG(arg7); STDARG_ARG(arg8);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5); STDARG_ARG(arg6); STDARG_ARG(arg7); STDARG_ARG(arg8); STDARG_ARG(arg9);
	STDARG_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ static void va_start(va_list &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA)
{
	STDARG_PREAMBLE;
	STDARG_ARG(arg1); STDARG_ARG(arg2); STDARG_ARG(arg3); STDARG_ARG(arg4); STDARG_ARG(arg5); STDARG_ARG(arg6); STDARG_ARG(arg7); STDARG_ARG(arg8); STDARG_ARG(arg9); STDARG_ARG(argA);
	STDARG_POSTAMBLE;
}

#undef STDARG_PREAMBLE
#undef STDARG_ARG
#undef STDARG_POSTAMBLE

#pragma endregion

//////////////////////
// THROW
#pragma region THROW

#define THROW_PREAMBLE \
	char *start, *end, *bufptr, *fmtstart; \
	if ((start = __runtimeMoveNext(end, bufptr)) == nullptr) return;
#define THROW_ARG(argname) \
	bufptr = __copyArg(bufptr, argname, end);
#define THROW_POSTAMBLE \
	fmtstart = bufptr; end = __runtimeWrite(bufptr, fmt, 0, end); \
	__runtimeWriteHeader(RUNTIMETYPE_THROW, start, (end ? fmtstart : nullptr)); \
	__THROW;

__device__ static void _throw(const char *fmt)
{
	THROW_PREAMBLE;
	THROW_POSTAMBLE;
}
template <typename T1> __device__ static void _throw(const char *fmt, T1 arg1)
{
	THROW_PREAMBLE;
	THROW_ARG(arg1);
	THROW_POSTAMBLE;
}
template <typename T1, typename T2> __device__ static void _throw(const char *fmt, T1 arg1, T2 arg2)
{
	THROW_PREAMBLE;
	THROW_ARG(arg1); THROW_ARG(arg2);
	THROW_POSTAMBLE;
}
template <typename T1, typename T2, typename T3> __device__ static void _throw(const char *fmt, T1 arg1, T2 arg2, T3 arg3)
{
	THROW_PREAMBLE;
	THROW_ARG(arg1); THROW_ARG(arg2); THROW_ARG(arg3);
	THROW_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4> __device__ static void _throw(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
{
	THROW_PREAMBLE;
	THROW_ARG(arg1); THROW_ARG(arg2); THROW_ARG(arg3); THROW_ARG(arg4);
	THROW_POSTAMBLE;
}

#undef THROW_PREAMBLE
#undef THROW_ARG
#undef THROW_POSTAMBLE

#pragma endregion

#define __toupper(x) ((x)&~(_runtimeCtypeMap[(unsigned char)(x)]&0x20))
#define _isspace(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x01)
#define _isalnum(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x06)
#define _isalpha(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x02)
#define _isdigit(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x04)
#define _isxdigit(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x08)
#define _isidchar(x) (_runtimeCtypeMap[(unsigned char)(x)]&0x46)
#define __tolower(x) (_runtimeUpperToLower[(unsigned char)(x)])

// array
template <typename T> struct array_t { int length; T *data; __device__ inline array_t() { data = nullptr; length = 0; } __device__ inline array_t(T *a) { data = a; length = 0; } __device__ inline array_t(T *a, int b) { data = a; length = b; } __device__ inline void operator=(T *a) { data = a; } __device__ inline operator T *() { return data; } };
template <typename TLength, typename T> struct array_t2 { TLength length; T *data; __device__ inline array_t2() { data = nullptr; length = 0; } __device__ inline array_t2(T *a) { data = a; length = 0; } __device__ inline array_t2(T *a, size_t b) { data = a; length = b; } __device__ inline void operator=(T *a) { data = a; } __device__ inline operator T *() { return data; } };
#define __arrayStaticLength(symbol) (sizeof(symbol) / sizeof(symbol[0]))

// skiputf8
template <typename T> __device__ inline void _skiputf8(const T *z)
{
	if (*(z++) >= 0xc0) while ((*z & 0xc0) == 0x80) { z++; }
}

// strcmp
#undef _fstrcmp
template <typename T> __device__ inline int _strcmp(const T *left, const T *right)
{
	register unsigned char *a, *b;
	a = (unsigned char *)left;
	b = (unsigned char *)right;
	while (*a != 0 && _runtimeUpperToLower[*a] == _runtimeUpperToLower[*b]) { a++; b++; }
	return _runtimeUpperToLower[*a] - _runtimeUpperToLower[*b];
}
#define _fstrcmp(x, y) (__tolower(*(unsigned char *)(x))==__tolower(*(unsigned char *)(y))&&!_strcmp((x)+1,(y)+1))

// strncmp
#undef _fstrncmp
template <typename T> __device__ inline int _strncmp(const T *left, const T *right, int n)
{
	register unsigned char *a, *b;
	a = (unsigned char *)left;
	b = (unsigned char *)right;
	while (n-- > 0 && *a != 0 && _runtimeUpperToLower[*a] == _runtimeUpperToLower[*b]) { a++; b++; }
	return (n < 0 ? 0 : _runtimeUpperToLower[*a] - _runtimeUpperToLower[*b]);
}
#define _fstrncmp(x, y) (__tolower(*(unsigned char *)(x))==__tolower(*(unsigned char *)(y))&&!_strcmp((x)+1,(y)+1))

// memcpy
template <typename T> __device__ inline void _memcpy(T *dest, const T *src, size_t length)
{
	register unsigned char *a, *b;
	a = (unsigned char *)dest;
	b = (unsigned char *)src;
	for (size_t i = 0; i < length; ++i, ++a, ++b)
		*a = *b;
}

// memset
template <typename T> __device__ inline void _memset(T *dest, const char value, size_t length)
{
	register unsigned char *a;
	a = (unsigned char *)dest;
	for (size_t i = 0; i < length; ++i, ++a)
		*a = value;
}

// memcmp
template <typename T, typename Y> __device__ inline int _memcmp(T *left, Y right, size_t length)
{
	register unsigned char *a, *b;
	a = (unsigned char *)left;
	b = (unsigned char *)right;
	while (*a != 0 && *a == *b) { a++; b++; }
	return *a - *b;
}

// strlen30
__device__ inline int _strlen30(const char *z)
{
	register const char *z2 = z;
	if (z == nullptr) return 0;
	while (*z2) { z2++; }
	return 0x3fffffff & (int)(z2 - z);
}

// hextobyte
__device__ inline unsigned char _hextobyte(char h)
{
	_assert((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F'));
	return (unsigned char)((h + 9*(1&(h>>6))) & 0xf);
}

#ifndef OMIT_FLOATING_POINT
__device__ inline int _isNaN(double x)
{
#if !defined(HAVE_ISNAN)
	// Systems that support the isnan() library function should probably make use of it by compiling with -DHAVE_ISNAN.  But we have
	// found that many systems do not have a working isnan() function so this implementation is provided as an alternative.
	//
	// This NaN test sometimes fails if compiled on GCC with -ffast-math. On the other hand, the use of -ffast-math comes with the following
	// warning:
	//
	//      This option [-ffast-math] should never be turned on by any -O option since it can result in incorrect output for programs
	//      which depend on an exact implementation of IEEE or ISO rules/specifications for math functions.
	//
	// Under MSVC, this NaN test may fail if compiled with a floating-point precision mode other than /fp:precise.  From the MSDN 
	// documentation:
	//
	//      The compiler [with /fp:precise] will properly handle comparisons involving NaN. For example, x != x evaluates to true if x is NaN 
#ifdef __FAST_MATH__
#error Runtime will not work correctly with the -ffast-math option of GCC.
#endif
	volatile double y = x;
	volatile double z = y;
	return (y != z);
#else
	return isnan(x);
#endif
}
#endif

#endif // __RUNTIME_CU_H__
