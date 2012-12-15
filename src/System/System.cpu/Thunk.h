#ifndef __SYSTEM_THUNK_H__
#define __SYSTEM_THUNK_H__

#include <string.h>
#include <stdlib.h>
#include <stddef.h>

typedef unsigned char		byte;		// 8 bits
typedef unsigned short		word;		// 16 bits
typedef unsigned int		dword;		// 32 bits
typedef unsigned int		uint;
typedef unsigned long		ulong;

typedef signed char			int8;
typedef unsigned char		uint8;
typedef short int			int16;
typedef unsigned short int	uint16;
typedef int					int32;
typedef unsigned int		uint32;
typedef long long			int64;
typedef unsigned long long	uint64;

// The C/C++ standard guarantees the size of an unsigned type is the same as the signed type.
// The exact size in bytes of several types is guaranteed here.
assert_sizeof(bool,		1);
assert_sizeof(char,		1);
assert_sizeof(short,	2);
assert_sizeof(int,		4);
assert_sizeof(float,	4);
assert_sizeof(byte,		1);
assert_sizeof(int8,		1);
assert_sizeof(uint8,	1);
assert_sizeof(int16,	2);
assert_sizeof(uint16,	2);
assert_sizeof(int32,	4);
assert_sizeof(uint32,	4);
assert_sizeof(int64,	8);
assert_sizeof(uint64,	8);

#define MAX_TYPE(x) ((((1 << ((sizeof(x) - 1) * 8 - 1)) - 1) << 8) | 255)
#define MIN_TYPE(x) (-MAX_TYPE(x) - 1)
#define MAX_UNSIGNED_TYPE(x) ((((1U << ((sizeof(x) - 1) * 8)) - 1) << 8) | 255U)
#define MIN_UNSIGNED_TYPE(x) 0

template<typename T> bool IsSignedType(const T t) { return _type_( -1 ) < 0; }
template<class T> T	Max(T x, T y) { return (x > y ? x : y); }
template<class T> T	Min(T x, T y) { return (x < y ? x : y); }


// C99 Standard
#ifndef nullptr
struct a__nullptr {
	// one pointer member initialized to zero so you can pass NULL as a vararg
	void *value; a__nullptr() : value(0) { }
	// implicit conversion to all pointer types
	template<typename T1> operator T1 * () const { return 0; }
	// implicit conversion to all pointer to member types
	template<typename T1, typename T2> operator T1 T2::* () const { return 0; }
};
#define nullptr	a__nullptr()		
#endif


// Assert
#if 0 & defined(_DEBUG) || defined(_lint)
#undef assert
// idassert is useful for cases where some external library (think MFC, etc.) decides it's a good idea to redefine assert on us
#define idassert(x) (void)((!!(x)) || (AssertFailed( __FILE__, __LINE__, #x)))
// We have the code analysis tools on the 360 compiler, so let it know what our asserts are. The VS ultimate editions also get it on win32, but not x86
#define assert(x) __analysis_assume(x); idassert(x)
#define verify(x) ((x) ? true : (AssertFailed( __FILE__, __LINE__, #x), false))
#else
#undef assert
#define idassert( x ) { ((void)0); }
#define assert(x) idassert(x)
#define verify(x) ((x) ? true : false)
#endif // _DEBUG

#endif /* __SYSTEM_THUNK_H__ */

// Includes
#include "System\Text\String.h"
#include "System\Class.h"
#include "System\Event.h"