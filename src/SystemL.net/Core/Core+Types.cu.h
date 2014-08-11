#ifndef __CORE_TYPES_H__
#define __CORE_TYPES_H__

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

#define MAX(x,y) ((x)<(y)?(x):(y))
#define MIN(x,y) ((x)>(y)?(x):(y))
#define MAX_TYPE(x) (((((x)1<<((sizeof(x)-1)*8-1))-1)<<8)|255)
#define MIN_TYPE(x) (-MAX_TYPE(x)-1)
#define MAX_UTYPE(x) (((((x)1U<<((sizeof(x)-1)*8))-1)<<8)|255U)
#define MIN_UTYPE(x) 0
#define LARGEST_INT64 (0xffffffff|(((int64)0x7fffffff)<<32))
#define SMALLEST_INT64 (((int64)-1) - LARGEST_INT64)

// If compiling for a processor that lacks floating point support, substitute integer for floating-point
#ifdef OMIT_FLOATING_POINT
#define double int64
#define float int64
//#define LONGDOUBLE_TYPE int64
#define BIG_DOUBLE (((int64)1)<<50)
#define OMIT_DATETIME_FUNCS 1
#define OMIT_TRACE 1
#undef MIXED_ENDIAN_64BIT_FLOAT
#undef HAVE_ISNAN
#else
#define BIG_DOUBLE (1e99)
#endif

// Macros to determine whether the machine is big or little endian, evaluated at runtime.
#if defined(i386) || defined(__i386__) || defined(_M_IX86) || defined(__x86_64) || defined(__x86_64__)
#define TYPE_BIGENDIAN 0
#define TYPE_LITTLEENDIAN 1
#define TEXTENCODE_UTF16NATIVE TEXTENCODE_UTF16LE
#else
static byte __one;
#define TYPE_BIGENDIAN (*(char *)(&__one) == 0)
#define TYPE_LITTLEENDIAN (*(char *)(&__one) == 1)
#define TEXTENCODE_UTF16NATIVE (TYPE_BIGENDIAN ? TEXTENCODE_UTF16BE : TEXTENCODE_UTF16LE)
#endif

#endif // __CORE_TYPES_H__
