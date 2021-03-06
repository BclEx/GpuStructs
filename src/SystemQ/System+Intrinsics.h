#ifndef __SYSTEM_SYSTEM_INTRINSICS_H__
#define __SYSTEM_SYSTEM_INTRINSICS_H__

#include <intrin.h> // needed for intrinsics like _mm_setzero_si28

// Scalar single precision floating-point intrinsics
extern inline float __fmuls(float a, float b)				{ return (a * b); }
extern inline float __fmadds(float a, float b, float c)		{ return (a * b + c); }
extern inline float __fnmsubs(float a, float b, float c)	{ return (c - a * b); }
extern inline float __fsels(float a, float b, float c)		{ return (a >= 0.0f ? b : c); }
extern inline float __frcps(float x)						{ return (1.0f / x ); }
extern inline float __fdivs(float x, float y)				{ return (x / y ); }
extern inline float __frsqrts(float x)						{ return (1.0f / sqrtf(x)); }
extern inline float __frcps16(float x)						{ return (1.0f / x); }
extern inline float __fdivs16(float x, float y)				{ return (x / y); }
extern inline float __frsqrts16(float x)					{ return (1.0f / sqrtf(x)); }
extern inline float __frndz(float x)						{ return (float)((int)(x)); }


// Zero cache line and prefetch intrinsics

// The code below assumes that a cache line is 64 bytes. We specify the cache line size as 128 here to make the code consistent with the consoles.
#define CACHE_LINE_SIZE	128

__forceinline void Prefetch(const void * ptr, int offset)
{
	//const char *bytePtr = ((const char *)ptr) + offset;
	//_mm_prefetch(bytePtr +  0, _MM_HINT_NTA);
	//_mm_prefetch(bytePtr + 64, _MM_HINT_NTA);
}
__forceinline void ZeroCacheLine(void *ptr, int offset)
{
	assert_128_byte_aligned(ptr);
	char *bytePtr = ((char *)ptr) + offset;
	__m128i zero = _mm_setzero_si128();
	_mm_store_si128((__m128i *)(bytePtr + 0*16), zero);
	_mm_store_si128((__m128i *)(bytePtr + 1*16), zero);
	_mm_store_si128((__m128i *)(bytePtr + 2*16), zero);
	_mm_store_si128((__m128i *)(bytePtr + 3*16), zero);
	_mm_store_si128((__m128i *)(bytePtr + 4*16), zero);
	_mm_store_si128((__m128i *)(bytePtr + 5*16), zero);
	_mm_store_si128((__m128i *)(bytePtr + 6*16), zero);
	_mm_store_si128((__m128i *)(bytePtr + 7*16), zero);
}
__forceinline void FlushCacheLine(const void *ptr, int offset)
{
	const char *bytePtr = ((const char *)ptr) + offset;
	_mm_clflush(bytePtr +  0);
	_mm_clflush(bytePtr + 64);
}

// Block Clear Macros
// number of additional elements that are potentially cleared when clearing whole cache lines at a time
__forceinline int CACHE_LINE_CLEAR_OVERFLOW_COUNT(int size)
{
	if ((size & (CACHE_LINE_SIZE - 1)) == 0)
		return 0;
	if (size > CACHE_LINE_SIZE)
		return 1;
	return (CACHE_LINE_SIZE / (size & (CACHE_LINE_SIZE - 1)));
}

// if the pointer is not on a cache line boundary this assumes the cache line the pointer starts in was already cleared
#define CACHE_LINE_CLEAR_BLOCK(ptr, size) \
	byte *startPtr = (byte *)((((UINT_PTR) (ptr)) + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1)); \
	byte *endPtr = (byte *)(((UINT_PTR) (ptr) + (size) - 1 ) & ~( CACHE_LINE_SIZE - 1)); \
	for (; startPtr <= endPtr; startPtr += CACHE_LINE_SIZE) { ZeroCacheLine(startPtr, 0); }

#define CACHE_LINE_CLEAR_BLOCK_AND_FLUSH(ptr, size) \
	byte *startPtr = (byte *)((((UINT_PTR)(ptr)) + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1)); \
	byte *endPtr = (byte *)(((UINT_PTR)(ptr) + (size) - 1) & ~(CACHE_LINE_SIZE - 1)); \
	for (; startPtr <= endPtr; startPtr += CACHE_LINE_SIZE) { ZeroCacheLine(startPtr, 0); FlushCacheLine(startPtr, 0); }

// PC Windows
#if !defined(R_SHUFFLE_D)
#define R_SHUFFLE_D(x, y, z, w) (((w) & 3) << 6 | ((z) & 3) << 4 | ((y) & 3) << 2 | ((x) & 3))
#endif

// make the intrinsics "type unsafe"
typedef union __declspec(intrin_type) _CRT_ALIGN(16) __m128c
{
	__m128c() { }
	__m128c(__m128 f) { m128 = f; }
	__m128c(__m128i i) { m128i = i; }
	operator __m128() { return m128; }
	operator __m128i() { return m128i; }
	__m128 m128;
	__m128i m128i;
} __m128c;
#define _mm_madd_ps(a, b, c) _mm_add_ps(_mm_mul_ps((a), (b)), (c))
#define _mm_nmsub_ps(a, b, c) _mm_sub_ps((c), _mm_mul_ps((a), (b)))
#define _mm_splat_ps(x, i) __m128c( _mm_shuffle_epi32(__m128c(x), _MM_SHUFFLE(i, i, i, i)))
#define _mm_perm_ps(x, perm) __m128c( _mm_shuffle_epi32(__m128c(x), perm))
#define _mm_sel_ps(a, b, c) _mm_or_ps(_mm_andnot_ps(__m128c(c), a), _mm_and_ps(__m128c(c), b))
#define _mm_sel_si128(a, b, c) _mm_or_si128(_mm_andnot_si128(__m128c(c), a), _mm_and_si128(__m128c(c), b))
#define _mm_sld_ps(x, y, imm) __m128c(_mm_or_si128(_mm_srli_si128(__m128c(x), imm), _mm_slli_si128(__m128c(y), 16 - imm)))
#define _mm_sld_si128(x, y, imm) _mm_or_si128(_mm_srli_si128(x, imm), _mm_slli_si128(y, 16 - imm))
extern __forceinline __m128 _mm_msum3_ps(__m128 a, __m128 b)
{
	__m128 c = _mm_mul_ps(a, b);
	return _mm_add_ps(_mm_splat_ps(c, 0), _mm_add_ps(_mm_splat_ps(c, 1), _mm_splat_ps(c, 2)));
}
extern __forceinline __m128 _mm_msum4_ps(__m128 a, __m128 b)
{
	__m128 c = _mm_mul_ps(a, b);
	c = _mm_add_ps(c, _mm_perm_ps(c, _MM_SHUFFLE(1, 0, 3, 2)));
	c = _mm_add_ps(c, _mm_perm_ps(c, _MM_SHUFFLE(2, 3, 0, 1)));
	return c;
}
#define _mm_shufmix_epi32(x, y, perm) __m128c( _mm_shuffle_ps(__m128c(x), __m128c(y), perm))
#define _mm_loadh_epi64(x, address) __m128c( _mm_loadh_pi(__m128c(x), (__m64 *)address))
#define _mm_storeh_epi64(address, x) _mm_storeh_pi((__m64 *)address, __m128c(x))
// floating-point reciprocal with close to full precision
extern __forceinline __m128 _mm_rcp32_ps(__m128 x)
{
	__m128 r = _mm_rcp_ps(x); // _mm_rcp_ps() has 12 bits of precision
	r = _mm_sub_ps(_mm_add_ps(r, r), _mm_mul_ps(_mm_mul_ps(x, r), r));
	r = _mm_sub_ps(_mm_add_ps(r, r), _mm_mul_ps(_mm_mul_ps(x, r), r));
	return r;
}
// floating-point reciprocal with at least 16 bits precision
extern __forceinline __m128 _mm_rcp16_ps(__m128 x)
{
	__m128 r = _mm_rcp_ps(x); // _mm_rcp_ps() has 12 bits of precision
	r = _mm_sub_ps(_mm_add_ps(r, r), _mm_mul_ps(_mm_mul_ps(x, r), r));
	return r;
}
// floating-point divide with close to full precision
extern __forceinline __m128 _mm_div32_ps(__m128 x, __m128 y) { return _mm_mul_ps(x, _mm_rcp32_ps(y)); }
// floating-point divide with at least 16 bits precision
extern __forceinline __m128 _mm_div16_ps(__m128 x, __m128 y) { return _mm_mul_ps(x, _mm_rcp16_ps(y)); }
// load Bounds::GetMins()
#define _mm_loadu_bounds_0(bounds) _mm_perm_ps(_mm_loadh_pi(_mm_load_ss(&bounds[0].x), (__m64 *)&bounds[0].y), _MM_SHUFFLE(1, 3, 2, 0))
// load Bounds::GetMaxs()
#define _mm_loadu_bounds_1(bounds) _mm_perm_ps(_mm_loadh_pi(_mm_load_ss(&bounds[1].x), (__m64 *)&bounds[1].y), _MM_SHUFFLE(1, 3, 2, 0))

#endif /* __SYSTEM_SYSTEM_INTRINSICS_H__ */
