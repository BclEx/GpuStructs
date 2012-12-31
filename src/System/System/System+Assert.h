#ifndef __SYSTEM_SYSTEM_ASSERT_H__
#define __SYSTEM_SYSTEM_ASSERT_H__

/// <summary>
/// Getting assert() to work as we want on all platforms and code analysis tools can be tricky.
/// </summary>
__device__ bool AssertFailed(const char *file, int line, const char *expression);

// idassert is useful for cases where some external library (think MFC, etc.) decides it's a good idea to redefine assert on us
// We have the code analysis tools on the 360 compiler, so let it know what our asserts are. The VS ultimate editions also get it on win32, but not x86
#if 0 & defined(_DEBUG) || defined(_lint)
#undef assert
#define __assert(x) (void)((!!(x)) || (AssertFailed(__FILE__, __LINE__, #x)))
#define assert(x) __analysis_assume(x); __assert(x)
#define verify(x) ((x) ? true : (AssertFailed(__FILE__, __LINE__, #x), false))
#else
#undef assert
#define __assert(x) { ((void)0); }
#define assert(x) __assert(x)
#define verify(x) ((x) ? true : false)
#endif // _DEBUG

#define __releaseassert(x) (void)(!!(x) || AssertFailed(__FILE__, __LINE__, #x));
#define release_assert(x) __releaseassert(x)

#define assert_2_byte_aligned(ptr) assert((((UINT_PTR)(ptr)) & 1) == 0)
#define assert_4_byte_aligned(ptr) assert((((UINT_PTR)(ptr)) & 3) == 0)
#define assert_8_byte_aligned(ptr) assert((((UINT_PTR)(ptr)) & 7) == 0)
#define assert_16_byte_aligned(ptr) assert((((UINT_PTR)(ptr)) & 15) == 0)
#define assert_32_byte_aligned(ptr) assert((((UINT_PTR)(ptr)) & 31) == 0)
#define assert_64_byte_aligned(ptr) assert((((UINT_PTR)(ptr)) & 63) == 0)
#define assert_128_byte_aligned(ptr) assert((((UINT_PTR)(ptr)) & 127) == 0)
#define assert_aligned_to_type_size(ptr) assert((((UINT_PTR)(ptr)) & (sizeof((ptr)[0]) - 1)) == 0)

#if !defined(__TYPEINFOGEN__) && !defined(_lint) // pcLint has problems with assert_offsetof()
template<bool> struct compile_time_assert_failed;
template<> struct compile_time_assert_failed<true> { };
template<int x> struct compile_time_assert_test { };
#define __compile_time_assert_join2(a, b) a##b
#define __compile_time_assert_join(a, b) __compile_time_assert_join2(a, b)
#define compile_time_assert(x) typedef compile_time_assert_test<sizeof(compile_time_assert_failed<(bool)(x)>)> __compile_time_assert_join(compile_time_assert_typedef_, __LINE__)
#define assert_sizeof(type, size) compile_time_assert(sizeof(type) == size)
#define assert_sizeof_8_byte_multiple(type) compile_time_assert((sizeof(type) &  7) == 0)
#define assert_sizeof_16_byte_multiple(type) compile_time_assert((sizeof(type) & 15) == 0)
#define assert_offsetof(type, field, offset) compile_time_assert(offsetof(type, field) == offset)
#define assert_offsetof_8_byte_multiple(type, field) compile_time_assert((offsetof(type, field) & 7) == 0)
#define assert_offsetof_16_byte_multiple(type, field) compile_time_assert((offsetof(type, field) & 15) == 0)
#else
#define compile_time_assert(x)
#define assert_sizeof(type, size)
#define assert_sizeof_8_byte_multiple(type)
#define assert_sizeof_16_byte_multiple(type)
#define assert_offsetof(type, field, offset)
#define assert_offsetof_8_byte_multiple(type, field)
#define assert_offsetof_16_byte_multiple(type, field)
#endif

// useful for verifying that an array of items has the same number of elements in it as an enum type
#define verify_array_size(_array_name_, _max_enum_) compile_time_assert(sizeof(_array_name_) == (_max_enum_) * sizeof(_array_name_[0]))

#endif /* __SYSTEM_SYSTEM_ASSERT_H__ */
