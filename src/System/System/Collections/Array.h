#ifndef __SYSTEM_ARRAY_H__
#define __SYSTEM_ARRAY_H__
namespace Sys { namespace Collections {

	/// <summary>
	/// Array is a replacement for a normal C array.
	/// int myArray[ARRAY_SIZE]; ==> Array<int, ARRAY_SIZE> myArray;
	/// </summary>
	/// <about>
	/// Has no performance overhead in release builds, but does index range checking in debug builds.
	/// Unlike TempArray, the memory is allocated inline with the object, rather than on the heap.
	/// Unlike StaticList, there are no fields other than the actual raw data, and the size is fixed.
	/// </about>
	template<class T, int Elements>
	class Array
	{
	public:
		/// <summary>
		/// returns number of elements in list
		/// </summary>
		int Num() const { return Elements; }

		/// <summary>
		/// returns the number of bytes the array takes up
		/// </summary>
		int ByteSize() const { return sizeof(_ptr); }

		/// <summary>
		/// memset the entire array to zero
		/// </summary>
		void Zero() { memset(_ptr, 0, sizeof(_ptr)); }

		/// <summary>
		/// memset the entire array to a specific value
		/// </summary>
		void Memset(const char fill) { memset(_ptr, fill, Elements * sizeof(*_ptr)); }

		/// <summary>
		/// array operators
		/// </summary>
		const T &operator[](int index) const { assert((unsigned)index < (unsigned)Elements); return _ptr[index]; }
		T_ &operator[](int index) { assert((unsigned)index < (unsigned)Elements); return _ptr[index]; }

		/// <summary>
		/// returns a pointer to the list
		/// </summary>
		const T *Ptr() const { return _ptr; }
		T *Ptr() { return _ptr; }

	private:
		T _ptr[Elements];
	};

#define ARRAY_COUNT(arrayName) (sizeof(arrayName) / sizeof(arrayName[0]))
#define ARRAY_DEF(arrayName) arrayName, ARRAY_COUNT(arrayName)

	/// <summary>
	/// Array2D is essentially a typedef (as close as we can get for templates before C++11 anyway) to make declaring two-dimensional idArrays easier.
	/// Usage: id2DArray< int, 5, 10 >::type someArray;
	/// </summary>
	template<class T, int Dim1, int Dim2>
	struct Array2D { typedef Array<idArray<T, Dim2 >, Dim1> type; };

	/// <summary>
	/// Generic way to get the size of a tuple-like type. Add specializations as needed. This is modeled after std::tuple_size from C++11, which works for std::arrays also.
	/// </summary>
	template<class T>
	struct TupleSize;
	template<class T, int Num>
	struct TupleSize<Array<T, Num>> { enum { value = Num }; };

}}
#endif /* __SYSTEM_ARRAY_H__ */
