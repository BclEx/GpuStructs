#ifndef __SYSTEM_LIST_H__
#define __SYSTEM_LIST_H__
//#include <new>
#include "..\System.h"
namespace Sys { namespace Collections {

	/// <summary>
	/// ListArrayNew
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline void * ListArrayNew(int num, bool zeroBuffer)
	{
		T *ptr = nullptr;
		if (zeroBuffer)
			ptr = (T *)Mem_ClearedAlloc(sizeof(T) * num, Tag);
		else
			ptr = (T *)Mem_Alloc(sizeof(T) * num, Tag);
		for (int i = 0; i < num; i++)
			new (&ptr[i]) T;
		return ptr;
	}

	/// <summary>
	/// ListArrayDelete
	/// </summary>
	template<typename T>
	__device__ inline void ListArrayDelete(void *ptr, int num)
	{
		// Call the destructors on all the elements
		for (int i = 0; i < num; i++)
			((T *)ptr)[i].~T();
		Mem_Free(ptr);
	}

	/// <summary>
	/// ListArrayResize
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline void *ListArrayResize(void * voldptr, int oldNum, int newNum, bool zeroBuffer)
	{
		T *oldptr = (T *)voldptr;
		T *newptr = nullptr;
		if (newNum > 0)
		{
			newptr = (T *)ListArrayNew<T, Tag>(newNum, zeroBuffer);
			int overlap = Min(oldNum, newNum);
			for (int i = 0; i < overlap; i++)
				newptr[i] = oldptr[i];
		}
		ListArrayDelete<T>(voldptr, oldNum);
		return newptr;
	}

	/// <summary>
	/// ListNewElement
	/// </summary>
	template<class T>
	__device__ inline T *ListNewElement() { return new T; }

	/// <summary>
	/// Does not allocate memory until the first item is added.
	/// </summary>
	template<typename T, memTag_t Tag = TAG_IDLIB_LIST>
	class List {
	public:
		typedef int cmp_t(const T *, const T *);
		typedef T new_t();

		__device__ List(int newgranularity = 16);
		__device__ List(const List &other);
		__device__ ~List();

		__device__ void Clear();								// clear the list
		__device__ int Num() const;							// returns number of elements in list
		__device__ int NumAllocated() const;					// returns number of elements allocated for
		__device__ void SetGranularity(int newgranularity);	// set new granularity
		__device__ int GetGranularity() const;					// get the current granularity

		__device__ size_t Allocated() const;			// returns total size of allocated memory
		__device__ size_t Size() const;				// returns total size of allocated memory including size of list _type_
		__device__ size_t MemoryUsed() const;			// returns size of the used elements in the list

		__device__ List<T, Tag> &operator=(const List<T, Tag> &other);
		__device__ const T &operator[](int index) const;
		__device__ T &operator[](int index);

		__device__ void Condense();										// resizes list to exactly the number of elements it contains
		__device__ void Resize( int newsize);								// resizes list to the given number of elements
		__device__ void Resize( int newsize, int newgranularity);			// resizes list and sets new granularity
		__device__ void SetNum( int newnum);								// set number of elements in list and resize to exactly this number if needed
		__device__ void AssureSize(int newSize);							// assure list has given number of elements, but leave them uninitialized
		__device__ void AssureSize(int newSize, const T &initValue);		// assure list has given number of elements and initialize any new elements
		__device__ void AssureSizeAlloc(int newSize, new_t *allocator);	// assure the pointer list has the given number of elements and allocate any new elements

		__device__ T *Ptr();										// returns a pointer to the list
		__device__ const T *Ptr() const;							// returns a pointer to the list
		__device__ T &Alloc();										// returns reference to a new data element at the end of the list
		__device__ int Append(const T & obj );						// append element
		__device__ int Append(const List &other );					// append list
		__device__ int AddUnique(const T &obj );					// add unique element
		__device__ int Insert(const T &obj, int index = 0);		// insert the element at the given index
		__device__ int FindIndex(const T &obj) const;				// find the index for the given element
		__device__ T *Find(T const &obj) const;					// find pointer to the given element
		__device__ int FindNull() const;							// find the index for the first NULL pointer in the list
		__device__ int IndexOf(const T *obj) const;				// returns the index for the pointer to an element in the list
		__device__ bool RemoveIndex(int index);					// remove the element at the given index
		// removes the element at the given index and places the last element into its spot - DOES NOT PRESERVE LIST ORDER
		__device__ bool RemoveIndexFast(int index);
		__device__ bool Remove(const T &obj);						// remove the element
		//__device__ void Sort(cmp_t *compare = (cmp_t *)&idListSortCompare<T, tag>);
		__device__ void SortWithTemplate(const Sorter<T> &sort = Sort_QuickDefault<T>());
		//__device__ void SortSubSection( int startIndex, int endIndex, cmp_t *compare = ( cmp_t * )&idListSortCompare<_type_> );
		__device__ void Swap(List &other);							// swap the contents of the lists
		__device__ void DeleteContents(bool clear = true);			// delete the contents of the list

		// auto-cast to other List types with a different memory tag
		template<memTag_t _tag>
		__device__ operator List<T, _tag> &() { return *reinterpret_cast<List<T, _tag> *>(this); }
		template<memTag_t _tag>
		__device__ operator const List<T, _tag> &() const { return *reinterpret_cast<const List<T, _tag> *>(this); }

		// memTag, Changing the memTag when the list has an allocated buffer will result in corruption of the memory statistics.
		__device__ memTag_t GetMemTag() const { return (memTag_t)_memTag; }
		__device__ void SetMemTag(memTag_t tag_) { _memTag = (byte)tag_; }

	private:
		int _num;
		int _size;
		int _granularity;
		T *_list;
		byte _memTag;
	};

	/// <summary>
	/// List(int)
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline List<T, Tag>::List(int newgranularity) { assert(newgranularity > 0); _list = nullptr; _granularity = newgranularity; _memTag = Tag; Clear(); }
	/// <summary>
	/// List(const List)
	/// </summary>
	template<typename T, memTag_t Tag >
	__device__ inline List<T, Tag>::List(const List &other) { _list = nullptr; *this = other; }
	/// <summary>
	/// ~List
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline List<T, Tag>::~List() { Clear(); }

	/// <summary>
	/// Frees up the memory allocated by the list.  Assumes that _type_ automatically handles freeing up memory.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline void List<T, Tag>::Clear()
	{
		if (_list)
			ListArrayDelete<T>(_list, _size);
		_list = nullptr;
		_num = 0;
		_size = 0;
	}

	/// <summary>
	/// Calls the destructor of all elements in the list.  Conditionally frees up memory used by the list. Note that this only works on lists containing pointers to objects and will cause a compiler error
	/// if called with non-pointers.  Since the list was not responsible for allocating the object, it has no information on whether the object still exists or not, so care must be taken to ensure that
	/// the pointers are still valid when this function is called.  Function will set all pointers in the list to NULL.
	/// <summary>
	template<typename T, memTag_t Tag>
	__device__ inline void List<T, Tag>::DeleteContents(bool clear)
	{
		for (int i = 0; i < _num; i++)
		{
			delete _list[i];
			_list[i] = nullptr;
		}
		if (clear)
			Clear();
		else
			memset(_list, 0, _size * sizeof(T));
	}

	/// <summary>
	/// return total memory allocated for the list in bytes, but doesn't take into account additional memory allocated by _type_
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline size_t List<T, Tag>::Allocated() const { return _size * sizeof(T); }

	/// <summary>
	/// return total size of list in bytes, but doesn't take into account additional memory allocated by _type_
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline size_t List<T, Tag>::Size() const { return sizeof(List<T, Tag>) + Allocated(); }

	/// <summary>
	/// MemoryUsed
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline size_t List<T, Tag>::MemoryUsed() const { return _num * sizeof(*_list); }

	/// <summary>
	/// Returns the number of elements currently contained in the list. Note that this is NOT an indication of the memory allocated.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline int List<T, Tag>::Num() const { return _num; }

	/// <summary>
	/// Returns the number of elements currently allocated for.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline int List<T, Tag>::NumAllocated() const { return _size; }

	/// <summary>
	/// SetNum
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline void List<T, Tag>::SetNum(int newnum) { assert(newnum >= 0); if (newnum > _size) Resize(newnum); _num = newnum; }

	/// <summary>
	/// Sets the base size of the array and resizes the array to match.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline void List<T, Tag>::SetGranularity(int newgranularity)
	{
		assert(newgranularity > 0);
		_granularity = newgranularity;
		if (_list)
		{
			// resize it to the closest level of granularity
			int newsize = _num + _granularity - 1;
			newsize -= newsize % _granularity;
			if (newsize != _size)
				Resize(newsize);
		}
	}

	/// <summary>
	/// Get the current granularity.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline int List<T, Tag>::GetGranularity() const { return _granularity; }

	/// <summary>
	/// Resizes the array to exactly the number of elements it contains or frees up memory if empty.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline void List<T, Tag>::Condense() {
		if (_list)
			if (_num)
				Resize(_num);
			else
				Clear();
	}

	/// <summary>
	/// Allocates memory for the amount of elements requested while keeping the contents intact. Contents are copied using their = operator so that data is correnctly instantiated.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline void List<T, Tag>::Resize(int newsize)
	{
		assert(newsize >= 0);
		// free up the list if no data is being reserved
		if (newsize <= 0)
		{
			Clear();
			return;
		}
		// not changing the size, so just exit
		if (newsize == _size)
			return;
		_list = (T *)ListArrayResize<T, Tag>(_list, _size, newsize, false);
		_size = newsize;
		if (_size < _num)
			_num = _size;
	}

	/// <summary>
	/// Allocates memory for the amount of elements requested while keeping the contents intact. Contents are copied using their = operator so that data is correnctly instantiated.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline void List<T, Tag>::Resize(int newsize, int newgranularity)
	{
		assert(newsize >= 0);
		assert(newgranularity > 0);
		_granularity = newgranularity;
		// free up the list if no data is being reserved
		if (newsize <= 0)
		{
			Clear();
			return;
		}
		_list = (T *)ListArrayResize<T, Tag>(_list, _size, newsize, false);
		_size = newsize;
		if (_size < _num)
			_num = _size;
	}

	/// <summary>
	/// Makes sure the list has at least the given number of elements.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline void List<T, Tag>::AssureSize(int newSize)
	{
		int newNum = newSize;
		if (newSize > _size)
		{
			// this is a hack to fix our memset classes
			if (_granularity == 0)
				_granularity = 16;
			newSize += _granularity - 1;
			newSize -= newSize % _granularity;
			Resize(newSize);
		}
		_num = newNum;
	}

	/// <summary>
	/// Makes sure the list has at least the given number of elements and initialize any elements not yet initialized.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline void List<T, Tag>::AssureSize(int newSize, const T &initValue)
	{
		int newNum = newSize;
		if (newSize > _size)
		{
			// this is a hack to fix our memset classes
			if (_granularity == 0)
				_granularity = 16;
			newSize += _granularity - 1;
			newSize -= newSize % _granularity;
			_num = size;
			Resize(newSize);
			for (int i = _num; i < newSize; i++)
				_list[i] = initValue;
		}
		_num = newNum;
	}

	/// <summary>
	/// Makes sure the list has at least the given number of elements and allocates any elements using the allocator.
	/// NOTE: This function can only be called on lists containing pointers. Calling it on non-pointer lists will cause a compiler error.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline void List<T, Tag>::AssureSizeAlloc(int newSize, new_t *allocator)
	{
		int newNum = newSize;
		if (newSize > size)
		{
			// this is a hack to fix our memset classes
			if (_granularity == 0)
				_granularity = 16;
			newSize += _granularity - 1;
			newSize -= newSize % _granularity;
			_num = _size;
			Resize(newSize);
			for (int i = _num; i < newSize; i++)
				_list[i] = (*allocator)();
		}
		_num = newNum;
	}

	/// <summary>
	/// Copies the contents and size attributes of another list.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline List<T, Tag> &List<T, Tag>::operator=(const List<T, Tag> &other)
	{
		Clear();
		_num = other._num;
		_size = other._size;
		_granularity = other._granularity;
		_memTag = other._memTag;
		if (size)
		{
			_list = (T *)ListArrayNew<T, Tag>(_size, false);
			for (int i = 0; i < num; i++)
				_list[i] = other._list[i];
		}
		return *this;
	}

	/// <summary>
	/// Access operator.  Index must be within range or an assert will be issued in debug builds. Release builds do no range checking.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline const T &List<T, Tag>::operator[](int index) const { assert(index >= 0 && index < _num); return _list[index]; }

	/// <summary>
	/// Access operator.  Index must be within range or an assert will be issued in debug builds. Release builds do no range checking.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline T &List<T, Tag>::operator[](int index) { assert(index >= 0 && index < _num); return _list[index]; }

	/// <summary>
	/// Returns a pointer to the begining of the array.  Useful for iterating through the list in loops.
	/// Note: may return NULL if the list is empty.
	/// FIXME: Create an iterator template for this kind of thing.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline T *List<T, Tag>::Ptr() { return _list; }

	/// <summary>
	/// Returns a pointer to the begining of the array.  Useful for iterating through the list in loops.
	/// Note: may return NULL if the list is empty.
	/// FIXME: Create an iterator template for this kind of thing.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ const inline T *List<T, Tag>::Ptr() const { return _list; }

	/// <summary>
	/// Returns a reference to a new data element at the end of the list.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline T &List<T, Tag>::Alloc()
	{
		if (!_list)
			Resize(_granularity);
		if (_num == _size)
			Resize(_size + _granularity);
		return _list[_num++];
	}

	/// <summary>
	/// Increases the size of the list by one element and copies the supplied data into it.
	/// Returns the index of the new element.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline int List<T, Tag>::Append(T const &obj)
	{
		if (!_list)
			Resize(_granularity);
		if (_num == _size)
		{
			// this is a hack to fix our memset classes
			if (_granularity == 0)
				_granularity = 16;
			int newsize = _size + _granularity;
			Resize(newsize - newsize % _granularity);
		}
		_list[_num] = obj;
		_num++;
		return _num - 1;
	}

	/// <summary>
	/// Increases the size of the list by at leat one element if necessary and inserts the supplied data into it.
	/// Returns the index of the new element.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline int List<T, Tag>::Insert(T const &obj, int index)
	{
		if (!_list)
			Resize(_granularity);
		if (_num == _size)
		{
			// this is a hack to fix our memset classes
			if (_granularity == 0)
				_granularity = 16;
			int newsize = _size + _granularity;
			Resize(newsize - newsize % _granularity);
		}
		if (index < 0)
			index = 0;
		else if (index > _num)
			index = _num;
		for (int i = _num; i > index; --i)
			_list[i] = _list[i - 1];
		_num++;
		_list[index] = obj;
		return index;
	}

	/// <summary>
	/// adds the other list to this one
	/// Returns the size of the new combined list
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline int List<T, Tag>::Append(const List<T, Tag> &other)
	{
		if (!_list) 
		{
			// this is a hack to fix our memset classes
			if (_granularity == 0)
				_granularity = 16;
			Resize(_granularity);
		}
		int n = other.Num();
		for (int i = 0; i < n; i++)
			Append(other[i]);
		return Num();
	}

	/// <summary>
	/// Adds the data to the list if it doesn't already exist.  Returns the index of the data in the list.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline int List<T, Tag>::AddUnique(T const &obj)
	{
		int index = FindIndex(obj);
		if (index < 0)
			index = Append(obj);
		return index;
	}

	/// <summary>
	/// Searches for the specified data in the list and returns it's index.  Returns -1 if the data is not found.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline int List<T, Tag>::FindIndex(T const &obj) const
	{
		for (int i = 0; i < _num; i++)
			if (_list[i] == obj)
				return i;
		// Not found
		return -1;
	}

	/// <summary>
	/// Searches for the specified data in the list and returns it's address. Returns NULL if the data is not found.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline T *List<T, Tag>::Find(T const &obj) const { int i = FindIndex(obj); return (i >= 0 ? &_list[i] : nullptr); }

	/// <summary>
	/// Searches for a NULL pointer in the list.  Returns -1 if NULL is not found.
	/// NOTE: This function can only be called on lists containing pointers. Calling it on non-pointer lists will cause a compiler error.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline int List<T, Tag>::FindNull() const
	{
		for (int i = 0; i < _num; i++)
			if (_list[i] == nullptr)
				return i;
		// Not found
		return -1;
	}

	/// <summary>
	/// Takes a pointer to an element in the list and returns the index of the element. This is NOT a guarantee that the object is really in the list. 
	/// Function will assert in debug builds if pointer is outside the bounds of the list, but remains silent in release builds.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline int List<T, Tag>::IndexOf(T const *objptr) const { int index = objptr - _list; assert(index >= 0 && index < _num); return index; }

	/// <summary>
	/// Removes the element at the specified index and moves all data following the element down to fill in the gap. The number of elements in the list is reduced by one.  Returns false if the index is outside the bounds of the list.
	/// Note that the element is not destroyed, so any memory used by it may not be freed until the destruction of the list.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline bool List<T, Tag>::RemoveIndex(int index)
	{
		assert(_list != nullptr);
		assert(index >= 0 && index < _num);
		if (index < 0 || index >= _num)
			return false;
		_num--;
		for (int i = index; i < _num; i++)
			_list[i] = _list[i + 1];
		return true;
	}

	/// <summary>
	/// Removes the element at the specified index and moves the last element into its spot, rather than moving the whole array down by one. Of course, this doesn't maintain the order of 
	/// elements! The number of elements in the list is reduced by one.  
	/// return:	bool - false if the data is not found in the list.
	/// NOTE:The element is not destroyed, so any memory used by it may not be freed until the destruction of the list.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline bool List<T, Tag>::RemoveIndexFast(int index)
	{
		if (index < 0 || index >= _num)
			return false;
		_num--;
		if (index != _num)
			_list[index] = _list[_num];
		return true;
	}

	/// <summary>
	/// Removes the element if it is found within the list and moves all data following the element down to fill in the gap. The number of elements in the list is reduced by one.  Returns false if the data is not found in the list.  Note that
	/// the element is not destroyed, so any memory used by it may not be freed until the destruction of the list.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline bool List<T, Tag>::Remove(T const &obj) { int index = FindIndex(obj); return (index >= 0 ? RemoveIndex(index) : false); }

	/// <summary>
	/// Performs a qsort on the list using the supplied comparison function.  Note that the data is merely moved around the list, so any pointers to data within the list may no longer be valid.
	/// </summary>
	//template<typename T, memTag_t Tag>
	//__device__ inline void List<T, Tag>::Sort(cmp_t *compare)
	//{
	//	if (!_list)
	//		return;
	//	typedef int cmp_c(const void *, const void *);
	//	cmp_c *vCompare = (cmp_c *)compare;
	//	qsort((void *)list, (size_t)num, sizeof(T), vCompare);
	//}

	/// <summary>
	/// Performs a QuickSort on the list using the supplied sort algorithm.  
	/// Note: The data is merely moved around the list, so any pointers to data within the list may  no longer be valid.
	/// </summary>
	template<typename T, memTag_t Tag>
	__device__ inline void List<T, Tag>::SortWithTemplate(const Sorter<T> &sort) { if (_list == nullptr) return; sort.Sort(Ptr(), Num()); }

	/// <summary>
	/// Sorts a subsection of the list.
	/// </summary>
	//template<typename T, memTag_t Tag>
	//__device__ inline void List<T, Tag>::SortSubSection(int startIndex, int endIndex, cmp_t *compare)
	//{
	//	if (!_list)
	//		return;
	//	if (startIndex < 0)
	//		startIndex = 0;
	//	if (endIndex >= _num)
	//		endIndex = _num - 1;
	//	if (startIndex >= endIndex)
	//		return;
	//	typedef int cmp_c(const void *, const void *);
	//	cmp_c *vCompare = (cmp_c *)compare;
	//	qsort((void *)(&list[startIndex]), (size_t)(endIndex - startIndex + 1), sizeof(T), vCompare);
	//}

	/// <summary>
	/// Finds an item in a list based on any another datatype.  Your _type_ must overload operator()== for the _type_. If your _type_ is a ptr, use the FindFromGenericPtr function instead.
	/// </summary>
	template<typename T, memTag_t Tag, typename TCompare>
	__device__ T *FindFromGeneric(List<T, Tag> &list, const TCompare &other)
	{
		for (int i = 0; i < list.Num(); i++)
			if (list[i] == other)
				return &list[i];
		return nullptr;
	}

	/// <summary>
	/// FindFromGenericPtr
	/// </summary>
	template<typename T, memTag_t Tag, typename TCompare>
	__device__ T *FindFromGenericPtr(List<T, Tag> &list, const TCompare &other)
	{
		for (int i = 0; i < list.Num(); i++)
			if (*list[i] == other)
				return &list[i];
		return nullptr;
	}

}}
#endif /* __SYSTEM_LIST_H__ */
