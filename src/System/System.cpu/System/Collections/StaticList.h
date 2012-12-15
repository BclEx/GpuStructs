#ifndef __SYSTEM_STATICLIST_H__
#define __SYSTEM_STATICLIST_H__
#include "List.h"
namespace System { namespace Collections {

	/// <summary>
	/// A non-growing, memset-able list using no memory allocation.
	/// </summary>
	template<class T, int Size>
	class StaticList
	{
	public:
		StaticList();
		StaticList(const StaticList<T, Size> &other);
		~StaticList<T, Size>();

		void Clear();					// marks the list as empty.  does not deallocate or intialize data.
		int Num() const;				// returns number of elements in list
		int Max() const;				// returns the maximum number of elements in the list
		void SetNum(int newnum);		// set number of elements in list
		/// <summary>
		/// sets the number of elements in list and initializes any newly allocated elements to the given value
		/// </summary>
		void SetNum(int newNum, const T &initValue);

		size_t Allocated() const;		// returns total size of allocated memory
		size_t Size() const;			// returns total size of allocated memory including size of list type
		size_t MemoryUsed() const;		// returns size of the used elements in the list

		const T &operator[](int index) const;
		T &operator[](int index);

		T *Ptr();										// returns a pointer to the list
		const T *Ptr() const;							// returns a pointer to the list
		T *Alloc();										// returns reference to a new data element at the end of the list.  returns NULL when full.
		int Append(const T &obj);						// append element
		int Append(const StaticList<T, Size> &other);	// append list
		int AddUnique(const T &obj);				// add unique element
		int Insert(const T &obj, int index = 0);		// insert the element at the given index
		int FindIndex(const T &obj) const;				// find the index for the given element
		T *Find(T const &obj) const;					// find pointer to the given element
		int FindNull() const;							// find the index for the first NULL pointer in the list
		int IndexOf(const T *obj) const;				// returns the index for the pointer to an element in the list
		bool RemoveIndex(int index);					// remove the element at the given index
		bool RemoveIndexFast(int index);				// remove the element at the given index
		bool Remove(const T &obj);						// remove the element
		void Swap(StaticList<T, Size> &other);			// swap the contents of the lists
		void DeleteContents(bool clear);				// delete the contents of the list

		void Sort(const Sort<T> &sort = Sort_QuickDefault<T>());

	private:
		int _num;
		T _list[Size];
		/// <summary>
		/// resizes list to the given number of elements
		/// </summary>
		void Resize(int newsize);
	};

	/// <summary>
	/// StaticList
	/// </summary>
	template<class T, int Size>
	inline StaticList<T, Size>::StaticList() { _num = 0; }

	/// <summary>
	/// StaticList
	/// </summary>
	template<class T, int Size>
	inline StaticList<T, Size>::StaticList(const StaticList<T, Size> &other) { *this = other; }

	/// <summary>
	/// ~StaticList
	/// </summary>
	template<class T, int Size>
	inline StaticList<T, Size>::~StaticList() { }

	/// <summary>
	/// Sets the number of elements in the list to 0.  Assumes that type automatically handles freeing up memory.
	/// </summary>
	template<class T, int Size>
	inline void StaticList<T, Size>::Clear() { _num = 0; }

	/// <summary>
	/// Performs a QuickSort on the list using the supplied sort algorithm.  
	/// Note: The data is merely moved around the list, so any pointers to data within the list may  no longer be valid.
	/// </summary>
	template<class T, int Size>
	inline void StaticList<T, Size>::Sort(const Sort<T> &sort)
	{
		if (_list == nullptr)
			return;
		sort.Sort(Ptr(), Num());
	}

	/// <summary>
	/// Calls the destructor of all elements in the list.  Conditionally frees up memory used by the list. Note that this only works on lists containing pointers to objects and will cause a compiler error
	/// if called with non-pointers.  Since the list was not responsible for allocating the object, it has no information on whether the object still exists or not, so care must be taken to ensure that
	/// the pointers are still valid when this function is called.  Function will set all pointers in the list to NULL.
	/// </summary>
	template<class T, int Size>
	inline void StaticList<T, Size>::DeleteContents(bool clear)
	{
		for (int i = 0; i < _num; i++)
		{
			delete _list[i];
			_list[i] = nullptr;
		}
		if (clear)
			Clear();
		else
			memset(_list, 0, sizeof(_list));
	}

	/// <summary>
	/// Returns the number of elements currently contained in the list.
	/// </summary>
	template<class T, int Size>
	inline int StaticList<T, Size>::Num() const { return _num; }

	/// <summary>
	/// Returns the maximum number of elements in the list.
	/// </summary>
	template<class T, int Size>
	inline int StaticList<T, Size>::Max() const { return Size; }

	/// <summary>
	/// Allocated
	/// </summary>
	template<class T, int Size>
	inline size_t StaticList<T, Size>::Allocated() const { return Size * sizeof(T); }

	/// <summary>
	/// Size
	/// </summary>
	template<class T, int Size>
	inline size_t StaticList<T, Size>::Size() const { return sizeof(StaticList<T, Size>) + Allocated(); }

	/// <summary>
	/// Num
	/// </summary>
	template<class T, int Size>
	inline size_t StaticList<T, Size>::MemoryUsed() const { return _num * sizeof(_list[0]); }

	/// <summary>
	/// Set number of elements in list.
	/// </summary>
	template<class type,int size>
	inline void StaticList<type,size>::SetNum(int newnum)
	{
		assert(newnum >= 0);
		assert(newnum <= Size);
		_num = newnum;
	}

	/// <summary>
	/// SetNum
	/// </summary>
	template<class T, int Size>
	inline void StaticList<T, Size>::SetNum(int newNum, const T &initValue)
	{
		assert(newNum >= 0);
		newNum = Min(newNum, Size);
		assert(newNum <= Size);
		for (int i = _num; i < newNum; i++)
			_list[i] = initValue;
		_num = newNum;
	}

	/// <summary>
	/// Access operator.  Index must be within range or an assert will be issued in debug builds. Release builds do no range checking.
	/// </summary>
	template<class T, int Size>
	inline const T &StaticList<T, Size>::operator[](int index) const
	{
		assert(index >= 0);
		assert(index < num);
		return _list[index];
	}

	/// <summary>
	/// Access operator.  Index must be within range or an assert will be issued in debug builds. Release builds do no range checking.
	/// </summary>
	template<class T, int Size>
	inline T &StaticList<T, Size>::operator[](int index)
	{
		assert(index >= 0);
		assert(index < _num);
		return _list[index];
	}

	/// <summary>
	/// Returns a pointer to the begining of the array.  Useful for iterating through the list in loops.
	/// Note: may return NULL if the list is empty.
	/// FIXME: Create an iterator template for this kind of thing.
	/// </summary>
	template<class T, int Size>
	inline T *StaticList<T, Size>::Ptr() { return &_list[0]; }

	/// <summary>
	/// Returns a pointer to the begining of the array.  Useful for iterating through the list in loops.
	/// Note: may return NULL if the list is empty.
	/// FIXME: Create an iterator template for this kind of thing.
	/// </summary>
	template<class T, int Size>
	inline const T *StaticList<T, Size>::Ptr() const { return &_list[0]; }

	/// <summary>
	/// Returns a pointer to a new data element at the end of the list.
	/// </summary>
	template<class T, int Size>
	inline T *StaticList<T, Size>::Alloc() { return (_num >= Size ? nullptr : &_list[_num++]); }

	/// <summary>
	/// Increases the size of the list by one element and copies the supplied data into it. Returns the index of the new element, or -1 when list is full.
	/// </summary>
	template<class T, int Size>
	inline int StaticList<T, Size>::Append(T const &obj)
	{
		assert(_num < Size);
		if (_num < Size)
		{
			_list[_num] = obj;
			_num++;
			return _num - 1;
		}
		return -1;
	}

	/// <summary>
	/// Increases the size of the list by at leat one element if necessary and inserts the supplied data into it. Returns the index of the new element, or -1 when list is full.
	/// </summary>
	template<class T, int Size>
	inline int StaticList<T, Size>::Insert(T const &obj, int index)
	{
		assert(_num < Size);
		if (_num >= Size)
			return -1;
		assert(index >= 0);
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
	/// Adds the other list to this one. Returns the size of the new combined list
	/// </summary>
	template<class T, int Size>
	inline int StaticList<T, Size>::Append(const StaticList<T, Size> &other)
	{
		int n = other.Num();
		if (_num + n > Size)
			n = Size - _num;
		for (int i = 0; i < n; i++)
			_list[i + _num] = other._list[i];
		_num += n;
		return Num();
	}

	/// <summary>
	/// Adds the data to the list if it doesn't already exist.  Returns the index of the data in the list.
	/// </summary>
	template<class T, int Size>
	inline int StaticList<T, Size>::AddUnique(T const &obj)
	{
		int index = FindIndex(obj);
		if (index < 0)
			index = Append(obj);
		return index;
	}

	/// <summary>
	/// Searches for the specified data in the list and returns it's index.  Returns -1 if the data is not found.
	/// </summary>
	template<class T, int Size>
	inline int StaticList<T, Size>::FindIndex(T const &obj) const
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
	template<class T, int Size>
	inline type *StaticList<T, Size>::Find(T const &obj) const
	{
		int i = FindIndex(obj);
		return (i >= 0 ? (T *)&_list[i] : nullptr);
	}

	/// <summary>
	/// Searches for a NULL pointer in the list.  Returns -1 if NULL is not found.
	/// NOTE: This function can only be called on lists containing pointers. Calling it on non-pointer lists will cause a compiler error.
	/// </summary>
	template<class T, int Size>
	inline int StaticList<T, Size>::FindNull() const
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
	template<class T, int Size>
	inline int idStaticList<T, Size>::IndexOf(t const *objptr) const
	{
		int index = objptr - _list;
		assert(index >= 0);
		assert(index < _num);
		return index;
	}

	/// <summary>
	/// Removes the element at the specified index and moves all data following the element down to fill in the gap.
	/// The number of elements in the list is reduced by one.  Returns false if the index is outside the bounds of the list.
	/// Note that the element is not destroyed, so any memory used by it may not be freed until the destruction of the list.
	/// </summary>
	template<class T, int Size>
	inline bool StaticList<T, Size>::RemoveIndex(int index)
	{
		assert(index >= 0);
		assert(index < _num);
		if (index < 0 || index >= _num)
			return false;
		_num--;
		for (int i = index; i < _num; i++)
			_list[i] = _list[i + 1];
		return true;
	}

	/// <summary>
	/// Removes the element at the specified index and moves the last element into its spot, rather than moving the whole array down by one. Of course, this doesn't maintain the order of 
	/// elements! The number of elements in the list is reduced by one.  Returns False if the data is not found in the list.  
	/// NOTE: The element is not destroyed, so any memory used by it may not be freed until the destruction of the list.
	/// </summary>
	template< typename _type_,int size >
	inline bool StaticList<T, Size>::RemoveIndexFast(int index)
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
	template<class T, int Size>
	inline bool StaticList<T, Size>::Remove(T const &obj)
	{
		int index = FindIndex(obj);
		return (index >= 0 ? RemoveIndex(index); : false);
	}

	/// <summary>
	/// Swaps the contents of two lists
	/// </summary>
	template<class T, int Size>
	inline void StaticList<T, Size>::Swap(StaticList<T, Size> &other)
	{
		StaticList<T, Size> temp = *this;
		*this = other;
		other = temp;
	}

	// debug tool to find uses of list that are dynamically growing Ideally, most lists on shipping titles will explicitly set their size correctly instead of relying on allocate-on-add
	void BreakOnListGrowth();
	void BreakOnListDefault();

	/// <summary>
	/// Allocates memory for the amount of elements requested while keeping the contents intact. Contents are copied using their = operator so that data is correctly instantiated.
	/// </summary>
	template<class T, int Size>
	inline void StaticList<T, Size>::Resize(int newsize)
	{
		assert(newsize >= 0);
		// free up the list if no data is being reserved
		if (newsize <= 0)
		{
			Clear();
			return;
		}
		// not changing the size, so just exit
		if (newsize == Size)
			return;
		assert(newsize < Size);
		return;
	}

}}
#endif /* __SYSTEM_STATICLIST_H__ */