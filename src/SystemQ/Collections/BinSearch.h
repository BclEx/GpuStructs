/// <summary>
/// The array elements have to be ordered in increasing order.
/// </summary>
#ifndef __SYSTEM_BINSEARCH_H__
#define __SYSTEM_BINSEARCH_H__
namespace Sys { namespace Collections {

	/// <summary>
	/// Finds the last array element which is smaller than the given value.
	/// </summary>
	template<class T>
	inline int BinSearch_Less(const T *array, const int arraySize, const T &value)
	{
		int len = arraySize;
		int mid = len;
		int offset = 0;
		while (mid > 0)
		{
			mid = len >> 1;
			if (array[offset + mid] < value)
				offset += mid;
			len -= mid;
		}
		return offset;
	}

	/// <summary>
	/// Finds the last array element which is smaller than or equal to the given value.
	/// </summary>
	template<class T>
	inline int BinSearch_LessEqual(const T *array, const int arraySize, const T &value)
	{
		int len = arraySize;
		int mid = len;
		int offset = 0;
		while (mid > 0)
		{
			mid = len >> 1;
			if (array[offset + mid] <= value)
				offset += mid;
			len -= mid;
		}
		return offset;
	}

	/// <summary>
	/// Finds the first array element which is greater than the given value.
	/// </summary>
	template<class T>
	inline int BinSearch_Greater(const T *array, const int arraySize, const T &value)
	{
		int len = arraySize;
		int mid = len;
		int offset = 0;
		int res = 0;
		while (mid > 0)
		{
			mid = len >> 1;
			if (array[offset + mid] > value)
				res = 0;
			else
			{
				offset += mid;
				res = 1;
			}
			len -= mid;
		}
		return offset+res;
	}

	/// <summary>
	/// Finds the first array element which is greater than or equal to the given value.
	/// </summary>
	template<class T>
	inline int BinSearch_GreaterEqual( const t *array, const int arraySize, const T &value)
	{
		int len = arraySize;
		int mid = len;
		int offset = 0;
		int res = 0;
		while (mid > 0)
		{
			mid = len >> 1;
			if (array[offset + mid] >= value)
				res = 0;
			else
			{
				offset += mid;
				res = 1;
			}
			len -= mid;
		}
		return offset + res;
	}

}}
#endif /* __SYSTEM_BINSEARCH_H__ */
