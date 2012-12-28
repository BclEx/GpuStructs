#ifndef __SYSTEM_SORT_H__
#define __SYSTEM_SORT_H__
namespace Sys { namespace Collections {

	/// <summary>
	/// SwapValues
	/// </summary>
	template<typename T>
	__device__ inline void SwapValues(T &a, T &b)
	{
		T c = a;
		a = b;
		b = c;
	}

	/// <summary>
	/// Sort is an abstract template class for sorting an array of objects of the specified data type. The array of objects is sorted such that: Compare( array[i], array[i+1] ) <= 0 for all i
	/// </summary>
	template<typename T>
	class Sorter
	{
	public:
		__device__ virtual ~Sorter() { }
		__device__ virtual void Sort(T *base, unsigned int num) const = 0;
	};

	/// <summary>
	/// Sort_Quick is a sort template that implements the quick-sort algorithm on an array of objects of the specified data type.
	/// </summary>
	template<typename T, typename TDerived>
	class Sort_Quick : public Sorter<T>
	{
	public:
		__device__ virtual void Sort(T *base, unsigned int num) const
		{
			if (num <= 0)
				return;
			const int64 MAX_LEVELS = 128;
			int64 lo[MAX_LEVELS], hi[MAX_LEVELS];
			// 'lo' is the lower index, 'hi' is the upper index of the region of the array that is being sorted.
			lo[0] = 0;
			hi[0] = num - 1;
			for (int64 level = 0; level >= 0;)
			{
				int64 i = lo[level];
				int64 j = hi[level];
				// Only use quick-sort when there are 4 or more elements in this region and we are below MAX_LEVELS. Otherwise fall back to an insertion-sort.
				if ((j - i) >= 4 && level < (MAX_LEVELS - 1))
				{
					// Use the center element as the pivot. The median of a multi point sample could be used but simply taking the center works quite well.
					int64 pi = (i + j) / 2;
					// Move the pivot element to the end of the region.
					SwapValues(base[j], base[pi]);
					// Get a reference to the pivot element.
					T &pivot = base[j--];
					// Partition the region.
					do
					{
						while (static_cast<const TDerived *>(this)->Compare(base[i], pivot) < 0) { if (++i >= j) break; }
						while (static_cast<const TDerived *>(this)->Compare(base[j], pivot) > 0) { if (--j <= i) break; }
						if (i >= j) break;
						SwapValues(base[i], base[j]);
					}
					while(++i < --j);
					// Without these iterations sorting of arrays with many duplicates may become really slow because the partitioning can be very unbalanced.
					// However, these iterations are unnecessary if all elements are unique.
					while (static_cast<const TDerived *>(this)->Compare(base[i], pivot) <= 0 && i < hi[level]) { i++; }
					while (static_cast<const TDerived *>(this)->Compare(base[j], pivot) >= 0 && lo[level] < j) { j--; }
					// Move the pivot element in place.
					SwapValues(pivot, base[i]);
					assert(level < MAX_LEVELS - 1);
					lo[level+1] = i;
					hi[level+1] = hi[level];
					hi[level] = j;
					level++;
				}
				else
				{
					// Insertion-sort of the remaining elements.
					for (; i < j; j--)
					{
						int64 m = i;
						for (int64 k = i + 1; k <= j; k++)
							if (static_cast<const TDerived *>(this)->Compare(base[k], base[m]) > 0) 
								m = k;
						SwapValues(base[m], base[j]);
					}
					level--;
				}
			}
		}
	};

	/// <summary>
	/// Default quick-sort comparison function that can be used to sort scalars from small to large.
	/// </summary>
	template<typename T>
	class Sort_QuickDefault : public Sort_Quick<T, Sort_QuickDefault<T>>
	{
	public:
		__device__ int Compare(const T &a, const T &b ) const { return a - b; }
	};

	/// <summary>
	/// Specialization for floating point values to avoid an float-to-int conversion for every comparison.
	/// </summary>
	template<>
	class Sort_QuickDefault<float> : public Sort_Quick<float, Sort_QuickDefault<float>>
	{
	public:
		__device__ int Compare(const float &a, const float &b) const { return (a < b ? - 1 : (a > b ? 1 : 0)); }
	};

	/// <summary>
	/// Sort_Heap is a sort template class that implements the heap-sort algorithm on an array of objects of the specified data type.
	/// </summary>
	template<typename T, typename TDerived>
	class Sort_Heap : public Sorter<T>
	{
	public:
		__device__ virtual void Sort(T *base, unsigned int num) const
		{
			// get all elements in heap order
#if 1
			// O( n )
			for (unsigned int i = num / 2; i > 0; i--)
			{
				// sift down
				unsigned int parent = i - 1;
				for (unsigned int child = parent * 2 + 1; child < num; child = parent * 2 + 1)
				{
					if (child + 1 < num && static_cast<const TDerived *>(this)->Compare(base[child + 1], base[child]) > 0) child++;
					if (static_cast<const TDerived *>(this)->Compare(base[child], base[parent]) <= 0) break;
					SwapValues(base[parent], base[child]);
					parent = child;
				}
			}
#else
			// O(n log n)
			for (unsigned int i = 1; i < num; i++)
			{
				// sift up
				for (unsigned int child = i; child > 0;)
				{
					unsigned int parent = (child - 1) / 2;
					if (static_cast<const TDerived *>(this)->Compare(base[parent], base[child]) > 0) break;
					SwapValues(base[child], base[parent]);
					child = parent;
				}
			}
#endif
			// get sorted elements while maintaining heap order
			for (unsigned int i = num - 1; i > 0; i--)
			{
				SwapValues(base[0], base[i]);
				// sift down
				unsigned int parent = 0;
				for (unsigned int child = parent * 2 + 1; child < i; child = parent * 2 + 1)
				{
					if (child + 1 < i && static_cast<const TDerived *>( this )->Compare(base[child + 1], base[child]) > 0 ) child++;
					if (static_cast<const TDerived *>(this)->Compare(base[child], base[parent]) <= 0) break;
					SwapValues( base[parent], base[child] );
					parent = child;
				}
			}
		}
	};

	/// <summary>
	/// Default heap-sort comparison function that can be used to sort scalars from small to large.
	/// </summary>
	template<typename T>
	class Sort_HeapDefault : public Sort_Heap<T, Sort_HeapDefault<T>>
	{
	public:
		__device__ int Compare(const T &a, const T &b) const { return a - b; }
	};

	/// <summary>
	/// Sort_Insertion is a sort template class that implements the insertion-sort algorithm on an array of objects of the specified data type.
	/// </summary>
	template<typename T, typename TDerived>
	class Sort_Insertion : public Sorter<T>
	{
	public:
		__device__ virtual void Sort(T *base, unsigned int num) const
		{
			T *lo = base;
			T *hi = base + (num - 1);
			while (hi > lo)
			{
				T *max = lo;
				for (T *p = lo + 1; p <= hi; p++)
					if (static_cast<const TDerived *>(this)->Compare((*p), (*max)) > 0) max = p;
				SwapValues(*max, *hi);
				hi--;
			}
		}
	};

	/// <summary>
	/// Default insertion-sort comparison function that can be used to sort scalars from small to large.
	/// </summary>
	template<typename T>
	class Sort_InsertionDefault : public Sort_Insertion<T, Sort_InsertionDefault<T>>
	{
	public:
		__device__ int Compare(const T &a, const T &b) const { return a - b; }
	};

}}
#endif /* __SYSTEM_SORT_H__ */
