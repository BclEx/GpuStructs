#ifndef __SYSTEM_VECTORSET_H__
#define __SYSTEM_VECTORSET_H__
#include "List.h"
namespace Sys { namespace Collections {

	/// <summary>
	/// Creates a set of vectors without duplicates.
	/// </summary>
	template<class T, int Dimension>
	class VectorSet : public List<T>
	{
	public:
		VectorSet();
		VectorSet(const T &mins, const T &maxs, const int boxHashSize, const int initialSize);

		/// <summary>
		/// returns total size of allocated memory
		/// </summary>
		size_t Allocated() const { return List<T>::Allocated() + _hash.Allocated(); }
		/// <summary>
		/// returns total size of allocated memory including size of type
		/// </summary>
		size_t Size() const { return sizeof(*this) + Allocated(); }

		void Init(const T &mins, const T &maxs, const int boxHashSize, const int initialSize);
		void ResizeIndex(const int newSize);
		void Clear();

		int FindVector(const T &v, const float epsilon);

	private:
		HashIndex _hash;
		T _mins;
		T _maxs;
		int _boxHashSize;
		float _boxInvSize[Dimension];
		float _boxHalfSize[Dimension];
	};

	/// <summary>
	/// Creates a set of vectors without duplicates.
	/// </summary>
	template<class T, int Dimension>
	inline VectorSet<T, Dimension>::VectorSet()
	{
		_hash.Clear(Math::IPow(_boxHashSize, Dimension), 128);
		_boxHashSize = 16;
		memset(_boxInvSize, 0, Dimension * sizeof(_boxInvSize[0]) );
		memset(_boxHalfSize, 0, Dimension * sizeof(_boxHalfSize[0]) );
	}

	/// <summary>
	/// Creates a set of vectors without duplicates.
	/// </summary>
	template<class T, int Dimension>
	inline VectorSet<T, Dimension>::VectorSet(const T &mins, const T &maxs, const int boxHashSize, const int initialSize) { Init(mins, maxs, boxHashSize, initialSize); }

	/// <summary>
	/// Creates a set of vectors without duplicates.
	/// </summary>
	template<class T, int Dimension>
	inline void VectorSet<T, Dimension>::Init(const T &mins, const T &maxs, const int boxHashSize, const int initialSize)
	{
		List<T>::AssureSize(initialSize);
		List<T>::SetNum(0, false);
		_hash.Clear(Math::IPow(boxHashSize, Dimension), initialSize);
		this->_mins = mins;
		this->_maxs = maxs;
		this->_boxHashSize = boxHashSize;
		for (int i = 0; i < Dimension; i++)
		{
			float boxSize = (maxs[i] - mins[i]) / (float)boxHashSize;
			boxInvSize[i] = 1.0f / boxSize;
			boxHalfSize[i] = boxSize * 0.5f;
		}
	}

	/// <summary>
	/// Creates a set of vectors without duplicates.
	/// </summary>
	template<class T, int Dimension>
	inline void VectorSet<T, Dimension>::ResizeIndex(const int newSize)
	{
		List<T>::Resize(newSize);
		_hash.ResizeIndex(newSize);
	}

	/// <summary>
	/// Creates a set of vectors without duplicates.
	/// </summary>
	template<class T, int Dimension>
	inline void VectorSet<T, Dimension>::Clear()
	{
		List<T>::Clear();
		_hash.Clear();
	}

	/// <summary>
	/// Creates a set of vectors without duplicates.
	/// </summary>
	template<class T, int Dimension>
	inline int VectorSet<T, Dimension>::FindVector(const T &v, const float epsilon)
	{
		int i, partialHashKey[Dimension];
		for (i = 0; i < Dimension; i++)
		{
			assert(epsilon <= _boxHalfSize[i]);
			partialHashKey[i] = (int)((v[i] - _mins[i] - _boxHalfSize[i]) * _boxInvSize[i]);
		}
		int hashKey;
		for (i = 0; i < (1 << Dimension); i++)
		{
			hashKey = 0;
			int j;
			for (j = 0; j < Dimension; j++)
			{
				hashKey *= _boxHashSize;
				hashKey += partialHashKey[j] + ((i >> j) & 1);
			}
			for (j = hash.First(hashKey); j >= 0; j = hash.Next(j))
			{
				const T &lv = (*this)[j];
				int k;
				for (k = 0; k < Dimension; k++)
					if (Math::Fabs(lv[k] - v[k]) > epsilon)
						break;
				if (k >= Dimension)
					return j;
			}
		}
		hashKey = 0;
		for (i = 0; i < Dimension; i++)
		{
			hashKey *= _boxHashSize;
			hashKey += (int)((v[i] - _mins[i]) * _boxInvSize[i]);
		}
		_hash.Add(hashKey, List<T>::Num());
		Append(v);
		return List<T>::Num() - 1;
	}

	/// <summary>
	/// Creates a subset without duplicates from an existing list with vectors.
	/// </summary>
	template<class T, int Dimension>
	class VectorSubset
	{
	public:
		VectorSubset();
		VectorSubset(const T &mins, const T &maxs, const int boxHashSize, const int initialSize);

		/// <summary>
		/// returns total size of allocated memory
		/// </summary>
		size_t Allocated() const { return List<T>::Allocated() + _hash.Allocated(); }
		/// <summary>
		/// returns total size of allocated memory including size of type
		/// </summary>
		size_t Size() const { return sizeof(*this) + Allocated(); }

		void Init(const T &mins, const T &maxs, const int boxHashSize, const int initialSize);
		void Clear();

		/// <summary>
		/// returns either vectorNum or an index to a previously found vector
		/// </summary>
		int FindVector(const T *vectorList, const int vectorNum, const float epsilon);

	private:
		HashIndex _hash;
		T _mins;
		T _maxs;
		int _boxHashSize;
		float _boxInvSize[Dimension];
		float _boxHalfSize[Ddimension];
	};

	/// <summary>
	/// VectorSubset
	/// </summary>
	template<class T, int Dimension>
	inline VectorSubset<T, Dimension>::VectorSubset()
	{
		_hash.Clear(Math::IPow(_boxHashSize, Dimension), 128);
		_boxHashSize = 16;
		memset(_boxInvSize, 0, Dimension * sizeof(_boxInvSize[0]));
		memset(_boxHalfSize, 0, Dimension * sizeof(_boxHalfSize[0]));
	}

	/// <summary>
	/// VectorSubset
	/// </summary>
	template<class T, int Dimension>
	inline VectorSubset<T, Dimension>::VectorSubset(const T &mins, const T &maxs, const int boxHashSize, const int initialSize) { Init(mins, maxs, boxHashSize, initialSize); }

	/// <summary>
	/// Init
	/// </summary>
	template<class T, int Dimension>
	inline void VectorSubset<T, Dimension>::Init(const T &mins, const T &maxs, const int boxHashSize, const int initialSize)
	{
		_hash.Clear(Math::IPow(boxHashSize, Dimension), initialSize);
		this->_mins = mins;
		this->_maxs = maxs;
		this->_boxHashSize = boxHashSize;
		for (int i = 0; i < Dimension; i++)
		{
			float boxSize = (maxs[i] - mins[i]) / (float)boxHashSize;
			boxInvSize[i] = 1.0f / boxSize;
			boxHalfSize[i] = boxSize * 0.5f;
		}
	}

	/// <summary>
	/// Creates a subset without duplicates from an existing list with vectors.
	/// </summary>
	template<class T, int Dimension>
	inline void VectorSubset<T, Dimension>::Clear()
	{
		List<T>::Clear();
		_hash.Clear();
	}

	/// <summary>
	/// FindVector
	/// </summary>
	template<class T, int Dimension>
	inline int VectorSubset<T, Dimension>::FindVector(const T *vectorList, const int vectorNum, const float epsilon)
	{
		const T &v = vectorList[vectorNum];
		int i, partialHashKey[Dimension];
		for (i = 0; i < Dimension; i++)
		{
			assert(epsilon <= _boxHalfSize[i]);
			partialHashKey[i] = (int)((v[i] - _mins[i] - _boxHalfSize[i]) * _boxInvSize[i]);
		}
		int hashkey;
		for (i = 0; i < (1 << Dimension); i++)
		{
			hashKey = 0;
			int j;
			for (j = 0; j < dimension; j++)
			{
				hashKey *= _boxHashSize;
				hashKey += partialHashKey[j] + ((i >> j) & 1);
			}
			for (j = hash.First(hashKey); j >= 0; j = hash.Next(j))
			{
				const type &lv = vectorList[j];
				int k;
				for (k = 0; k < Dimension; k++)
					if (Math::Fabs(lv[k] - v[k]) > epsilon)
						break;
				if (k >= Dimension)
					return j;
			}
		}
		hashKey = 0;
		for (i = 0; i < Dimension; i++)
		{
			hashKey *= _boxHashSize;
			hashKey += (int)((v[i] - _mins[i]) * _boxInvSize[i]);
		}
		_hash.Add(hashKey, vectorNum);
		return vectorNum;
	}

}}
#endif /* __SYSTEM_VECTORSET_H__ */
