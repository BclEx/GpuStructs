#include "HashIndex.h"
namespace System { namespace Collections {

	int HashIndex::INVALID_INDEX[1] = { -1 };

	/// <summary>
	/// Init
	/// </summary>
	void HashIndex::Init(const int initialHashSize, const int initialIndexSize)
	{
		assert(Math::IsPowerOfTwo(initialHashSize));
		_hashSize = initialHashSize;
		_hash = INVALID_INDEX;
		_indexSize = initialIndexSize;
		_indexChain = INVALID_INDEX;
		_granularity = DEFAULT_HASH_GRANULARITY;
		_hashMask = _hashSize - 1;
		_lookupMask = 0;
	}

	/// <summary>
	/// Allocate
	/// </summary>
	void HashIndex::Allocate(const int newHashSize, const int newIndexSize)
	{
		assert(Math::IsPowerOfTwo( newHashSize ) );
		Free();
		_hashSize = newHashSize;
		_hash = new (TAG_IDLIB_HASH) int[_hashSize];
		memset(_hash, 0xff, _hashSize * sizeof(_hash[0]));
		_indexSize = newIndexSize;
		_indexChain = new (TAG_IDLIB_HASH) int[_indexSize];
		memset(_indexChain, 0xff, _indexSize * sizeof(_indexChain[0]));
		_hashMask = _hashSize - 1;
		_lookupMask = -1;
	}

	/// <summary>
	/// Free
	/// </summary>
	void HashIndex::Free() {
		if (_hash != INVALID_INDEX)
		{
			delete[] _hash;
			_hash = INVALID_INDEX;
		}
		if (_indexChain != INVALID_INDEX)
		{
			delete[] _indexChain;
			indexChain = INVALID_INDEX;
		}
		_lookupMask = 0;
	}

	/// <summary>
	/// ResizeIndex
	/// </summary>
	void HashIndex::ResizeIndex(const int newIndexSize)
	{
		if (newIndexSize <= _indexSize)
			return;
		int mod = newIndexSize % _granularity;
		int newSize = (!mod ? newIndexSize : newIndexSize + _granularity - mod);
		if (_indexChain == INVALID_INDEX)
		{
			_indexSize = newSize;
			return;
		}
		int *oldIndexChain = _indexChain;
		_indexChain = new (TAG_IDLIB_HASH) int[newSize];
		memcpy(_indexChain, oldIndexChain, _indexSize * sizeof(int));
		memset(_indexChain + _indexSize, 0xff, (newSize - _indexSize) * sizeof(int));
		delete[] oldIndexChain;
		_indexSize = newSize;
	}

	/// <summary>
	/// GetSpread
	/// </summary>
	int HashIndex::GetSpread() const
	{
		if (_hash == INVALID_INDEX)
			return 100;
		int totalItems = 0;
		int *numHashItems = new (TAG_IDLIB_HASH) int[_hashSize];
		int i;
		for (i = 0; i < _hashSize; i++)
		{
			numHashItems[i] = 0;
			for (int index = _hash[i]; index >= 0; index = _indexChain[index])
				numHashItems[i]++;
			totalItems += numHashItems[i];
		}
		// if no items in hash
		if (totalItems <= 1)
		{
			delete[] numHashItems;
			return 100;
		}
		int average = totalItems / _hashSize;
		int error = 0;
		for (i = 0; i < _hashSize; i++)
		{
			int e = abs(numHashItems[i] - average);
			if (e > 1)
				error += e - 1;
		}
		delete[] numHashItems;
		return 100 - (error * 100 / totalItems);
	}

}}