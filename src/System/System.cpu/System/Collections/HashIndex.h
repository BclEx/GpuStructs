#ifndef __SYSTEM_HASHINDEX_H__
#define __SYSTEM_HASHINDEX_H__
namespace System { namespace Collections {

#define DEFAULT_HASH_SIZE			1024
#define DEFAULT_HASH_GRANULARITY	1024

	/// <summary>
	/// Fast hash table for indexes and arrays. Does not allocate memory until the first key/index pair is added.
	/// </summary>
	class HashIndex {
	public:
		static const int NULL_INDEX = -1;
		HashIndex();
		HashIndex( const int initialHashSize, const int initialIndexSize );
		~HashIndex();

		/// <summary>
		/// returns total size of allocated memory
		/// </summary>
		size_t Allocated() const;
		/// <summary>
		/// returns total size of allocated memory including size of hash index type
		/// </summary>
		size_t Size() const;

		HashIndex &operator=(const HashIndex &other);
		/// <summary>
		/// add an index to the hash, assumes the index has not yet been added to the hash
		/// </summary>
		void Add(const int key, const int index);
		/// <summary>
		/// remove an index from the hash
		/// </summary>
		void Remove(const int key, const int index);
		/// <summary>
		/// get the first index from the hash, returns -1 if empty hash entry
		/// </summary>
		int First(const int key) const;
		/// <summary>
		/// get the next index from the hash, returns -1 if at the end of the hash chain
		/// </summary>
		int Next(const int index) const;

		/// <summary>
		/// For porting purposes...
		/// </summary>
		int GetFirst(const int key) const { return First(key); }
		int GetNext(const int index) const { return Next(index); }

		/// <summary>
		/// insert an entry into the index and add it to the hash, increasing all indexes >= index
		/// </summary>
		void InsertIndex( const int key, const int index );
		/// <summary>
		/// remove an entry from the index and remove it from the hash, decreasing all indexes >= index
		/// </summary>
		void RemoveIndex( const int key, const int index );
		/// <summary>
		/// clear the hash
		/// </summary>
		void Clear();
		/// <summary>
		/// clear and resize
		/// </summary>
		void Clear(const int newHashSize, const int newIndexSize);
		/// <summary>
		/// free allocated memory
		/// </summary>
		void Free();
		/// <summary>
		/// get size of hash table
		/// </summary>
		int GetHashSize() const;
		/// <summary>
		/// get size of the index
		/// </summary>
		int GetIndexSize() const;
		/// <summary>
		/// set granularity
		/// </summary>
		void SetGranularity(const int newGranularity);
		/// <summary>
		/// force resizing the index, current hash table stays intact
		/// </summary>
		void ResizeIndex(const int newIndexSize);
		/// <summary>
		/// returns number in the range [0-100] representing the spread over the hash table
		/// </summary>
		int GetSpread() const;
		/// <summary>
		/// returns a key for a string
		/// </summary>
		int GenerateKey(const char *string, bool caseSensitive = true) const;
		/// <summary>
		/// returns a key for two integers
		/// </summary>
		int GenerateKey(const int n1, const int n2) const;
		/// <summary>
		/// returns a key for a single integer
		/// </summary>
		int GenerateKey(const int n) const;

	private:
		int _hashSize;
		int *_hash;
		int _indexSize;
		int *_indexChain;
		int _granularity;
		int _hashMask;
		int _lookupMask;

		static int INVALID_INDEX[1];

		void Init(const int initialHashSize, const int initialIndexSize);
		void Allocate(const int newHashSize, const int newIndexSize);
	};

	/// <summary>
	/// HashIndex
	/// </summary>
	inline HashIndex::HashIndex() { Init(DEFAULT_HASH_SIZE, DEFAULT_HASH_SIZE); }

	/// <summary>
	/// HashIndex
	/// </summary>
	inline HashIndex::HashIndex(const int initialHashSize, const int initialIndexSize) { Init(initialHashSize, initialIndexSize); }

	/// <summary>
	/// ~HashIndex
	/// </summary>
	inline HashIndex::~HashIndex() { Free(); }

	/// <summary>
	/// Allocated
	/// </summary>
	inline size_t HashIndex::Allocated() const { return _hashSize * sizeof(int) + _indexSize * sizeof(int); }

	/// <summary>
	/// Size
	/// </summary>
	inline size_t HashIndex::Size() const { return sizeof(*this) + Allocated(); }

	/// <summary>
	/// operator=
	/// </summary>
	inline HashIndex &HashIndex::operator=(const HashIndex &other)
	{
		_granularity = other._granularity;
		_hashMask = other._hashMask;
		_lookupMask = other._lookupMask;
		if (other._lookupMask == 0)
		{
			_hashSize = other._hashSize;
			_indexSize = other._indexSize;
			Free();
		}
		else
		{
			if (other._hashSize != _hashSize || _hash == INVALID_INDEX)
			{
				if (_hash != INVALID_INDEX )
					delete[] _hash;
				_hashSize = other._hashSize;
				_hash = new (TAG_IDLIB_HASH) int[_hashSize];
			}
			if (other._indexSize != _indexSize || _indexChain == INVALID_INDEX)
			{
				if (_indexChain != INVALID_INDEX)
					delete[] _indexChain;
				_indexSize = other._indexSize;
				_indexChain = new (TAG_IDLIB_HASH) int[_indexSize];
			}
			memcpy(_hash, other._hash, _hashSize * sizeof(_hash[0]));
			memcpy(_indexChain, other._indexChain, _indexSize * sizeof(_indexChain[0]));
		}
		return *this;
	}

	/// <summary>
	/// Add
	/// </summary>
	inline void HashIndex::Add(const int key, const int index)
	{
		assert(index >= 0);
		if (_hash == INVALID_INDEX)
			Allocate(_hashSize, index >= _indexSize ? index + 1 : _indexSize);
		else if (index >= _indexSize)
			ResizeIndex(index + 1);
		int h = (key & _hashMask);
		_indexChain[index] = _hash[h];
		_hash[h] = index;
	}

	/// <summary>
	/// Remove
	/// </summary>
	inline void HashIndex::Remove(const int key, const int index)
	{
		int k = (key & _hashMask);
		if (_hash == INVALID_INDEX)
			return;
		if (_hash[k] == index)
			_hash[k] = _indexChain[index];
		else
		{
			for (int i = _hash[k]; i != -1; i = _indexChain[i])
				if (_indexChain[i] == index)
				{
					_indexChain[i] = _indexChain[index];
					break;
				}
		}
		_indexChain[index] = -1;
	}


	/// <summary>
	/// First
	/// </summary>
	inline int HashIndex::First(const int key) const { return _hash[key & _hashMask & _lookupMask]; }

	/// <summary>
	/// Next
	/// </summary>
	inline int HashIndex::Next(const int index) const { assert(index >= 0 && index < _indexSize); return _indexChain[index & _lookupMask]; }

	/// <summary>
	/// InsertIndex
	/// </summary>
	inline void HashIndex::InsertIndex(const int key, const int index)
	{
		if (_hash != INVALID_INDEX)
		{
			int i;
			int max = index;
			for (i = 0; i < _hashSize; i++)
			{
				if (_hash[i] >= index)
				{
					_hash[i]++;
					if (_hash[i] > max)
						max = _hash[i];
				}
			}
			for (i = 0; i < _indexSize; i++)
			{
				if (_indexChain[i] >= index)
				{
					_indexChain[i]++;
					if (_indexChain[i] > max)
						max = _indexChain[i];
				}
			}
			if (max >= _indexSize)
				ResizeIndex(max + 1);
			for (i = max; i > index; i--)
				_indexChain[i] = _indexChain[i - 1];
			_indexChain[index] = -1;
		}
		Add(key, index);
	}

	/// <summary>
	/// RemoveIndex
	/// </summary>
	inline void HashIndex::RemoveIndex(const int key, const int index)
	{
		Remove(key, index);
		if (_hash != INVALID_INDEX)
		{
			int i;
			int max = index;
			for (i = 0; i < _hashSize; i++)
			{
				if (_hash[i] >= index)
				{
					if (_hash[i] > max)
						max = _hash[i];
					_hash[i]--;
				}
			}
			for (i = 0; i < _indexSize; i++)
			{
				if (_indexChain[i] >= index)
				{
					if (_indexChain[i] > max)
						max = _indexChain[i];
					_indexChain[i]--;
				}
			}
			for (i = index; i < max; i++)
				_indexChain[i] = _indexChain[i + 1];
			_indexChain[max] = -1;
		}
	}

	/// <summary>
	/// Clear
	/// </summary>
	inline void HashIndex::Clear()
	{
		// only clear the hash table because clearing the indexChain is not really needed
		if (_hash != INVALID_INDEX)
			memset(_hash, 0xff, _hashSize * sizeof(_hash[0]));
	}

	/// <summary>
	/// Clear
	/// </summary>
	inline void HashIndex::Clear(const int newHashSize, const int newIndexSize)
	{
		Free();
		_hashSize = newHashSize;
		_indexSize = newIndexSize;
	}

	/// <summary>
	/// GetHashSize
	/// </summary>
	inline int HashIndex::GetHashSize() const { return _hashSize; }

	/// <summary>
	/// GetIndexSize
	/// </summary>
	inline int HashIndex::GetIndexSize() const { return _indexSize; }

	/// <summary>
	/// SetGranularity
	/// </summary>
	inline void HashIndex::SetGranularity(const int newGranularity) { assert(newGranularity > 0); _granularity = newGranularity; }

	/// <summary>
	/// GenerateKey
	/// </summary>
	inline int HashIndex::GenerateKey(const char *string, bool caseSensitive) const { return ((caseSensitive ? String::Hash(string) : String::IHash(string)) & _hashMask); }

	/// <summary>
	/// GenerateKey
	/// </summary>
	inline int HashIndex::GenerateKey(const int n1, const int n2) const { return ((n1 + n2) & _hashMask); }

	/// <summary>
	/// GenerateKey
	/// </summary>
	inline int HashIndex::GenerateKey(const int n) const { return (n & _hashMask); }

}}
#endif /* __SYSTEM_HASHINDEX_H__ */
