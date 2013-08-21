#ifndef __SYSTEM_HASHTABLE_H__
#define __SYSTEM_HASHTABLE_H__
namespace Sys { namespace Collections {

	/// <summary>
	/// HashNode is a generic node for a HashTable. It is specialized by the StringHashNode and CStringHashNode template classes.
	/// </summary>
	template<typename TKey, class TValue> 
	class HashNode
	{
	public:
		HashNode() : next( nullptr ) { }
		HashNode(const TKey &key, const TValue &value, HashNode *next) : Key(key), Value(value), Next(next) { }
		static int GetHash(const TKey &key, const int tableMask) { return key & tableMask; }
		static int Compare(const TKey &key1, const TKey &key2) { return (key1 < key2 ? -1 : (key1 > key2 ? 1 : 0)); }

	public:
		TKey Key;
		TValue Value;
		HashNode<TKey, TValue> *Next;
	};

	/// <summary>
	/// HashNode is a HashNode that provides for partial specialization for the HashTable, allowing the String class's Cmp function to be used for inserting values in sorted order.
	/// </summary>
	template<class TValue>
	class HashNode<String, TValue>
	{
	public:
		HashNode(const String &key, const TValue &value, HashNode *next) :	Key(key), Value(value), Next(next) { }
		static int GetHash(const String &key, const int tableMask) { return ( String::Hash(key) & tableMask); }
		static int Compare(const String &key1, const String &key2) { return String::Icmp(key1, key2); }

	public:
		String Key;
		TValue Value;
		HashNode<String, TValue> *Next;
	};

	/// <summary>
	/// HashNode is a HashNode that provides for a partial specialization for the HashTable, allowing the String class's Cmp function to be used for inserting values in sorted order. It also ensures that a copy of the the key is 
	/// stored in a String (to more closely model the original implementation of the HashTable).
	/// </summary>
	template<class TValue>
	class HashNode<const char*, TValue>
	{
	public:
		HashNode(const char* const &key, const TValue &value, HashNode *next) : Key(key), Value(value), Next(next) { } 
		static int	GetHash(const char* const &key, const int tableMask) { return ( String::Hash(key) & tableMask); }
		static int	Compare(const char* const &key1, const char* const &key2) { return String::Icmp(key1, key2); }

	public:
		String Key;	// char * keys must still get stored in an String
		TValue Value;
		HashNode<const char *, TValue> *Next;
	};

	/// <summary>
	/// HashTable is a general implementation of a hash table data type. It is slower than the HashIndex, but it can also be used for LinkedLists and other data structures, rather than just indexes and arrays. 
	/// It uses an arbitrary key type. For String keys, use the StringHashTable template specialization.
	/// </summary>
	template<typename TKey, class TValue>
	class HashTable
	{
	public:
		HashTable(const int tableSize = 256);
		HashTable(const HashTable &other);
		~HashTable();

		size_t Allocated() const; // returns total size of allocated memory
		size_t Size() const; // returns total size of allocated memory including size of hash table type

		TValue &Set(const TKey &key, const TValue &value);
		bool Get(const TKey &key, TValue **value = nullptr);
		bool Get(const TKey &key, const TValue **value = nullptr) const;
		bool Remove(const TKey &key);
		void Clear();
		void DeleteContents();

		int Num() const;
		TValue *GetIndex(const int index) const;
		bool GetIndexKey(const int index, TKey &key) const;

		int GetSpread() const;

		HashTable &operator=(const HashTable &other);

	protected:
		void Copy(const HashTable &other);

	private:
		typedef HashNode<TKey, TValue> hashnode_t;
		hashnode_t **_heads;
		int _tableSize;
		int _numEntries;
		int _tableSizeMask;
	};

	/// <summary>
	/// General hash table. Slower than HashIndex but it can also be used for linked lists and other data structures than just indexes or arrays.
	/// </summary>
	template<class TValue>
	class HashTable
	{
	public:
		HashTable(int tableSize = 256);
		HashTable(const HashTable &other);
		~HashTable();

		size_t Allocated() const; // returns total size of allocated memory
		size_t Size() const; // returns total size of allocated memory including size of hash table type

		void Set(const char *key, const TValue &value);
		bool Get(const char *key, TValue **value = nullptr) const;
		bool Remove(const char *key);
		void Clear();
		void DeleteContents();

		int Num() const; // the entire contents can be itterated over, but note that the exact index for a given element may change when new elements are added
		TValue *GetIndex(const int index) const;

		int GetSpread() const;

	protected:
		void Copy(const HashTable &other);

	private:
		struct hashnode_s
		{
			String Key;
			TValue Value;
			hashnode_s *Next;
			hashnode_s(const String &key, TValue value, hashnode_s *next) : Key( k ), Value(value), Next(next) { };
			hashnode_s(const char *key, TValue value, hashnode_s *next) : Key(key), Value(value), Next(next) { };
			int GetHash(const char *key) const;
		};
		hashnode_s **_heads;
		int _tableSize;
		int _numEntries;
		int _tablesizeMask;
	};


	/// <summary>
	/// HashTable
	/// </summary>
	template<typename TKey, class TValue>
	inline HashTable<TKey, TValue>::HashTable(const int tableSize)
	{
		assert(tableSize > 0);
		assert(Math::IsPowerOfTwo(tableSize));
		_tableSize = tableSize;
		_heads = new (TAG_IDLIB_HASH) hashnode_t *[tableSize];
		memset(_heads, 0, sizeof(hashnode_t *) * tableSize);
		_numEntries = 0;
		_tableSizeMask = tableSize - 1;
	}
	template<class TValue>
	inline HashTable<TValue>::HashTable(const int tableSize)
	{
		assert(tableSize > 0);
		assert(Math::IsPowerOfTwo(tableSize));
		_tableSize = tableSize;
		_heads = new (TAG_IDLIB_HASH) hashnode_s *[tableSize];
		memset(_heads, 0, sizeof(hashnode_s *) * tableSize);
		_numEntries = 0;
		_tablesizeMask = tableSize - 1;
	}

	/// <summary>
	/// HashTable
	/// </summary>
	template<typename TKey, class TValue>
	inline HashTable<TKey, TValue>::HashTable(const HashTable &other) { Copy(other); }
	template<class TValue>
	inline HashTable<TValue>::HashTable(const HashTable &other) { Copy(other); }

	/// <summary>
	/// ~HashTable
	/// </summary>
	template<typename TKey, class TValue>
	inline HashTable<TKey, TValue>::~HashTable()
	{
		Clear();
		delete [] _heads;
		_heads = nullptr;
		_tableSize = 0;
		_tableSizeMask = 0;
		_numEntries = 0;
	}
	template<class TValue>
	inline HashTable<TValue>::~HashTable()
	{
		Clear();
		delete[] _heads;
	}

	/// <summary>
	/// Allocated
	/// </summary>
	template<typename TKey, class TValue>
	inline size_t HashTable<TKey, TValue>::Allocated() const { return sizeof(_heads) * _tableSize + sizeof(hashnode_t*) * _numEntries; }
	template<class TValue>
	inline size_t HashTable<TValue>::Allocated() const { return sizeof(_heads) * _tableSize + sizeof(*_heads) * _numEntries; }

	/// <summary>
	/// Size
	/// </summary>
	template<typename TKey, class TValue>
	inline size_t HashTable<TKey, TValue>::Size() const { return sizeof(HashTable) + sizeof(_heads) * _tableSize + sizeof(hashnode_t*) * _numEntries; }
	template<class TValue>
	inline size_t HashTable<TValue>::Size() const { return sizeof(HashTable) + sizeof(_heads) * _tablesize + sizeof(*_heads) * _numEntries; }

	/// <summary>
	/// GetHash
	/// </summary>
	template<class TValue>
	inline int HashTable<TValue>::GetHash(const char *key) const { return (String::Hash(key) & _tableSizeMask); }

	/// <summary>
	/// Set
	/// </summary>
	template<typename TKey, class TValue>
	inline TValue &HashTable<TKey, TValue>::Set(const TKey &key, const TValue &value)
	{
		int hash = hashnode_t::GetHash(key, tableSizeMask);
		hashnode_t **nextPtr = &(_heads[hash]);
		hashnode_t *node = *nextPtr;
		for (; node != nullptr; nextPtr = &(node->Next), node = *nextPtr)
		{
			int s = node->Compare(node->Key, key);
			if (s == 0)
			{
				node->Value = value;
				return node->Value;
			}
			if (s > 0)
				break;
		}
		_numEntries++;
		*nextPtr = new (TAG_IDLIB_HASH) hashnode_t(key, value, _heads[hash]);
		(*nextPtr)->Next = node;
		return (*nextPtr)->Value;
	}
	template<class TValue>
	inline void HashTable<TValue>::Set(const char *key, const TValue &value)
	{
		int hash = hashnode_s::GetHash(key, _tableSizeMask);
		hashnode_s **nextPtr = &(_heads[hash])
			hashnode_s *node = *nextPtr
			for (; node != nullptr; nextPtr = &(node->Next), node = *nextPtr)
			{
				int s = node->Key.Cmp(key);
				if (s == 0)
				{
					node->Value = value;
					return;
				}
				if (s > 0) 
					break;
			}
			_numEntries++;
			*nextPtr = new (TAG_IDLIB_HASH) hashnode_s(key, value, _heads[hash]);
			(*nextPtr)->Next = node;
	}

	/// <summary>
	/// Get
	/// </summary>
	template<typename TKey, class TValue>
	inline bool HashTable<TKey, TValue>::Get(const TKey &key, TValue **value)
	{
		int hash = hashnode_t::GetHash(key, _tableSizeMask);
		for (hashnode_t * node = _heads[hash]; node != nullptr; node = node->Next)
		{
			int s = node->Compare(node->Key, key);
			if (s == 0)
			{
				if (value)
					*value = &node->Value;
				return true;
			}
			if (s > 0) 
				break;
		}
		if (value) 
			*value = nullptr;
		return false;
	}
	template<class TValue>
	inline bool HashTable<TValue>::Get(const char *key, TValue **value) const
	{
		int hash = hashnode_s::GetHash(key, _tableSizeMask);
		for (hashnode_s *node = _heads[hash]; node != nullptr; node = node->Next)
		{
			int s = node->Key.Cmp(key);
			if (s == 0)
			{
				if (value)
					*value = &node->Value;
				return true;
			}
			if (s > 0)
				break;
		}
		if (value)
			*value = nullptr;
		return false;
	}

	/// <summary>
	/// Get
	/// </summary>
	template<typename TKey, class TValue>
	inline bool HashTable<TKey, TValue>::Get(const TKey &key, const TValue **value ) const
	{
		int hash = hashnode_t::GetHash(key, _tableSizeMask);
		for (hashnode_t *node = _heads[hash]; node != nullptr; node = node->Next)
		{
			int s = node->Compare(node->Key, key);
			if (s == 0) {
				if (value)
					*value = &node->Value;
				return true;
			}
			if (s > 0)
				break;
		}
		if (value)
			*value = nullptr;
		return false;
	}

	/// <summary>
	/// GetIndex
	/// the entire contents can be itterated over, but note that the exact index for a given element may change when new elements are added
	/// </summary>
	template<typename TKey, class TValue>
	inline TValue *HashTable<TKey, TValue>::GetIndex(const int index) const
	{
		if (index < 0 || index > _numEntries)
		{
			assert(0);
			return nullptr;
		}
		int count = 0;
		for (int i = 0; i < _tableSize; i++)
		{
			for (hashnode_t *node = _heads[i]; node != nullptr; node = node->Next)
			{
				if (count == index)
					return &node->Value;
				count++;
			}
		}
		return nullptr;
	}
	template<class TValue>
	inline TValue *HashTable<TValue>::GetIndex(const int index) const
	{
		if (index < 0  ||  index > _numEntries)
		{
			assert(0);
			return nullptr;
		}
		int count = 0;
		for (int i = 0; i < _tablesize; i++)
		{
			for (hashnode_s	* node = _heads[i]; node != nullptr; node = node->Next)
			{
				if (count == index)
					return &node->value;
				count++;
			}
		}
		return nullptr;
	}

	/// <summary>
	/// GetIndexKey
	/// </summary>
	template<typename TKey, class TValue>
	inline bool HashTable<TKey, TValue>::GetIndexKey(const int index, TKey &key) const
	{
		if (index < 0 || index > _numEntries)
		{
			assert(0);
			return false;
		}
		int count = 0;
		for (int i = 0; i < _tableSize; i++)
		{
			for (hashnode_t *node = _heads[i]; node != nullptr; node = node->Next)
			{
				if (count == index)
				{
					key = node->Key;
					return true;
				}
				count++;
			}
		}
		return false;
	}

	/// <summary>
	/// Remove
	/// </summary>
	template<typename TKey, class TValue>
	inline bool HashTable<TKey, TValue>::Remove(const TKey &key)
	{
		int hash = hashnode_t::GetHash(key, _tableSizeMask);
		hashnode_t **head = &_heads[hash];
		if (*head)
		{
			hashnode_t *prev = nullptr;
			hashnode_t *node = *head;
			for (; node != nullptr; prev = node, node = node->Next)
			{
				if (node->Key == key)
				{
					if (prev)
						prev->Next = node->Next;
					else
						*head = node->Next;
					delete node;
					_numEntries--;
					return true;
				}
			}
		}
		return false;
	}
	template<class TValue>
	inline bool HashTable<TValue>::Remove(const char *key)
	{
		int hash = hashnode_s::GetHash(key, _tableSizeMask);
		hashnode_s **head = &_heads[hash];
		if (*head)
		{
			hashnode_s *prev == nullptr;
			hashnode_s *node = *head;
			for (; node != nullptr; prev = node, node = node->Next)
			{
				if (node->Key == key)
				{
					if (prev)
						prev->Next = node->Next;
					else
						*head = node->Next;
					delete node;
					_numEntries--;
					return true;
				}
			}
		}
		return false;
	}

	/// <summary>
	/// Clear
	/// </summary>
	template<typename TKey, class TValue>
	inline void HashTable<TKey, TValue>::Clear()
	{
		for (int i = 0; i < _tableSize; i++)
		{
			hashnode_t *next = _heads[i];
			while (next != nullptr)
			{
				hashnode_t *node = next;
				next = next->Next;
				delete node;
			}
			_heads[i] = nullptr;
		}
		_numEntries = 0;
	}
	template<class TValue>
	inline void HashTable<TValue>::Clear()
	{
		for (int i = 0; i < _tablesize; i++)
		{
			hashnode_s *next = _heads[i];
			while (next != nullptr)
			{
				hashnode_s *node = next;
				next = next->Next;
				delete node;
			}
			_heads[i] = nullptr;
		}
		_numEntries = 0;
	}

	/// <summary>
	/// DeleteContents
	/// </summary>
	template<typename TKey, class TValue>
	inline void HashTable<TKey, TValue>::DeleteContents()
	{
		for (int i = 0; i < _tableSize; i++)
		{
			hashnode_t *next = _heads[i];
			while (next != nullptr)
			{
				hashnode_t *node = next;
				next = next->Next;
				delete node->Value;
				delete node;
			}
			_heads[i] = nullptr;
		}
		_numEntries = 0;
	}
	template<class TValue>
	inline void HashTable<TValue>::DeleteContents()
	{
		for (int i = 0; i < _tablesize; i++)
		{
			hashnode_s *next = _heads[i];
			while (next != nullptr)
			{
				hashnode_s *node = next;
				next = next->Next;
				delete node->Value;
				delete node;
			}
			_heads[i] = nullptr;
		}
		_numEntries = 0;
	}

	/// <summary>
	/// Num
	/// </summary>
	template<typename TKey, class TValue>
	inline int HashTable<TKey, TValue>::Num() const { return _numEntries; }
	template<class TValue>
	inline int HashTable<TValue>::Num() const { return _numEntries; }

#if defined(ID_TYPEINFO)
#define __GNUC__ 99
#endif

	/// <summary>
	/// GetSpread
	/// </summary>
	template<typename TKey, class TValue>
	inline int HashTable<TKey, TValue>::GetSpread() const
	{
		if (!_numEntries)
			return 100;
		int average = _numEntries / _tableSize;
		int error = 0;
		for (int i = 0; i < _tableSize; i++)
		{
			int numItems = 0;
			for (hashnode_t *node = _heads[i]; node != nullptr; node = node->Next)
				numItems++;
			int e = abs(numItems - average);
			if (e > 1) 
				error += e - 1;
		}
		return 100 - (error * 100 / _numEntries);
	}
#if !defined(__GNUC__) || __GNUC__ < 4
	/// <summary>
	/// GetSpread
	/// </summary>
	template<class TValue>
	inline int HashTable<TValue>::GetSpread() const
	{
		if (!_numEntries)
			return 100;
		int average = _numEntries / _tableSize;
		int error = 0;
		for (int i = 0; i < _tableSize; i++)
		{
			int numItems = 0;
			for (hashnode_s *node = _heads[i]; node != nullptr; node = node->Next)
				numItems++;
			int e = abs(numItems - average);
			if (e > 1)
				error += e - 1;
		}
		return 100 - (error * 100 / _numEntries);
	}
#endif

#if defined(ID_TYPEINFO)
#undef __GNUC__
#endif

	/// <summary>
	/// operator=
	/// </summary>
	template<typename TKey, class TValue>
	inline HashTable<TKey, TValue> &HashTable<TKey, TValue>::operator=(const HashTable &other) { Copy(other); return *this; }

	/// <summary>
	/// Copy
	/// </summary>
	template<typename TKey, class TValue>
	inline void HashTable<TKey, TValue>::Copy(const HashTable &other)
	{
		if (&other == this)
			return;
		assert(other._tableSize > 0);

		_tableSize = other._tableSize;
		_heads = new (TAG_IDLIB_HASH) hashnode_t *[_tableSize];
		_numEntries = other._numEntries;
		_tableSizeMask = other._tableSizeMask;
		for (int i = 0; i < _tableSize; i++)
		{
			if (!other._heads[i])
			{
				_heads[i] = nullptr;
				continue;
			}
			hashnode_t **prev = &_heads[i];
			for (hashnode_t *node = other._heads[i]; node != nullptr; node = node->Next)
			{
				*prev = new (TAG_IDLIB_HASH) hashnode_t(node->Key, node->Value, nullptr);
				prev = &(*prev)->Next;
			}
		}
	}
	template<class TValue>
	inline void HashTable<TValue>::Copy(const HashTable &other)
	{
		if (&other == this)
			return;
		assert(other._tableSize > 0);

		_tableSize = other._tableSize;
		_heads = new (TAG_IDLIB_HASH) hashnode_s *[_tableSize];
		_numEntries = other._numEntries;
		_tableSizeMask = other._tableSizeMask;
		for (int i = 0; i < _tableSize; i++)
		{
			if (!other._heads[i])
			{
				_heads[i] = nullptr;
				continue;
			}
			hashnode_s **prev = &_heads[i];
			for (hashnode_s *node = other._heads[i]; node != nullptr; node = node->Next)
			{
				*prev = new (TAG_IDLIB_HASH) hashnode_s(node->Key, node->Value, nullptr);
				prev = &(*prev)->Next;
			}
		}
	}

}}
#endif /* __SYSTEM_HASHINDEX_H__ */
