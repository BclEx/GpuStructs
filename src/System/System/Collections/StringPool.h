#ifndef __SYSTEM_STRINGPOOL_H__
#define __SYSTEM_STRINGPOOL_H__
#include "..\Text\String.h"
#include "List.h"
#include "HashIndex.h"
using namespace Sys::Text;
namespace Sys { namespace Collections {

	/// <summary>
	/// PoolString 
	/// </summary>
	class StringPool;
	class PoolString : public String
	{
	public:
		PoolString() { _numUsers = 0; }
		~PoolString() { assert(_numUsers == 0); }

		// returns total size of allocated memory
		size_t Allocated() const { return String::Allocated(); }
		// returns total size of allocated memory including size of string pool type
		size_t Size() const { return sizeof(*this) + Allocated(); }
		// returns a pointer to the pool this string was allocated from
		const StringPool *GetPool() const { return _pool; }

	private:
		friend class StringPool;
		StringPool *_pool;
		mutable int _numUsers;
	};

	/// <summary>
	/// StringPool
	/// </summary>
	class StringPool {
	public:
		StringPool() { _caseSensitive = true; }

		void SetCaseSensitive(bool caseSensitive);

		int Num() const { return _pool.Num(); }
		size_t Allocated() const;
		size_t Size() const;

		const PoolString *operator[](int index) const { return _pool[index]; }

		const PoolString *AllocString(const char *string);
		void FreeString(const PoolString *poolStr);
		const PoolString *CopyString(const PoolString *poolStr);
		void Clear();

	private:
		bool _caseSensitive;
		List<PoolString *> _pool;
		HashIndex _poolHash;
	};

	/// <summary>
	/// StringPool
	/// </summary>
	inline void StringPool::SetCaseSensitive(bool caseSensitive) { this->_caseSensitive = caseSensitive; }

	/// <summary>
	/// AllocString
	/// </summary>
	inline const PoolString *StringPool::AllocString(const char *string)
	{
		int hash = _poolHash.GenerateKey(string, _caseSensitive);
		if (_caseSensitive)
		{
			for (int i = _poolHash.First(hash); i != -1; i = _poolHash.Next(i))
				if (_pool[i]->Cmp(string) == 0)
				{
					_pool[i]->_numUsers++;
					return _pool[i];
				}
		} else {
			for (int i = _poolHash.First(hash); i != -1; i = _poolHash.Next(i)) 
				if (_pool[i]->Icmp(string) == 0)
				{
					_pool[i]->_numUsers++;
					return _pool[i];
				}
		}
		PoolString *poolString = new (TAG_IDLIB_STRING) PoolString;
		*static_cast<String *>(poolString) = string;
		poolString->_pool = this;
		poolString->_numUsers = 1;
		_poolHash.Add(hash, _pool.Append(poolString));
		return poolString;
	}

	/// <summary>
	/// FreeString
	/// </summary>
	inline void StringPool::FreeString(const PoolString *poolString)
	{
		assert(poolString->_numUsers >= 1);
		assert(poolString->_pool == this);
		poolString->_numUsers--;
		if (poolString->_numUsers <= 0)
		{
			int i;
			int hash = _poolHash.GenerateKey(poolString->c_str(), _caseSensitive);
			if (_caseSensitive)
			{ 
				for (i = _poolHash.First(hash); i != -1; i = _poolHash.Next(i))
					if (_pool[i]->Cmp(poolString->c_str()) == 0)
						break;
			} else {
				for (i = _poolHash.First(hash); i != -1; i = _poolHash.Next(i)) 
					if (_pool[i]->Icmp(poolString->c_str()) == 0)
						break;
			}
			assert(i != -1);
			assert(_pool[i] == poolString);
			delete _pool[i];
			_pool.RemoveIndex(i);
			_poolHash.RemoveIndex(hash, i);
		}
	}

	/// <summary>
	/// CopyString
	/// </summary>
	inline const PoolString *StringPool::CopyString(const PoolString *poolString)
	{
		assert(poolString->_numUsers >= 1);
		if (poolString->_pool == this)
		{
			// the string is from this pool so just increase the user count
			poolString->_numUsers++;
			return poolString;
		}
		// the string is from another pool so it needs to be re-allocated from this pool.
		return AllocString(poolString->c_str());
	}

	/// <summary>
	/// Clear
	/// </summary>
	inline void StringPool::Clear()
	{
		for (int i = 0; i < _pool.Num(); i++)
			_pool[i]->_numUsers = 0;
		_pool.DeleteContents(true);
		_poolHash.Free();
	}

	/// <summary>
	/// Allocated
	/// </summary>
	inline size_t StringPool::Allocated() const
	{
		size_t size = _pool.Allocated() + _poolHash.Allocated();
		for (int i = 0; i < _pool.Num(); i++)
			size += _pool[i]->Allocated();
		return size;
	}

	/// <summary>
	/// Size
	/// </summary>
	inline size_t StringPool::Size() const
	{
		size_t size = _pool.Size() + _poolHash.Size();
		for (int i = 0; i < _pool.Num(); i++)
			size += _pool[i]->Size();
		return size;
	}

}}
#endif /* __SYSTEM_STRINGPOOL_H__ */