//hash.c
#include "Core.cu.h"

namespace Core
{

	__device__ void Hash::Hash()
	{
		First = nullptr;
		Count = 0;
		TableSize = 0;
		Table = nullptr;
	}

	__device__ void Hash::Clear()
	{
		HashElem *elem = First; // For looping over all elements of the table
		First = nullptr;
		SysEx::Free(Table); Table = nullptr;
		TableSize = 0;
		while (elem)
		{
			HashElem *nextElem = elem->Next;
			SysEx::Free(elem);
			elem = nextElem;
		}
		Count = 0;
	}

	/*
	** The hashing function.
	*/
	static unsigned int strHash(const char *z, int keyLength)
	{
		_assert(keyLength >= 0);
		int h = 0;
		while (keyLength > 0) { h = (h<<3) ^ h ^ __tolower[(unsigned char)*z++]; keyLength--; }
		return h;
	}


	/* Link pNew element into the hash table pH.  If pEntry!=0 then also
	** insert pNew into the pEntry hash bucket.
	*/
	static void insertElement(Hash *pH, struct _ht *pEntry, HashElem *pNew)
	{
		HashElem *pHead;       // First element already in pEntry
		if( pEntry ){
			pHead = pEntry->count ? pEntry->chain : 0;
			pEntry->count++;
			pEntry->chain = pNew;
		}else{
			pHead = 0;
		}
		if( pHead ){
			pNew->next = pHead;
			pNew->prev = pHead->prev;
			if( pHead->prev ){ pHead->prev->next = pNew; }
			else             { pH->first = pNew; }
			pHead->prev = pNew;
		}else{
			pNew->next = pH->first;
			if( pH->first ){ pH->first->prev = pNew; }
			pNew->prev = 0;
			pH->first = pNew;
		}
	}


	/* Resize the hash table so that it cantains "new_size" buckets.
	**
	** The hash table might fail to resize if sqlite3_malloc() fails or
	** if the new size is the same as the prior size.
	** Return TRUE if the resize occurs and false if not.
	*/
	static int rehash(Hash *pH, unsigned int new_size)
	{
		struct _ht *new_ht;            /* The new hash table */
		HashElem *elem, *next_elem;    /* For looping over existing elements */

#if MALLOC_SOFT_LIMIT>0
		if( new_size*sizeof(struct _ht)>SQLITE_MALLOC_SOFT_LIMIT ){
			new_size = SQLITE_MALLOC_SOFT_LIMIT/sizeof(struct _ht);
		}
		if( new_size==pH->htsize ) return 0;
#endif

		/* The inability to allocates space for a larger hash table is
		** a performance hit but it is not a fatal error.  So mark the
		** allocation as a benign. Use sqlite3Malloc()/memset(0) instead of 
		** sqlite3MallocZero() to make the allocation, as sqlite3MallocZero()
		** only zeroes the requested number of bytes whereas this module will
		** use the actual amount of space allocated for the hash table (which
		** may be larger than the requested amount).
		*/
		sqlite3BeginBenignMalloc();
		new_ht = (struct _ht *)sqlite3Malloc( new_size*sizeof(struct _ht) );
		sqlite3EndBenignMalloc();

		if( new_ht==0 ) return 0;
		sqlite3_free(pH->ht);
		pH->ht = new_ht;
		pH->htsize = new_size = sqlite3MallocSize(new_ht)/sizeof(struct _ht);
		memset(new_ht, 0, new_size*sizeof(struct _ht));
		for(elem=pH->first, pH->first=0; elem; elem = next_elem){
			unsigned int h = strHash(elem->pKey, elem->nKey) % new_size;
			next_elem = elem->next;
			insertElement(pH, &new_ht[h], elem);
		}
		return 1;
	}

	/* This function (for internal use only) locates an element in an
	** hash table that matches the given key.  The hash for this key has
	** already been computed and is passed as the 4th parameter.
	*/
	static HashElem *findElementGivenHash(
		const Hash *pH,     /* The pH to be searched */
		const char *pKey,   /* The key we are searching for */
		int nKey,           /* Bytes in key (not counting zero terminator) */
		unsigned int h      /* The hash for this key. */
		){
			HashElem *elem;                /* Used to loop thru the element list */
			int count;                     /* Number of elements left to test */

			if( pH->ht ){
				struct _ht *pEntry = &pH->ht[h];
				elem = pEntry->chain;
				count = pEntry->count;
			}else{
				elem = pH->first;
				count = pH->count;
			}
			while( count-- && ALWAYS(elem) ){
				if( elem->nKey==nKey && sqlite3StrNICmp(elem->pKey,pKey,nKey)==0 ){ 
					return elem;
				}
				elem = elem->next;
			}
			return 0;
	}

	/* Remove a single entry from the hash table given a pointer to that
	** element and a hash on the element's key.
	*/
	static void removeElementGivenHash(
		Hash *pH,         /* The pH containing "elem" */
		HashElem* elem,   /* The element to be removed from the pH */
		unsigned int h    /* Hash value for the element */
		){
			struct _ht *pEntry;
			if( elem->prev ){
				elem->prev->next = elem->next; 
			}else{
				pH->first = elem->next;
			}
			if( elem->next ){
				elem->next->prev = elem->prev;
			}
			if( pH->ht ){
				pEntry = &pH->ht[h];
				if( pEntry->chain==elem ){
					pEntry->chain = elem->next;
				}
				pEntry->count--;
				assert( pEntry->count>=0 );
			}
			sqlite3_free( elem );
			pH->count--;
			if( pH->count==0 ){
				assert( pH->first==0 );
				assert( pH->count==0 );
				sqlite3HashClear(pH);
			}
	}

	__device__ static void *Find(const Hash *pH, const char *pKey, int nKey)
	{
		HashElem *elem;    /* The element that matches key */
		unsigned int h;    /* A hash on key */

		assert( pH!=0 );
		assert( pKey!=0 );
		assert( nKey>=0 );
		if( pH->ht ){
			h = strHash(pKey, nKey) % pH->htsize;
		}else{
			h = 0;
		}
		elem = findElementGivenHash(pH, pKey, nKey, h);
		return (elem ? elem->Data : 0);
	}

	__device__ void *Hash::Insert(const char *key, int keyLength, void *data)
	{
		HashElem *newElem;   // New element added to the pH

		_assert(key != nullptr);
		_assert(keyLength >= 0);
		unsigned int hashCode = (TableSize ? getHashCode(key, keyLength) % TableSize : 0); // the hash of the key modulo hash table size
		HashElem *elem = findElementGivenHash(key, keyLength, hashCode); // Used to loop thru the element list
		if (elem)
		{
			void *oldData = elem->Data;
			if (data == 0)
				removeElementGivenHash(elem, hashCode);
			else
			{
				elem->Data = data;
				elem->Key = key;
				_assert(keyLength == elem->KeyLength);
			}
			return oldData;
		}
		if (data == 0) return 0;
		HashElem *newElem = (HashElem*)sqlite3Malloc( sizeof(HashElem) );
		if (newElem == nullptr) return data;
		newElem->Key = key;
		newElem->KeyLength = keyLength;
		newElem->Data = data;
		Count++;
		if (Count >= 10 && Count > 2 * TableSize)
		{
			if (rehash(pH, pH->count*2))
			{
				_assert(TableSize > 0);
				hashCode = getHashCode(key, keyLength) % TableSize;
			}
		}
		if( pH->ht ){
			insertElement(pH, &pH->ht[h], new_elem);
		}else{
			insertElement(pH, 0, new_elem);
		}
		return 0;
	}


}
