#ifndef __SYSTEM_HEAP_H__
#define __SYSTEM_HEAP_H__

// memory tag names are used to sort allocations for sys_dumpMemory and other reporting functions
enum memTag_t {
#define MEM_TAG(x) TAG_##x,
#include "Heap+Tags.h"
	TAG_NUM_TAGS,
};

__constant__ static const int MAX_TAGS = 256;
__device__ void *Mem_Alloc16(const int size, const memTag_t tag);
__device__ void Mem_Free16(void *ptr);
__device__ inline void *Mem_Alloc(const int size, const memTag_t tag) { return Mem_Alloc16(size, tag); }
__device__ inline void Mem_Free(void *ptr) { Mem_Free16(ptr); }
__device__ void *Mem_ClearedAlloc(const int size, const memTag_t tag);
__device__ char *Mem_CopyString(const char *in);
__device__ inline void *operator new(size_t s) { return Mem_Alloc(s, TAG_NEW); }
__device__ inline void *operator new(size_t s, memTag_t tag) { return Mem_Alloc(s, tag); }
__device__ inline void *operator new[](size_t s) { return Mem_Alloc(s, TAG_NEW); }
__device__ inline void *operator new[](size_t s, memTag_t tag) { return Mem_Alloc(s, tag); }
__device__ inline void operator delete(void *p) { Mem_Free(p); }
__device__ inline void operator delete(void *p, memTag_t tag) { Mem_Free(p); }
__device__ inline void operator delete[](void *p) { Mem_Free(p); }
__device__ inline void operator delete[](void *p, memTag_t tag) { Mem_Free(p); }

// Define replacements for the PS3 library's aligned new operator.
// Without these, allocations of objects with 32 byte or greater alignment may not go through our memory system.
namespace Sys {

	/// <summary>
	/// TempArray
	/// </summary>
	/// <about>
	/// TempArray is an array that is automatically free'd when it goes out of scope. There is no "cast" operator because these are very unsafe.
	/// The template parameter MUST BE POD!
	/// Compile time asserting POD-ness of the template parameter is complicated due to our vector classes that need a default constructor but are otherwise considered POD.
	/// </about>
	template <class T>
	class TempArray
	{
	public:
		__device__ TempArray(TempArray<T> &other);
		__device__ TempArray(unsigned int num);
		__device__ ~TempArray();

		__device__ T &operator [](unsigned int i) { assert(i < num); return _buffer[i]; }
		__device__ const T &operator [](unsigned int i) const { assert(i < num); return _buffer[i]; }

		__device__ T *Ptr() { return _buffer; }
		__device__ const T *Ptr() const { return _buffer; }
		__device__ size_t Size() const { return _num * sizeof(T); }
		__device__ unsigned int Num() const { return num; }
		__device__ void Zero() { memset(Ptr(), 0, Size()); }

	private:
		T *_buffer; // Ensure this buffer comes first, so this == &this->buffer
		unsigned int _num;
	};

	/// <summary>
	/// TempArray
	/// </summary>
	template <class T>
	__device__ inline TempArray<T>::TempArray(TempArray<T> &other) { _num = other._num; _buffer = other._buffer; other._num = 0; other._buffer = nullptr; }
	template <class T>
	__device__ inline TempArray<T>::TempArray(unsigned int num) { _num = num; _buffer = (T *)Mem_Alloc(num * sizeof(T), TAG_TEMP); }
	template <class T>
	__device__ inline TempArray<T>::~TempArray() { Mem_Free(_buffer); }


	/// <summary>
	/// BlockAlloc is a block-based allocator for fixed-size objects. All objects are properly constructed and destructed.
	/// </summary>
	/// <about>
	/// Block based allocator for fixed size objects.
	/// All objects of the 'type' are properly constructed and destructed when reused.
	/// </about>
#define BLOCK_ALLOC_ALIGNMENT 16
	//#define FORCE_DISCRETE_BLOCK_ALLOCS // Define this to force all block allocators to act like normal new/delete allocation for tool checking.
	template<class T, int BlockSize, memTag_t MemTag = TAG_BLOCKALLOC>
	class BlockAlloc
	{
	public:
		__device__ inline BlockAlloc(bool clear = false);
		__device__ inline ~BlockAlloc();

		__device__ size_t Allocated() const { return _total * sizeof(T); } // returns total size of allocated memory
		__device__ size_t Size() const { return sizeof(*this) + Allocated(); } // returns total size of allocated memory including size of (*this)

		__device__ inline void Shutdown();
		__device__ inline void SetFixedBlocks(int numBlocks);
		__device__ inline void FreeEmptyBlocks();
		__device__ inline T *Alloc();
		__device__ inline void Free(T *element);

		__device__ int GetTotalCount() const { return _total; }
		__device__ int GetAllocCount() const { return _active; }
		__device__ int GetFreeCount() const { return _total - _active; }

	private:
		/// <summary>
		/// element_t
		/// </summary>
		union element_t
		{
			T *Data; // this is a hack to make sure the save game system marks _type_ as saveable
			element_t *Next;
			byte Buffer[(CONST_MAX(sizeof(T), sizeof(element_t *)) + (BLOCK_ALLOC_ALIGNMENT - 1)) & ~(BLOCK_ALLOC_ALIGNMENT - 1)];
		};

		/// <summary>
		/// Block
		/// </summary>
		class Block
		{
		public:
			element_t Elements[BlockSize];
			Block *Next;
			element_t *Free; // list with free elements in this block (temp used only by FreeEmptyBlocks)
			int FreeCount; // number of free elements in this block (temp used only by FreeEmptyBlocks)
		};

		Block *_blocks;
		element_t *_free;
		int _total;
		int _active;
		bool _allowAllocs;
		bool _clearAllocs;

		__device__ inline void AllocNewBlock();
	};

	/// <summary>
	/// BlockAlloc
	/// </summary>
	template<class T, int BlockSize, memTag_t MemTag>
	__device__ inline BlockAlloc<T, BlockSize, MemTag>::BlockAlloc(bool clear) : _blocks(nullptr), _free(nullptr), _total(0), _active(0), _allowAllocs(true), _clearAllocs(clear) { }
	template<class T, int BlockSize, memTag_t MemTag>
	__device__ inline BlockAlloc<T, BlockSize, MemTag>::~BlockAlloc() { Shutdown(); }

	/// <summary>
	/// Alloc
	/// </summary>
	template<class T, int BlockSize, memTag_t MemTag>
	__device__ inline T *BlockAlloc<T, BlockSize, MemTag>::Alloc()
	{
#ifdef FORCE_DISCRETE_BLOCK_ALLOCS
		// for debugging tools
		return new T;
#else
		if (_free == nullptr) {
			if (!_allowAllocs) 
				return nullptr;
			AllocNewBlock();
		}
		_active++;
		element_t *element = _free;
		_free = _free->Next;
		element->Next = nullptr;

		T *t = (T *)element->Buffer;
		if (_clearAllocs)
			memset(t, 0, sizeof(T));
		//new (t) T;
		return t;
#endif
	}

	/// <summary>
	/// Free
	/// </summary>
	template<class T, int BlockSize, memTag_t MemTag>
	__device__ inline void BlockAlloc<T, BlockSize, MemTag>::Free(T *t)
	{
#ifdef FORCE_DISCRETE_BLOCK_ALLOCS
		// for debugging tools
		delete t;
#else
		if (t == nullptr)
			return;
		t->~T();
		element_t *element = (element_t *)(t);
		element->Next = _free;
		_free = element;
		_active--;
#endif
	}

	/// <summary>
	/// Shutdown
	/// </summary>
	template<class T, int BlockSize, memTag_t MemTag>
	__device__ inline void BlockAlloc<T, BlockSize, MemTag>::Shutdown()
	{
		while (_blocks != nullptr)
		{
			Block *block = _blocks;
			_blocks = _blocks->Next;
			Mem_Free(block);
		}
		_blocks = nullptr;
		_free = nullptr;
		_total = _active = 0;
	}

	/// <summary>
	/// SetFixedBlocks
	/// </summary>
	template<class T, int BlockSize, memTag_t MemTag>
	__device__ inline void BlockAlloc<T, BlockSize, MemTag>::SetFixedBlocks(int numBlocks)
	{
		int currentNumBlocks = 0;
		for (Block *block = _blocks; block != nullptr; block = block->Next)
			currentNumBlocks++;
		for (int i = currentNumBlocks; i < numBlocks; i++)
			AllocNewBlock();
		_allowAllocs = false;
	}


	/// <summary>
	/// AllocNewBlock
	/// </summary>
	template<class T, int BlockSize, memTag_t MemTag>
	__device__ inline void BlockAlloc<T, BlockSize, MemTag>::AllocNewBlock()
	{
		Block *block = (Block *)Mem_Alloc(sizeof(Block), MemTag);
		block->Next = _blocks;
		_blocks = block;
		for (int i = 0; i < BlockSize; i++)
		{
			block->Elements[i].Next = _free;
			_free = &block->Elements[i];
			assert((((UINT_PTR)free) & (BLOCK_ALLOC_ALIGNMENT - 1)) == 0);
		}
		_total += BlockSize;
	}

	/// <summary>
	/// FreeEmptyBlocks
	/// </summary>
	template<class T, int BlockSize, memTag_t MemTag>
	__device__ inline void BlockAlloc<T, BlockSize, MemTag>::FreeEmptyBlocks()
	{
		// first count how many free elements are in each block and build up a free chain per block
		for (Block *block = _blocks; block != nullptr; block = block->Next)
		{
			block->Free = nullptr;
			block->FreeCount = 0;
		}
		for (element_t *element = _free; element != nullptr;)
		{
			element_t *next = element->Next;
			for (Block *block = _blocks; block != nullptr; block = block->Next)
			{
				if (element >= block->Elements && element < block->Elements + BlockSize)
				{
					element->Next = block->Free;
					block->Free = element;
					block->FreeCount++;
					break;
				}
			}
			// if this assert fires, we couldn't find the element in any block
			assert(element->Next != next);
			element = next;
		}
		// now free all blocks whose free count == BlockSize
		Block *prevBlock = nullptr;
		for (Block *block = _blocks; block != nullptr;)
		{
			Block *next = block->Next;
			if (block->FreeCount == BlockSize)
			{
				if (prevBlock == nullptr)
				{
					assert(_blocks == block);
					_blocks = block->Next;
				}
				else
				{
					assert(prevBlock->Next == block);
					prevBlock->Next = block->Next;
				}
				Mem_Free(block);
				_total -= BlockSize;
			}
			else
				prevBlock = block;
			block = next;
		}
		// now rebuild the free chain
		_free = nullptr;
		for (Block *block = _blocks; block != nullptr; block = block->Next)
		{
			for (element_t *element = block->Free; element != nullptr;)
			{
				element_t *next = element->Next;
				element->Next = _free;
				_free = element;
				element = next;
			}
		}
	}

	/// <summary>
	/// Dynamic allocator, simple wrapper for normal allocations which can be interchanged with DynamicBlockAlloc.
	/// </summary>
	/// <about>
	/// No constructor is called for the 'type'.
	/// Allocated blocks are always 16 byte aligned.
	/// </about>
	template<class T, int BaseBlockSize, int MinBlockSize>
	class DynamicAlloc
	{
	public:
		__device__ DynamicAlloc();
		__device__ ~DynamicAlloc();

		__device__ void Init();
		__device__ void Shutdown();
		__device__ void SetFixedBlocks(int numBlocks) { }
		__device__ void SetLockMemory(bool lock) { }
		__device__ void FreeEmptyBaseBlocks() { }

		__device__ T *Alloc(const int num);
		__device__ T *Resize(T *ptr, const int num);
		__device__ void Free(T *ptr);
		__device__ const char *CheckMemory(const T *ptr) const;

		__device__ int GetNumBaseBlocks() const { return 0; }
		__device__ int GetBaseBlockMemory() const { return 0; }
		__device__ int GetNumUsedBlocks() const { return _numUsedBlocks; }
		__device__ int GetUsedBlockMemory() const { return _usedBlockMemory; }
		__device__ int GetNumFreeBlocks() const { return 0; }
		__device__ int GetFreeBlockMemory() const { return 0; }
		__device__ int GetNumEmptyBaseBlocks() const { return 0; }

	private:
		int _numUsedBlocks;		// number of used blocks
		int _usedBlockMemory;	// total memory in used blocks
		int _numAllocs;
		int _numResizes;
		int _numFrees;

		__device__ void Clear();
	};

	/// <summary>
	/// DynamicAlloc
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize>
	__device__ DynamicAlloc<T, BaseBlockSize, MinBlockSize>::DynamicAlloc() { Clear(); }
	template<class T, int BaseBlockSize, int MinBlockSize>
	__device__ DynamicAlloc<T, BaseBlockSize, MinBlockSize>::~DynamicAlloc() { Shutdown(); }

	/// <summary>
	/// Init
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize>
	__device__ void DynamicAlloc<T, BaseBlockSize, MinBlockSize>::Init() { }

	/// <summary>
	/// Shutdown
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize>
	__device__ void DynamicAlloc<T, BaseBlockSize, MinBlockSize>::Shutdown() { Clear(); }

	/// <summary>
	/// Alloc
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize>
	__device__ T *DynamicAlloc<T, BaseBlockSize, MinBlockSize>::Alloc(const int num)
	{
		_numAllocs++;
		if (num <= 0)
			return nullptr;
		_numUsedBlocks++;
		_usedBlockMemory += num * sizeof(T);
		return (T *)Mem_Alloc16(num * sizeof(T), TAG_BLOCKALLOC);
	}

	/// <summary>
	/// Resize
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize>
	__device__ T *DynamicAlloc<T, BaseBlockSize, MinBlockSize>::Resize(T *ptr, const int num)
	{
		_numResizes++;
		if (ptr == nullptr)
			return Alloc(num);
		if (num <= 0)
		{
			Free(ptr);
			return nullptr;
		}
		assert(0);
		return ptr;
	}

	/// <summary>
	/// Free
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize>
	__device__ void DynamicAlloc<T, BaseBlockSize, MinBlockSize>::Free(T *ptr)
	{
		_numFrees++;
		if (ptr == nullptr)
			return;
		Mem_Free16(ptr);
	}

	/// <summary>
	/// CheckMemory
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize>
	__device__ const char *DynamicAlloc<T, BaseBlockSize, MinBlockSize>::CheckMemory(const T *ptr) const { return nullptr; }

	/// <summary>
	/// Clear
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize>
	__device__ void DynamicAlloc<T, BaseBlockSize, MinBlockSize>::Clear()
	{
		_numUsedBlocks = 0;
		_usedBlockMemory = 0;
		_numAllocs = 0;
		_numResizes = 0;
		_numFrees = 0;
	}


	/// <summary>
	/// DynamicBlock
	/// Fast dynamic block allocator.
	/// </summary>
	/// <about>
	/// No constructor is called for the 'type'.
	/// Allocated blocks are always 16 byte aligned.
	/// </about>
	//#define DYNAMIC_BLOCK_ALLOC_CHECK
#include "Collections/BTree.h"
	template<class T>
	class DynamicBlock
	{
	public:
		__device__ T *GetMemory() const { return (T *)(((byte *)this) + sizeof(DynamicBlock<T>)); }
		__device__ int GetSize() const { return abs(Size); }
		__device__ void SetSize(int s, bool isBaseBlock) { Size = (isBaseBlock ? -s : s); }
		__device__ bool IsBaseBlock() const { return (Size < 0); }

#ifdef DYNAMIC_BLOCK_ALLOC_CHECK
		int ID[3];
		void *Allocator;
#endif
		int Size; // size in bytes of the block
		DynamicBlock<T> *Prev; // previous memory block
		DynamicBlock<T> *Next; // next memory block
		Sys::Collections::BTreeNode<DynamicBlock<T>, int> *Node; // node in the B-Tree with free blocks
	};

	/// <summary>
	/// DynamicBlockAlloc
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag = TAG_BLOCKALLOC>
	class DynamicBlockAlloc
	{
	public:
		__device__ DynamicBlockAlloc();
		__device__ ~DynamicBlockAlloc();

		__device__ void Init();
		__device__ void Shutdown();
		__device__ void SetFixedBlocks(int numBlocks);
		__device__ void SetLockMemory(bool lock);
		__device__ void FreeEmptyBaseBlocks();

		__device__ T *Alloc(const int num);
		__device__ T *Resize(T *ptr, const int num);
		__device__ void Free(T *ptr);
		__device__ const char *CheckMemory(const T *ptr) const;

		__device__ int GetNumBaseBlocks() const { return _numBaseBlocks; }
		__device__ int GetBaseBlockMemory() const { return _baseBlockMemory; }
		__device__ int GetNumUsedBlocks() const { return _numUsedBlocks; }
		__device__ int GetUsedBlockMemory() const { return _usedBlockMemory; }
		__device__ int GetNumFreeBlocks() const { return _numFreeBlocks; }
		__device__ int GetFreeBlockMemory() const { return _freeBlockMemory; }
		__device__ int GetNumEmptyBaseBlocks() const;

	private:
		DynamicBlock<T> *_firstBlock;	// first block in list in order of increasing address
		DynamicBlock<T> *_lastBlock;	// last block in list in order of increasing address
		Sys::Collections::BTree<DynamicBlock<T>, int, 4> _freeTree; // B-Tree with free memory blocks
		bool _allowAllocs;		// allow base block allocations
		bool _lockMemory;		// lock memory so it cannot get swapped out
#ifdef DYNAMIC_BLOCK_ALLOC_CHECK
		int BlockId[3];
#endif
		int _numBaseBlocks;		// number of base blocks
		int _baseBlockMemory;	// total memory in base blocks
		int _numUsedBlocks;		// number of used blocks
		int _usedBlockMemory;	// total memory in used blocks
		int _numFreeBlocks;		// number of free blocks
		int _freeBlockMemory;	// total memory in free blocks

		int _numAllocs;
		int _numResizes;
		int _numFrees;

		memTag_t _tag;

		__device__ void Clear();
		__device__ DynamicBlock<T> *AllocInternal(const int num);
		__device__ DynamicBlock<T> *ResizeInternal(DynamicBlock<T> *block, const int num);
		__device__ void FreeInternal(DynamicBlock<T> *block);
		__device__ void LinkFreeInternal(DynamicBlock<T> *block);
		__device__ void UnlinkFreeInternal(DynamicBlock<T> *block);
		__device__ void CheckMemory() const;
	};

	/// <summary>
	/// DynamicBlockAlloc
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::DynamicBlockAlloc() { _tag = Tag; Clear(); }
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::~DynamicBlockAlloc() { Shutdown(); }

	/// <summary>
	/// Init
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ void DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::Init() { _freeTree.Init(); }

	/// <summary>
	/// Shutdown
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ void DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::Shutdown()
	{
		for (DynamicBlock<T> *block = _firstBlock; block != nullptr; block = block->Next)
		{
			if (block->node == nullptr)
				FreeInternal(block);
		}
		for (DynamicBlock<T> *block = _firstBlock; block != nullptr; block = _firstBlock)
		{
			_firstBlock = block->Next;
			assert(block->IsBaseBlock());
			//if (_lockMemory)
			//	Lib::sys->UnlockMemory(block, block->GetSize() + (int)sizeof(DynamicBlock<T>));
			Mem_Free16(block);
		}
		_freeTree.Shutdown();
		Clear();
	}

	/// <summary>
	/// SetFixedBlocks
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ void DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::SetFixedBlocks(int numBlocks)
	{
		for (int i = _numBaseBlocks; i < numBlocks; i++)
		{
			DynamicBlock<T> *block = (DynamicBlock<T> *)Mem_Alloc16(BaseBlockSize, Tag);
			//if (_lockMemory)
			//	Lib::sys->LockMemory(block, BaseBlockSize);
#ifdef DYNAMIC_BLOCK_ALLOC_CHECK
			memcpy(block->ID, _blockId, sizeof(block->ID));
			block->Allocator = (void*)this;
#endif
			block->SetSize(BaseBlockSize - (int)sizeof(DynamicBlock<T>), true);
			block->Next = nullptr;
			block->Prev = lastBlock;
			if (_lastBlock)
				_lastBlock->next = block;
			else
				_firstBlock = block;
			lastBlock = block;
			block->Node = nullptr;
			FreeInternal(block);
			_numBaseBlocks++;
			_baseBlockMemory += BaseBlockSize;
		}
		_allowAllocs = false;
	}

	/// <summary>
	/// SetLockMemory
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ void DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::SetLockMemory(bool lock) { _lockMemory = lock; }

	/// <summary>
	/// FreeEmptyBaseBlocks
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ void DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::FreeEmptyBaseBlocks()
	{
		DynamicBlock<T> *next;
		for (DynamicBlock<T> *block = firstBlock; block != nullptr; block = next)
		{
			next = block->Next;
			if (block->IsBaseBlock() && block->Node != nullptr && (next == nullptr || next->IsBaseBlock()))
			{
				UnlinkFreeInternal(block);
				if (block->Prev)
					block->Prev->Next = block->Next;
				else 
					_firstBlock = block->Next;
				if (block->Next)
					block->Next->Prev = block->Prev;
				else
					_lastBlock = block->Prev;
				//if (_lockMemory) 
				//	Lib::sys->UnlockMemory(block, block->GetSize() + (int)sizeof(DynamicBlock<T>));
				_numBaseBlocks--;
				_baseBlockMemory -= block->GetSize() + (int)sizeof(DynamicBlock<T>);
				Mem_Free16(block);
			}
		}
#ifdef DYNAMIC_BLOCK_ALLOC_CHECK
		CheckMemory();
#endif
	}

	/// <summary>
	/// GetNumEmptyBaseBlocks
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ int DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::GetNumEmptyBaseBlocks() const
	{
		int numEmptyBaseBlocks = 0;
		for (DynamicBlock<T> *block = _firstBlock; block != nullptr; block = block->Next)
			if (block->IsBaseBlock() && block->Node != nullptr && (block->next == nullptr || block->Next->IsBaseBlock()))
				numEmptyBaseBlocks++;
		return numEmptyBaseBlocks;
	}

	/// <summary>
	/// Alloc
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ T *DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::Alloc(const int num)
	{
		_numAllocs++;
		if (num <= 0)
			return nullptr;
		DynamicBlock<T> *block = AllocInternal(num);
		if (block == nullptr)
			return nullptr;
		block = ResizeInternal(block, num);
		if (block == nullptr)
			return nullptr;
#ifdef DYNAMIC_BLOCK_ALLOC_CHECK
		CheckMemory();
#endif
		_numUsedBlocks++;
		_usedBlockMemory += block->GetSize();
		return block->GetMemory();
	}

	/// <summary>
	/// Resize
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ T *DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::Resize(T *ptr, const int num)
	{
		_numResizes++;
		if (ptr == nullptr)
			return Alloc(num);
		if (num <= 0)
		{
			Free(ptr);
			return null;
		}
		DynamicBlock<T> *block = (DynamicBlock<T> *)(((byte *)ptr) - (int)sizeof(DynamicBlock<T>));
		_usedBlockMemory -= block->GetSize();
		block = ResizeInternal(block, num);
		if (block == nullptr)
			return nullptr;
#ifdef DYNAMIC_BLOCK_ALLOC_CHECK
		CheckMemory();
#endif
		_usedBlockMemory += block->GetSize();
		return block->GetMemory();
	}

	/// <summary>
	/// Free
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ void DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::Free(T *ptr)
	{
		_numFrees++;
		if (ptr == nullptr)
			return;
		DynamicBlock<T> *block = (DynamicBlock<T> *)(((byte *)ptr) - (int)sizeof(DynamicBlock<T>));
		_numUsedBlocks--;
		_usedBlockMemory -= block->GetSize();
		FreeInternal(block);
#ifdef DYNAMIC_BLOCK_ALLOC_CHECK
		CheckMemory();
#endif
	}

	/// <summary>
	/// CheckMemory
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ const char *DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::CheckMemory(const T *ptr) const
	{
		if (ptr == nullptr)
			return nullptr;
		DynamicBlock<T> *block = (DynamicBlock<T> *)(((byte *)ptr) - (int)sizeof(DynamicBlock<T>));
		if (block->Node != nullptr)
			return "memory has been freed";
#ifdef DYNAMIC_BLOCK_ALLOC_CHECK
		if (block->ID[0] != 0x11111111 || block->ID[1] != 0x22222222 || block->ID[2] != 0x33333333)
			return "memory has invalid id";
		if (block->Allocator != (void*)this)
			return "memory was allocated with different allocator";
#endif
		// base blocks can be larger than baseBlockSize which can cause this code to fail
		/*
		DynamicBlock<T> *base;
		for (base = _firstBlock; base != nullptr; base = base->Next)
		if (base->IsBaseBlock() && ((int)block) >= ((int)base) && ((int)block) < ((int)base) + baseBlockSize)
		break;
		if (base == nullptr)
		return "no base block found for memory";
		*/
		return nullptr;
	}

	/// <summary>
	/// Clear
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ void DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::Clear()
	{
		_firstBlock = _lastBlock = nullptr;
		_allowAllocs = true;
		_lockMemory = false;
		_numBaseBlocks = 0;
		_baseBlockMemory = 0;
		_numUsedBlocks = 0;
		_usedBlockMemory = 0;
		_numFreeBlocks = 0;
		_freeBlockMemory = 0;
		_numAllocs = 0;
		_numResizes = 0;
		_numFrees = 0;
#ifdef DYNAMIC_BLOCK_ALLOC_CHECK
		_blockId[0] = 0x11111111;
		_blockId[1] = 0x22222222;
		_blockId[2] = 0x33333333;
#endif
	}

	/// <summary>
	/// AllocInternal
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ DynamicBlock<T> *DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::AllocInternal(const int num)
	{
		int alignedBytes = (num * sizeof(T) + 15) & ~15;
		DynamicBlock<T> *block = _freeTree.FindSmallestLargerEqual(alignedBytes);
		if (block != nullptr)
			UnlinkFreeInternal( block );
		else if (_allowAllocs)
		{
			int allocSize = Max(BaseBlockSize, alignedBytes + (int)sizeof(DynamicBlock<T>));
			block = (DynamicBlock<T> *)Mem_Alloc16(allocSize, Tag);
			//if (_lockMemory)
			//	Lib::sys->LockMemory(block, BaseBlockSize);
#ifdef DYNAMIC_BLOCK_ALLOC_CHECK
			memcpy(block->ID, _blockId, sizeof(block->ID));
			block->Allocator = (void*)this;
#endif
			block->SetSize(allocSize - (int)sizeof(DynamicBlock<T>), true);
			block->Next = nullptr;
			block->Prev = _lastBlock;
			if (_lastBlock)
				_lastBlock->Next = block;
			else 
				_firstBlock = block;
			_lastBlock = block;
			block->Node = nullptr;
			_numBaseBlocks++;
			_baseBlockMemory += allocSize;
		}
		return block;
	}

	/// <summary>
	/// ResizeInternal
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ DynamicBlock<T> *DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::ResizeInternal(DynamicBlock<T> *block, const int num)
	{
#ifdef DYNAMIC_BLOCK_ALLOC_CHECK
		assert(block->ID[0] == 0x11111111 && block->ID[1] == 0x22222222 && block->ID[2] == 0x33333333 && block->Allocator == (void*)this);
#endif
		int alignedBytes = (num * sizeof(T) + 15) & ~15;
		// if the new size is larger
		if (alignedBytes > block->GetSize())
		{
			DynamicBlock<T> *nextBlock = block->Next;
			// try to annexate the next block if it's free
			if (nextBlock && !nextBlock->IsBaseBlock() && nextBlock->node != nullptr && block->GetSize() + (int)sizeof(DynamicBlock<T>) + nextBlock->GetSize() >= alignedBytes)
			{
				UnlinkFreeInternal(nextBlock);
				block->SetSize(block->GetSize() + (int)sizeof(DynamicBlock<T>) + nextBlock->GetSize(), block->IsBaseBlock());
				block->Next = nextBlock->Next;
				if (nextBlock->Next)
					nextBlock->Next->Prev = block;
				else
					_lastBlock = block;
			}
			else
			{
				// allocate a new block and copy
				DynamicBlock<T> *oldBlock = block;
				block = AllocInternal(num);
				if (block == nullptr)
					return null;
				memcpy(block->GetMemory(), oldBlock->GetMemory(), oldBlock->GetSize());
				FreeInternal(oldBlock);
			}
		}
		// if the unused space at the end of this block is large enough to hold a block with at least one element
		if (block->GetSize() - alignedBytes - (int)sizeof(DynamicBlock<T>) < Max(minBlockSize, (int)sizeof(type)))
			return block;
		DynamicBlock<T> *newBlock;
		newBlock = (DynamicBlock<T> *)(((byte *)block) + (int)sizeof(DynamicBlock<T>) + alignedBytes);
#ifdef DYNAMIC_BLOCK_ALLOC_CHECK
		memcpy(newBlock->ID, _blockId, sizeof(newBlock->ID));
		newBlock->Allocator = (void*)this;
#endif
		newBlock->SetSize(block->GetSize() - alignedBytes - (int)sizeof(DynamicBlock<T>), false);
		newBlock->Next = block->Next;
		newBlock->Prev = block;
		if (newBlock->Next)
			newBlock->Next->Prev = newBlock;
		else
			_lastBlock = newBlock;
		newBlock->Node = nullptr;
		block->Next = newBlock;
		block->SetSize(alignedBytes, block->IsBaseBlock());
		FreeInternal(newBlock);
		return block;
	}

	/// <summary>
	/// FreeInternal
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ void DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::FreeInternal(DynamicBlock<T> *block)
	{
#ifdef DYNAMIC_BLOCK_ALLOC_CHECK
		assert(block->ID[0] == 0x11111111 && block->ID[1] == 0x22222222 && block->ID[2] == 0x33333333 && block->Allocator == (void*)this);
#endif
		assert(block->Node == nullptr);
		// try to merge with a next free block
		DynamicBlock<T> *nextBlock = block->Next;
		if (nextBlock && !nextBlock->IsBaseBlock() && nextBlock->Node != nullptr)
		{
			UnlinkFreeInternal(nextBlock);
			block->SetSize(block->GetSize() + (int)sizeof(DynamicBlock<T>) + nextBlock->GetSize(), block->IsBaseBlock());
			block->Next = nextBlock->Next;
			if (nextBlock->Next)
				nextBlock->Next->Prev = block;
			else
				_lastBlock = block;
		}
		// try to merge with a previous free block
		DynamicBlock<T> *prevBlock = block->prev;
		if (prevBlock && !block->IsBaseBlock() && prevBlock->node != nullptr)
		{
			UnlinkFreeInternal(prevBlock);
			prevBlock->SetSize(prevBlock->GetSize() + (int)sizeof(DynamicBlock<T>) + block->GetSize(), prevBlock->IsBaseBlock());
			prevBlock->Next = block->Next;
			if (block->Next)
				block->Next->Prev = prevBlock;
			else
				_lastBlock = prevBlock;
			LinkFreeInternal(prevBlock);
		}
		else
			LinkFreeInternal(block);
	}

	/// <summary>
	/// LinkFreeInternal
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ inline void DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::LinkFreeInternal(DynamicBlock<T> *block) { block->Node = _freeTree.Add(block, block->GetSize()); _numFreeBlocks++; _freeBlockMemory += block->GetSize(); }

	/// <summary>
	/// UnlinkFreeInternal
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ inline void DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::UnlinkFreeInternal(DynamicBlock<T> *block) { _freeTree.Remove(block->Node); block->node = nullptr; _numFreeBlocks--; _freeBlockMemory -= block->GetSize(); }

	/// <summary>
	/// CheckMemory
	/// </summary>
	template<class T, int BaseBlockSize, int MinBlockSize, memTag_t Tag>
	__device__ void DynamicBlockAlloc<T, BaseBlockSize, MinBlockSize, Tag>::CheckMemory() const
	{
		for (DynamicBlock<T> *block = _firstBlock; block != nullptr; block = block->Next)
		{
			// make sure the block is properly linked
			if (block->Prev == nullptr) { assert(firstBlock == block); }
			else { assert(block->Prev->Next == block); }
			if (block->Next == nullptr) { assert(lastBlock == block); }
			else { assert(block->Next->Prev == block); }
		}
	}

}
#endif /* __SYSTEM_HEAP_H__ */
