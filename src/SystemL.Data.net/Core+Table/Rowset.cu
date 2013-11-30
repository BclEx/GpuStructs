#include "Core+Table.cu.h"

namespace Core
{
	struct RowSetEntry
	{            
		int64 V;                    // ROWID value for this entry
		struct RowSetEntry *Right;	// Right subtree (larger entries) or list
		struct RowSetEntry *Left;	// Left subtree (smaller entries)
	};

#define ROWSET_ALLOCATION_SIZE 1024
#define ROWSET_ENTRY_PER_CHUNK \ ((ROWSET_ALLOCATION_SIZE-8)/sizeof(struct RowSetEntry))

	struct RowSetChunk
	{
		struct RowSetChunk *NextChunk; // Next chunk on list of them all
		struct RowSetEntry Entrys[ROWSET_ENTRY_PER_CHUNK]; // Allocated entries
	};

	enum ROWSET : uint8
	{
		ROWSET_SORTED = 0x01,   // True if RowSet.pEntry is sorted
		ROWSET_NEXT = 0x02,		// True if sqlite3RowSetNext() has been called
	};

	struct RowSet
	{
		struct RowSetChunk *Chunk;		// List of all chunk allocations
		Context *Db;					// The database connection
		struct RowSetEntry *Entry;		// List of entries using pRight
		struct RowSetEntry *Last;		// Last entry on the pEntry list
		array_t<struct RowSetEntry> Fresh;  // Source of new entry objects
		struct RowSetEntry *Forest;		// List of binary trees of entries
		ROWSET Flags;                    // Various flags
		uint8 Batch;                    // Current insert batch
	};

	__device__ RowSet *sqlite3RowSetInit(Context *db, void *space, unsigned int n)
	{
		_assert(n >= SysEx_ROUND8(sizeof(RowSet **)));
		RowSet *p = (RowSet *)space;
		p->Chunk = nullptr;
		p->Db = db;
		p->Entry = nullptr;
		p->Last = nullptr;
		p->Forest = nullptr;
		p->Fresh = (struct RowSetEntry *)(SysEx_ROUND8(sizeof(*p)) + (char *)p);
		p->Fresh.length = (uint16)((n - SysEx_ROUND8(sizeof(*p))) / sizeof(struct RowSetEntry));
		p->Flags = ROWSET_SORTED;
		p->Batch = 0;
		return p;
	}

	__device__ void sqlite3RowSetClear(RowSet *p)
	{
		struct RowSetChunk *chunk, *nextChunk;
		for (chunk = p->Chunk; chunk; chunk = nextChunk)
		{
			nextChunk = chunk->NextChunk;
			SysEx::TagFree(p->Db, chunk);
		}
		p->Chunk = nullptr;
		p->Fresh.length = 0;
		p->Entry = nullptr;
		p->Last = nullptr;
		p->Forest = nullptr;
		p->Flags = ROWSET_SORTED;
	}

	__device__ static struct RowSetEntry *rowSetEntryAlloc(RowSet *p)
	{
		_assert(p);
		if (!p->Fresh.length)
		{
			struct RowSetChunk *newChunk = (struct RowSetChunk *)SysEx::TagAlloc(p->Db, sizeof(*newChunk));
			if (!newChunk)
				return nullptr;
			newChunk->NextChunk = p->Chunk;
			p->Chunk = newChunk;
			p->Fresh = newChunk->Entrys;
			p->Fresh.length = ROWSET_ENTRY_PER_CHUNK;
		}
		p->Fresh.length--;
		return p->Fresh.data++;
	}

	__device__ void sqlite3RowSetInsert(RowSet *p, int64 rowid)
	{
		// This routine is never called after sqlite3RowSetNext()
		_assert(p && (p->Flags & ROWSET_NEXT) == 0);

		struct RowSetEntry *entry = rowSetEntryAlloc(p); // The new entry
		if (!entry) return;
		entry->V = rowid;
		entry->Right = nullptr;
		struct RowSetEntry *last = p->Last; // The last prior entry
		if (last)
		{
			if ((p->Flags & ROWSET_SORTED) != 0 && rowid <= last->V)
				p->Flags &= ~ROWSET_SORTED;
			last->Right = entry;
		}
		else
			p->Entry = entry;
		p->Last = entry;
	}

	__device__ static struct RowSetEntry *rowSetEntryMerge(struct RowSetEntry *a, struct RowSetEntry *b)
	{
		struct RowSetEntry head;
		struct RowSetEntry *tail = &head;
		while (a && b)
		{
			_assert(!a->Right || a->V <= a->Right->V);
			_assert(!b->Right || b->V <= b->Right->V);
			if (a->V < b->V)
			{
				tail->Right = a;
				a = a->Right;
				tail = tail->Right;
			}
			else if (b->V < a->V)
			{
				tail->Right = b;
				b = b->Right;
				tail = tail->Right;
			}
			else
				a = a->Right;
		}
		if (a)
		{
			_assert(!a->Right || a->V <= a->Right->V);
			tail->Right = a;
		}
		else
		{
			_assert(!b || !b->Right || b->V <= b->Right->V);
			tail->Right = b;
		}
		return head.Right;
	}

	__device__ static struct RowSetEntry *rowSetEntrySort(struct RowSetEntry *p)
	{
		unsigned int i;
		struct RowSetEntry *next, *buckets[40];
		_memset(buckets, 0, sizeof(buckets));
		while (p)
		{
			next = p->Right;
			p->Right = nullptr;
			for (i = 0; buckets[i]; i++)
			{
				p = rowSetEntryMerge(buckets[i], p);
				buckets[i] = nullptr;
			}
			buckets[i] = p;
			p = next;
		}
		p = nullptr;
		for (i = 0; i < __arrayStaticLength(buckets); i++)
			p = rowSetEntryMerge(p, buckets[i]);
		return p;
	}

	__device__ static void rowSetTreeToList(struct RowSetEntry *parent, struct RowSetEntry **first, struct RowSetEntry **last)
	{
		_assert(parent);
		if (parent->Left)
		{
			struct RowSetEntry *p;
			rowSetTreeToList(parent->Left, first, &p);
			p->Right = parent;
		}
		else
			*first = parent;
		if (parent->Right)
			rowSetTreeToList(parent->Right, &parent->Right, last);
		else
			*last = parent;
		_assert(!(*last)->Right);
	}

	__device__ static struct RowSetEntry *rowSetNDeepTree(struct RowSetEntry **list, int depth)
	{
		if (!*list)
			return nullptr;
		struct RowSetEntry *p; // Root of the new tree
		struct RowSetEntry *left; // Left subtree
		if (depth == 1)
		{
			p = *list;
			*list = p->Right;
			p->Left = p->Right = nullptr;
			return p;
		}
		left = rowSetNDeepTree(list, depth - 1);
		p = *list;
		if (!p)
			return left;
		p->Left = left;
		*list = p->Right;
		p->Right = rowSetNDeepTree(list, depth - 1);
		return p;
	}

	__device__ static struct RowSetEntry *rowSetListToTree(struct RowSetEntry *list)
	{
		_assert(list);
		struct RowSetEntry *p = list; // Current tree root
		list = p->Right;
		p->Left = p->Right = nullptr;
		for (int depth = 1; list; depth++)
		{
			struct RowSetEntry *left = p; // Left subtree
			p = list;
			list = p->Right;
			p->Left = left;
			p->Right = rowSetNDeepTree(&list, depth);
		}
		return p;
	}

	__device__ static void rowSetToList(RowSet *p)
	{
		// This routine is called only once
		_assert(p && (p->Flags & ROWSET_NEXT) == 0);
		if ((p->Flags & ROWSET_SORTED) == 0)
			p->Entry = rowSetEntrySort(p->Entry);
		// While this module could theoretically support it, sqlite3RowSetNext() is never called after sqlite3RowSetText() for the same RowSet.  So
		// there is never a forest to deal with.  Should this change, simply remove the assert() and the #if 0.
		_assert(!p->Forest);
#if 0
		while (p->Forest)
		{
			struct RowSetEntry *tree = p->Forest->Left;
			if (tree)
			{
				struct RowSetEntry *head, *tail;
				rowSetTreeToList(tree, &head, &tail);
				p->Entry = rowSetEntryMerge(p->Entry, head);
			}
			p->Forest = p->Forest->Right;
		}
#endif
		p->Flags |= ROWSET_NEXT; // Verify this routine is never called again
	}

	__device__ bool sqlite3RowSetNext(RowSet *p, int64 *rowid)
	{
		_assert(p);

		// Merge the forest into a single sorted list on first call
		if ((p->Flags & ROWSET_NEXT) == 0) rowSetToList(p);

		// Return the next entry on the list
		if (p->Entry)
		{
			*rowid = p->Entry->V;
			p->Entry = p->Entry->Right;
			if (!p->Entry)
				sqlite3RowSetClear(p);
			return true;
		}
		return false;
	}

	__device__ bool sqlite3RowSetTest(RowSet *rowSet, uint8 batch, int64 rowid)
	{
		// This routine is never called after sqlite3RowSetNext()
		_assert(rowSet && (rowSet->Flags & ROWSET_NEXT) == 0);

		// Sort entries into the forest on the first test of a new batch 
		struct RowSetEntry *p, *tree;
		if (batch != rowSet->Batch)
		{
			p = rowSet->Entry;
			if (p)
			{
				struct RowSetEntry **prevTree = &rowSet->Forest;
				if ((rowSet->Flags & ROWSET_SORTED) == 0)
					p = rowSetEntrySort(p);
				for (tree = rowSet->Forest; tree; tree = tree->Right)
				{
					prevTree = &tree->Right;
					if (!tree->Left)
					{
						tree->Left = rowSetListToTree(p);
						break;
					}
					else
					{
						struct RowSetEntry *aux, *tail;
						rowSetTreeToList(tree->Left, &aux, &tail);
						tree->Left = nullptr;
						p = rowSetEntryMerge(aux, p);
					}
				}
				if (!tree)
				{
					*prevTree = tree = rowSetEntryAlloc(rowSet);
					if (tree)
					{
						tree->V = 0;
						tree->Right = nullptr;
						tree->Left = rowSetListToTree(p);
					}
				}
				rowSet->Entry = nullptr;
				rowSet->Last = nullptr;
				rowSet->Flags |= ROWSET_SORTED;
			}
			rowSet->Batch = batch;
		}

		// Test to see if the iRowid value appears anywhere in the forest. Return 1 if it does and 0 if not.
		for (tree = rowSet->Forest; tree; tree = tree->Right)
		{
			p = tree->Left;
			while (p)
			{
				if (p->V < rowid) p = p->Right;
				else if( p->V > rowid) p = p->Left;
				else return true;
			}
		}
		return false;
	}

}