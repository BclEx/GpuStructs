using System;
using System.Diagnostics;

namespace Core
{
    public class RowSetEntry
    {
        public long V;             // ROWID value for this entry
        public RowSetEntry Right; // Right subtree (larger entries) or list
        public RowSetEntry Left;  // Left subtree (smaller entries)
    }

    public class RowSetChunk
    {
        public RowSetChunk NextChunk; // Next chunk on list of them all
        public RowSetEntry[] Entrys = new RowSetEntry[RowSet.ROWSET_ENTRY_PER_CHUNK]; // Allocated entries
    }

    [Flags]
    public enum ROWSET : byte
    {
        SORTED = 0x01,   // True if RowSet.pEntry is sorted
        NEXT = 0x02,		// True if sqlite3RowSetNext() has been called
    }

    public class RowSet
    {
        internal const int ROWSET_ALLOCATION_SIZE = 1024;
        internal const int ROWSET_ENTRY_PER_CHUNK = 63; //#define ROWSET_ENTRY_PER_CHUNK \ ((ROWSET_ALLOCATION_SIZE-8)/sizeof(struct RowSetEntry))

        public RowSetChunk Chunk;             // List of all chunk allocations
        public Context Db;                    // The database connection
        public RowSetEntry Entry;             // List of entries using pRight
        public RowSetEntry Last;              // Last entry on the pEntry list
        public RowSetEntry[] Fresh;           // Source of new entry objects
        public RowSetEntry Forest;              // Binary tree of entries
        public int FreshLength;                    // Number of objects on pFresh
        public ROWSET Flags;                 // True if pEntry is sorted
        public byte Batch;                    // Current insert batch

        public RowSet(Context db, int n)
        {
            Chunk = null;
            Db = db;
            Entry = null;
            Last = null;
            Fresh = new RowSetEntry[n];
            FreshLength = n;
            Forest = null;
            Flags = ROWSET.SORTED;
            Batch = 0;
        }

        static RowSet sqlite3RowSetInit(Context db, object space, uint n)
        {
            RowSet p = new RowSet(db, (int)n);
            return p;
        }

        static void sqlite3RowSetClear(RowSet p)
        {
            RowSetChunk chunk, nextChunk;
            for (chunk = p.Chunk; chunk != null; chunk = nextChunk)
            {
                nextChunk = chunk.NextChunk;
                C._tagfree(p.Db, ref chunk);
            }
            p.Chunk = null;
            p.FreshLength = 0;
            p.Entry = null;
            p.Last = null;
            p.Forest = null;
            p.Flags = ROWSET.SORTED;
        }

        static RowSetEntry rowSetEntryAlloc(RowSet p)
        {
            Debug.Assert(p != null);
            if (p.FreshLength == 0)
            {
                RowSetChunk newChunk = new RowSetChunk();
                if (newChunk == null)
                    return null;
                newChunk.NextChunk = p.Chunk;
                p.Chunk = newChunk;
                p.Fresh = newChunk.Entrys;
                p.FreshLength = ROWSET_ENTRY_PER_CHUNK;
            }
            p.FreshLength--;
            return p.Fresh[p.Fresh.Length - p.FreshLength] = new RowSetEntry();
        }

        static void sqlite3RowSetInsert(RowSet p, long rowid)
        {
            // This routine is never called after sqlite3RowSetNext()
            Debug.Assert(p != null && (p.Flags & ROWSET.NEXT) == 0);

            RowSetEntry entry = rowSetEntryAlloc(p); // The new entry
            if (entry == null) return;
            entry.V = rowid;
            entry.Right = null;
            RowSetEntry last = p.Last; // The last prior entry
            if (last != null)
            {
                if ((p.Flags & ROWSET.SORTED) != 0 && rowid <= last.V)
                    p.Flags &= ~ROWSET.SORTED;
                last.Right = entry;
            }
            else
                p.Entry = entry;
            p.Last = entry;
        }

        static RowSetEntry rowSetEntryMerge(RowSetEntry a, RowSetEntry b)
        {
            RowSetEntry head = new RowSetEntry();
            RowSetEntry tail = head;
            while (a != null && b != null)
            {
                Debug.Assert(a.Right == null || a.V <= a.Right.V);
                Debug.Assert(b.Right == null || b.V <= b.Right.V);
                if (a.V < b.V)
                {
                    tail.Right = a;
                    a = a.Right;
                    tail = tail.Right;
                }
                else if (b.V < a.V)
                {
                    tail.Right = b;
                    b = b.Right;
                    tail = tail.Right;
                }
                else
                    a = a.Right;
            }
            if (a != null)
            {
                Debug.Assert(a.Right == null || a.V <= a.Right.V);
                tail.Right = a;
            }
            else
            {
                Debug.Assert(b == null || b.Right == null || b.V <= b.Right.V);
                tail.Right = b;
            }
            return head.Right;
        }

        static RowSetEntry rowSetEntrySort(RowSetEntry p)
        {
            uint i;
            RowSetEntry next; RowSetEntry[] buckets = new RowSetEntry[40];
            while (p != null)
            {
                next = p.Right;
                p.Right = null;
                for (i = 0; buckets[i] != null; i++)
                {
                    p = rowSetEntryMerge(buckets[i], p);
                    buckets[i] = null;
                }
                buckets[i] = p;
                p = next;
            }
            p = null;
            for (i = 0; i < buckets.Length; i++)
                p = rowSetEntryMerge(p, buckets[i]);
            return p;
        }

        static void rowSetTreeToList(RowSetEntry parent, ref RowSetEntry first, ref RowSetEntry last)
        {
            Debug.Assert(parent != null);
            if (parent.Left != null)
            {
                RowSetEntry p = new RowSetEntry();
                rowSetTreeToList(parent.Left, ref first, ref p);
                p.Right = parent;
            }
            else
                first = parent;
            if (parent.Right != null)
                rowSetTreeToList(parent.Right, ref parent.Right, ref last);
            else
                last = parent;
            Debug.Assert((last).Right == null);
        }

        static RowSetEntry rowSetNDeepTree(ref RowSetEntry list, int depth)
        {
            if (list == null)
                return null;
            RowSetEntry p; // Root of the new tree
            RowSetEntry left; // Left subtree
            if (depth == 1)
            {
                p = list;
                list = p.Right;
                p.Left = p.Right = null;
                return p;
            }
            left = rowSetNDeepTree(ref list, depth - 1);
            p = list;
            if (p == null)
                return left;
            p.Left = left;
            list = p.Right;
            p.Right = rowSetNDeepTree(ref list, depth - 1);
            return p;
        }

        static RowSetEntry rowSetListToTree(RowSetEntry list)
        {
            Debug.Assert(list != null);
            RowSetEntry p = list; // Current tree root
            list = p.Right;
            p.Left = p.Right = null;
            for (int depth = 1; list != null; depth++)
            {
                RowSetEntry left = p; // Left subtree
                p = list;
                list = p.Right;
                p.Left = left;
                p.Right = rowSetNDeepTree(ref list, depth);
            }
            return p;
        }

        static void rowSetToList(RowSet p)
        {
            // This routine is called only once
            Debug.Assert(p != null && (p.Flags & ROWSET.NEXT) == 0);
            if ((p.Flags & ROWSET.SORTED) == 0)
                p.Entry = rowSetEntrySort(p.Entry);
            // While this module could theoretically support it, sqlite3RowSetNext() is never called after sqlite3RowSetText() for the same RowSet.  So
            // there is never a forest to deal with.  Should this change, simply remove the assert() and the #if 0.
            Debug.Assert(p.Forest == null);
            p.Flags |= ROWSET.NEXT; // Verify this routine is never called again
        }

        static bool sqlite3RowSetNext(RowSet p, ref long rowid)
        {
            Debug.Assert(p != null);

            // Merge the forest into a single sorted list on first call
            if ((p.Flags & ROWSET.NEXT) == 0) rowSetToList(p);

            // Return the next entry on the list
            if (p.Entry != null)
            {
                rowid = p.Entry.V;
                p.Entry = p.Entry.Right;
                if (p.Entry == null)
                    sqlite3RowSetClear(p);
                return true;
            }
            return false;
        }

        static bool sqlite3RowSetTest(RowSet rowSet, byte batch, long rowid)
        {
            // This routine is never called after sqlite3RowSetNext()
            Debug.Assert(rowSet != null && (rowSet.Flags & ROWSET.NEXT) == 0);

            // Sort entries into the forest on the first test of a new batch 
            RowSetEntry p, tree;
            if (batch != rowSet.Batch)
            {
                p = rowSet.Entry;
                if (p != null)
                {
                    RowSetEntry prevTree = rowSet.Forest;
                    if ((rowSet.Flags & ROWSET.SORTED) == 0)
                        p = rowSetEntrySort(p);
                    for (tree = rowSet.Forest; tree != null; tree = tree.Right)
                    {
                        prevTree = tree.Right;
                        if (tree.Left == null)
                        {
                            tree.Left = rowSetListToTree(p);
                            break;
                        }
                        else
                        {
                            RowSetEntry aux = new RowSetEntry(), tail = new RowSetEntry();
                            rowSetTreeToList(tree.Left, ref aux, ref tail);
                            tree.Left = null;
                            p = rowSetEntryMerge(aux, p);
                        }
                    }
                    if (tree == null)
                    {
                        prevTree = tree = rowSetEntryAlloc(rowSet);
                        if (tree != null)
                        {
                            tree.V = 0;
                            tree.Right = null;
                            tree.Left = rowSetListToTree(p);
                        }
                    }
                    rowSet.Entry = null;
                    rowSet.Last = null;
                    rowSet.Flags |= ROWSET.SORTED;
                }
                rowSet.Batch = batch;
            }

            // Test to see if the iRowid value appears anywhere in the forest. Return 1 if it does and 0 if not.
            for (tree = rowSet.Forest; tree != null; tree = tree.Right)
            {
                p = tree.Left;
                while (p != null)
                {
                    if (p.V < rowid) p = p.Right;
                    else if (p.V > rowid) p = p.Left;
                    else return true;
                }
            }
            return false;
        }
    }
}
