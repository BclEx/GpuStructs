// vdbesort.c
#include "VdbeInt.cu.h"

namespace Core
{
	struct VdbeSorterIter;
	struct SorterRecord;
	struct FileWriter;

#pragma region Structs

	struct VdbeSorter
	{
		int64 WriteOffset;				// Current write offset within file pTemp1
		int64 ReadOffset;				// Current read offset within file pTemp1
		int InMemory;					// Current size of pRecord list as PMA
		int PMAs;                       // Number of PMAs stored in pTemp1
		int MinPmaSize;                 // Minimum PMA size, in bytes
		int MaxPmaSize;                 // Maximum PMA size, in bytes.  0==no limit
		VdbeSorterIter *Iters;          // Array of iterators to merge
		array_t<int> Trees;             // Current state of incremental merge / Used size of aTree/aIter (power of 2)
		VFile *Temp1;					// PMA file 1
		SorterRecord *Record;			// Head of in-memory record list
		UnpackedRecord *Unpacked;		// Used to unpack keys
	};

	struct VdbeSorterIter
	{
		int64 ReadOffset;				// Current read offset
		int64 Eof;						// 1 byte past EOF for this iterator
		VFile *File;					// File iterator is reading from
		array_t<uint8> Alloc;			// Allocated space
		array_t<uint8> Key;			// Pointer to current key
		array_t<uint8> Buffer;			// Current read buffer
	};

	struct FileWriter
	{
		int FWErr;						// Non-zero if in an error state
		array_t<uint8> Buffer;			// Pointer to write buffer
		int BufStart;					// First byte of buffer to write
		int BufEnd;						// Last byte of buffer to write
		int64 WriteOffset;				// Offset of start of buffer in file
		VFile *File;					// File to write to
	};

	struct SorterRecord
	{
		void *P;
		int N;
		SorterRecord *Next;
	};

#define SORTER_MIN_WORKING 10 // Minimum allowable value for the VdbeSorter.nWorking variable
#define SORTER_MAX_MERGE_COUNT 16 // Maximum number of segments to merge in a single pass.

#pragma endregion

#pragma region Sorter Iter

	__device__ static void vdbeSorterIterZero(Context *db, VdbeSorterIter *iter)
	{
		_tagfree(db, iter->Alloc);
		_tagfree(db, iter->Buffer);
		_memset(iter, 0, sizeof(VdbeSorterIter));
	}

	__device__ static RC vdbeSorterIterRead(Context *db, VdbeSorterIter *p, int bytes, uint8 **out_)
	{
		_assert(p->Buffer);

		// If there is no more data to be read from the buffer, read the next p->nBuffer bytes of data from the file into it. Or, if there are less
		// than p->nBuffer bytes remaining in the PMA, read all remaining data.
		int bufferIdx = (int)(p->ReadOffset % p->Buffer.length); // Offset within buffer to read from
		if (bufferIdx == 0)
		{
			// Determine how many bytes of data to read.
			int read = (int)((p->Eof - p->ReadOffset) > p->Buffer.length ? p->Buffer.length : p->Eof - p->ReadOffset); // Bytes to read from disk
			_assert(read > 0);

			// Read data from the file. Return early if an error occurs.
			RC rc = p->File->Read(p->Buffer, read, p->ReadOffset);
			_assert(rc != RC_IOERR_SHORT_READ);
			if (rc != RC_OK) return rc;
		}
		int avail = p->Buffer.length - bufferIdx; // Bytes of data available in buffer

		if (bytes <= avail)
		{
			// The requested data is available in the in-memory buffer. In this case there is no need to make a copy of the data, just return a 
			// pointer into the buffer to the caller.
			*out_ = &p->Buffer[bufferIdx];
			p->ReadOffset += bytes;
		}
		// The requested data is not all available in the in-memory buffer. In this case, allocate space at p->aAlloc[] to copy the requested
		// range into. Then return a copy of pointer p->aAlloc to the caller.
		else
		{
			// Extend the p->aAlloc[] allocation if required.
			if (p->Alloc.length < bytes)
			{
				int newSize = p->Alloc.length * 2;
				while (bytes > newSize) newSize = newSize * 2;
				p->Alloc = (uint8 *)SysEx::TagRealloc(db, p->Alloc, newSize);
				if (!p->Alloc) return RC_NOMEM;
				p->Alloc.length = newSize;
			}

			// Copy as much data as is available in the buffer into the start of p->aAlloc[].
			_memcpy(p->Alloc.data, &p->Buffer[bufferIdx], avail);
			p->ReadOffset += avail;

			// The following loop copies up to p->nBuffer bytes per iteration into the p->aAlloc[] buffer.
			int remaining = bytes - avail; // Bytes remaining to copy
			while (remaining > 0)
			{
				int copy = remaining; // Number of bytes to copy
				if (remaining > p->Buffer.length) copy = p->Buffer.length;
				uint8 *next; // Pointer to buffer to copy data from
				RC rc = vdbeSorterIterRead(db, p, copy, &next);
				if (rc != RC_OK) return rc;
				_assert(next != p->Alloc);
				_memcpy(&p->Alloc[bytes - remaining], next, copy);
				remaining -= copy;
			}

			*out_ = p->Alloc;
		}
		return RC_OK;
	}

	__device__ static RC vdbeSorterIterVarint(Context *db, VdbeSorterIter *p, uint64 *out_)
	{
		int bufferIdx = (int)(p->ReadOffset % p->Buffer.length);
		if (bufferIdx && (p->Buffer.length - bufferIdx) >= 9)
			p->ReadOffset += ConvertEx::GetVarint(&p->Buffer[bufferIdx], out_);
		else
		{
			uint8 varint[16], *a;
			int i = 0;
			do
			{
				RC rc = vdbeSorterIterRead(db, p, 1, &a);
				if (rc) return rc;
				varint[(i++) & 0xf] = a[0];
			} while ((a[0] & 0x80) != 0);
			ConvertEx::GetVarint(varint, out_);
		}
		return RC_OK;
	}

	__device__ static RC vdbeSorterIterNext(Context *db, VdbeSorterIter *iter)
	{
		if (iter->ReadOffset >= iter->Eof)
		{
			vdbeSorterIterZero(db, iter); // This is an EOF condition
			return RC_OK;
		}
		uint64 recordSize; // Size of record in bytes
		RC rc = vdbeSorterIterVarint(db, iter, &recordSize);
		if (rc == RC_OK)
		{
			iter->Key.length = (int)recordSize;
			rc = vdbeSorterIterRead(db, iter, (int)recordSize, &iter->Key.data);
		}
		return rc;
	}

	__device__ static RC vdbeSorterIterInit(Context *db, const VdbeSorter *sorter, int64 start, VdbeSorterIter *iter, int64 *bytes)
	{
		_assert(sorter->WriteOffset > start);
		_assert(!iter->Alloc);
		_assert(!iter->Buffer);
		int bufferLength = db->DBs[0].Bt->GetPageSize();
		iter->File = sorter->Temp1;
		iter->ReadOffset = start;
		iter->Alloc.length = 128;
		iter->Alloc = (uint8 *)_tagalloc(db, iter->Alloc.length);
		iter->Buffer.length = bufferLength;
		iter->Buffer = (uint8 *)_tagalloc(db, bufferLength);
		RC rc = RC_OK;
		if (!iter->Buffer)
			rc = RC_NOMEM;
		else
		{
			int bufferIdx = start % bufferLength;
			if (bufferIdx)
			{
				int read = bufferLength - bufferIdx;
				if ((start + read) > sorter->WriteOffset)
					read = (int)(sorter->WriteOffset - start);
				rc = sorter->Temp1->Read(&iter->Buffer[bufferIdx], read, start);
				_assert(rc != RC_IOERR_SHORT_READ);
			}
			if (rc == RC_OK)
			{
				iter->Eof = sorter->WriteOffset;
				uint64 bytes2; // Size of PMA in bytes
				rc = vdbeSorterIterVarint(db, iter, &bytes2);
				iter->Eof = iter->ReadOffset + bytes2;
				*bytes += bytes2;
			}
		}
		if (rc == RC_OK)
			rc = vdbeSorterIterNext(db, iter);
		return rc;
	}

#pragma endregion

#pragma region Sorter Compare/Merge

	__device__ static void vdbeSorterCompare(const VdbeCursor *cursor, bool omitRowid, const void *key1, int key1Length, const void *key2, int key2Length, int *out_)
	{
		KeyInfo *keyInfo = cursor->KeyInfo;
		VdbeSorter *sorter = cursor->Sorter;
		UnpackedRecord *r2 = sorter->Unpacked;
		if (key2)
			Vdbe::RecordUnpack(keyInfo, key2Length, key2, r2);
		if (omitRowid)
		{
			r2->Fields = keyInfo->Fields;
			_assert(r2->Fields > 0);
			for (int i = 0; i < r2->Fields; i++)
				if (r2->Mems[i].Flags & MEM_Null)
				{
					*out_ = -1;
					return;
				}
				r2->Flags |= UNPACKED_PREFIX_MATCH;
		}
		*out_ = Vdbe::RecordCompare(key1Length, key1, r2);
	}

	__device__ static RC vdbeSorterDoCompare(const VdbeCursor *cursor, int idx)
	{
		VdbeSorter *sorter = cursor->Sorter;
		_assert(idx < sorter->Trees.length && idx > 0);

		int i1;
		int i2;
		if (idx >= (sorter->Trees.length / 2))
		{
			i1 = (idx - sorter->Trees.length / 2) * 2;
			i2 = i1 + 1;
		}
		else
		{
			i1 = sorter->Trees[idx * 2];
			i2 = sorter->Trees[idx * 2 + 1];
		}
		VdbeSorterIter *p1 = &sorter->Iters[i1];
		VdbeSorterIter *p2 = &sorter->Iters[i2];

		int i;
		if (!p1->File)
			i = i2;
		else if (!p2->File)
			i = i1;
		else
		{
			_assert(sorter->Unpacked); // allocated in vdbeSorterMerge()
			int r;
			vdbeSorterCompare(cursor, 0, p1->Key, p1->Key.length, p2->Key, p2->Key.length, &r);
			i = (r <= 0 ? i1 : i2);
		}
		sorter->Trees[idx] = i;
		return RC_OK;
	}

	__device__ RC Vdbe::SorterInit(Context *db, VdbeCursor *cursor)
	{
		_assert(cursor->KeyInfo && !cursor->Bt);
		VdbeSorter *sorter; // The new sorter
		cursor->Sorter = sorter = (VdbeSorter *)_tagalloc(db, sizeof(VdbeSorter));
		if (!sorter)
			return RC_NOMEM;

		char *d;
		sorter->Unpacked = Vdbe::AllocUnpackedRecord(cursor->KeyInfo, 0, 0, &d);
		if (!sorter->Unpacked) return RC_NOMEM;
		_assert(sorter->Unpacked == (UnpackedRecord *)d);

		if (!db->TempInMemory())
		{
			int pageSize = db->DBs[0].Bt->GetPageSize(); // Page size of main database
			sorter->MinPmaSize = SORTER_MIN_WORKING * pageSize;
			int maxCache = db->DBs[0].Schema->CacheSize; // Cache size
			if (maxCache < SORTER_MIN_WORKING) maxCache = SORTER_MIN_WORKING;
			sorter->MaxPmaSize = maxCache * pageSize;
		}
		return RC_OK;
	}

	__device__ static void vdbeSorterRecordFree(Context *db, SorterRecord *record)
	{
		SorterRecord *next;
		for (SorterRecord *p = record; p; p = next)
		{
			next = p->Next;
			_tagfree(db, p);
		}
	}

	__device__ void Vdbe::SorterClose(Context *db, VdbeCursor *cursor)
	{
		VdbeSorter *sorter = cursor->Sorter;
		if (sorter)
		{
			if (sorter->Iters)
			{
				for (int i = 0; i < sorter->Trees.length; i++)
					vdbeSorterIterZero(db, &sorter->Iters[i]);
				_tagfree(db, sorter->Iters);
			}
			if (sorter->Temp1)
				sorter->Temp1->CloseAndFree();
			vdbeSorterRecordFree(db, sorter->Record);
			_tagfree(db, sorter->Unpacked);
			_tagfree(db, sorter);
			cursor->Sorter = nullptr;
		}
	}

	__device__ static RC vdbeSorterOpenTempFile(Context *db, VFile **file)
	{
		VSystem::OPEN outFlags;
		return db->Vfs->OpenAndAlloc(nullptr, file, (VSystem::OPEN)(VSystem::OPEN_TEMP_JOURNAL | VSystem::OPEN_READWRITE | VSystem::OPEN_CREATE | VSystem::OPEN_EXCLUSIVE | VSystem::OPEN_DELETEONCLOSE), &outFlags);
	}

	__device__ static void vdbeSorterMerge(const VdbeCursor *cursor, SorterRecord *p1, SorterRecord *p2, SorterRecord **out_)
	{
		SorterRecord *result = nullptr;
		SorterRecord **pp = &result;
		void *p2P = (p2 ? p2->P : nullptr);
		while (p1 && p2)
		{
			int r;
			vdbeSorterCompare(cursor, false, p1->P, p1->N, p2P, p2->N, &r);
			if (r <= 0)
			{
				*pp = p1;
				pp = &p1->Next;
				p1 = p1->Next;
				p2P = nullptr;
			}
			else
			{
				*pp = p2;
				pp = &p2->Next;
				p2 = p2->Next;
				if (!p2) break;
				p2P = p2->P;
			}
		}
		*pp = (p1 ? p1 : p2);
		*out_ = result;
	}

	__device__ static RC vdbeSorterSort(const VdbeCursor *cursor)
	{
		SorterRecord **slots = (SorterRecord **)_alloc(64 * sizeof(SorterRecord *));
		if (!slots)
			return RC_NOMEM;
		VdbeSorter *sorter = cursor->Sorter;
		SorterRecord *p = sorter->Record;
		int i;
		while (p)
		{
			SorterRecord *next = p->Next;
			p->Next = nullptr;
			for (i = 0; slots[i]; i++)
			{
				vdbeSorterMerge(cursor, p, slots[i], &p);
				slots[i] = nullptr;
			}
			slots[i] = p;
			p = next;
		}
		p = nullptr;
		for (i = 0; i < 64; i++)
			vdbeSorterMerge(cursor, p, slots[i], &p);
		sorter->Record = p;
		_free(slots);
		return RC_OK;
	}

#pragma endregion

#pragma region FileWriter

	__device__ static void fileWriterInit(Context *db, VFile *file, FileWriter *p, int64 start)
	{
		_memset(p, 0, sizeof(FileWriter));
		int pageSize = db->DBs[0].Bt->GetPageSize();
		p->Buffer = (uint8 *)_tagalloc(db, pageSize);
		if (!p->Buffer)
			p->FWErr = RC_NOMEM;
		else
		{
			p->BufEnd = p->BufStart = (start % pageSize);
			p->WriteOffset = start - p->BufStart;
			p->Buffer.length = pageSize;
			p->File = file;
		}
	}

	__device__ static void fileWriterWrite(FileWriter *p, uint8 *data, int dataLength)
	{
		int remain = dataLength;
		while (remain > 0 && p->FWErr == 0)
		{
			int copy = remain;
			if (copy > (p->Buffer.length - p->BufEnd))
				copy = p->Buffer.length - p->BufEnd;
			_memcpy(&p->Buffer[p->BufEnd], &data[dataLength - remain], copy);
			p->BufEnd += copy;
			if (p->BufEnd == p->Buffer.length)
			{
				p->FWErr = p->File->Write(&p->Buffer[p->BufStart], p->BufEnd - p->BufStart, p->WriteOffset + p->BufStart);
				p->BufStart = p->BufEnd = 0;
				p->WriteOffset += p->Buffer.length;
			}
			_assert(p->BufEnd < p->Buffer.length);
			remain -= copy;
		}
	}

	__device__ static RC fileWriterFinish(Context *db, FileWriter *p, int64 *eof)
	{
		if (p->FWErr == 0 && _ALWAYS(p->Buffer) && p->BufEnd > p->BufStart)
			p->FWErr = p->File->Write(&p->Buffer[p->BufStart], p->BufEnd - p->BufStart, p->WriteOffset + p->BufStart);
		*eof = (p->WriteOffset + p->BufEnd);
		_tagfree(db, p->Buffer);
		RC rc = (RC)p->FWErr;
		_memset(p, 0, sizeof(FileWriter));
		return rc;
	}

	__device__ static void fileWriterWriteVarint(FileWriter *p, uint64 value)
	{
		uint8 bytes[10];
		int length = ConvertEx::PutVarint(bytes, value);
		fileWriterWrite(p, bytes, length);
	}

	__device__ static RC vdbeSorterListToPMA(Context *db, const VdbeCursor *cursor)
	{
		VdbeSorter *sorter = cursor->Sorter;
		FileWriter writer;
		_memset(&writer, 0, sizeof(FileWriter));
		if (sorter->InMemory == 0)
		{
			_assert(!sorter->Record);
			return RC_OK;
		}
		RC rc = vdbeSorterSort(cursor);
		// If the first temporary PMA file has not been opened, open it now.
		if (rc == RC_OK && !sorter->Temp1)
		{
			rc = vdbeSorterOpenTempFile(db, &sorter->Temp1);
			_assert(rc != RC_OK || sorter->Temp1);
			_assert(sorter->WriteOffset == 0);
			_assert(sorter->PMAs == 0);
		}
		if (rc == RC_OK)
		{
			fileWriterInit(db, sorter->Temp1, &writer, sorter->WriteOffset);
			sorter->PMAs++;
			fileWriterWriteVarint(&writer, sorter->InMemory);
			SorterRecord *p;
			SorterRecord *next = nullptr;
			for (p = sorter->Record; p; p = next)
			{
				next = p->Next;
				fileWriterWriteVarint(&writer, p->N);
				fileWriterWrite(&writer, (uint8 *)p->P, p->N);
				_tagfree(db, p);
			}
			sorter->Record = p;
			rc = fileWriterFinish(db, &writer, &sorter->WriteOffset);
		}
		return rc;
	}

	__device__ RC Vdbe::SorterWrite(Context *db, const VdbeCursor *cursor, Mem *mem)
	{
		VdbeSorter *sorter = cursor->Sorter;
		_assert(sorter);
		sorter->InMemory += ConvertEx::GetVarintLength(mem->N) + mem->N;
		SorterRecord *newRecord = (SorterRecord *)_tagalloc(db, mem->N + sizeof(SorterRecord)); // New list element
		RC rc = RC_OK;
		if (!newRecord)
			rc = RC_NOMEM;
		else
		{
			newRecord->P = (void *)&newRecord[1];
			_memcpy((char *)newRecord->P, mem->Z, mem->N);
			newRecord->N = mem->N;
			newRecord->Next = sorter->Record;
			sorter->Record = newRecord;
		}
		// See if the contents of the sorter should now be written out. They are written out when either of the following are true:
		//   * The total memory allocated for the in-memory list is greater than (page-size * cache-size), or
		//   * The total memory allocated for the in-memory list is greater than (page-size * 10) and sqlite3HeapNearlyFull() returns true.
		if (rc == RC_OK && sorter->MaxPmaSize > 0 && ((sorter->InMemory > sorter->MaxPmaSize) || (sorter->InMemory > sorter->MaxPmaSize && SysEx::HeapNearlyFull()))){
#ifdef _DEBUG
			int64 expect = sorter->WriteOffset + ConvertEx::GetVarintLength(sorter->InMemory) + sorter->InMemory;
#endif
			rc = vdbeSorterListToPMA(db, cursor);
			sorter->InMemory = 0;
			_assert(rc != RC_OK || expect == sorter->WriteOffset);
		}
		return rc;
	}

	__device__ static RC vdbeSorterInitMerge(Context *db, const VdbeCursor *cursor, int64 *bytes)
	{
		VdbeSorter *sorter = cursor->Sorter;
		int64 bytes2 = 0; // Total bytes in all opened PMAs
		RC rc = RC_OK;
		// Initialize the iterators.
		int i; // Used to iterator through aIter[]
		for (i = 0; i < SORTER_MAX_MERGE_COUNT; i++)
		{
			VdbeSorterIter *iter = &sorter->Iters[i];
			rc = vdbeSorterIterInit(db, sorter, sorter->ReadOffset, iter, &bytes2);
			sorter->ReadOffset = iter->Eof;
			_assert(rc != RC_OK || sorter->ReadOffset <= sorter->WriteOffset);
			if (rc != RC_OK || sorter->ReadOffset >= sorter->WriteOffset) break;
		}
		// Initialize the aTree[] array.
		for (i = sorter->Trees.length - 1; rc == RC_OK && i > 0; i--)
			rc = vdbeSorterDoCompare(cursor, i);
		*bytes = bytes2;
		return rc;
	}

	__device__ RC Vdbe::SorterRewind(Context *db, const VdbeCursor *cursor, bool *eof)
	{
		VdbeSorter *sorter = cursor->Sorter;
		_assert(sorter);

		// If no data has been written to disk, then do not do so now. Instead, sort the VdbeSorter.pRecord list. The vdbe layer will read data directly
		// from the in-memory list.
		if (sorter->PMAs == 0)
		{
			*eof = !sorter->Record;
			assert(!sorter->Trees.data);
			return vdbeSorterSort(cursor);
		}

		// Write the current in-memory list to a PMA.
		RC rc = vdbeSorterListToPMA(db, cursor);
		if (rc != RC_OK) return rc;

		// Allocate space for aIter[] and aTree[].
		int iters = sorter->PMAs; // Number of iterators used
		if (iters > SORTER_MAX_MERGE_COUNT) iters = SORTER_MAX_MERGE_COUNT;
		_assert(iters > 0);
		int n = 2; while (n < iters) n += n; // Power of 2 >= iters
		int bytes = n * (sizeof(int) + sizeof(VdbeSorterIter)); // Bytes of space required for aIter/aTree
		sorter->Iters = (VdbeSorterIter *)_tagalloc(db, bytes);
		if (!sorter->Iters) return RC_NOMEM;
		sorter->Trees = (int *)&sorter->Iters[n];
		sorter->Trees.length = n;

		int newIdx; // Index of new, merged, PMA
		VFile *temp2 = nullptr; // Second temp file to use
		int64 write2 = 0; // Write offset for pTemp2
		do
		{
			for (newIdx = 0; rc == RC_OK && newIdx * SORTER_MAX_MERGE_COUNT < sorter->PMAs; newIdx++)
			{
				FileWriter writer; _memset(&writer, 0, sizeof(FileWriter)); // Object used to write to disk
				// If there are SORTER_MAX_MERGE_COUNT or less PMAs in file pTemp1, initialize an iterator for each of them and break out of the loop.
				// These iterators will be incrementally merged as the VDBE layer calls sqlite3VdbeSorterNext().
				//
				// Otherwise, if pTemp1 contains more than SORTER_MAX_MERGE_COUNT PMAs, initialize interators for SORTER_MAX_MERGE_COUNT of them. These PMAs
				// are merged into a single PMA that is written to file pTemp2.
				int64 writes; // Number of bytes in new PMA
				rc = vdbeSorterInitMerge(db, cursor, &writes);
				_assert(rc != RC_OK || sorter->Iters[sorter->Trees[1]].File);
				if (rc != RC_OK || sorter->PMAs <= SORTER_MAX_MERGE_COUNT)
					break;
				// Open the second temp file, if it is not already open.
				if (!temp2)
				{
					_assert(write2 == 0);
					rc = vdbeSorterOpenTempFile(db, &temp2);
				}
				if (rc == RC_OK)
				{
					bool eof = false;
					fileWriterInit(db, temp2, &writer, write2);
					fileWriterWriteVarint(&writer, writes);
					while (rc == RC_OK && !eof)
					{
						VdbeSorterIter *iter = &sorter->Iters[sorter->Trees[1]];
						_assert(iter->File);
						fileWriterWriteVarint(&writer, iter->Key.length);
						fileWriterWrite(&writer, iter->Key, iter->Key.length);
						rc = Vdbe::SorterNext(db, cursor, &eof);
					}
					RC rc2 = fileWriterFinish(db, &writer, &write2);
					if (rc == RC_OK) rc = rc2;
				}
			}
			if (sorter->PMAs <= SORTER_MAX_MERGE_COUNT)
				break;
			else
			{
				VFile *tmp = sorter->Temp1;
				sorter->PMAs = newIdx;
				sorter->Temp1 = temp2;
				temp2 = tmp;
				sorter->WriteOffset = write2;
				sorter->ReadOffset = 0;
				write2 = 0;
			}
		} while (rc == RC_OK);
		if (temp2)
			temp2->CloseAndFree();
		*eof = !sorter->Iters[sorter->Trees[1]].File;
		return rc;
	}

	__device__ RC Vdbe::SorterNext(Context *db, const VdbeCursor *cursor, bool *eof)
	{
		VdbeSorter *sorter = cursor->Sorter;
		if (sorter->Trees)
		{
			int prevI = sorter->Trees[1]; // Index of iterator to advance
			RC rc = vdbeSorterIterNext(db, &sorter->Iters[prevI]);
			for (int i = (sorter->Trees.length + prevI) / 2; rc == RC_OK && i > 0; i /= 2) // Index of aTree[] to recalculate
				rc = vdbeSorterDoCompare(cursor, i);
			*eof = !sorter->Iters[sorter->Trees[1]].File;
			return rc;
		}
		SorterRecord *free = sorter->Record;
		sorter->Record = free->Next;
		free->Next = nullptr;
		vdbeSorterRecordFree(db, free);
		*eof = !sorter->Record;
		return RC_OK;
	}

	__device__ static void *vdbeSorterRowkey(const VdbeSorter *sorter, int *keyLength)
	{
		if (sorter->Trees.data)
		{
			VdbeSorterIter *iter = &sorter->Iters[sorter->Trees.data[1]];
			*keyLength = iter->Key.length;
			return iter->Key;
		}
		*keyLength = sorter->Record->N;
		return sorter->Record->P;
	}

	__device__ RC Vdbe::SorterRowkey(const VdbeCursor *cursor, Mem *mem)
	{
		int keyLength;
		void *key = vdbeSorterRowkey(cursor->Sorter, &keyLength);
		if (Vdbe::MemGrow(mem, keyLength, 0))
			return RC_NOMEM;
		mem->N = keyLength;
		MemSetTypeFlag(mem, MEM_Blob);
		_memcpy(mem->Z, (char *)key, keyLength);
		return RC_OK;
	}

	__device__ RC Vdbe::SorterCompare(const VdbeCursor *cursor, Mem *mem, int *r)
	{
		int keyLength;
		void *key = vdbeSorterRowkey(cursor->Sorter, &keyLength);
		vdbeSorterCompare(cursor, 1, mem->Z, mem->N, key, keyLength, r);
		return RC_OK;
	}

#pragma endregion

}