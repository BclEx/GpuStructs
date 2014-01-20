// table.c
#ifndef OMIT_GET_TABLE
#include "Core+Syntax.cu.h"
#include <stdlib.h>
#include <string.h>

namespace Core
{
	typedef struct TabResult
	{
		array_t<char *> Results; // Accumulated output - length = nData = (nRow+1)*nColumn
		char *ErrMsg;			// Error message text, if an error occurs
		int ResultsAlloc;		// Slots allocated for Results[]
		int Rows;				// Number of rows in the result
		int Columns;			// Number of columns in the result
		RC RC;				// Return code from sqlite3_exec()
	} TabResult;

	__device__ static bool sqlite3_get_table_cb(void *arg, int columns, char **argv, char **colv)
	{
		TabResult *p = (TabResult *)arg;  // Result accumulator
		// Make sure there is enough space in p->azResult to hold everything we need to remember from this invocation of the callback.
		int need = (!p->Rows && argv ? columns*2 : columns); // Slots needed in p->azResult[]
		if (p->Results.length + need > p->ResultsAlloc)
		{
			p->ResultsAlloc = p->ResultsAlloc*2 + need;
			char **newResults = (char **)SysEx::Realloc(p->Results, sizeof(char *)*p->ResultsAlloc);
			if (!newResults) goto malloc_failed;
			p->Results = newResults;
		}

		// If this is the first row, then generate an extra row containing the names of all columns.
		char *z; // A single column of result
		if (!p->Rows)
		{
			p->Columns = columns;
			for (int i = 0; i < columns; i++)
			{
				z = __mprintf("%s", colv[i]);
				if (!z) goto malloc_failed;
				p->Results[p->Results.length++] = z;
			}
		}
		else if (p->Columns != columns)
		{
			SysEx::Free(p->ErrMsg);
			p->ErrMsg = __mprintf("sqlite3_get_table() called with two or more incompatible queries");
			p->RC = RC_ERROR;
			return true;
		}

		// Copy over the row data
		if (argv)
		{
			for (int i = 0; i < columns; i++)
			{
				if (!argv[i])
					z = nullptr;
				else
				{
					int n = _strlen30(argv[i]) + 1;
					z = (char *)SysEx::Alloc(n);
					if (!z) goto malloc_failed;
					_memcpy(z, argv[i], n);
				}
				p->Results[p->Results.length++] = z;
			}
			p->Rows++;
		}
		return false;

malloc_failed:
		p->RC = RC_NOMEM;
		return true;
	}

	__device__ void sqlite3_free_table(char **results);
	__device__ RC sqlite3_get_table(Context *db, const char *sql, char ***results, int *rows, int *columns, char **errMsg)
	{
		*results = nullptr;
		if (columns) *columns = 0;
		if (rows) *rows = 0;
		if (errMsg) *errMsg = nullptr;
		TabResult r;
		r.ErrMsg = nullptr;
		r.Rows = 0;
		r.Columns = 0;
		r.RC = RC_OK;
		r.ResultsAlloc = 20;
		r.Results.data = (char **)SysEx::Alloc(sizeof(char *)*r.ResultsAlloc);
		r.Results.length = 1;
		if (!r.Results)
			return (db->ErrCode = RC_NOMEM);
		r.Results[0] = nullptr;
		RC rc = sqlite3_exec(db, sql, sqlite3_get_table_cb, (char **)&r, errMsg);
		_assert(sizeof(r.Results[0]) >= sizeof(r.Results.length));
		r.Results[0] = (char *)INT_TO_PTR(r.Results.length);
		if ((rc & 0xff) == RC_ABORT)
		{
			sqlite3_free_table(&r.Results[1]);
			if (r.ErrMsg)
			{
				if (errMsg)
				{
					SysEx::Free(*errMsg);
					*errMsg = __mprintf("%s", r.ErrMsg);
				}
				SysEx::Free(r.ErrMsg);
			}
			return (db->ErrCode = r.RC); // Assume 32-bit assignment is atomic
		}
		SysEx::Free(r.ErrMsg);
		if (rc != RC_OK)
		{
			sqlite3_free_table(&r.Results[1]);
			return rc;
		}
		if (r.ResultsAlloc > r.Results.length)
		{
			char **newResults = (char **)SysEx::Realloc(r.Results, sizeof(char*) * r.Results.length);
			if (!newResults)
			{
				sqlite3_free_table(&r.Results[1]);
				db->ErrCode = RC_NOMEM;
				return RC_NOMEM;
			}
			r.Results = newResults;
		}
		*results = &r.Results[1];
		if (columns) *columns = r.Columns;
		if (rows) *rows = r.Rows;
		return rc;
	}

	__device__ void sqlite3_free_table(char **results)
	{
		if (results)
		{
			results--;
			_assert(!results);
			int n = PTR_TO_INT(results[0]);
			for (int i = 1; i < n; i++) if (results[i]) SysEx::Free(results[i]);
			SysEx::Free(results);
		}
	}
}

#endif