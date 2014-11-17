// func.c
#include "..\Core+Vdbe.cu.h"
#include "..\VdbeInt.h"
#include <stdlib.h>
#include <assert.h>

namespace Core { namespace Command
{
	__device__ static CollSeq *Func::GetFuncCollSeq(FuncContext *fctx)
	{
		return fctx->Coll;
	}

	__device__ static void Func::SkipAccumulatorLoad(FuncContext *fctx)
	{
		fctx->SkipFlag = true;
	}

	__device__ static void MinMaxFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc > 1);
		int mask = (sqlite3_user_data(fctx) == 0 ? 0 : -1); // 0 for min() or 0xffffffff for max()
		CollSeq *coll = sqlite3GetFuncCollSeq(fctx);
		_assert(coll);
		_assert(mask == -1 || mask == 0);
		int best = 0;
		if (sqlite3_value_type(argv[0]) == TYPE_NULL) return;
		for (int i = 1; i < argc; i++)
		{
			if (sqlite3_value_type(argv[i]) == TYPE_NULL) return;
			if ((sqlite3MemCompare(argv[best], argv[i], coll)^mask) >= 0)
			{
				ASSERTCOVERAGE(mask == 0);
				best = i;
			}
		}
		sqlite3_result_value(fctx, argv[best]);
	}

	__device__ static void TypeofFunc(FuncContext *fctx, int notUsed1, Mem **argv)
	{
		const char *z;
		switch (sqlite3_value_type(argv[0]))
		{
		case TYPE_INTEGER: z = "integer"; break;
		case TYPE_TEXT: z = "text"; break;
		case TYPE_FLOAT: z = "real"; break;
		case TYPE_BLOB: z = "blob"; break;
		default: z = "null"; break;
		}
		sqlite3_result_text(fctx, z, -1, DESTRUCTOR_STATIC);
	}

	__device__ static void LengthFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		int len;
		_assert(argc == 1);
		switch (sqlite3_value_type(argv[0]))
		{
		case TYPE_BLOB:
		case TYPE_INTEGER:
		case TYPE_FLOAT: {
			sqlite3_result_int(fctx, sqlite3_value_bytes(argv[0]));
			break; }
		case TYPE_TEXT: {
			const unsigned char *z = sqlite3_value_text(argv[0]);
			if (!z) return;
			len = 0;
			while (*z)
			{
				len++;
				_strskiputf8(z);
			}
			sqlite3_result_int(fctx, len);
			break; }
		default: {
			sqlite3_result_null(fctx);
			break; }
		}
	}

	__device__ static void AbsFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		switch (sqlite3_value_type(argv[0]) ){
		case TYPE_INTEGER: {
			int64 iVal = sqlite3_value_int64(argv[0]);
			if (iVal < 0)
			{
				if ((iVal<<1) == 0)
				{
					// IMP: R-35460-15084 If X is the integer -9223372036854775807 then abs(X) throws an integer overflow error since there is no
					// equivalent positive 64-bit two complement value.
					sqlite3_result_error(fctx, "integer overflow", -1);
					return;
				}
				iVal = -iVal;
			} 
			sqlite3_result_int64(fctx, iVal);
			break; }
		case TYPE_NULL: {
			// IMP: R-37434-19929 Abs(X) returns NULL if X is NULL.
			sqlite3_result_null(fctx);
			break; }
		default: {
			// Because sqlite3_value_double() returns 0.0 if the argument is not something that can be converted into a number, we have:
			// IMP: R-57326-31541 Abs(X) return 0.0 if X is a string or blob that cannot be converted to a numeric value. 
			double rVal = sqlite3_value_double(argv[0]);
			if (rVal < 0) rVal = -rVal;
			sqlite3_result_double(fctx, rVal);
			break; }
		}
	}

	__device__ static void InstrFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		TYPE typeHaystack = sqlite3_value_type(argv[0]);
		TYPE typeNeedle = sqlite3_value_type(argv[1]);
		if (typeHaystack == TYPE_NULL || typeNeedle == TYPE_NULL) return;
		int haystackLength = sqlite3_value_bytes(argv[0]);
		int needleLength = sqlite3_value_bytes(argv[1]);
		const unsigned char *haystack;
		const unsigned char *needle;
		bool isText;
		if (typeHaystack == TYPE_BLOB && typeNeedle == TYPE_BLOB)
		{
			haystack = sqlite3_value_blob(argv[0]);
			needle = sqlite3_value_blob(argv[1]);
			isText = false;
		}
		else
		{
			haystack = sqlite3_value_text(argv[0]);
			needle = sqlite3_value_text(argv[1]);
			isText = true;
		}
		int n = 1;
		while (needleLength <= haystackLength && _memcmp(haystack, needle, needleLength) != 0)
		{
			n++;
			do
			{
				haystackLength--;
				haystack++;
			} while (isText && (haystack[0] & 0xc0) == 0x80);
		}
		if (needleLength > haystackLength) n = 0;
		sqlite3_result_int(fctx, n);
	}

	__device__ static void SubstrFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 3 || argc == 2);
		if (sqlite3_value_type(argv[1]) == TYPE_NULL || (argc == 3 && sqlite3_value_type(argv[2]) == TYPE_NULL))
			return;
		TYPE p0type = sqlite3_value_type(argv[0]);
		int64 p1 = sqlite3_value_int(argv[1]);
		int len;
		const unsigned char *z;
		const unsigned char *z2;
		if (p0type == TYPE_BLOB)
		{
			len = sqlite3_value_bytes(argv[0]);
			z = sqlite3_value_blob(argv[0]);
			if (!z) return;
			//_assert(len == sqlite3_value_bytes(argv[0]));
		}
		else
		{
			z = sqlite3_value_text(argv[0]);
			if (!z) return;
			len = 0;
			if (p1 < 0)
				for (z2 = z; *z2; len++)
					_strskiputf8(z2);
		}
		int64 p2;
		bool negP2 = false;
		if (argc == 3)
		{
			p2 = sqlite3_value_int(argv[2]);
			if (p2 < 0)
			{
				p2 = -p2;
				negP2 = true;
			}
		}
		else
			p2 = sqlite3_context_db_handle(fctx)->Limits[LIMIT_LENGTH];
		if (p1 < 0)
		{
			p1 += len;
			if (p1 < 0)
			{
				p2 += p1;
				if (p2 < 0) p2 = 0;
				p1 = 0;
			}
		}
		else if (p1 > 0)
			p1--;
		else if (p2 > 0)
			p2--;
		if (negP2)
		{
			p1 -= p2;
			if (p1 < 0)
			{
				p2 += p1;
				p1 = 0;
			}
		}
		_assert(p1 >= 0 && p2 >= 0);
		if (p0type != TYPE_BLOB)
		{
			while (*z && p1)
			{
				_strskiputf8(z);
				p1--;
			}
			for (z2 = z; *z2 && p2; p2--)
				_strskiputf8(z2);
			sqlite3_result_text(fctx, (char*)z, (int)(z2-z), SQLITE_TRANSIENT);
		}
		else
		{
			if (p1 + p2 > len)
			{
				p2 = len-p1;
				if (p2 < 0) p2 = 0;
			}
			sqlite3_result_blob(fctx, (char*)&z[p1], (int)p2, SQLITE_TRANSIENT);
		}
	}

#ifndef OMIT_FLOATING_POINT
	__device__ static void RoundFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1 || argc == 2);
		int n = 0;
		if (argc == 2)
		{
			if (sqlite3_value_type(argv[1]) == TYPE_NULL) return;
			n = sqlite3_value_int(argv[1]);
			if (n > 30) n = 30;
			if (n < 0) n = 0;
		}
		if (sqlite3_value_type(argv[0]) == TYPE_NULL) return;
		double r = sqlite3_value_double(argv[0]);
		// If Y==0 and X will fit in a 64-bit int, handle the rounding directly, otherwise use printf.
		if (n == 0 && r >= 0 && r < LARGEST_INT64-1)
			r = (double)((int64)(r+0.5));
		else if (n == 0 && r < 0 && (-r) < LARGEST_INT64-1)
			r = -(double)((int64)((-r)+0.5));
		else
		{
			char *buf = _mprintf("%.*f",n,r);
			if (!buf)
			{
				sqlite3_result_error_nomem(fctx);
				return;
			}
			ConvertEx::Atof(buf, &r, _strlen30(buf), TEXTENCODE_UTF8);
			_free(buf);
		}
		sqlite3_result_double(fctx, r);
	}
#endif

	__device__ static void *ContextMalloc(FuncContext *fctx, int64 bytes)
	{

		Context *ctx = sqlite3_context_db_handle(fctx);
		_assert(bytes > 0);
		ASSERTCOVERAGE(bytes == ctx->Limits[LIMIT_LENGTH]);
		ASSERTCOVERAGE(bytes == ctx->Limits[LIMIT_LENGTH]+1);
		char *z;
		if (bytes > ctx->Limits[LIMIT_LENGTH])
		{
			sqlite3_result_error_toobig(fctx);
			z = nullptr;
		}
		else
		{
			z = sqlite3Malloc((int)bytes);
			if (!z)
				sqlite3_result_error_nomem(fctx);
		}
		return z;
	}

	__device__ static void UpperFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		const char *z2 = (char*)sqlite3_value_text(argv[0]);
		int n = sqlite3_value_bytes(argv[0]);
		// Verify that the call to _bytes() does not invalidate the _text() pointer
		_assert(z2 == (char *)sqlite3_value_text(argv[0]));
		if (z2)
		{
			char *z1 = ContextMalloc(fctx, ((int64)n)+1);
			if (z1)
			{
				for(int i = 0; i < n; i++)
					z1[i] = _toupper(z2[i]);
				sqlite3_result_text(fctx, z1, n, _free);
			}
		}
	}

	__device__ static void LowerFunc(FuncContext *fctx, int argc, Mem **argv){
		const char *z2 = (char*)sqlite3_value_text(argv[0]);
		int n = sqlite3_value_bytes(argv[0]);
		// Verify that the call to _bytes() does not invalidate the _text() pointer
		_assert(z2 == (char *)sqlite3_value_text(argv[0]));
		if (z2)
		{
			char *z1 = ContextMalloc(fctx, ((int64)n)+1);
			if (z1)
			{
				for (int i = 0; i < n; i++)
					z1[i] = _tolower(z2[i]);
				sqlite3_result_text(fctx, z1, n, _free);
			}
		}
	}

#define ifnullFunc versionFunc   // Substitute function - never called

	__device__ static void RandomFunc(FuncContext *fctx, int notUsed1, Mem **notUsed2)
	{
		int64 r;
		sqlite3_randomness(sizeof(r), &r);
		if (r < 0)
		{
			// We need to prevent a random number of 0x8000000000000000 (or -9223372036854775808) since when you do abs() of that
			// number of you get the same value back again.  To do this in a way that is testable, mask the sign bit off of negative
			// values, resulting in a positive value.  Then take the 2s complement of that positive value.  The end result can
			// therefore be no less than -9223372036854775807.
			r = -(r & LARGEST_INT64);
		}
		sqlite3_result_int64(fctx, r);
	}

	__device__ static void RandomBlob(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		int n = sqlite3_value_int(argv[0]);
		if (n < 1)
			n = 1;
		unsigned char *p = ContextMalloc(fctx, n);
		if (p)
		{
			sqlite3_randomness(n, p);
			sqlite3_result_blob(fctx, (char *)p, n, _free);
		}
	}

	__device__ static void LastInsertRowid(FuncContext *fctx, int notUsed1, Mem **notUsed2)
	{
		Context *ctx = sqlite3_context_db_handle(fctx);
		// IMP: R-51513-12026 The last_insert_rowid() SQL function is a wrapper around the sqlite3_last_insert_rowid() C/C++ interface function.
		sqlite3_result_int64(fctx, sqlite3_last_insert_rowid(ctx));
	}

	__device__ static void Changes(FuncContext *fctx, int notUsed1, Mem **notUsed2)
	{
		Context *ctx = sqlite3_context_db_handle(fctx);
		sqlite3_result_int(fctx, sqlite3_changes(ctx));
	}

	__device__ static void TotalChanges(FuncContext *fctx, int notUsed1, Mem **notUsed2)
	{
		Context *ctx = sqlite3_context_db_handle(fctx);
		// IMP: R-52756-41993 This function is a wrapper around the sqlite3_total_changes() C/C++ interface.
		sqlite3_result_int(fctx, sqlite3_total_changes(db));
	}

	struct CompareInfo
	{
		uint8 MatchAll;
		uint8 MatchOne;
		uint8 MatchSet;
		bool NoCase;
	};

	__device__ static const CompareInfo _globInfo = { '*', '?', '[', false };
	__device__ static const CompareInfo _likeInfoNorm = { '%', '_',   0, true }; // The correct SQL-92 behavior is for the LIKE operator to ignore case.  Thus  'a' LIKE 'A' would be true.
	__device__ static const CompareInfo _likeInfoAlt = { '%', '_',   0, false }; // If SQLITE_CASE_SENSITIVE_LIKE is defined, then the LIKE operator is case sensitive causing 'a' LIKE 'A' to be false

	__device__ static bool PatternCompare(const uint8 *pattern, const uint8 *string, const CompareInfo *info, uint32 escape)
	{
		uint32 c, c2;
		int invert;
		int seen;
		uint8 matchOne = info->MatchOne;
		uint8 matchAll = info->MatchAll;
		uint8 matchSet = info->MatchSet;
		bool noCase = info->NoCase; 
		bool prevEscape = false; // True if the previous character was 'escape'

		while ((c = SysEx::Utf8Read(&pattern)) != 0)
		{
			if (c == matchAll && !prevEscape)
			{
				while ((c = SysEx::Utf8Read(&pattern)) == matchAll || c == matchOne)
					if (c == matchOne && SysEx::Utf8Read(&string) == 0)
						return false;
				if (c == 0)
					return true;
				else if (c == escape)
				{
					c = SysEx::Utf8Read(&pattern);
					if (c == 0)
						return false;
				}
				else if (c == matchSet)
				{
					_assert(escape == 0); // This is GLOB, not LIKE
					_assert(matchSet < 0x80); // '[' is a single-byte character
					while (*string && !PatternCompare(&pattern[-1], string, info, escape))
						_strskiputf8(string);
					return (*string != 0);
				}
				while ((c2 = SysEx::Utf8Read(&string)) != 0)
				{
					if (noCase)
					{
						c2 = _tolower(c2);
						c = _tolower(c);
						while (c2 != 0 && c2 != c)
						{
							c2 = SysEx::Utf8Read(&string);
							c2 = _tolower(c2);
						}
					}
					else
					{
						while (c2 != 0 && c2 != c)
							c2 = SysEx::Utf8Read(&string);
					}
					if (c2 == 0) return false;
					if (PatternCompare(pattern, string, info, escape)) return true;
				}
				return false;
			}
			else if (c == matchOne && !prevEscape)
			{
				if (SysEx::Utf8Read(&string) == 0)
					return false;
			}
			else if (c == matchSet)
			{
				uint32 prior_c = 0;
				_assert(escape == 0); // This only occurs for GLOB, not LIKE
				seen = 0;
				invert = 0;
				c = SysEx::Utf8Read(&string);
				if (c == 0) return false;
				c2 = SysEx::Utf8Read(&pattern);
				if (c2 == '^')
				{
					invert = 1;
					c2 = SysEx::Utf8Read(&pattern);
				}
				if (c2 == ']')
				{
					if (c == ']') seen = 1;
					c2 = SysEx::Utf8Read(&pattern);
				}
				while (c2 && c2 != ']')
				{
					if (c2 == '-' && pattern[0] != ']' && pattern[0] != 0 && prior_c > 0)
					{
						c2 = SysEx::Utf8Read(&pattern);
						if (c >= prior_c && c <= c2) seen = 1;
						prior_c = 0;
					}
					else
					{
						if (c == c2)
							seen = 1;
						prior_c = c2;
					}
					c2 = SysEx::Utf8Read(&pattern);
				}
				if (c2 == 0 || (seen ^ invert) == 0)
					return false;
			}
			else if (escape == c && !prevEscape)
				prevEscape = true;
			else
			{
				c2 = SysEx::Utf8Read(&string);
				if (noCase)
				{
					c = _tolower(c);
					c2 = _tolower(c2);
				}
				if (c != c2)
					return false;
				prevEscape = false;
			}
		}
		return (*string == 0);
	}

#ifdef TEST
	__device__ static int _likeCount = 0;
#endif
	__device__ static void LikeFunc(FuncContext *fctx,  int argc,  Mem **argv)
	{
		uint32 escape = 0;
		Context *ctx = sqlite3_context_db_handle(fctx);

		const unsigned char *zB = sqlite3_value_text(argv[0]);
		const unsigned char *zA = sqlite3_value_text(argv[1]);

		// Limit the length of the LIKE or GLOB pattern to avoid problems of deep recursion and N*N behavior in patternCompare().
		int pats = sqlite3_value_bytes(argv[0]);
		ASSERTCOVERAGE(pats == ctx->Limits[LIMIT_LIKE_PATTERN_LENGTH]);
		ASSERTCOVERAGE(pats == ctx->Limits[LIMIT_LIKE_PATTERN_LENGTH]+1);
		if (pats > ctx->Limits[LIMIT_LIKE_PATTERN_LENGTH])
		{
			sqlite3_result_error(fctx, "LIKE or GLOB pattern too complex", -1);
			return;
		}
		_assert(zB == sqlite3_value_text(argv[0])); // Encoding did not change

		if (argc == 3)
		{
			// The escape character string must consist of a single UTF-8 character. Otherwise, return an error.
			const unsigned char *zEscape = sqlite3_value_text(argv[2]);
			if (!zEscape) return;
			if (SysEx::Utf8CharLen((char *)zEscape, -1) != 1)
			{
				sqlite3_result_error(fctx, "ESCAPE expression must be a single character", -1);
				return;
			}
			escape = SysEx::Utf8Read(&zEscape);
		}
		if (zA && zB)
		{
			CompareInfo *info = sqlite3_user_data(fctx);
#ifdef TEST
			_likeCount++;
#endif
			sqlite3_result_int(fctx, PatternCompare(zB, zA, info, escape));
		}
	}

	__device__ static void NullifFunc(FuncContext *fctx, int notUsed1, Mem **argv)
	{
		CollSeq *coll = Func::GetFuncCollSeq(fctx);
		if (sqlite3MemCompare(argv[0], argv[1], coll) != 0)
			sqlite3_result_value(fctx, argv[0]);
	}

	__device__ static void VersionFunc(FuncContext *fctx, int notUsed1, Mem **notUsed2)
	{
		// IMP: R-48699-48617 This function is an SQL wrapper around the sqlite3_libversion() C-interface.
		sqlite3_result_text(fctx, sqlite3_libversion(), -1, DESTRUCTOR_STATIC);
	}

	__device__ static void SourceidFunc(FuncContext *fctx, int notUsed1, Mem **notUsed2)
	{
		// IMP: R-24470-31136 This function is an SQL wrapper around the sqlite3_sourceid() C interface.
		sqlite3_result_text(fctx, sqlite3_sourceid(), -1, DESTRUCTOR_STATIC);
	}

	__device__ static void ErrlogFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		SysEx_LOG(sqlite3_value_int(argv[0]), "%s", sqlite3_value_text(argv[1]));
	}


#ifndef OMIT_COMPILEOPTION_DIAGS
	extern __device__ bool CompileTimeOptionUsed(const char *optName);
	__device__ static void CompileoptionusedFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		// IMP: R-39564-36305 The sqlite_compileoption_used() SQL function is a wrapper around the sqlite3_compileoption_used() C/C++ function.
		const char *optName;
		if ((optName = (const char*)sqlite3_value_text(argv[0])) != nullptr)
			sqlite3_result_int(fctx, CompileTimeOptionUsed(optName));
	}

	__device__ static void CompileoptiongetFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		assert(argc == 1);
		// IMP: R-04922-24076 The sqlite_compileoption_get() SQL function is a wrapper around the sqlite3_compileoption_get() C/C++ function.
		int n = sqlite3_value_int(argv[0]);
		sqlite3_result_text(fctx, CompileTimeGet(n), -1, DESTRUCTOR_STATIC);
	}
#endif

	// [EXPERIMENTAL]
	__device__ static const char _hexdigits[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'  };
	__device__ static void QuoteFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		switch (sqlite3_value_type(argv[0]))
		{
		case TYPE_FLOAT: {
			double r1 = sqlite3_value_double(argv[0]);
			char b[50];
			__snprintf(sizeof(b), b, "%!.15g", r1);
			double r2;
			ConvertEx::Atof(b, &r2, 20, TEXTENCODE_UTF8);
			if (r1 != r2)
				__snprintf(sizeof(b), b, "%!.20e", r1);
			sqlite3_result_text(fctx, b, -1, DESTRUCTOR_TRANSIENT);
			break; }

		case TYPE_INTEGER: {
			sqlite3_result_value(fctx, argv[0]);
			break; }

		case TYPE_BLOB: {
			char *z = 0;
			char const *blob = sqlite3_value_blob(argv[0]);
			int blobLength = sqlite3_value_bytes(argv[0]);
			_assert(blob == sqlite3_value_blob(argv[0])); // No encoding change
			z = (char *)ContextMalloc(fctx, (2*(int64)blobLength)+4); 
			if (z)
			{
				for (int i = 0; i < blobLength; i++)
				{
					z[(i*2)+2] = _hexdigits[(blob[i]>>4)&0x0F];
					z[(i*2)+3] = _hexdigits[(blob[i])&0x0F];
				}
				z[(blobLength*2)+2] = '\'';
				z[(blobLength*2)+3] = '\0';
				z[0] = 'X';
				z[1] = '\'';
				sqlite3_result_text(fctx, z, -1, DESTRUCTOR_TRANSIENT);
				_free(z);
			}
			break; }

		case TYPE_TEXT: {
			const unsigned char *zArg = sqlite3_value_text(argv[0]);
			if (zArg == nullptr) return;
			int i, j;
			uint64 n;
			for (i = 0, n = 0; zArg[i]; i++) { if (zArg[i] == '\'') n++; }
			char *z = (char *)ContextMalloc(fctx, ((int64)i)+((int64)n)+3);
			if (z)
			{
				z[0] = '\'';
				for (i = 0, j = 1; zArg[i]; i++)
				{
					z[j++] = zArg[i];
					if (zArg[i] == '\'')
						z[j++] = '\'';
				}
				z[j++] = '\'';
				z[j] = 0;
				sqlite3_result_text(fctx, z, j, _free);
			}
			break; }

		default: {
			assert( sqlite3_value_type(argv[0]) == TYPE_NULL);
			sqlite3_result_text(fctx, "NULL", 4, DESTRUCTOR_STATIC);
			break; }
		}
	}

	__device__ static void UnicodeFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		const unsigned char *z = sqlite3_value_text(argv[0]);
		if (z && z[0]) sqlite3_result_int(fctx, SysEx::Utf8Read(&z));
	}

	__device__ static void CharFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		unsigned char *z, *z2;
		z = z2 = sqlite3_malloc(argc*4);
		if (!z)
		{
			sqlite3_result_error_nomem(fctx);
			return;
		}
		for (int i = 0; i < argc; i++)
		{
			int64 x = sqlite3_value_int64(argv[i]);
			if (x < 0 || x > 0x10ffff) x = 0xfffd;
			unsigned c = (unsigned)(x & 0x1fffff);
			if (c < 0x00080)
				*z2++ = (uint8)(c & 0xFF);
			else if (c < 0x00800)
			{
				*z2++ = 0xC0 + (uint8)((c>>6) & 0x1F);
				*z2++ = 0x80 + (uint8)(c & 0x3F);
			}
			else if (c < 0x10000)
			{
				*z2++ = 0xE0 + (uint8)((c>>12)&0x0F);
				*z2++ = 0x80 + (uint8)((c>>6) & 0x3F);
				*z2++ = 0x80 + (uint8)(c & 0x3F);
			}
			else
			{
				*z2++ = 0xF0 + (uint8)((c>>18) & 0x07);
				*z2++ = 0x80 + (uint8)((c>>12) & 0x3F);
				*z2++ = 0x80 + (uint8)((c>>6) & 0x3F);
				*z2++ = 0x80 + (uint8)(c & 0x3F);
			}
		}
		sqlite3_result_text(fctx, (char *)z, (int)(z2 - z), _free);
	}

	__device__ static void HexFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		const unsigned char *blob = sqlite3_value_blob(argv[0]);
		int n = sqlite3_value_bytes(argv[0]);
		_assert(blob == sqlite3_value_blob(argv[0])); // No encoding change
		char *zHex, *z;
		zHex = z = (char *)ContextMalloc(fctx, ((int64)n)*2 + 1);
		if (zHex)
		{
			for (int i = 0; i < n; i++, blob++)
			{
				unsigned char c = *blob;
				*(z++) = _hexdigits[(c>>4)&0xf];
				*(z++) = _hexdigits[c&0xf];
			}
			*z = 0;
			sqlite3_result_text(fctx, zHex, n*2, _free);
		}
	}

	__device__ static void ZeroblobFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		Context *ctx = sqlite3_context_db_handle(fctx);
		int64 n = sqlite3_value_int64(argv[0]);
		ASSERTCOVERAGE(n == ctx->Limits[LIMIT_LENGTH]);
		ASSERTCOVERAGE(n == ctx->Limits[LIMIT_LENGTH]+1);
		if (n > ctx->Limits[LIMIT_LENGTH])
			sqlite3_result_error_toobig(fctx);
		else
			sqlite3_result_zeroblob(fctx, (int)n); // IMP: R-00293-64994
	}

	__device__ static void ReplaceFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 3);
		const unsigned char *string = sqlite3_value_text(argv[0]); // The input string A
		if (!string) return;
		int stringLength = sqlite3_value_bytes(argv[0]); // Size of string
		_assert(string == sqlite3_value_text(argv[0])); // No encoding change
		const unsigned char *pattern = sqlite3_value_text(argv[1]);
		if (!pattern)
		{
			_assert(sqlite3_value_type(argv[1]) == TYPE_NULL || sqlite3_context_db_handle(fctx)->MallocFailed);
			return;
		}
		if (pattern[0] == 0)
		{
			_assert(sqlite3_value_type(argv[1]) != TYPE_NULL);
			sqlite3_result_value(fctx, argv[0]);
			return;
		}
		int patternLength = sqlite3_value_bytes(argv[1]); // Size of pattern
		_assert(pattern == sqlite3_value_text(argv[1])); // No encoding change
		const unsigned char *replacement = sqlite3_value_text(argv[2]); // The replacement string C
		if (!replacement) return;
		int replacementLength = sqlite3_value_bytes(argv[2]); // Size of replacement
		_assert(replacement == sqlite3_value_text(argv[2]));
		int64 outLength = stringLength + 1; // Maximum size of out
		_assert(outLength < CORE_MAX_LENGTH);
		unsigned char *out = (unsigned char *)ContextMalloc(fctx, (int64)outLength); // The output
		if (!out)
			return;
		int loopLimit = stringLength - patternLength; // Last string[] that might match pattern[]
		int i, j;
		for (i = j = 0; i <= loopLimit; i++)
		{
			if (string[i] != pattern[0] || _memcmp(&string[i], pattern, patternLength))
				out[j++] = string[i];
			else
			{
				Context *ctx = sqlite3_context_db_handle(fctx);
				outLength += replacementLength - patternLength;
				ASSERTCOVERAGE(outLength-1 == ctx->Limits[LIMIT_LENGTH]);
				ASSERTCOVERAGE(outLength-2 == ctx->Limits[LIMIT_LENGTH]);
				if (outLength-1 > ctx->Limits[LIMIT_LENGTH])
				{
					sqlite3_result_error_toobig(fctx);
					_free(out);
					return;
				}
				uint8 *oldOut = out;
				out = (unsigned char *)SysEx::Realloc(out, (int)outLength);
				if (!out)
				{
					sqlite3_result_error_nomem(fctx);
					_free(oldOut);
					return;
				}
				_memcpy(&out[j], replacement, replacementLength);
				j += replacementLength;
				i += patternLength-1;
			}
		}
		_assert(j+stringLength-i+1 == outLength);
		_memcpy(&out[j], &string[i], stringLength-i);
		j += stringLength - i;
		_assert(j <= outLength);
		out[j] = 0;
		sqlite3_result_text(fctx, (char*)out, j, _free);
	}

	__device__ static const unsigned char _trimOneLength[] = { 1 };
	__device__ static unsigned char * const _trimOne[] = { (uint8 *)" " };
	__device__ static void TrimFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		const unsigned char *charSet; // Set of characters to trim
		int charSetLength; // Number of characters in charSet
		unsigned char *charsLength = 0; // Length of each character in charSet
		unsigned char **chars = 0; // Individual characters in charSet
		int flags; // 1: trimleft  2: trimright  3: trim

		if (sqlite3_value_type(argv[0]) == TYPE_NULL)
			return;
		const unsigned char *in = sqlite3_value_text(argv[0]); // Input string
		if (!in) return;
		int inLength = sqlite3_value_bytes(argv[0]); // Number of bytes in input
		//? _assert(in == sqlite3_value_text(argv[0]));
		if (argc == 1)
		{
			charSetLength = 1;
			charsLength = (uint8 *)_trimOneLength;
			chars = (unsigned char **)_trimOne;
			charSet = 0;
		}
		else if ((charSet = sqlite3_value_text(argv[1])) == nullptr)
			return;
		else
		{
			const unsigned char *z;
			for (z = charSet, charSetLength = 0; *z; charSetLength++)
				_strskiputf8(z);
			if (charSetLength > 0)
			{
				chars = (unsigned char **)ContextMalloc(fctx, ((int64)charSetLength)*(sizeof(char*)+1));
				if (!chars)
					return;
				charsLength = (unsigned char *)&chars[charSetLength];
				for (z = charSet, charSetLength = 0; *z; charSetLength++)
				{
					chars[charSetLength] = (unsigned char *)z;
					_strskiputf8(z);
					charsLength[charSetLength] = (uint8)(z - chars[charSetLength]);
				}
			}
		}
		int i;
		if (charSetLength > 0)
		{
			flags = PTR_TO_INT(sqlite3_user_data(fctx));
			if (flags & 1)
			{
				while (inLength > 0)
				{
					int len = 0;
					for (i = 0; i < charSetLength; i++)
					{
						len = charsLength[i];
						if (len <= inLength && !_memcmp(in, chars[i], len)) break;
					}
					if (i >= charSetLength) break;
					in += len;
					inLength -= len;
				}
			}
			if (flags & 2)
			{
				while (inLength > 0)
				{
					int len = 0;
					for (i = 0; i < charSetLength; i++)
					{
						len = charsLength[i];
						if (len <= inLength && !_memcmp(&in[inLength-len], chars[i], len)) break;
					}
					if (i >= charSetLength) break;
					inLength -= len;
				}
			}
			if (charSet)
				_free(chars);
		}
		sqlite3_result_text(fctx, (char *)in, inLength, DESTRUCTOR_TRANSIENT);
	}

#ifdef SOUNDEX
	__device__ static const unsigned char _soundexCode[] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 2, 3, 0, 1, 2, 0, 0, 2, 2, 4, 5, 5, 0,
		1, 2, 6, 2, 3, 0, 1, 0, 2, 0, 2, 0, 0, 0, 0, 0,
		0, 0, 1, 2, 3, 0, 1, 2, 0, 0, 2, 2, 4, 5, 5, 0,
		1, 2, 6, 2, 3, 0, 1, 0, 2, 0, 2, 0, 0, 0, 0, 0,
	};

	__device__ static void SoundexFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		const uint8 *z = (uint8 *)sqlite3_value_text(argv[0]);
		if (!z) z = (uint8 *)"";
		int i;
		for (i = 0; z[i] && !_isalpha(z[i]); i++) { }
		char r[8];
		if (z[i])
		{
			uint8 prevcode = _soundexCode[z[i]&0x7f];
			r[0] = _toupper(z[i]);
			int j;
			for (j = 1; j < 4 && z[i]; i++)
			{
				uint8 code = _soundexCode[z[i]&0x7f];
				if (code > 0)
				{
					if (code != prevcode)
					{
						prevcode = code;
						r[j++] = code + '0';
					}
				}
				else
					prevcode = 0;
			}
			while (j < 4)
				r[j++] = '0';
			r[j] = 0;
			sqlite3_result_text(fctx, r, 4, DESTRUCTOR_TRANSIENT);
		}
		else
			sqlite3_result_text(fctx, "?000", 4, DESTRUCTOR_STATIC); // IMP: R-64894-50321 The string "?000" is returned if the argument is NULL or contains no ASCII alphabetic characters. */
	}
#endif

#ifndef OMIT_LOAD_EXTENSION
	__device__ static void LoadExtFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		const char *file = (const char *)sqlite3_value_text(argv[0]);
		Context *ctx = sqlite3_context_db_handle(fctx);
		char *errMsg = nullptr;
		const char *proc = (argc == 2 ? (const char *)sqlite3_value_text(argv[1]) : nullptr);
		if (file && sqlite3_load_extension(ctx, file, proc, &errMsg))
		{
			sqlite3_result_error(fctx, errMsg, -1);
			_free(errMsg);
		}
	}
#endif

	struct SumCtx
	{
		double RSum;	// Floating point sum
		int64 ISum;		// Integer sum
		int64 Count;	// Number of elements summed
		bool overflow;	// True if integer overflow seen
		bool approx;	// True if non-integer value was input to the sum
	};

	__device__ static void SumStep(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1);
		SumCtx *p = sqlite3_aggregate_context(fctx, sizeof(*p));
		TYPE type = sqlite3_value_numeric_type(argv[0]);
		if (p && type != TYPE_NULL)
		{
			p->cnt++;
			if (type == TYPE_INTEGER)
			{
				int64 v = sqlite3_value_int64(argv[0]);
				p->RSum += v;
				if (!(p->Approx | p->Overflow) && sqlite3AddInt64(&p->iSum, v))
					p->Overflow = true;
			}
			else
			{
				p->RSum += sqlite3_value_double(argv[0]);
				p->Approx = true;
			}
		}
	}

	__device__ static void SumFinalize(FuncContext *fctx)
	{
		SumCtx *p = sqlite3_aggregate_context(fctx, 0);
		if (p && p->Count > 0)
		{
			if (p->Overflow)
				sqlite3_result_error(fctx, "integer overflow", -1);
			else if (p->Approx)
				sqlite3_result_double(fctx, p->rSum);
			else
				sqlite3_result_int64(fctx, p->iSum);
		}
	}

	__device__ static void AvgFinalize(FuncContext *fctx)
	{
		SumCtx *p = sqlite3_aggregate_context(fctx, 0);
		if (p && p->Cnt > 0)
			sqlite3_result_double(fctx, p->rSum/(double)p->cnt);
	}

	__device__ static void TotalFinalize(FuncContext *fctx)
	{
		SumCtx *p = sqlite3_aggregate_context(fctx, 0);
		sqlite3_result_double(fctx, p ? p->rSum : (double)0); // (double)0 In case of OMIT_FLOATING_POINT...
	}

	struct CountCtx
	{
		int64 N;
	};

	__device__ static void CountStep(FuncContext *fctx, int argc, Mem **argv)
	{
		CountCtx *p = sqlite3_aggregate_context(fctx, sizeof(*p));
		if ((argc == 0 || TYPE_NULL != sqlite3_value_type(argv[0])) && p)
			p->N++;
	}

	__device__ static void CountFinalize(FuncContext *fctx)
	{
		CountCtx *p = sqlite3_aggregate_context(fctx, 0);
		sqlite3_result_int64(fctx, p ? p->N : 0);
	}

	__device__ static void MinMaxStep(FuncContext *fctx, int notUsed1, Mem **argv)
	{
		Mem *arg = (Mem *)argv[0];
		Mem *best = (Mem *)sqlite3_aggregate_context(fctx, sizeof(*best));
		if (!best) return;
		if (sqlite3_value_type(argv[0]) == TYPE_NULL)
		{
			if (best->Flags) sqlite3SkipAccumulatorLoad(fctx);
		}
		else if (best->Flags)
		{
			CollSeq *coll = sqlite3GetFuncCollSeq(fctx);
			// This step function is used for both the min() and max() aggregates, the only difference between the two being that the sense of the
			// comparison is inverted. For the max() aggregate, the sqlite3_user_data() function returns (void *)-1. For min() it
			// returns (void *)db, where db is the sqlite3* database pointer. Therefore the next statement sets variable 'max' to 1 for the max()
			// aggregate, or 0 for min().
			bool max = (sqlite3_user_data(fctx) != 0);
			int cmp = sqlite3MemCompare(best, arg, coll);
			if ((max && cmp < 0) || (!max && cmp > 0))
				sqlite3VdbeMemCopy(best, arg);
			else
				sqlite3SkipAccumulatorLoad(fctx);
		}
		else
		{
			sqlite3VdbeMemCopy(best, arg);
		}
	}

	__device__ static void MinMaxFinalize(FuncContext *fctx)
	{
		Mem *r = (Mem *)sqlite3_aggregate_context(fctx, 0);
		if (r)
		{
			if (r->Flags)
				sqlite3_result_value(fctx, r);
			sqlite3VdbeMemRelease(r);
		}
	}

	__device__ static void GroupConcatStep(FuncContext *fctx, int argc, Mem **argv)
	{
		_assert(argc == 1 || argc == 2);
		if (sqlite3_value_type(argv[0]) == TYPE_NULL) return;
		TextBuilder *b = (TextBuilder *)sqlite3_aggregate_context(fctx, sizeof(*b));
		if (b)
		{
			Context *ctx = sqlite3_context_db_handle(fctx);
			bool firstTerm = (b->AllocType == 0);
			b->AllocType = 2;
			b->MaxSize = ctx->Limits[LIMIT_LENGTH];
			if (!firstTerm)
			{
				const char *zSep;
				int nSep;
				if (argc == 2)
				{
					zSep = (char *)sqlite3_value_text(argv[1]);
					nSep = sqlite3_value_bytes(argv[1]);
				}
				else
				{
					zSep = ",";
					nSep = 1;
				}
				b->Append(zSep, nSep);
			}
			const char *zVal = (char *)sqlite3_value_text(argv[0]);
			int nVal = sqlite3_value_bytes(argv[0]);
			b->Append(zVal, nVal);
		}
	}

	__device__ static void GroupConcatFinalize(FuncContext *fctx)
	{
		TextBuilder *b = sqlite3_aggregate_context(fctx, 0);
		if (b)
		{
			if (b->Overflowed)
				sqlite3_result_error_toobig(fctx);
			else if (b->AllocFailed)
				sqlite3_result_error_nomem(fctx);
			else   
				sqlite3_result_text(fctx, b->ToString(), -1,  _free);
		}
	}

	__device__ void Func::RegisterBuiltinFunctions(Context *ctx)
	{
		RC rc = sqlite3_overload_function(ctx, "MATCH", 2);
		_assert(rc == RC_NOMEM || rc == RC_OK);
		if (rc == RC_NOMEM)
			ctx->MallocFailed = true;
	}

	static void SetLikeOptFlag(Context *ctx, const char *name, uint8 flagVal)
	{
		FuncDef *def = sqlite3FindFunction(ctx, name, _strlen30(name), 2, TEXTENCODE_UTF8, 0);
		if (_ALWAYS(def))
			def->Flags = flagVal;
	}

	__device__ void Func::RegisterLikeFunctions(Context *ctx, bool caseSensitive)
	{
		CompareInfo *info = (caseSensitive ? (CompareInfo *)&_likeInfoAlt : (CompareInfo *)&_likeInfoNorm);
		sqlite3CreateFunc(ctx, "like", 2, TEXTENCODE_UTF8, info, LikeFunc, 0, 0, 0);
		sqlite3CreateFunc(ctx, "like", 3, TEXTENCODE_UTF8, info, LikeFunc, 0, 0, 0);
		sqlite3CreateFunc(ctx, "glob", 2, TEXTENCODE_UTF8, (CompareInfo *)&globInfo, LikeFunc, 0, 0, 0);
		SetLikeOptFlag(ctx, "glob", FUNC_LIKE | FUNC_CASE);
		SetLikeOptFlag(ctx, "like", caseSensitive ? (FUNC_LIKE | FUNC_CASE) : FUNC_LIKE);
	}

	__device__ bool Func::IsLikeFunction(Context *ctx, Expr *expr, int *isNocase, char *wc)
	{
		if (expr->OP != TK_FUNCTION  || !expr->x.List || expr->x.List->Exprs != 2)
			return false;
		_assert(!ExprHasProperty(expr, EP_xIsSelect));
		FuncDef *def = sqlite3FindFunction(ctx, expr->u.Token, _strlen30(expr->u.Token), 2, TEXTENCODE_UTF8, 0);
		if (_NEVER(def == nullptr) || (def->Flags & FUNC_LIKE) == 0)
			return false;
		// The memcpy() statement assumes that the wildcard characters are the first three statements in the compareInfo structure.  The asserts() that follow verify that assumption
		_memcpy(wc, def->UserData, 3);
		_assert((char *)&_likeInfoAlt == (char*)&_likeInfoAlt.MatchAll);
		_assert(&((char *)&_likeInfoAlt)[1] == (char*)&_likeInfoAlt.MatchOne);
		_assert(&((char *)&_likeInfoAlt)[2] == (char*)&_likeInfoAlt.MatchSet);
		*isNoCase = ((def->Flags & FUNC_CASE) == 0);
		return true;
	}

	// The following array holds FuncDef structures for all of the functions defined in this file.
	//
	// The array cannot be constant since changes are made to the FuncDef.pHash elements at start-time.  The elements of this array are read-only after initialization is complete.
	__device__ static FuncDef _builtinFuncs[] = {
		FUNCTION(ltrim,              1, 1, 0, TrimFunc         ),
		FUNCTION(ltrim,              2, 1, 0, TrimFunc         ),
		FUNCTION(rtrim,              1, 2, 0, TrimFunc         ),
		FUNCTION(rtrim,              2, 2, 0, TrimFunc         ),
		FUNCTION(trim,               1, 3, 0, TrimFunc         ),
		FUNCTION(trim,               2, 3, 0, TrimFunc         ),
		FUNCTION(min,               -1, 0, 1, MinMaxFunc       ),
		FUNCTION(min,                0, 0, 1, nullptr          ),
		AGGREGATE(min,               1, 0, 1, MinMaxStep,      MinMaxFinalize),
		FUNCTION(max,               -1, 1, 1, MinMaxFunc       ),
		FUNCTION(max,                0, 1, 1, nullptr          ),
		AGGREGATE(max,               1, 1, 1, MinMaxStep,      MinMaxFinalize),
		FUNCTION2(typeof,            1, 0, 0, TypeOfFunc,  FUNC_TYPEOF),
		FUNCTION2(length,            1, 0, 0, LengthFunc,  FUNC_LENGTH),
		FUNCTION(instr,              2, 0, 0, InstrFunc        ),
		FUNCTION(substr,             2, 0, 0, SubstrFunc       ),
		FUNCTION(substr,             3, 0, 0, SubstrFunc       ),
		FUNCTION(unicode,            1, 0, 0, UnicodeFunc      ),
		FUNCTION(char,              -1, 0, 0, CharFunc         ),
		FUNCTION(abs,                1, 0, 0, AbsFunc          ),
#ifndef FLOATING_POINT
		FUNCTION(round,              1, 0, 0, RoundFunc        ),
		FUNCTION(round,              2, 0, 0, RoundFunc        ),
#endif
		FUNCTION(upper,              1, 0, 0, UpperFunc        ),
		FUNCTION(lower,              1, 0, 0, LowerFunc        ),
		FUNCTION(coalesce,           1, 0, 0, nullptr          ),
		FUNCTION(coalesce,           0, 0, 0, nullptr          ),
		FUNCTION2(coalesce,         -1, 0, 0, IfNullFunc,  FUNC_COALESCE),
		FUNCTION(hex,                1, 0, 0, HexFunc          ),
		FUNCTION2(ifnull,            2, 0, 0, IfNullFunc,  FUNC_COALESCE),
		FUNCTION(random,             0, 0, 0, RandomFunc       ),
		FUNCTION(randomblob,         1, 0, 0, RandomBlob       ),
		FUNCTION(nullif,             2, 0, 1, NullIfFunc       ),
		FUNCTION(sqlite_version,     0, 0, 0, VersionFunc      ),
		FUNCTION(sqlite_source_id,   0, 0, 0, SourceIdFunc     ),
		FUNCTION(sqlite_log,         2, 0, 0, ErrlogFunc       ),
#ifndef OMIT_COMPILEOPTION_DIAGS
		FUNCTION(sqlite_compileoption_used,1, 0, 0, CompileoptionusedFunc),
		FUNCTION(sqlite_compileoption_get, 1, 0, 0, CompileoptiongetFunc),
#endif
		FUNCTION(quote,              1, 0, 0, QuoteFunc        ),
		FUNCTION(last_insert_rowid,  0, 0, 0, LastInsertRowid  ),
		FUNCTION(changes,            0, 0, 0, Changes          ),
		FUNCTION(total_changes,      0, 0, 0, TotalChanges     ),
		FUNCTION(replace,            3, 0, 0, ReplaceFunc      ),
		FUNCTION(zeroblob,           1, 0, 0, ZeroBlobFunc     ),
#ifdef SOUNDEX
		FUNCTION(soundex,            1, 0, 0, SoundexFunc      ),
#endif
#ifndef OMIT_LOAD_EXTENSION
		FUNCTION(load_extension,     1, 0, 0, LoadExtFunc      ),
		FUNCTION(load_extension,     2, 0, 0, LoadExtFunc      ),
#endif
		AGGREGATE(sum,               1, 0, 0, SumStep,         SumFinalize),
		AGGREGATE(total,             1, 0, 0, SumStep,         TotalFinalize),
		AGGREGATE(avg,               1, 0, 0, SumStep,         AvgFinalize),
		//AGGREGATE(count,             0, 0, 0, CountStep,       CountFinalize),
		{0, TEXTENCODE_UTF8, FUNC_COUNT, 0, 0, 0, CountStep, CountFinalize, "count", 0, 0}, 
		AGGREGATE(count,             1, 0, 0, CountStep,       CountFinalize),
		AGGREGATE(group_concat,      1, 0, 0, GroupConcatStep, GroupConcatFinalize),
		AGGREGATE(group_concat,      2, 0, 0, GroupConcatStep, GroupConcatFinalize),
		LIKEFUNC(glob, 2, &globInfo, FUNC_LIKE|FUNC_CASE),
#ifdef CASE_SENSITIVE_LIKE
		LIKEFUNC(like, 2, &_likeInfoAlt, FUNC_LIKE|FUNC_CASE),
		LIKEFUNC(like, 3, &_likeInfoAlt, FUNC_LIKE|FUNC_CASE),
#else
		LIKEFUNC(like, 2, &_likeInfoNorm, FUNC_LIKE),
		LIKEFUNC(like, 3, &_likeInfoNorm, FUNC_LIKE),
#endif
	};

	__device__ void Func::RegisterGlobalFunctions()
	{
		FuncDefHash *hash = Context::GlobalFunctions;
		FuncDef *func = (FuncDef *)_builtinFunc;
		for (int i = 0; i < _lengthof(_builtinFunc); i++)
			hash->Insert(&func[i]);
		Date_::RegisterDateTimeFunctions();
#ifndef OMIT_ALTERTABLE
		Alter::Functions();
#endif
	}

}}