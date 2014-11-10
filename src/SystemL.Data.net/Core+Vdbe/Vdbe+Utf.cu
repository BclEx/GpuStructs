#ifndef OMIT_UTF16
#include "Core.cu.h"

namespace Core
{

	/* #define TRANSLATE_TRACE 1 */

	__device__ int Vdbe::VdbeMemTranslate(Mem *mem, TEXTENCODE desiredEncode)
	{
		int len; // Maximum length of output string in bytes
		unsigned char *zOut; // Output buffer
		unsigned char *zIn; // Input iterator
		unsigned char *zTerm; // End of input
		unsigned char *z; // Output iterator
		unsigned int c;

		_assert(mem->Ctx == nullptr || MutexEx::Held(mem->Ctx->Mutex));
		_assert(mem->Flags & MEM_Str);
		_assert(mem->Encode != desiredEncode);
		_assert(mem->Encode != 0);
		_assert(mem->n >= 0);

#if defined(TRANSLATE_TRACE) && defined(_DEBUG)
		{
			char buf[100];
			sqlite3VdbeMemPrettyPrint(mem, buf);
			fprintf(stderr, "INPUT:  %s\n", buf);
		}
#endif

		// If the translation is between UTF-16 little and big endian, then all that is required is to swap the byte order. This case is handled differently from the others.
		if (mem->Encode != TEXTENCODE_UTF8 && desiredEncode != TEXTENCODE_UTF8)
		{
			RC rc = sqlite3VdbeMemMakeWriteable(mem);
			if (rc != RC_OK)
			{
				_assert(rc == RC_NOMEM);
				return RC_NOMEM;
			}
			zIn = (uint8*)mem->z;
			zTerm = &zIn[mem->n&~1];
			while (zIn < zTerm)
			{
				c = *zIn;
				*zIn = *(zIn+1);
				zIn++;
				*zIn++ = c;
			}
			mem->Encode = desiredEncode;
			goto translate_out;
		}

		// Set len to the maximum number of bytes required in the output buffer.
		if (desiredEncode == TEXTENCODE_UTF8)
		{
			// When converting from UTF-16, the maximum growth results from translating a 2-byte character to a 4-byte UTF-8 character.
			// A single byte is required for the output string nul-terminator.
			mem->n &= ~1;
			len = mem->n * 2 + 1;
		}
		else
		{
			// When converting from UTF-8 to UTF-16 the maximum growth is caused when a 1-byte UTF-8 character is translated into a 2-byte UTF-16
			// character. Two bytes are required in the output buffer for the nul-terminator.
			len = mem->n * 2 + 2;
		}

		// Set zIn to point at the start of the input buffer and zTerm to point 1 byte past the end.
		// Variable zOut is set to point at the output buffer, space obtained from sqlite3_malloc().
		zIn = (u8*)mem->z; // Input iterator
		zTerm = &zIn[mem->n]; // End of input
		zOut = _tagalloc(mem->Ctx, len); // Output buffer
		if (!zOut)
			return RC_NOMEM;
		z = zOut; // Output iterator

		if (mem->Encode == TEXTENCODE_UTF8)
		{
			if (desiredEncode == TEXT_UTF16LE)
			{
				// UTF-8 -> UTF-16 Little-endian
				while (zIn < zTerm)
				{
					READ_UTF8(zIn, zTerm, c);
					WRITE_UTF16LE(z, c);
				}
			}
			else
			{
				_assert(desiredEncode == TEXT_UTF16BE);
				// UTF-8 -> UTF-16 Big-endian
				while (zIn < zTerm)
				{
					READ_UTF8(zIn, zTerm, c);
					WRITE_UTF16BE(z, c);
				}
			}
			mem->n = (int)(z - zOut);
			*z++ = 0;
		}
		else
		{
			_assert(desiredEncode == TEXTENCODE_UTF8);
			if (mem->Encode == TEXTENCODE_UTF16LE)
			{
				// UTF-16 Little-endian -> UTF-8
				while (zIn < zTerm)
				{
					READ_UTF16LE(zIn, zIn<zTerm, c); 
					WRITE_UTF8(z, c);
				}
			}
			else
			{
				// UTF-16 Big-endian -> UTF-8 
				while (zIn < zTerm)
				{
					READ_UTF16BE(zIn, zIn<zTerm, c); 
					WRITE_UTF8(z, c);
				}
			}
			mem->n = (int)(z - zOut);
		}
		*z = nullptr;
		_assert((mem->n+(desiredEncode == TEXTENCODE_UTF8 ? 1 : 2)) <= len);

		Vdbe::MemRelease(mem);
		mem->Flags &= ~(MEM_Static|MEM_Dyn|MEM_Ephem);
		mem->Encode = desiredEncode;
		mem->Flags |= (MEM_Term|MEM_Dyn);
		mem->z = (char*)zOut;
		mem->zMalloc = mem->z;

translate_out:
#if defined(TRANSLATE_TRACE) && defined(_DEBUG)
		{
			char buf[100];
			sqlite3VdbeMemPrettyPrint(mem, buf);
			fprintf(stderr, "OUTPUT: %s\n", buf);
		}
#endif
		return RC_OK;
	}

	__device__ int Vdbe::VdbeMemHandleBom(Mem *mem)
	{
		int rc = RC_OK;
		uint8 bom = 0;
		_assert(mem->n >= 0);
		if (mem->n > 1)
		{
			uint8 b1 = *(uint8 *)mem->z;
			uint8 b2 = *(((uint8 *)mem->z) + 1);
			if (b1 == 0xFE && b2 == 0xFF)
				bom = TEXTENCODE_UTF16BE;
			if (b1 == 0xFF && b2 == 0xFE)
				bom = TEXTENCODE_UTF16LE;
		}
		if (bom)
		{
			rc = sqlite3VdbeMemMakeWriteable(mem);
			if (rc == RC_OK)
			{
				mem->n -= 2;
				memmove(mem->z, &mem->z[2], mem->n);
				mem->z[mem->n] = '\0';
				mem->z[mem->n+1] = '\0';
				mem->Flags |= MEM_Term;
				mem->Encode = bom;
			}
		}
		return rc;
	}
#endif

#ifndef OMIT_UTF16
	__device__ char *Vdbe::Utf16to8(Context *ctx, const void *z, int nByte, TEXTENCODE encode)
	{
		Mem m;
		_memset(&m, 0, sizeof(m));
		m.Ctx = ctx;
		sqlite3VdbeMemSetStr(&m, z, bytes, encode, SQLITE_STATIC);
		sqlite3VdbeChangeEncoding(&m, TEXTENCODE_UTF8);
		if (ctx->MallocFailed)
		{
			sqlite3VdbeMemRelease(&m);
			m.z = nullptr;
		}
		_assert((m.Flags & MEM_Term) != 0 || ctx->MallocFailed);
		_assert((m.Flags & MEM_Str) != 0 || ctx->MallocFailed);
		_assert((m.Flags & MEM_Dyn) != 0 || ctx->MallocFailed);
		_assert(m.z || ctx->MallocFailed);
		return m.z;
	}

#ifdef ENABLE_STAT3
	__device__ char *Vdbe::Utf8to16(Context *ctx, TEXTENCODE encode, char *z, int n, int *pnOut)
	{
		Mem m;
		_memset(&m, 0, sizeof(m));
		m.Ctx = ctx;
		sqlite3VdbeMemSetStr(&m, z, n, TEXTENCODE_UTF8, SQLITE_STATIC);
		if (sqlite3VdbeMemTranslate(&m, encode))
		{
			assert(ctx->MallocFailed);
			return 0;
		}
		_assert(m.z == m.zMalloc);
		*pnOut = m.n;
		return m.z;
	}
#endif


}
#endif