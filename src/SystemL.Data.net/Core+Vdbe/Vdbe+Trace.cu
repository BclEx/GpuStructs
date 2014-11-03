// vdbetrace.c
#include "VdbeInt.cu.h"

namespace Core {

#pragma region Trace
#ifndef OMIT_TRACE

	__device__ static int FindNextHostParameter(const char *sql, int *tokens)
	{
		int total = 0;
		*tokens = 0;
		while (sql[0])
		{
			int tokenType;
			int n = Parse::GetToken((uint8 *)sql, &tokenType);
			_assert(n > 0 && tokenType != TK_ILLEGAL);
			if (tokenType == TK_VARIABLE)
			{
				*tokens = n;
				break;
			}
			total += n;
			sql += n;
		}
		return total;
	}

	__device__ char *Vdbe::ExpandSql(const char *rawSql)
	{
		Context *ctx = Ctx; // The database connection
		char bBase[100]; // Initial working space
		Text::StringBuilder b; // Accumulate the output here
		Text::StringBuilder::Init(&b, bBase, sizeof(bBase), ctx->Limits[LIMIT_LENGTH]);
		b.Ctx = ctx;
		int nextIndex = 1; // Index of next ? host parameter
		int idx = 0; // Index of a host parameter
		if (ctx->VdbeExecCnt > 1)
			while (*rawSql)
			{
				const char *start = rawSql;
				while (*(rawSql++) != '\n' && *rawSql);
				b.Append("-- ", 3);
				b.Append(start, (int)(rawSql - start));
			}
		else
			while (rawSql[0])
			{
				int tokenLength; // Length of the parameter token
				int n = findNextHostParameter(rawSql, &tokenLength); // Length of a token prefix
				_assert(n > 0);
				b.Append(rawSql, n);
				rawSql += n;
				_assert(rawSql[0] || !tokenLength);
				if (!tokenLength) break;
				if (rawSql[0] == '?')
				{
					if (tokenLength > 1)
					{
						_assert(_isdigit(rawSql[1]));
						ConvertEx::Atoi(&rawSql[1], &idx);
					}
					else
						idx = nextIndex;
				}
				else
				{
					_assert(rawSql[0] == ':' || rawSql[0] == '$' || rawSql[0] == '@');
					ASSERTCOVERAGE(rawSql[0] == ':');
					ASSERTCOVERAGE(rawSql[0] == '$');
					ASSERTCOVERAGE(rawSql[0] == '@');
					idx = ParameterIndex(rawSql, tokenLength);
					_assert(idx > 0);
				}
				rawSql += tokenLength;
				nextIndex = idx + 1;
				_assert(idx > 0 && idx <= Vars.length);
				Mem *var = &Vars.data[idx - 1]; // Value of a host parameter
				if (var->Flags & MEM_Null) b.Append("NULL", 4);
				else if (var->Flags & MEM_Int) sqlite3XPrintf(&b, "%lld", var->u.I);
				else if (var->Flags & MEM_Real) sqlite3XPrintf(&b, "%!.15g", var->R);
				else if (var->Flags & MEM_Str)
				{
#ifndef OMIT_UTF16
					TEXTENCODE encode = CTXENCODE(ctx);
					if (encode != TEXTENCODE_UTF8)
					{
						Mem utf8;
						_memset(&utf8, 0, sizeof(utf8));
						utf8.Ctx = ctx;
						Vdbe::MemSetStr(&utf8, var->Z, var->N, encode, DESTRUCTOR_STATIC);
						Vdbe::ChangeEncoding(&utf8, TEXTENCODE_UTF8);
						sqlite3XPrintf(&b, "'%.*q'", utf8.N, utf8.Z);
						Vdbe::MemRelease(&utf8);
					}
					else
#endif
						sqlite3XPrintf(&b, "'%.*q'", var->N, var->Z);
				}
				else if (var->Flags & MEM_Zero) sqlite3XPrintf(&b, "zeroblob(%d)", var->u.Zero);
				else
				{
					_assert(var->Flags & MEM_Blob);
					b.Append("x'", 2);
					for (int i = 0; i < var->N; i++)
						sqlite3XPrintf(&b, "%02x", var->Z[i] & 0xff);
					b.Append("'", 1);
				}
			}
			return b.ToString();
	}

#endif
#pragma endregion

#pragma region Explain
#if defined(ENABLE_TREE_EXPLAIN)

	__device__ void sqlite3ExplainBegin(Vdbe *vdbe)
	{
		if (vdbe)
		{
			SysEx::BeginBenignAlloc();
			Explain *p = (Explain *)SysEx::Alloc(sizeof(Explain), true);
			if (p)
			{
				p->Vdbe = vdbe;
				SysEx::Free(vdbe->Explain);
				vdbe->Explain = p;
				Text::StringBuilder::Init(&p->Str, p->ZBase, sizeof(p->ZBase), CORE_MAX_LENGTH);
				p->Str.UseMalloc = 2;
			}
			else
				SysEx::EndBenignAlloc();
		}
	}

	__device__ inline static int endsWithNL(Explain *p)
	{
		return (p && p->Str.zText && p->Str.nChar && p->Str.zText[p->Str.nChar-1]=='\n');
	}

	__device__ void sqlite3ExplainPrintf(Vdbe *pVdbe, const char *zFormat, ...)
	{
		Explain *p;
		if (vdbe && (p = vdbe->Explain) != nullptr)
		{
			va_list ap;
			if (p->Indents && endsWithNL(p))
			{
				int n = p->Indents;
				if (n > __arrayStaticLength(p->Indents)) n = __arrayStaticLength(p->Indents);
				sqlite3AppendSpace(&p->Str, p->Indents[n-1]);
			}   
			va_start(ap, format);
			sqlite3VXPrintf(&p->Str, 1, zFormat, ap);
			va_end(ap);
		}
	}

	__device__ void sqlite3ExplainNL(Vdbe *vdbe)
	{
		Explain *p;
		if (vdbe && (p = vdbe->Explain) != nullptr && !endsWithNL(p))
			p->Str.Append("\n", 1);
	}

	__device__ void sqlite3ExplainPush(Vdbe *vdbe)
	{
		Explain *p;
		if (vdbe && (p = vdbe->Explain)!=0 ){
			if (p->Str.zText && p->Indents.Length < __arrayStaticLength(p->Indents))
			{
				const char *z = p->str.zText;
				int i = p->str.nChar-1;
				int x;
				while( i>=0 && z[i]!='\n' ){ i--; }
				x = (p->str.nChar - 1) - i;
				if( p->nIndent && x<p->aIndent[p->nIndent-1] ){
					x = p->aIndent[p->nIndent-1];
				}
				p->aIndent[p->nIndent] = x;
			}
			p->nIndent++;
		}
	}

	__device__ void sqlite3ExplainPop(Vdbe *p)
	{
		if (p && p->Explain) p->Explain->Indents.Length--;
	}

	__device__ void sqlite3ExplainFinish(Vdbe *vdbe)
	{
		if (vdbe && vdbe->Explain)
		{
			SysEx::Free(vdbe->ExplainString);
			sqlite3ExplainNL(vdbe);
			vdbe->ExplainString = vdbe->Explain->Str.ToString();
			SysEx::Free(vdbe->Explain); vdbe->Explain = nullptr;
			SysEx::EndBenignAlloc();
		}
	}

	__device__ const char *sqlite3VdbeExplanation(Vdbe *vdbe)
	{
		return (vdbe && vdbe->ExplainString ? vdbe->ExplainString : nullptr);
	}

#endif
#pragma endregion
}
