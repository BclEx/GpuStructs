using System;
using System.Diagnostics;

namespace Core
{
    public partial class Vdbe
    {
        #region Trace
#if !OMIT_TRACE

        static int findNextHostParameter(string sql, int sqlIdx, ref int tokens)
        {
            int total = 0;
            tokens = 0;
            while (sqlIdx < sql.Length)
            {
                int tokenType = 0;
                int n = sqlite3GetToken(sql, sqlIdx, ref tokenType);
                Debug.Assert(n > 0 && tokenType != TK_ILLEGAL);
                if (tokenType == TK_VARIABLE)
                {
                    tokens = n;
                    break;
                }
                total += n;
                sqlIdx += n;
            }
            return total;
        }

        static string sqlite3VdbeExpandSql(Vdbe p, string rawSql)
        {
            Text.StringBuilder b = new Text.StringBuilder(); // Accumulate the _output here
            Text.StringBuilder.Init(b, 100, db.Limits[LIMIT_LENGTH]);
            Context db = p.Db; // The database connection
            b.Db = db;
            int rawSqlIdx = 0;
            int nextIndex = 1; // Index of next ? host parameter
            int idx = 0; // Index of a host parameter
            if (db.VdbeExecCnt > 1)
                while (rawSqlIdx < rawSql.Length)
                {
                    while (rawSql[rawSqlIdx++] != '\n' && rawSqlIdx < rawSql.Length) ;
                    b.Append("-- ", 3);
                    b.Append(rawSql, (int)rawSqlIdx);
                }
            else
                while (rawSqlIdx < rawSql.Length)
                {
                    int tokenLength = 0; // Length of the parameter token
                    int n = findNextHostParameter(rawSql, rawSqlIdx, ref tokenLength); // Length of a token prefix
                    Debug.Assert(n > 0);
                    b.Append(rawSql.Substring(rawSqlIdx, n), n);
                    rawSqlIdx += n;
                    Debug.Assert(rawSqlIdx < rawSql.Length || tokenLength == 0);
                    if (tokenLength == 0) break;
                    if (rawSql[rawSqlIdx] == '?')
                    {
                        if (tokenLength > 1)
                        {
                            Debug.Assert(char.IsDigit(rawSql[rawSqlIdx + 1]));
                            sqlite3GetInt32(rawSql, rawSqlIdx + 1, ref idx);
                        }
                        else
                            idx = nextIndex;
                    }
                    else
                    {
                        Debug.Assert(rawSql[rawSqlIdx] == ':' || rawSql[rawSqlIdx] == '$' || rawSql[rawSqlIdx] == '@');
                        ASSERTCOVERAGE(rawSql[rawSqlIdx] == ':');
                        ASSERTCOVERAGE(rawSql[rawSqlIdx] == '$');
                        ASSERTCOVERAGE(rawSql[rawSqlIdx] == '@');
                        idx = p.ParameterIndex(rawSql.Substring(rawSqlIdx, tokenLength), tokenLength);
                        Debug.Assert(idx > 0);
                    }
                    rawSqlIdx += tokenLength;
                    nextIndex = idx + 1;
                    Debug.Assert(idx > 0 && idx <= p.Vars.length);
                    Mem var = p.Vars[idx - 1]; // Value of a host parameter
                    if ((var.flags & MEM_Null) != 0) b.Append("NULL", 4);
                    else if ((var.flags & MEM_Int) != 0) sqlite3XPrintf(b, "%lld", var.u.I);
                    else if ((var.flags & MEM_Real) != 0) sqlite3XPrintf(b, "%!.15g", var.R);
                    else if ((var.flags & MEM_Str) != 0)
                    {
#if !OMIT_UTF16
                        byte enc = ENC(db);
                        if (enc != SQLITE_UTF8)
                        {
                            Mem utf8;
                            memset(&utf8, 0, sizeof(utf8));
                            utf8.db = db;
                            Vdbe.MemSetStr(&utf8, var.z, var.n, enc, SQLITE_STATIC);
                            Vdbe.ChangeEncoding(&utf8, SQLITE_UTF8);
                            sqlite3XPrintf(b, "'%.*q'", utf8.n, utf8.z);
                            Vdbe.MemRelease(&utf8);
                        }
                        else
#endif
                            sqlite3XPrintf(b, "'%.*q'", var.n, var.z);
                    }
                    else if ((var.Flags & MEM_Zero) != 0) sqlite3XPrintf(b, "zeroblob(%d)", var.u.Zero);
                    else
                    {
                        Debug.Assert((var.flags & MEM_Blob) != 0);
                        b.Append("x'", 2);
                        for (int i = 0; i < var.N; i++) sqlite3XPrintf(b, "%02x", var.zBLOB[i] & 0xff);
                        b.Append("'", 1);
                    }
                }
            return b.ToString();
        }

#endif
        #endregion

        #region Explain
#if ENABLE_TREE_EXPLAIN

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
				Text::StringBuilder::Init(&p->Str, p->ZBase, sizeof(p->ZBase), MAX_LENGTH);
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
        #endregion
    }
}