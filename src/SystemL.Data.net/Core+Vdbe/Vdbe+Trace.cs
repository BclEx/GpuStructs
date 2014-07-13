using System;
using System.Diagnostics;

namespace Core
{
    public partial class Vdbe
    {
        #region Trace
#if !OMIT_TRACE

        static int FindNextHostParameter(string sql, int sqlIdx, ref int tokens)
        {
            int total = 0;
            tokens = 0;
            while (sqlIdx < sql.Length)
            {
                TK tokenType = 0;
                int n = Parse.GetToken(sql, sqlIdx, ref tokenType);
                Debug.Assert(n > 0 && tokenType != TK.ILLEGAL);
                if (tokenType == TK.VARIABLE)
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
            Context ctx = p.Ctx; // The database connection
            Text.StringBuilder b = new Text.StringBuilder(); // Accumulate the _output here
            Text.StringBuilder.Init(b, 100, ctx.Limits[(int)LIMIT.LENGTH]);
            b.Ctx = ctx;
            int rawSqlIdx = 0;
            int nextIndex = 1; // Index of next ? host parameter
            int idx = 0; // Index of a host parameter
            if (ctx.VdbeExecCnt > 1)
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
                    int n = FindNextHostParameter(rawSql, rawSqlIdx, ref tokenLength); // Length of a token prefix
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
                            ConvertEx.Atoi(rawSql, rawSqlIdx + 1, ref idx);
                        }
                        else
                            idx = nextIndex;
                    }
                    else
                    {
                        Debug.Assert(rawSql[rawSqlIdx] == ':' || rawSql[rawSqlIdx] == '$' || rawSql[rawSqlIdx] == '@');
                        SysEx.ASSERTCOVERAGE(rawSql[rawSqlIdx] == ':');
                        SysEx.ASSERTCOVERAGE(rawSql[rawSqlIdx] == '$');
                        SysEx.ASSERTCOVERAGE(rawSql[rawSqlIdx] == '@');
                        idx = p.ParameterIndex(rawSql.Substring(rawSqlIdx, tokenLength), tokenLength);
                        Debug.Assert(idx > 0);
                    }
                    rawSqlIdx += tokenLength;
                    nextIndex = idx + 1;
                    Debug.Assert(idx > 0 && idx <= p.Vars.length);
                    Mem var = p.Vars[idx - 1]; // Value of a host parameter
                    if ((var.Flags & MEM.Null) != 0) b.Append("NULL", 4);
                    else if ((var.Flags & MEM.Int) != 0) sqlite3XPrintf(b, "%lld", var.u.I);
                    else if ((var.Flags & MEM.Real) != 0) sqlite3XPrintf(b, "%!.15g", var.R);
                    else if ((var.Flags & MEM.Str) != 0)
                    {
#if !OMIT_UTF16
                        TEXTENCODE encode = Context.CTXENCODE(ctx);
                        if (encode != TEXTENCODE.UTF8)
                        {
                            Mem utf8;
                            memset(&utf8, 0, sizeof(utf8));
                            utf8.Ctx = ctx;
                            Vdbe.MemSetStr(&utf8, var.Z, var.N, encode, SysEx.DESTRUCTOR_STATIC);
                            Vdbe.ChangeEncoding(&utf8, TEXTENCODE.UTF8);
                            sqlite3XPrintf(b, "'%.*q'", utf8.N, utf8.Z);
                            Vdbe.MemRelease(&utf8);
                        }
                        else
#endif
                            sqlite3XPrintf(b, "'%.*q'", var.N, var.Z);
                    }
                    else if ((var.Flags & MEM.Zero) != 0) sqlite3XPrintf(b, "zeroblob(%d)", var.u.Zero);
                    else
                    {
                        Debug.Assert((var.Flags & MEM.Blob) != 0);
                        b.Append("x'", 2);
                        for (int i = 0; i < var.N; i++) sqlite3XPrintf(b, "%02x", var.u.Zero[i] & 0xff);
                        b.Append("'", 1);
                    }
                }
            return b.ToString();
        }

#endif
        #endregion

        #region Explain
#if false && ENABLE_TREE_EXPLAIN

	public static void ExplainBegin(Vdbe vdbe)
	{
		if (vdbe != null)
		{
			SysEx.BeginBenignAlloc();
			Explain p = new Explain();
			if (p)
			{
				p.Vdbe = vdbe;
				vdbe.Explain = p;
				Text.StringBuilder.Init(p.Str, p.ZBase, sizeof(p.ZBase), MAX_LENGTH);
				p.Str.UseMalloc = 2;
			}
			else
				SysEx.EndBenignAlloc();
		}
	}

	static int EndsWithNL(Explain p)
	{
		return (p != null && p.Str.zText && p.Str.nChar && p.Str.zText[p.Str.nChar-1]=='\n');
	}

	public static void sqlite3ExplainPrintf(Vdbe vdbe, string format, params object[] args)
	{
		Explain p;
		if (vdbe != null && (p = vdbe.Explain) != null)
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
