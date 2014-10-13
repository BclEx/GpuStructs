// func.c
using System;
using System.Diagnostics;
using System.Text;
using SystemStringBuilder = System.Text.StringBuilder;

namespace Core.Command
{
    public static class Func
    {
        public static CollSeq GetFuncCollSeq(FuncContext fctx)
        {
            return fctx.Coll;
        }

        public static void SkipAccumulatorLoad(FuncContext fctx)
        {
            fctx.SkipFlag = true;
        }

        static void MinMaxFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc > 1);
            int mask = (sqlite3_user_data(fctx) == 0 ? 0 : -1); // 0 for min() or 0xffffffff for max()
            CollSeq coll = GetFuncCollSeq(fctx);
            Debug.Assert(coll != null);
            Debug.Assert(mask == -1 || mask == 0);
            SysEx.ASSERTCOVERAGE(mask == 0);
            int best = 0;
            if (sqlite3_value_type(argv[0]) == TYPE.NULL) return;
            for (int i = 1; i < argc; i++)
            {
                if (sqlite3_value_type(argv[i]) == TYPE.NULL) return;
                if ((sqlite3MemCompare(argv[best], argv[i], coll) ^ mask) >= 0)
                {
                    SysEx.ASSERTCOVERAGE(mask == 0);
                    best = i;
                }
            }
            sqlite3_result_value(fctx, argv[best]);
        }

        static void TypeofFunc(FuncContext fctx, int notUsed1, Mem[] argv)
        {
            string z;
            switch (sqlite3_value_type(argv[0]))
            {
                case TYPE.INTEGER: z = "integer"; break;
                case TYPE.TEXT: z = "text"; break;
                case TYPE.FLOAT: z = "real"; break;
                case TYPE.BLOB: z = "blob"; break;
                default: z = "null"; break;
            }
            sqlite3_result_text(fctx, z, -1, SQLITE_STATIC);
        }


        static void LengthFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            int len;
            Debug.Assert(argc == 1);
            switch (sqlite3_value_type(argv[0]))
            {
                case TYPE.BLOB:
                case TYPE.INTEGER:
                case TYPE.FLOAT:
                    {
                        sqlite3_result_int(fctx, sqlite3_value_bytes(argv[0]));
                        break;
                    }
                case TYPE.TEXT:
                    {
                        byte[] z = sqlite3_value_blob(argv[0]);
                        if (z == null) return;
                        len = 0;
                        int zIdx = 0;
                        while (zIdx < z.Length && z[zIdx] != '\0')
                        {
                            len++;
                            SQLITE_SKIP_UTF8(z, ref zIdx);
                        }
                        sqlite3_result_int(fctx, len);
                        break;
                    }
                default:
                    {
                        sqlite3_result_null(fctx);
                        break;
                    }
            }
        }

        static void AbsFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1);
            switch (sqlite3_value_type(argv[0]))
            {
                case TYPE.INTEGER:
                    {
                        long iVal = sqlite3_value_int64(argv[0]);
                        if (iVal < 0)
                        {
                            if ((iVal << 1) == 0)
                            {
                                // IMP: R-35460-15084 If X is the integer -9223372036854775807 then abs(X) throws an integer overflow error since there is no
                                // equivalent positive 64-bit two complement value.
                                sqlite3_result_error(fctx, "integer overflow", -1);
                                return;
                            }
                            iVal = -iVal;
                        }
                        sqlite3_result_int64(fctx, iVal);
                        break;
                    }
                case TYPE.NULL:
                    {
                        // IMP: R-37434-19929 Abs(X) returns NULL if X is NULL.
                        sqlite3_result_null(fctx);
                        break;
                    }
                default:
                    {
                        // Because sqlite3_value_double() returns 0.0 if the argument is not something that can be converted into a number, we have:
                        // IMP: R-57326-31541 Abs(X) return 0.0 if X is a string or blob that cannot be converted to a numeric value. 
                        double rVal = sqlite3_value_double(argv[0]);
                        if (rVal < 0) rVal = -rVal;
                        sqlite3_result_double(fctx, rVal);
                        break;
                    }
            }
        }

        static void InstrFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            TYPE typeHaystack = sqlite3_value_type(argv[0]);
            TYPE typeNeedle = sqlite3_value_type(argv[1]);
            if (typeHaystack == TYPE.NULL || typeNeedle == TYPE.NULL) return;
            int haystackLength = sqlite3_value_bytes(argv[0]);
            int needleLength = sqlite3_value_bytes(argv[1]);
            byte[] haystack;
            byte[] needle;
            bool isText;
            if (typeHaystack == TYPE.BLOB && typeNeedle == TYPE.BLOB)
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

        static void SubstrFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 3 || argc == 2);
            if (sqlite3_value_type(argv[1]) == TYPE.NULL || (argc == 3 && sqlite3_value_type(argv[2]) == TYPE.NULL))
                return;
            TYPE p0type = sqlite3_value_type(argv[0]);
            int p1 = sqlite3_value_int(argv[1]);
            int len;
            string z = string.Empty;
            string z2;
            byte[] zAsBlob_ = null;
            if (p0type == TYPE.BLOB)
            {
                len = sqlite3_value_bytes(argv[0]);
                zAsBlob_ = argv[0].zBLOB;
                if (zAsBlob_ == null) return;
                //Debug.Assert(len == zAsBlob_.Length);
            }
            else
            {
                z = sqlite3_value_text(argv[0]);
                if (z == null) return;
                len = 0;
                if (p1 < 0)
                    for (z2 = z; z2 != string.Empty; len++)
                        SQLITE_SKIP_UTF8(ref z2);
            }
            long p2;
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
                p2 = (sqlite3_context_db_handle(fctx)).Limits[(int)LIMIT.LENGTH];
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
                p1 -= (int)p2;
                if (p1 < 0)
                {
                    p2 += p1;
                    p1 = 0;
                }
            }
            Debug.Assert(p1 >= 0 && p2 >= 0);
            if (p0type != TYPE.BLOB)
            {
                while (z != null && p1 != 0)
                {
                    SQLITE_SKIP_UTF8(ref z);
                    p1--;
                }
                for (z2 = z; z2 != null && p2 != 0; p2--)
                    SQLITE_SKIP_UTF8(ref z2);
                sqlite3_result_text(fctx, z, p1, p2 <= z.Length - p1 ? p2 : z.Length - p1, SQLITE_TRANSIENT);
            }
            else
            {
                if (p1 + p2 > len)
                {
                    p2 = len - p1;
                    if (p2 < 0) p2 = 0;
                }
                StringBuilder sb = new StringBuilder(zAsBlob_.Length);
                if (zAsBlob_.Length == 0 || p1 > zAsBlob_.Length)
                    sb.Length = 0;
                else
                    for (int i = p1; i < p1 + p2; i++)
                        sb.Append((char)zAsBlob_[i]);
                sqlite3_result_blob(fctx, sb.ToString(), (int)p2, SQLITE_TRANSIENT);
            }
        }

#if !OMIT_FLOATING_POINT
        static void RoundFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1 || argc == 2);
            int n = 0;
            if (argc == 2)
            {
                if (sqlite3_value_type(argv[1]) == TYPE.NULL) return;
                n = sqlite3_value_int(argv[1]);
                if (n > 30) n = 30;
                if (n < 0) n = 0;
            }
            if (sqlite3_value_type(argv[0]) == TYPE.NULL)
                return;
            double r = sqlite3_value_double(argv[0]);
            // If Y==0 and X will fit in a 64-bit int, handle the rounding directly, otherwise use printf.
            if (n == 0 && r >= 0 && r < LARGEST_INT64 - 1)
                r = (double)((long)(r + 0.5));
            else if (n == 0 && r < 0 && (-r) < LARGEST_INT64 - 1)
                r = -(double)((long)((-r) + 0.5));
            else
            {
                string buf = _mprintf("%.*f", n, r);
                if (buf == null)
                {
                    sqlite3_result_error_nomem(fctx);
                    return;
                }
                ConvertEx.Atof(buf, ref r, buf.Length, TEXTENCODE.UTF8);
                SysEx.Free(ref buf);
            }
            sqlite3_result_double(fctx, r);
        }
#endif

        static T[] ContextMalloc<T>(FuncContext fctx, long bytes)
            where T : struct
        {
            Context ctx = sqlite3_context_db_handle(fctx);
            Debug.Assert(bytes > 0);
            SysEx.ASSERTCOVERAGE(bytes == ctx.Limits[(int)LIMIT.LENGTH]);
            SysEx.ASSERTCOVERAGE(bytes == ctx.Limits[(int)LIMIT.LENGTH] + 1);
            T[] z;
            if (bytes > ctx.Limits[(int)LIMIT.LENGTH])
            {
                sqlite3_result_error_toobig(fctx);
                z = null;
            }
            else
            {
                z = new T[(int)bytes];
                if (z == null)
                    sqlite3_result_error_nomem(fctx);
            }
            return z;
        }

        static void UpperFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            string z2 = sqlite3_value_text(argv[0]);
            int n = sqlite3_value_bytes(argv[0]);
            // Verify that the call to _bytes() does not invalidate the _text() pointer
            Debug.Assert(z2 == sqlite3_value_text(argv[0]));
            if (z2 != null)
            {
                string z1 = (z2.Length == 0 ? string.Empty : z2.Substring(0, n).ToUpperInvariant()); //: Many
                sqlite3_result_text(fctx, z1, -1, null); //: SysEx::Free
            }
        }

        static void LowerFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            string z2 = sqlite3_value_text(argv[0]);
            int n = sqlite3_value_bytes(argv[0]);
            // Verify that the call to _bytes() does not invalidate the _text() pointer
            Debug.Assert(z2 == sqlite3_value_text(argv[0]));
            if (z2 != null)
            {
                string z1 = (z2.Length == 0 ? string.Empty : z2.Substring(0, n).ToLowerInvariant());
                sqlite3_result_text(fctx, z1, -1, null); //: SysEx::Free
            }
        }

        //: #define ifnullFunc versionFunc // Substitute function - never called

        static void RandomFunc(FuncContext fctx, int notUsed1, Mem[] notUsed2)
        {
            long r = 0;
            sqlite3_randomness(sizeof(long), ref r);
            if (r < 0)
            {
                // We need to prevent a random number of 0x8000000000000000 (or -9223372036854775808) since when you do abs() of that
                // number of you get the same value back again.  To do this in a way that is testable, mask the sign bit off of negative
                // values, resulting in a positive value.  Then take the 2s complement of that positive value.  The end result can
                // therefore be no less than -9223372036854775807.
                r = -(r ^ (((long)1) << 63));
            }
            sqlite3_result_int64(fctx, r);
        }

        static void RandomBlob(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1);
            int n = sqlite3_value_int(argv[0]);
            if (n < 1)
                n = 1;
            char[] p = ContextMalloc<char>(fctx, n);
            if (p != null)
            {
                long p_ = 0;
                for (int i = 0; i < n; i++)
                {
                    sqlite3_randomness(sizeof(char), ref p_);
                    p[i] = (char)(p_ & 0x7F);
                }
                sqlite3_result_blob(fctx, new string(p), n, null); //: SysEx::Free
            }
        }

        static void LastInsertRowid(FuncContext fctx, int notUsed1, Mem[] notUsed2
        )
        {
            Context ctx = sqlite3_context_db_handle(fctx);
            // IMP: R-51513-12026 The last_insert_rowid() SQL function is a wrapper around the sqlite3_last_insert_rowid() C/C++ interface function.
            sqlite3_result_int64(fctx, sqlite3_last_insert_rowid(ctx));
        }

        static void Changes(FuncContext fctx, int notUsed1, Mem[] notUsed2)
        {
            Context ctx = sqlite3_context_db_handle(fctx);
            sqlite3_result_int(fctx, sqlite3_changes(ctx));
        }

        static void TotalChanges(FuncContext fctx, int notUsed1, Mem[] notUsed2)
        {
            Context ctx = sqlite3_context_db_handle(fctx);
            // IMP: R-52756-41993 This function is a wrapper around the sqlite3_total_changes() C/C++ interface.
            sqlite3_result_int(fctx, sqlite3_total_changes(ctx));
        }

        struct CompareInfo
        {
            public char MatchAll;
            public char MatchOne;
            public char MatchSet;
            public bool NoCase;
        };

        static CompareInfo _globInfo = new CompareInfo { MatchAll = '*', MatchOne = '?', MatchSet = '[', NoCase = false };
        static CompareInfo _likeInfoNorm = new CompareInfo { MatchAll = '%', MatchOne = '_', MatchSet = '\0', NoCase = true }; // The correct SQL-92 behavior is for the LIKE operator to ignore case.  Thus  'a' LIKE 'A' would be true.
        static CompareInfo _likeInfoAlt = new CompareInfo { MatchAll = '%', MatchOne = '_', MatchSet = '\0', NoCase = false }; // If SQLITE_CASE_SENSITIVE_LIKE is defined, then the LIKE operator is case sensitive causing 'a' LIKE 'A' to be false

        static bool PatternCompare(string pattern, string string_, CompareInfo info, uint escape)
        {
            uint c, c2;
            int invert;
            int seen;
            int matchOne = (int)info.MatchOne;
            int matchAll = (int)info.MatchAll;
            int matchSet = (int)info.MatchSet;
            bool noCase = info.NoCase;
            bool prevEscape = false; // True if the previous character was 'escape'

            string inPattern = pattern; // Entered Pattern

            while ((c = SysEx.Utf8Read(pattern, ref pattern)) != 0)
            {
                if (c == matchAll && !prevEscape)
                {
                    while ((c = SysEx.Utf8Read(pattern, ref pattern)) == matchAll || c == matchOne)
                        if (c == matchOne && SysEx.Utf8Read(string_, ref string_) == 0)
                            return false;
                    if (c == 0)
                        return true;
                    else if (c == escape)
                    {
                        c = SysEx.Utf8Read(pattern, ref pattern);
                        if (c == 0)
                            return false;
                    }
                    else if (c == matchSet)
                    {
                        Debug.Assert(escape == 0); // This is GLOB, not LIKE
                        Debug.Assert(matchSet < 0x80); // '[' is a single-byte character
                        int stringIdx = 0;
                        while (stringIdx < string_.Length && !PatternCompare(inPattern.Substring(inPattern.Length - pattern.Length - 1), string_.Substring(stringIdx), info, escape))
                            SysEx.SKIP_UTF8(string_, ref stringIdx);
                        return (stringIdx < string_.Length);
                    }
                    while ((c2 = SysEx.Utf8Read(string_, ref string_)) != 0)
                    {
                        if (noCase)
                        {
                            c2 = (uint)char.ToLowerInvariant((char)c2);
                            c = (uint)char.ToLowerInvariant((char)c);
                            while (c2 != 0 && c2 != c)
                            {
                                c2 = SysEx.Utf8Read(string_, ref string_);
                                c2 = (uint)char.ToLowerInvariant((char)c2);
                            }
                        }
                        else
                        {
                            while (c2 != 0 && c2 != c)
                                c2 = SysEx.Utf8Read(string_, ref string_);
                        }
                        if (c2 == 0) return false;
                        if (PatternCompare(pattern, string_, info, escape)) return true;
                    }
                    return false;
                }
                else if (c == matchOne && !prevEscape)
                {
                    if (SysEx.Utf8Read(string_, ref string_) == 0)
                        return false;
                }
                else if (c == matchSet)
                {
                    uint prior_c = 0;
                    Debug.Assert(escape == 0); // This only occurs for GLOB, not LIKE
                    seen = 0;
                    invert = 0;
                    c = SysEx.Utf8Read(string_, ref string_);
                    if (c == 0) return false;
                    c2 = SysEx.Utf8Read(pattern, ref pattern);
                    if (c2 == '^')
                    {
                        invert = 1;
                        c2 = SysEx.Utf8Read(pattern, ref pattern);
                    }
                    if (c2 == ']')
                    {
                        if (c == ']') seen = 1;
                        c2 = SysEx.Utf8Read(pattern, ref pattern);
                    }
                    while (c2 != 0 && c2 != ']')
                    {
                        if (c2 == '-' && pattern[0] != ']' && pattern[0] != 0 && prior_c > 0)
                        {
                            c2 = SysEx.Utf8Read(pattern, ref pattern);
                            if (c >= prior_c && c <= c2) seen = 1;
                            prior_c = 0;
                        }
                        else
                        {
                            if (c == c2)
                                seen = 1;
                            prior_c = c2;
                        }
                        c2 = SysEx.Utf8Read(pattern, ref pattern);
                    }
                    if (c2 == 0 || (seen ^ invert) == 0)
                        return false;
                }
                else if (escape == c && !prevEscape)
                    prevEscape = true;
                else
                {
                    c2 = SysEx.Utf8Read(string_, ref string_);
                    if (noCase)
                    {
                        c = (uint)char.ToLowerInvariant((char)c);
                        c2 = (uint)char.ToLowerInvariant((char)c2);
                    }
                    if (c != c2)
                        return false;
                    prevEscape = false;
                }
            }
            return (string_.Length == 0);
        }


#if TEST
        static int _likeCount = 0;
#endif

        static void LikeFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            uint escape = 0;
            Context ctx = sqlite3_context_db_handle(fctx);

            string zB = sqlite3_value_text(argv[0]);
            string zA = sqlite3_value_text(argv[1]);

            // Limit the length of the LIKE or GLOB pattern to avoid problems of deep recursion and N*N behavior in patternCompare().
            int pats = sqlite3_value_bytes(argv[0]);
            SysEx.ASSERTCOVERAGE(pats == ctx.Limits[(int)LIMIT.LIKE_PATTERN_LENGTH]);
            SysEx.ASSERTCOVERAGE(pats == ctx.Limits[(int)LIMIT.LIKE_PATTERN_LENGTH] + 1);
            if (pats > ctx.Limits[(int)LIMIT.LIKE_PATTERN_LENGTH])
            {
                sqlite3_result_error(fctx, "LIKE or GLOB pattern too complex", -1);
                return;
            }
            Debug.Assert(zB == sqlite3_value_text(argv[0])); // Encoding did not change

            if (argc == 3)
            {
                // The escape character string must consist of a single UTF-8 character. Otherwise, return an error.
                string zEscape = sqlite3_value_text(argv[2]);
                if (zEscape == null) return;
                if (SysEx.Utf8CharLen(zEscape, -1) != 1)
                {
                    sqlite3_result_error(fctx, "ESCAPE expression must be a single character", -1);
                    return;
                }
                escape = SysEx.Utf8Read(zEscape, ref zEscape);
            }
            if (zA != null && zB != null)
            {
                CompareInfo info = (CompareInfo)sqlite3_user_data(fctx);
#if TEST
                _likeCount++;
#endif
                sqlite3_result_int(fctx, PatternCompare(zB, zA, info, escape) ? 1 : 0);
            }
        }

        static void NullifFunc(FuncContext fctx, int ntUsed1, Mem[] argv)
        {
            CollSeq coll = GetFuncCollSeq(fctx);
            if (sqlite3MemCompare(argv[0], argv[1], coll) != 0)
                sqlite3_result_value(fctx, argv[0]);
        }

        static void versionFunc(FuncContext fctx, int notUsed1, Mem[] notUsed2)
        {
            // IMP: R-48699-48617 This function is an SQL wrapper around the sqlite3_libversion() C-interface.
            sqlite3_result_text(fctx, sqlite3_libversion(), -1, SysEx.DESTRUCTOR_STATIC);
        }

        static void SourceidFunc(FuncContext fctx, int notUsed1, Mem[] notUsed2)
        {
            // IMP: R-24470-31136 This function is an SQL wrapper around the sqlite3_sourceid() C interface.
            sqlite3_result_text(fctx, sqlite3_sourceid(), -1, SysEx.DESTRUCTOR_STATIC);
        }

        static void ErrlogFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            SysEx.LOG(sqlite3_value_int(argv[0]), "%s", sqlite3_value_text(argv[1]));
        }

#if !OMIT_COMPILEOPTION_DIAGS
        static void CompileoptionusedFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1);
            // IMP: R-39564-36305 The sqlite_compileoption_used() SQL function is a wrapper around the sqlite3_compileoption_used() C/C++ function.
            string optName;
            if ((optName = sqlite3_value_text(argv[0])) != null)
                sqlite3_result_int(fctx, CompileTimeOptionUsed(optName));
        }
        static void CompileoptiongetFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1);
            // IMP: R-04922-24076 The sqlite_compileoption_get() SQL function is a wrapper around the sqlite3_compileoption_get() C/C++ function.
            int n = sqlite3_value_int(argv[0]);
            sqlite3_result_text(fctx, CompileTimeGet(n), -1, SysEx.DESTRUCTOR_STATIC);
        }
#endif

        static char[] _hexdigits = new char[] { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };

        // [EXPERIMENTAL]
        static void QuoteFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1);
            switch (sqlite3_value_type(argv[0]))
            {
                case TYPE.FLOAT:
                    {
                        SystemStringBuilder b = new SystemStringBuilder(50);
                        b.AppendFormat("%!.15g", r1);
                        double r1 = sqlite3_value_double(argv[0]);
                        double r2;
                        ConvertEx.Atof(b.ToString(), ref r2, 20, TEXTENCODE.UTF8);
                        b.Length = 0;
                        if (r1 != r2)
                            b.AppendFormat("%!.20e", r1);
                        sqlite3_result_text(fctx, b.ToString(), -1, DESTRUCTOR_TRANSIENT);
                        break;
                    }
                case TYPE.INTEGER:
                    {
                        sqlite3_result_value(fctx, argv[0]);
                        break;
                    }
                case TYPE.BLOB:
                    {
                        byte[] blob = sqlite3_value_blob(argv[0]);
                        int blobLength = sqlite3_value_bytes(argv[0]);
                        Debug.Assert(blob.Length == sqlite3_value_blob(argv[0]).Length); // No encoding change
                        SystemStringBuilder z = new StringBuilder(2 * blobLength + 4); //: ContextMalloc(fctx, (2*(int64)blobLength)+4);
                        if (z != null)
                        {
                            for (int i = 0; i < blobLength; i++)
                            {
                                z.Append(_hexdigits[(blob[i] >> 4) & 0x0F]);
                                z.Append(_hexdigits[(blob[i]) & 0x0F]);
                            }
                            z.Append[(blobLength * 2) + 2] = '\'';
                            z.Append[(blobLength * 2) + 3] = '\0';
                            z.Append[0] = 'X';
                            z.Append[1] = '\'';
                            sqlite3_result_text(fctx, z, -1, SysEx.DESTRUCTOR_TRANSIENT);
                            SysEx.Free(ref z);
                        }
                        break;
                    }
                case TYPE.TEXT:
                    {

                        string zArg = sqlite3_value_text(argv[0]);
                        if (zArg == null || zArg.Length == 0) return;
                        int i, j;
                        ulong n;
                        for (i = 0, n = 0; i < zArg.Length; i++) { if (zArg[i] == '\'') n++; }
                        StringBuilder z = new StringBuilder(i + n + 3); //: ContextMalloc(fctx, ((int64)i)+((int64)n)+3);
                        if (z != null)
                        {
                            z.Append('\'');
                            for (i = 0, j = 1; i < zArg.Length && zArg[i] != 0; i++)
                            {
                                z.Append((char)zArg[i]);
                                j++;
                                if (zArg[i] == '\'')
                                {
                                    z.Append('\'');
                                    j++;
                                }
                            }
                            z.Append('\'');
                            j++;
                            //: z[j] = 0;
                            sqlite3_result_text(fctx, z, j, null); //: SysEx.Free
                        }
                        break;
                    }
                default:
                    {
                        Debug.Assert(sqlite3_value_type(argv[0]) == SQLITE_NULL);
                        sqlite3_result_text(fctx, "NULL", 4, SQLITE_STATIC);
                        break;
                    }
            }
        }


        static void UnicodeFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            string z = sqlite3_value_text(argv[0]);
            if (z != null && z[0] != 0) sqlite3_result_int(fctx, SysEx.Utf8Read(z, ref z));
        }

        //static void CharFunc(FuncContext fctx, int argc, Mem[] argv)
        //{
        //    string z, z2;
        //    z = z2 = sqlite3_malloc(argc * 4);
        //    if (z == null)
        //    {
        //        sqlite3_result_error_nomem(fctx);
        //        return;
        //    }
        //    for (int i = 0; i < argc; i++)
        //    {
        //        long x = sqlite3_value_int64(argv[i]);
        //        if (x < 0 || x > 0x10ffff) x = 0xfffd;
        //        ulong c = (ulong)(x & 0x1fffff);
        //        if (c < 0x00080)
        //            *z2++ = (uint8)(c & 0xFF);
        //        else if (c < 0x00800)
        //        {
        //            *z2++ = 0xC0 + (uint8)((c >> 6) & 0x1F);
        //            *z2++ = 0x80 + (uint8)(c & 0x3F);
        //        }
        //        else if (c < 0x10000)
        //        {
        //            *z2++ = 0xE0 + (uint8)((c >> 12) & 0x0F);
        //            *z2++ = 0x80 + (uint8)((c >> 6) & 0x3F);
        //            *z2++ = 0x80 + (uint8)(c & 0x3F);
        //        }
        //        else
        //        {
        //            *z2++ = 0xF0 + (uint8)((c >> 18) & 0x07);
        //            *z2++ = 0x80 + (uint8)((c >> 12) & 0x3F);
        //            *z2++ = 0x80 + (uint8)((c >> 6) & 0x3F);
        //            *z2++ = 0x80 + (uint8)(c & 0x3F);
        //        }
        //    }
        //    sqlite3_result_text(fctx, (char*)z, (int)(z2 - z), null); // SysEx::Free
        //}

        static void HexFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Debug.Assert(argc == 1);
            byte[] blob = sqlite3_value_blob(argv[0]);
            int n = sqlite3_value_bytes(argv[0]);
            Debug.Assert(n == (blob == null ? 0 : blob.Length)); // No encoding change
            StringBuilder zHex = new StringBuilder(n * 2 + 1); //: ContextMalloc(fctx, ((int64)n)*2 + 1);
            if (zHex != null)
            {
                for (int i = 0; i < n; i++)
                {
                    byte c = blob[i];
                    zHex.Append(hexdigits[(c >> 4) & 0xf]);
                    zHex.Append(hexdigits[c & 0xf]);
                }
                //: *z = 0;
                sqlite3_result_text(fctx, zHex, n * 2, null); //: SysEx.Free
            }
        }

        static void ZeroblobFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            Context ctx = sqlite3_context_db_handle(fctx);
            Debug.Assert(argc == 1);
            long n = sqlite3_value_int64(argv[0]);
            SysEx.ASSERTCOVERAGE(n == ctx.Limits[(int)LIMIT.LENGTH]);
            SysEx.ASSERTCOVERAGE(n == ctx.Limits[(int)LIMIT.LENGTH] + 1);
            if (n > ctx.Limits[(int)LIMIT.LENGTH])
                sqlite3_result_error_toobig(fctx);
            else
                sqlite3_result_zeroblob(fctx, (int)n); // IMP: R-00293-64994
        }

        static void ReplaceFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            //int loopLimit;    // Last zStr[] that might match zPattern[]
            Debug.Assert(argc == 3);
            string string_ = sqlite3_value_text(argv[0]); // The input string A
            if (string_ == null) return;
            int stringLength = sqlite3_value_bytes(argv[0]); // Size of zStr
            Debug.Assert(string_ == sqlite3_value_text(argv[0]));  /* No encoding change */
            string pattern = sqlite3_value_text(argv[1]); // The pattern string B
            if (pattern == null)
            {
                Debug.Assert(sqlite3_value_type(argv[1]) == TYPE.NULL || sqlite3_context_db_handle(fctx).MallocFailed);
                return;
            }
            if (pattern == string.Empty)
            {
                Debug.Assert(sqlite3_value_type(argv[1]) != TYPE.NULL);
                sqlite3_result_value(fctx, argv[0]);
                return;
            }
            int patternLength = sqlite3_value_bytes(argv[1]); // Size of zPattern
            Debug.Assert(pattern == sqlite3_value_text(argv[1]));  /* No encoding change */
            string replacement = sqlite3_value_text(argv[2]); // The replacement string C
            if (replacement == null) return;
            int replacementLength = sqlite3_value_bytes(argv[2]); // Size of zRep
            Debug.Assert(replacement == sqlite3_value_text(argv[2]));
            long outLength = stringLength + 1; // Maximum size of zOut
            Debug.Assert(outLength < SQLITE_MAX_LENGTH);
            string out_ = null; // The output
            if (outLength <= sqlite3_context_db_handle(fctx).LimitS[(int)LIMIT.LENGTH])
                try { out_ = string_.Replace(pattern, replacement); j = out_.Length; }
                catch { j = 0; }
            if (j == 0 || j > sqlite3_context_db_handle(fctx).Limits[(int)LIMIT.LENGTH])
                sqlite3_result_error_toobig(fctx);
            else
                sqlite3_result_text(fctx, out_, j, null); //: SysEx::Free
        }

        static void TrimFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            in_;
            string charSet; // Set of characters to trim
            int inLength;
            int izIn = 0;         // C# string pointer
            int flags;            // 1: trimleft  2: trimright  3: trim
            int i;                // Loop counter
            int[] aLen = null;    // Length of each character in zCharSet
            byte[][] azChar = null; // Individual characters in zCharSet
            int nChar = 0;          // Number of characters in zCharSet
            byte[] zBytes = null;

            if (sqlite3_value_type(argv[0]) == TYPE.NULL)
                return;
            string in_ = sqlite3_value_text(argv[0]); // Input string
            if (in_ == null) return;
            int inLength = sqlite3_value_bytes(argv[0]); // Number of bytes in input
            //? Debug.Assert(in_ == sqlite3_value_text(argv[0]));
            byte[] zBlob = sqlite3_value_blob(argv[0]);

            if (argc == 1)
            {
                int[] lenOne = new int[] { 1 };
                byte[] azOne = new byte[] { (u8)' ' };//static unsigned char * const azOne[] = { (u8*)" " };
                nChar = 1;
                aLen = lenOne;
                azChar = new byte[1][];
                azChar[0] = azOne;
                charSet = null;
            }
            else if ((charSet = sqlite3_value_text(argv[1])) == null)
            {
                return;
            }
            else
            {
                if ((zBytes = sqlite3_value_blob(argv[1])) != null)
                {
                    int iz = 0;
                    for (nChar = 0; iz < zBytes.Length; nChar++)
                    {
                        SQLITE_SKIP_UTF8(zBytes, ref iz);
                    }
                    if (nChar > 0)
                    {
                        azChar = new byte[nChar][];//contextMalloc(context, ((i64)nChar)*(sizeof(char*)+1));
                        if (azChar == null)
                        {
                            return;
                        }
                        aLen = new int[nChar];

                        int iz0 = 0;
                        int iz1 = 0;
                        for (int ii = 0; ii < nChar; ii++)
                        {
                            SQLITE_SKIP_UTF8(zBytes, ref iz1);
                            aLen[ii] = iz1 - iz0;
                            azChar[ii] = new byte[aLen[ii]];
                            Buffer.BlockCopy(zBytes, iz0, azChar[ii], 0, azChar[ii].Length);
                            iz0 = iz1;
                        }
                    }
                }
            }
            if (nChar > 0)
            {
                flags = (int)sqlite3_user_data(fctx); // flags = SQLITE_PTR_TO_INT(sqlite3_user_data(context));
                if ((flags & 1) != 0)
                {
                    while (inLength > 0)
                    {
                        int len = 0;
                        for (i = 0; i < nChar; i++)
                        {
                            len = aLen[i];
                            if (len <= inLength && memcmp(zBlob, izIn, azChar[i], len) == 0)
                                break;
                        }
                        if (i >= nChar)
                            break;
                        izIn += len;
                        inLength -= len;
                    }
                }
                if ((flags & 2) != 0)
                {
                    while (inLength > 0)
                    {
                        int len = 0;
                        for (i = 0; i < nChar; i++)
                        {
                            len = aLen[i];
                            if (len <= inLength && memcmp(zBlob, izIn + inLength - len, azChar[i], len) == 0)
                                break;
                        }
                        if (i >= nChar)
                            break;
                        inLength -= len;
                    }
                }
                if (charSet != null)
                {
                    //sqlite3_free( ref azChar );
                }
            }
            StringBuilder sb = new StringBuilder(inLength);
            for (i = 0; i < inLength; i++)
                sb.Append((char)zBlob[izIn + i]);
            sqlite3_result_text(fctx, sb, inLength, SQLITE_TRANSIENT);
        }

        /* IMP: R-25361-16150 This function is omitted from SQLite by default. It
        ** is only available if the SQLITE_SOUNDEX compile-time option is used
        ** when SQLite is built.
        */
#if SQLITE_SOUNDEX
/*
** Compute the soundex encoding of a word.
**
** IMP: R-59782-00072 The soundex(X) function returns a string that is the
** soundex encoding of the string X. 
*/
static void soundexFunc(
FuncContext context,
int argc,
sqlite3_value[] argv
)
{
Debug.Assert(false); // TODO -- func_c
char zResult[8];
const u8 *zIn;
int i, j;
static const unsigned char iCode[] = {
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 2, 3, 0, 1, 2, 0, 0, 2, 2, 4, 5, 5, 0,
1, 2, 6, 2, 3, 0, 1, 0, 2, 0, 2, 0, 0, 0, 0, 0,
0, 0, 1, 2, 3, 0, 1, 2, 0, 0, 2, 2, 4, 5, 5, 0,
1, 2, 6, 2, 3, 0, 1, 0, 2, 0, 2, 0, 0, 0, 0, 0,
};
Debug.Assert( argc==1 );
zIn = (u8*)sqlite3_value_text(argv[0]);
if( zIn==0 ) zIn = (u8*)"";
for(i=0; zIn[i] && !sqlite3Isalpha(zIn[i]); i++){}
if( zIn[i] ){
u8 prevcode = iCode[zIn[i]&0x7f];
zResult[0] = sqlite3Toupper(zIn[i]);
for(j=1; j<4 && zIn[i]; i++){
int code = iCode[zIn[i]&0x7f];
if( code>0 ){
if( code!=prevcode ){
prevcode = code;
zResult[j++] = code + '0';
}
}else{
prevcode = 0;
}
}
while( j<4 ){
zResult[j++] = '0';
}
zResult[j] = 0;
sqlite3_result_text(context, zResult, 4, SQLITE_TRANSIENT);
}else{
/* IMP: R-64894-50321 The string "?000" is returned if the argument
** is NULL or contains no ASCII alphabetic characters. */
sqlite3_result_text(context, "?000", 4, SQLITE_STATIC);
}
}
#endif //* SQLITE_SOUNDEX */

#if !SQLITE_OMIT_LOAD_EXTENSION
        /*
** A function that loads a shared-library extension then returns NULL.
*/
        static void loadExt(
        FuncContext context,
        int argc,
        sqlite3_value[] argv
        )
        {
            string zFile = sqlite3_value_text(argv[0]);
            string zProc;
            sqlite3 db = (sqlite3)sqlite3_context_db_handle(context);
            string zErrMsg = "";

            if (argc == 2)
            {
                zProc = sqlite3_value_text(argv[1]);
            }
            else
            {
                zProc = "";
            }
            if (zFile != null && sqlite3_load_extension(db, zFile, zProc, ref zErrMsg) != 0)
            {
                sqlite3_result_error(context, zErrMsg, -1);
                sqlite3DbFree(db, ref zErrMsg);
            }
        }
#endif

        /*
** An instance of the following structure holds the context of a
** sum() or avg() aggregate computation.
*/
        //typedef struct SumCtx SumCtx;
        public class SumCtx
        {
            public double rSum;      /* Floating point sum */
            public i64 iSum;         /* Integer sum */
            public i64 cnt;          /* Number of elements summed */
            public int overflow;     /* True if integer overflow seen */
            public bool approx;      /* True if non-integer value was input to the sum */
            public Mem _M;
            public Mem Context
            {
                get
                {
                    return _M;
                }
                set
                {
                    _M = value;
                    if (_M == null || _M.z == null)
                        iSum = 0;
                    else
                        iSum = Convert.ToInt64(_M.z);
                }
            }
        };

        /*
        ** Routines used to compute the sum, average, and total.
        **
        ** The SUM() function follows the (broken) SQL standard which means
        ** that it returns NULL if it sums over no inputs.  TOTAL returns
        ** 0.0 in that case.  In addition, TOTAL always returns a float where
        ** SUM might return an integer if it never encounters a floating point
        ** value.  TOTAL never fails, but SUM might through an exception if
        ** it overflows an integer.
        */
        static void sumStep(
        FuncContext context,
        int argc,
        sqlite3_value[] argv
        )
        {
            SumCtx p;

            int type;
            Debug.Assert(argc == 1);
            UNUSED_PARAMETER(argc);
            Mem pMem = sqlite3_aggregate_context(context, 1);//sizeof(*p));
            if (pMem._SumCtx == null)
                pMem._SumCtx = new SumCtx();
            p = pMem._SumCtx;
            if (p.Context == null)
                p.Context = pMem;
            type = sqlite3_value_numeric_type(argv[0]);
            if (p != null && type != SQLITE_NULL)
            {
                p.cnt++;
                if (type == SQLITE_INTEGER)
                {
                    i64 v = sqlite3_value_int64(argv[0]);
                    p.rSum += v;
                    if (!(p.approx | p.overflow != 0) && 0 != sqlite3AddInt64(ref p.iSum, v))
                    {
                        p.overflow = 1;
                    }
                }
                else
                {
                    p.rSum += sqlite3_value_double(argv[0]);
                    p.approx = true;
                }
            }
        }
        static void sumFinalize(FuncContext context)
        {
            SumCtx p = null;
            Mem pMem = sqlite3_aggregate_context(context, 0);
            if (pMem != null)
                p = pMem._SumCtx;
            if (p != null && p.cnt > 0)
            {
                if (p.overflow != 0)
                {
                    sqlite3_result_error(context, "integer overflow", -1);
                }
                else if (p.approx)
                {
                    sqlite3_result_double(context, p.rSum);
                }
                else
                {
                    sqlite3_result_int64(context, p.iSum);
                }
                p.cnt = 0; // Reset for C#
            }
        }

        static void avgFinalize(FuncContext context)
        {
            SumCtx p = null;
            Mem pMem = sqlite3_aggregate_context(context, 0);
            if (pMem != null)
                p = pMem._SumCtx;
            if (p != null && p.cnt > 0)
            {
                sqlite3_result_double(context, p.rSum / (double)p.cnt);
            }
        }

        static void totalFinalize(FuncContext context)
        {
            SumCtx p = null;
            Mem pMem = sqlite3_aggregate_context(context, 0);
            if (pMem != null)
                p = pMem._SumCtx;
            /* (double)0 In case of SQLITE_OMIT_FLOATING_POINT... */
            sqlite3_result_double(context, p != null ? p.rSum : (double)0);
        }

        /*
        ** The following structure keeps track of state information for the
        ** count() aggregate function.
        */
        //typedef struct CountCtx CountCtx;
        public class CountCtx
        {
            i64 _n;
            Mem _M;
            public Mem Context
            {
                get
                {
                    return _M;
                }
                set
                {
                    _M = value;
                    if (_M == null || _M.z == null)
                        _n = 0;
                    else
                        _n = Convert.ToInt64(_M.z);
                }
            }
            public i64 n
            {
                get
                {
                    return _n;
                }
                set
                {
                    _n = value;
                    if (_M != null)
                        _M.z = _n.ToString();
                }
            }
        }

        /*
        ** Routines to implement the count() aggregate function.
        */
        static void countStep(
        FuncContext context,
        int argc,
        sqlite3_value[] argv
        )
        {
            CountCtx p = new CountCtx();
            p.Context = sqlite3_aggregate_context(context, 1);//sizeof(*p));
            if ((argc == 0 || SQLITE_NULL != sqlite3_value_type(argv[0])) && p.Context != null)
            {
                p.n++;
            }
#if !SQLITE_OMIT_DEPRECATED
            /* The sqlite3_aggregate_count() function is deprecated.  But just to make
** sure it still operates correctly, verify that its count agrees with our
** internal count when using count(*) and when the total count can be
** expressed as a 32-bit integer. */
            Debug.Assert(argc == 1 || p == null || p.n > 0x7fffffff
            || p.n == sqlite3_aggregate_count(context));
#endif
        }

        static void countFinalize(FuncContext context)
        {
            CountCtx p = new CountCtx();
            p.Context = sqlite3_aggregate_context(context, 0);
            sqlite3_result_int64(context, p != null ? p.n : 0);
        }

        /*
        ** Routines to implement min() and max() aggregate functions.
        */
        static void minmaxStep(
        FuncContext context,
        int NotUsed,
        sqlite3_value[] argv
        )
        {
            Mem pArg = (Mem)argv[0];
            Mem pBest;
            UNUSED_PARAMETER(NotUsed);

            if (sqlite3_value_type(argv[0]) == SQLITE_NULL)
                return;
            pBest = (Mem)sqlite3_aggregate_context(context, 1);//sizeof(*pBest));
            //if ( pBest == null ) return;

            if (pBest.flags != 0)
            {
                bool max;
                int cmp;
                CollSeq pColl = GetFuncCollSeq(context);
                /* This step function is used for both the min() and max() aggregates,
                ** the only difference between the two being that the sense of the
                ** comparison is inverted. For the max() aggregate, the
                ** sqlite3_context_db_handle() function returns (void *)-1. For min() it
                ** returns (void *)db, where db is the sqlite3* database pointer.
                ** Therefore the next statement sets variable 'max' to 1 for the max()
                ** aggregate, or 0 for min().
                */
                max = sqlite3_context_db_handle(context) != null && (int)sqlite3_user_data(context) != 0;
                cmp = sqlite3MemCompare(pBest, pArg, pColl);
                if ((max && cmp < 0) || (!max && cmp > 0))
                {
                    sqlite3VdbeMemCopy(pBest, pArg);
                }
            }
            else
            {
                sqlite3VdbeMemCopy(pBest, pArg);
            }
        }

        static void minMaxFinalize(FuncContext context)
        {
            sqlite3_value pRes;
            pRes = (sqlite3_value)sqlite3_aggregate_context(context, 0);
            if (pRes != null)
            {
                if (ALWAYS(pRes.flags != 0))
                {
                    sqlite3_result_value(context, pRes);
                }
                sqlite3VdbeMemRelease(pRes);
            }
        }

        /*
        ** group_concat(EXPR, ?SEPARATOR?)
        */
        static void groupConcatStep(
        FuncContext context,
        int argc,
        sqlite3_value[] argv
        )
        {
            string zVal;
            //StrAccum pAccum;
            string zSep;
            int nVal, nSep;
            Debug.Assert(argc == 1 || argc == 2);
            if (sqlite3_value_type(argv[0]) == SQLITE_NULL)
                return;
            Mem pMem = sqlite3_aggregate_context(context, 1);//sizeof(*pAccum));
            if (pMem._StrAccum == null)
                pMem._StrAccum = new StrAccum(100);
            //pAccum = pMem._StrAccum;

            //if ( pMem._StrAccum != null )
            //{
            sqlite3 db = sqlite3_context_db_handle(context);
            //int firstTerm = pMem._StrAccum.useMalloc == 0 ? 1 : 0;
            //pMem._StrAccum.useMalloc = 2;
            pMem._StrAccum.mxAlloc = db.aLimit[SQLITE_LIMIT_LENGTH];
            if (pMem._StrAccum.Context == null) // first term
                pMem._StrAccum.Context = pMem;
            else
            {
                if (argc == 2)
                {
                    zSep = sqlite3_value_text(argv[1]);
                    nSep = sqlite3_value_bytes(argv[1]);
                }
                else
                {
                    zSep = ",";
                    nSep = 1;
                }
                sqlite3StrAccumAppend(pMem._StrAccum, zSep, nSep);
            }
            zVal = sqlite3_value_text(argv[0]);
            nVal = sqlite3_value_bytes(argv[0]);
            sqlite3StrAccumAppend(pMem._StrAccum, zVal, nVal);
            //}
        }

        static void groupConcatFinalize(FuncContext context)
        {
            //StrAccum pAccum = null;
            Mem pMem = sqlite3_aggregate_context(context, 0);
            if (pMem != null)
            {
                if (pMem._StrAccum == null)
                    pMem._StrAccum = new StrAccum(100);
                StrAccum pAccum = pMem._StrAccum;
                //}
                //if ( pAccum != null )
                //{
                if (pAccum.tooBig)
                {
                    sqlite3_result_error_toobig(context);
                }
                //else if ( pAccum.mallocFailed != 0 )
                //{
                //  sqlite3_result_error_nomem( context );
                //}
                else
                {
                    sqlite3_result_text(context, sqlite3StrAccumFinish(pAccum), -1,
                    null); //sqlite3_free );
                }
            }
        }

        /*
        ** This routine does per-connection function registration.  Most
        ** of the built-in functions above are part of the global function set.
        ** This routine only deals with those that are not global.
        */
        public struct sFuncs
        {
            public string zName;
            public sbyte nArg;
            public u8 argType;           /* 1: 0, 2: 1, 3: 2,...  N:  N-1. */
            public u8 eTextRep;          /* 1: UTF-16.  0: UTF-8 */
            public u8 needCollSeq;
            public dxFunc xFunc; //(FuncContext*,int,sqlite3_value **);

            // Constructor
            public sFuncs(string zName, sbyte nArg, u8 argType, u8 eTextRep, u8 needCollSeq, dxFunc xFunc)
            {
                this.zName = zName;
                this.nArg = nArg;
                this.argType = argType;
                this.eTextRep = eTextRep;
                this.needCollSeq = needCollSeq;
                this.xFunc = xFunc;
            }
        };

        public struct sAggs
        {
            public string zName;
            public sbyte nArg;
            public u8 argType;
            public u8 needCollSeq;
            public dxStep xStep; //(FuncContext*,int,sqlite3_value**);
            public dxFinal xFinalize; //(FuncContext*);
            // Constructor
            public sAggs(string zName, sbyte nArg, u8 argType, u8 needCollSeq, dxStep xStep, dxFinal xFinalize)
            {
                this.zName = zName;
                this.nArg = nArg;
                this.argType = argType;
                this.needCollSeq = needCollSeq;
                this.xStep = xStep;
                this.xFinalize = xFinalize;
            }
        }
        static void sqlite3RegisterBuiltinFunctions(sqlite3 db)
        {
            int rc = sqlite3_overload_function(db, "MATCH", 2);
            Debug.Assert(rc == SQLITE_NOMEM || rc == SQLITE_OK);
            if (rc == SQLITE_NOMEM)
            {
                ////        db.mallocFailed = 1;
            }
        }

        /*
        ** Set the LIKEOPT flag on the 2-argument function with the given name.
        */
        static void setLikeOptFlag(sqlite3 db, string zName, int flagVal)
        {
            FuncDef pDef;
            pDef = sqlite3FindFunction(db, zName, sqlite3Strlen30(zName),
            2, SQLITE_UTF8, 0);
            if (ALWAYS(pDef != null))
            {
                pDef.flags = (byte)flagVal;
            }
        }

        /*
        ** Register the built-in LIKE and GLOB functions.  The caseSensitive
        ** parameter determines whether or not the LIKE operator is case
        ** sensitive.  GLOB is always case sensitive.
        */
        static void sqlite3RegisterLikeFunctions(sqlite3 db, int caseSensitive)
        {
            CompareInfo pInfo;
            if (caseSensitive != 0)
            {
                pInfo = _likeInfoAlt;
            }
            else
            {
                pInfo = _likeInfoNorm;
            }
            sqlite3CreateFunc(db, "like", 2, SQLITE_UTF8, pInfo, (dxFunc)likeFunc, null, null, null);
            sqlite3CreateFunc(db, "like", 3, SQLITE_UTF8, pInfo, (dxFunc)likeFunc, null, null, null);
            sqlite3CreateFunc(db, "glob", 2, SQLITE_UTF8,
            _globInfo, (dxFunc)likeFunc, null, null, null);
            setLikeOptFlag(db, "glob", SQLITE_FUNC_LIKE | SQLITE_FUNC_CASE);
            setLikeOptFlag(db, "like",
            caseSensitive != 0 ? (SQLITE_FUNC_LIKE | SQLITE_FUNC_CASE) : SQLITE_FUNC_LIKE);
        }

        /*
        ** pExpr points to an expression which implements a function.  If
        ** it is appropriate to apply the LIKE optimization to that function
        ** then set aWc[0] through aWc[2] to the wildcard characters and
        ** return TRUE.  If the function is not a LIKE-style function then
        ** return FALSE.
        */
        static bool sqlite3IsLikeFunction(sqlite3 db, Expr pExpr, ref bool pIsNocase, char[] aWc)
        {
            FuncDef pDef;
            if (pExpr.op != TK_FUNCTION
            || null == pExpr.x.pList
            || pExpr.x.pList.nExpr != 2
            )
            {
                return false;
            }
            Debug.Assert(!ExprHasProperty(pExpr, EP_xIsSelect));
            pDef = sqlite3FindFunction(db, pExpr.u.zToken, sqlite3Strlen30(pExpr.u.zToken),
            2, SQLITE_UTF8, 0);
            if (NEVER(pDef == null) || (pDef.flags & SQLITE_FUNC_LIKE) == 0)
            {
                return false;
            }

            /* The memcpy() statement assumes that the wildcard characters are
            ** the first three statements in the compareInfo structure.  The
            ** Debug.Asserts() that follow verify that assumption
            */
            //memcpy( aWc, pDef.pUserData, 3 );
            aWc[0] = ((CompareInfo)pDef.pUserData).MatchAll;
            aWc[1] = ((CompareInfo)pDef.pUserData).MatchOne;
            aWc[2] = ((CompareInfo)pDef.pUserData).MatchSet;
            // Debug.Assert((char*)&likeInfoAlt == (char*)&likeInfoAlt.matchAll);
            // Debug.Assert(&((char*)&likeInfoAlt)[1] == (char*)&likeInfoAlt.matchOne);
            // Debug.Assert(&((char*)&likeInfoAlt)[2] == (char*)&likeInfoAlt.matchSet);
            pIsNocase = (pDef.flags & SQLITE_FUNC_CASE) == 0;
            return true;
        }

        /*
        ** All all of the FuncDef structures in the aBuiltinFunc[] array above
        ** to the global function hash table.  This occurs at start-time (as
        ** a consequence of calling sqlite3_initialize()).
        **
        ** After this routine runs
        */
        static void sqlite3RegisterGlobalFunctions()
        {
            /*
            ** The following array holds FuncDef structures for all of the functions
            ** defined in this file.
            **
            ** The array cannot be constant since changes are made to the
            ** FuncDef.pHash elements at start-time.  The elements of this array
            ** are read-only after initialization is complete.
            */
            FuncDef[] aBuiltinFunc =  {
FUNCTION("ltrim",              1, 1, 0, trimFunc         ),
FUNCTION("ltrim",              2, 1, 0, trimFunc         ),
FUNCTION("rtrim",              1, 2, 0, trimFunc         ),
FUNCTION("rtrim",              2, 2, 0, trimFunc         ),
FUNCTION("trim",               1, 3, 0, trimFunc         ),
FUNCTION("trim",               2, 3, 0, trimFunc         ),
FUNCTION("min",               -1, 0, 1, minmaxFunc       ),
FUNCTION("min",                0, 0, 1, null                ),
AGGREGATE("min",               1, 0, 1, minmaxStep,      minMaxFinalize ),
FUNCTION("max",               -1, 1, 1, minmaxFunc       ),
FUNCTION("max",                0, 1, 1, null                ),
AGGREGATE("max",               1, 1, 1, minmaxStep,      minMaxFinalize ),
FUNCTION("typeof",             1, 0, 0, typeofFunc       ),
FUNCTION("length",             1, 0, 0, lengthFunc       ),
FUNCTION("substr",             2, 0, 0, substrFunc       ),
FUNCTION("substr",             3, 0, 0, substrFunc       ),
FUNCTION("abs",                1, 0, 0, absFunc          ),
#if !SQLITE_OMIT_FLOATING_POINT
FUNCTION("round",              1, 0, 0, roundFunc        ),
FUNCTION("round",              2, 0, 0, roundFunc        ),
#endif
FUNCTION("upper",              1, 0, 0, upperFunc        ),
FUNCTION("lower",              1, 0, 0, lowerFunc        ),
FUNCTION("coalesce",           1, 0, 0, null             ),
FUNCTION("coalesce",           0, 0, 0, null             ),
/*  FUNCTION(coalesce,          -1, 0, 0, ifnullFunc       ), */
// use versionFunc here just for a dummy placeholder
new FuncDef(-1,SQLITE_UTF8,SQLITE_FUNC_COALESCE,null,null,versionFunc,null,null,"coalesce",null,null), 
FUNCTION("hex",                1, 0, 0, hexFunc          ),
/*  FUNCTION(ifnull,             2, 0, 0, ifnullFunc       ), */
// use versionFunc here just for a dummy placeholder
new FuncDef(2,SQLITE_UTF8,SQLITE_FUNC_COALESCE,null,null,versionFunc,null,null,"ifnull",null,null),
FUNCTION("random",             0, 0, 0, randomFunc       ),
FUNCTION("randomblob",         1, 0, 0, randomBlob       ),
FUNCTION("nullif",             2, 0, 1, nullifFunc       ),
FUNCTION("sqlite_version",     0, 0, 0, versionFunc      ),
FUNCTION("sqlite_source_id",   0, 0, 0, sourceidFunc     ),
FUNCTION("sqlite_log",         2, 0, 0, errlogFunc       ),
#if !SQLITE_OMIT_COMPILEOPTION_DIAGS
FUNCTION("sqlite_compileoption_used",1, 0, 0, compileoptionusedFunc  ),
FUNCTION("sqlite_compileoption_get", 1, 0, 0, compileoptiongetFunc  ),
#endif //* SQLITE_OMIT_COMPILEOPTION_DIAGS */
FUNCTION("quote",              1, 0, 0, quoteFunc        ),
FUNCTION("last_insert_rowid",  0, 0, 0, last_insert_rowid),
FUNCTION("changes",            0, 0, 0, changes          ),
FUNCTION("total_changes",      0, 0, 0, total_changes    ),
FUNCTION("replace",            3, 0, 0, replaceFunc      ),
FUNCTION("zeroblob",           1, 0, 0, zeroblobFunc     ),
#if SQLITE_SOUNDEX
FUNCTION("soundex",            1, 0, 0, soundexFunc      ),
#endif
#if !SQLITE_OMIT_LOAD_EXTENSION
FUNCTION("load_extension",     1, 0, 0, loadExt          ),
FUNCTION("load_extension",     2, 0, 0, loadExt          ),
#endif
AGGREGATE("sum",               1, 0, 0, sumStep,         sumFinalize    ),
AGGREGATE("total",             1, 0, 0, sumStep,         totalFinalize    ),
AGGREGATE("avg",               1, 0, 0, sumStep,         avgFinalize    ),
/*AGGREGATE("count",             0, 0, 0, countStep,       countFinalize  ), */
/* AGGREGATE(count,             0, 0, 0, countStep,       countFinalize  ), */
new FuncDef( 0,SQLITE_UTF8,SQLITE_FUNC_COUNT,null,null,null,countStep,countFinalize,"count",null,null),
AGGREGATE("count",             1, 0, 0, countStep,       countFinalize  ),
AGGREGATE("group_concat",      1, 0, 0, groupConcatStep, groupConcatFinalize),
AGGREGATE("group_concat",      2, 0, 0, groupConcatStep, groupConcatFinalize),

LIKEFUNC("glob", 2, _globInfo, SQLITE_FUNC_LIKE|SQLITE_FUNC_CASE),
#if SQLITE_CASE_SENSITIVE_LIKE
LIKEFUNC("like", 2, likeInfoAlt, SQLITE_FUNC_LIKE|SQLITE_FUNC_CASE),
LIKEFUNC("like", 3, likeInfoAlt, SQLITE_FUNC_LIKE|SQLITE_FUNC_CASE),
#else
LIKEFUNC("like", 2, _likeInfoNorm, SQLITE_FUNC_LIKE),
LIKEFUNC("like", 3, _likeInfoNorm, SQLITE_FUNC_LIKE),
#endif
FUNCTION("regexp",                2, 0, 0, regexpFunc          ),};
            int i;
#if SQLITE_OMIT_WSD
FuncDefHash pHash = GLOBAL( FuncDefHash, sqlite3GlobalFunctions );
FuncDef[] aFunc = (FuncDef[])GLOBAL( FuncDef, aBuiltinFunc );
#else
            FuncDefHash pHash = sqlite3GlobalFunctions;
            FuncDef[] aFunc = aBuiltinFunc;
#endif
            for (i = 0; i < ArraySize(aBuiltinFunc); i++)
            {
                sqlite3FuncDefInsert(pHash, aFunc[i]);
            }
            sqlite3RegisterDateTimeFunctions();
#if !SQLITE_OMIT_ALTERTABLE
            sqlite3AlterFunctions();
#endif
        }
    }
}
