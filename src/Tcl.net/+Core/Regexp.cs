using System;
using System.Text;
namespace Core
{
    public class Regexp
    {
        //[STAThread]
        //public static void  Main(string[] args)
        //{
        //  if ((args.Length == 2) && (args[0].Equals("compile")))
        //  {
        //    System.Diagnostics.Debug.WriteLine(new Regexp(args[1]));
        //  }
        //  else if ((args.Length == 3) && (args[0].Equals("match")))
        //  {
        //    Regexp r = new Regexp(args[1]);
        //    string[] substrs = new string[r.subspecs()];
        //    bool match = r.match(args[2], substrs);
        //    System.Diagnostics.Debug.WriteLine("match:\t" + match);
        //    for (int i = 0; i < substrs.Length; i++)
        //    {
        //      System.Diagnostics.Debug.WriteLine((i + 1) + ":\t" + substrs[i]);
        //    }
        //  }
        //  else if ((args.Length == 4) && (args[0].Equals("sub")))
        //  {
        //    Regexp r = new Regexp(args[1]);
        //    System.Diagnostics.Debug.WriteLine(r.subAll(args[2], args[3]));
        //  }
        //  else
        //  {
        //    System.Diagnostics.Debug.WriteLine("usage:");
        //    System.Diagnostics.Debug.WriteLine("\tRegexp match <pattern> <string>");
        //    System.Diagnostics.Debug.WriteLine("\tRegexp sub <pattern> <string> <subspec>");
        //    System.Diagnostics.Debug.WriteLine("\tRegexp compile <pattern>");
        //  }
        //}

        // Structure for regexp "program".  This is essentially a linear encoding of a nondeterministic finite-state machine (aka syntax charts or
        // "railroad normal form" in parsing technology).  Each node is an opcode plus a "next" pointer, possibly plus an operand.  "Next" pointers of
        // all nodes except BRANCH implement concatenation; a "next" pointer with a BRANCH on both ends of it is connecting two alternatives.  (Here we
        // have one of the subtle syntax dependencies:  an individual BRANCH (as opposed to a collection of them) is never concatenated with anything
        // because of operator precedence.)  The operand of some types of node is a literal string; for others, it is a node leading into a sub-FSM.  In
        // particular, the operand of a BRANCH node is the first node of the branch. (NB this is *not* a tree structure:  the tail of the branch connects
        // to the thing following the set of BRANCHes.)  The opcodes are:
        internal const int NSUBEXP = 100;

        internal const char END = (char)(0); // no	End of program.
        internal const char BOL = (char)(1); // no	Match "" at beginning of line.
        internal const char EOL = (char)(2); // no	Match "" at end of line.
        internal const char ANY = (char)(3); // no	Match any one character.
        internal const char ANYOF = (char)(4); // str	Match any character in this string.
        internal const char ANYBUT = (char)(5); // str	Match any character not in this string.
        internal const char BRANCH = (char)(6); // node	Match this alternative, or the next...
        internal const char BACK = (char)(7); // no	Match "", "next" ptr points backward.
        internal const char EXACTLY = (char)(8); // str	Match this string.
        internal const char NOTHING = (char)(9); // no	Match empty string.
        internal const char STAR = (char)(10); // node	Match this (simple) thing 0 or more times.
        internal const char PLUS = (char)(11); // node	Match this (simple) thing 1 or more times.
        internal const char OPEN = (char)(20); // no	Mark this point in input as start of #n.
        // OPEN+1 is number 1, etc.
        internal static readonly char CLOSE = (char)(OPEN + NSUBEXP); // no	Analogous to OPEN.
        internal static readonly string[] _opnames = new string[] { "END", "BOL", "EOL", "ANY", "ANYOF", "ANYBUT", "BRANCH", "BACK", "EXACTLY", "NOTHING", "STAR", "PLUS" };

        // 
        // A node is one char of opcode followed by one char of "next" pointer. The value is a positive offset from the opcode of the node containing
        // it.  An operand, if any, simply follows the node.  (Note that much of the code generation knows about this implicit relationship.)
        // 
        // Opcode notes:
        // 
        // BRANCH	The set of branches constituting a single choice are hooked together with their "next" pointers, since precedence prevents
        // 		anything being concatenated to any individual branch.  The "next" pointer of the last BRANCH in a choice points to the
        // 		thing following the whole choice.  This is also where the final "next" pointer of each individual branch points; each
        // 		branch starts with the operand node of a BRANCH node.
        // 
        // ANYOF, ANYBUT, EXACTLY
        // 		The format of a string operand is one char of length followed by the characters making up the string.
        // 
        // BACK	Normal "next" pointers all implicitly point forward; BACK
        // 		exists to make loop structures possible.
        // 
        // STAR, PLUS
        // 		'?', and complex '*' and '+' are implemented as circular BRANCH structures using BACK.  Simple cases (one character
        // 		per match) are implemented with STAR and PLUS for speed and to minimize recursive plunges.
        // 
        // OPENn, CLOSEn
        // 		are numbered at compile time.
        // 

        internal char[] _program; // The bytecodes making up the regexp program.
        internal bool _ignoreCase; // Whether the regexp matching should be case insensitive.
        internal int _npar; // The number of parenthesized subexpressions in the regexp pattern, plus 1 for the match of the whole pattern itself.
        internal bool _anchored; // true if the pattern must match the beginning of the string, so we don't have to waste time matching against all possible starting locations in the string.
        internal int _startChar;
        internal string _must;

        /// <summary>
        /// Compiles a new Regexp object from the given regular expression pattern.
        /// It takes a certain amount of time to parse and validate a regular expression pattern before it can be used to perform matches
        /// or substitutions.  If the caller caches the new Regexp object, that parsing time will be saved because the same Regexp can be used with
        /// respect to many different strings.
        /// </summary>
        /// <param name="pat">
        /// The string holding the regular expression pattern.
        /// @throws	IllegalArgumentException if the pattern is malformed.
        /// The detail message for the exception will be set to a
        /// string indicating how the pattern was malformed.
        /// </param>
        public Regexp(string pat)
        {
            compile(pat);
        }

        /// <summary>
        /// Compiles a new Regexp object from the given regular expression pattern.
        /// </summary>
        /// <param name="pat">
        /// The string holding the regular expression pattern.
        /// </param>
        /// <param name="ignoreCase">
        /// If <code>true</code> then this regular expression will do case-insensitive matching.  If <code>false</code>, then
        /// the matches are case-sensitive.  Regular expressions generated by <code>Regexp(String)</code> are case-sensitive.
        /// 
        /// @throws	IllegalArgumentException if the pattern is malformed. The detail message for the exception will be set to a
        /// string indicating how the pattern was malformed.
        /// </param>
        public Regexp(string pat, bool ignoreCase)
        {
            _ignoreCase = ignoreCase;
            if (ignoreCase)
                pat = pat.ToLower();
            compile(pat);
        }

        /// <summary>
        /// Returns the number of parenthesized subexpressions in this regular expression, plus one more for this expression itself.
        /// </summary>
        /// <returns>
        /// The number.
        /// </returns>
        public int subspecs()
        {
            return _npar;
        }

        /// <summary>
        /// Matches the given string against this regular expression.
        /// </summary>
        /// <param name="str">
        /// The string to match.
        /// </param>
        /// <returns>
        /// The substring of <code>str</code> that matched the entire regular expression, or <code>null</code> if the string did not
        /// match this regular expression.
        /// </returns>
        public string match(string str)
        {
            Match m = exec(str, 0, 0);
            if (m == null)
                return null;
            return str.Substring(m._indices[0], (m._indices[1]) - (m._indices[0]));
        }

        /// <summary>
        /// Matches the given string against this regular expression, and computes the set of substrings that matched the parenthesized subexpressions.
        ///
        /// <code>substrs[0]</code> is set to the range of <code>str</code> that matched the entire regular expression.
        /// 
        /// <code>substrs[1]</code> is set to the range of <code>str</code> that matched the first (leftmost) parenthesized subexpression.
        /// <code>substrs[n]</code> is set to the range that matched the <code>n</code><i>th</i> subexpression, and so on.
        /// 
        /// If subexpression <code>n</code> did not match, then <code>substrs[n]</code> is set to <code>null</code>.  Not to
        /// be confused with "", which is a valid value for a subexpression that matched 0 characters.
        /// 
        /// The length that the caller should use when allocating the <code>substr</code> array is the return value of
        /// <code>Regexp.subspecs</code>.  The array can be shorter (in which case not all the information will
        /// be returned), or longer (in which case the remainder of the elements are initialized to <code>null</code>), or
        /// <code>null</code> (to ignore the subexpressions).
        /// </summary>
        /// <param name="str">
        /// The string to match.
        /// </param>
        /// <param name="substrs">
        /// An array of strings allocated by the caller, and filled in with information about the portions of <code>str</code> that
        /// matched the regular expression.  May be <code>null</code>.
        /// </param>
        /// <returns>
        /// <code>true</code> if <code>str</code> that matched this regular expression, <code>false</code> otherwise.
        /// If <code>false</code> is returned, then the contents of <code>substrs</code> are unchanged.
        /// </returns>
        /// <seealso cref="#subspecs" />
        public bool match(string str, string[] substrs)
        {
            Match m = exec(str, 0, 0);
            if (m == null)
                return false;
            if (substrs != null)
            {
                int max = Math.Min(substrs.Length, _npar);
                int i;
                int j = 0;
                for (i = 0; i < max; i++)
                {
                    int start = m._indices[j++];
                    int end = m._indices[j++];
                    substrs[i] = (start < 0 ? null : str.Substring(start, (end) - (start)));
                }
                for (; i < substrs.Length; i++)
                    substrs[i] = null;
            }
            return true;
        }

        /// <summary>
        /// Matches the given string against this regular expression, and computes the set of substrings that matched the parenthesized subexpressions.
        /// 
        /// For the indices specified below, the range extends from the character at the starting index up to, but not including, the character at the
        /// ending index.
        /// 
        /// <code>indices[0]</code> and <code>indices[1]</code> are set to starting and ending indices of the range of <code>str</code>
        /// that matched the entire regular expression.
        /// 
        /// <code>indices[2]</code> and <code>indices[3]</code> are set to the starting and ending indices of the range of <code>str</code> that
        /// matched the first (leftmost) parenthesized subexpression. <code>indices[n * 2]</code> and <code>indices[n * 2 + 1]</code>
        /// are set to the range that matched the <code>n</code><i>th</i> subexpression, and so on.
        /// 
        /// If subexpression <code>n</code> did not match, then <code>indices[n * 2]</code> and <code>indices[n * 2 + 1]</code>
        /// are both set to <code>-1</code>.
        /// 
        /// The length that the caller should use when allocating the <code>indices</code> array is twice the return value of
        /// <code>Regexp.subspecs</code>.  The array can be shorter (in which case not all the information will
        /// be returned), or longer (in which case the remainder of the elements are initialized to <code>-1</code>), or
        /// <code>null</code> (to ignore the subexpressions).
        /// </summary>
        /// <param name="str">
        /// The string to match.
        /// </param>
        /// <param name="indices">
        /// An array of integers allocated by the caller, and filled in with information about the portions of <code>str</code> that
        /// matched all the parts of the regular expression. May be <code>null</code>.
        /// </param>
        /// <returns>
        /// <code>true</code> if the string matched the regular expression, <code>false</code> otherwise.  If <code>false</code> is
        /// returned, then the contents of <code>indices</code> are unchanged.
        /// </returns>
        /// <seealso cref="#subspecs" />
        public bool match(string str, int[] indices)
        {
            Match m = exec(str, 0, 0);
            if (m == null)
                return false;
            if (indices != null)
            {
                int max = Math.Min(indices.Length, _npar * 2);
                Array.Copy((Array)m._indices, 0, (Array)indices, 0, max);
                for (int i = max; i < indices.Length; i++)
                    indices[i] = -1;
            }
            return true;
        }

        /// <summary>
        /// Matches a string against a regular expression and replaces the first match with the string generated from the substitution parameter.
        /// </summary>
        /// <param name="str">
        /// The string to match against this regular expression.
        /// </param>
        /// <param name="subspec">
        /// The substitution parameter, described in <a href=#regsub>REGULAR EXPRESSION SUBSTITUTION</a>.
        /// </param>
        /// <returns>
        /// The string formed by replacing the first match in <code>str</code> with the string generated from
        /// <code>subspec</code>.  If no matches were found, then the return value is <code>null</code>.
        /// </returns>
        public string sub(string str, string subspec)
        {
            Regsub rs = new Regsub(this, str);
            if (rs.nextMatch())
            {
                StringBuilder sb = new StringBuilder(rs.skipped());
                applySubspec(rs, subspec, sb);
                sb.Append(rs.rest());
                return sb.ToString();
            }
            return null;
        }

        /// <summary>
        /// Matches a string against a regular expression and replaces all matches with the string generated from the substitution parameter.
        /// After each substutition is done, the portions of the string already examined, including the newly substituted region, are <b>not</b> checked
        /// again for new matches -- only the rest of the string is examined.
        /// </summary>
        /// <param name="str">
        /// The string to match against this regular expression.
        /// </param>
        /// <param name="subspec">
        /// The substitution parameter, described in <a href=#regsub> REGULAR EXPRESSION SUBSTITUTION</a>.
        /// </param>
        /// <returns>
        /// The string formed by replacing all the matches in <code>str</code> with the strings generated from
        /// <code>subspec</code>.  If no matches were found, then the return value is a copy of <code>str</code>.
        /// </returns>
        public string subAll(string str, string subspec)
        {
            return sub(str, new SubspecFilter(subspec, true));
        }

        /// <summary>
        /// Utility method to give access to the standard substitution algorithm used by <code>sub</code> and <code>subAll</code>.  Appends to the
        /// string buffer the string generated by applying the substitution parameter to the matched region.
        /// </summary>
        /// <param name="rs">
        /// Information about the matched region.
        /// </param>
        /// <param name="subspec">
        /// The substitution parameter.
        /// </param>
        /// <param name="sb">
        /// StringBuffer to which the generated string is appended.
        /// </param>
        public static void applySubspec(Regsub rs, string subspec, StringBuilder sb)
        {
            try
            {
                int len = subspec.Length;
                for (int i = 0; i < len; i++)
                {
                    char ch = subspec[i];
                    switch (ch)
                    {
                        case '&':
                            sb.Append(rs.matched());
                            break;
                        case '\\':
                            i++;
                            ch = subspec[i];
                            if ((ch >= '0') && (ch <= '9'))
                            {
                                string match = rs.submatch(ch - '0');
                                if ((object)match != null)
                                    sb.Append(match);
                                break;
                            }
                            // fall through.
                            goto default;
                        default:
                            sb.Append(ch);
                            break;

                    }
                }
            }
            catch (IndexOutOfRangeException) { } // Ignore malformed substitution pattern. Return string matched so far.
        }

        public string sub(string str, IFilter rf)
        {
            Regsub rs = new Regsub(this, str);
            if (!rs.nextMatch())
                return str;

            StringBuilder sb = new StringBuilder();
            do
            {
                sb.Append(rs.skipped());
                if (!rf.filter(rs, sb))
                    break;
            }
            while (rs.nextMatch());
            sb.Append(rs.rest());
            return sb.ToString();
        }

        /// <summary>
        /// This interface is used by the <code>Regexp</code> class to generate the replacement string for each pattern match found in the source string.
        /// </summary>
        public interface IFilter
        {
            /// <summary>
            /// Given the current state of the match, generate the replacement string.  This method will be called for each match found in
            /// the source string, unless this filter decides not to handle any more matches.
            /// 
            /// The implementation can use whatever rules it chooses to generate the replacement string.  For example, here is an
            /// example of a filter that replaces the first <b>5</b> occurrences of "%XX" in a string with the ASCII character
            /// represented by the hex digits "XX":
            /// <pre>
            /// String str = ...;
            /// Regexp re = new Regexp("%[a-fA-F0-9][a-fA-F0-9]");
            /// Regexp.Filter rf = new Regexp.Filter() {
            /// int count = 5;
            /// public boolean filter(Regsub rs, StringBuffer sb) {
            /// String match = rs.matched();
            /// int hi = Character.digit(match.charAt(1), 16);
            /// int lo = Character.digit(match.charAt(2), 16);
            /// sb.append((char) ((hi &lt;&lt; 4) | lo));
            /// return (--count > 0);
            /// }
            /// }
            /// String result = re.sub(str, rf);
            /// </pre>
            /// </summary>
            /// <param name="rs">
            /// <code>Regsub</code> containing the state of the current match.
            /// </param>
            /// <param name="sb">
            /// The string buffer that this filter should append the generated string to.  This string buffer actually
            /// contains the results the calling <code>Regexp</code> has
            /// generated up to this point.
            /// </param>
            /// <returns>
            /// <code>false</code> if no further matches should be considered in this string, <code>true</code> to allow
            /// <code>Regexp</code> to continue looking for further matches.
            /// </returns>
            bool filter(Regsub rs, StringBuilder sb);
        }

        private class SubspecFilter : IFilter
        {
            internal string _subspec;
            internal bool _all;

            public SubspecFilter(string subspec, bool all)
            {
                _subspec = subspec;
                _all = all;
            }

            public bool filter(Regsub rs, StringBuilder sb)
            {
                Regexp.applySubspec(rs, _subspec, sb);
                return _all;
            }
        }

        /// <summary>
        /// Returns a string representation of this compiled regular expression.  The format of the string representation is a
        /// symbolic dump of the bytecodes.
        /// </summary>
        /// <returns>
        /// A string representation of this regular expression.
        /// </returns>
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("# subs:  " + _npar + "\n");
            sb.Append("anchor:  " + _anchored + "\n");
            sb.Append("start:   " + (char)_startChar + "\n");
            sb.Append("must:    " + _must + "\n");
            for (int i = 0; i < _program.Length; )
            {
                sb.Append(i + ":\t");
                int op = _program[i];
                if (op >= CLOSE)
                    sb.Append("CLOSE" + (op - CLOSE));
                else if (op >= OPEN)
                    sb.Append("OPEN" + (op - OPEN));
                else
                    sb.Append(_opnames[op]);
                int offset = (int)_program[i + 1];
                if (offset == 0)
                    sb.Append('\t');
                else if (op == BACK)
                    sb.Append("\t-" + offset + "," + (i - offset));
                else
                    sb.Append("\t+" + offset + "," + (i + offset));
                if (op == ANYOF || op == ANYBUT || op == EXACTLY)
                {
                    sb.Append("\t'");
                    sb.Append(_program, i + 3, _program[i + 2]);
                    sb.Append("'");
                    i += 3 + _program[i + 2];
                }
                else
                    i += 2;
                sb.Append('\n');
            }
            return sb.ToString();
        }

        private void compile(string exp)
        {
            Compiler rcstate = new Compiler();
            rcstate._parse = exp.ToCharArray();
            rcstate._off = 0;
            rcstate._npar = 1;
            rcstate._code = new StringBuilder();
            rcstate.reg(false);
            _program = rcstate._code.ToString().ToCharArray();
            _npar = rcstate._npar;
            _startChar = -1;

            // optimize
            if (_program[rcstate.regnext(0)] == END)
            {
                if (_program[2] == BOL)
                    _anchored = true;
                else if (_program[2] == EXACTLY)
                    _startChar = (int)_program[5];
            }

            //
            // If there's something expensive in the r.e., find the longest literal string that must appear and make it the
            // regmust.  Resolve ties in favor of later strings, since the regstart check works with the beginning of the r.e.
            // and avoiding duplication strengthens checking.  Not a strong reason, but sufficient in the absence of others.
            /*
            if ((rcstate.flagp & Compiler.SPSTART) != 0) {
            int index = -1;
            int longest = 0;
			
            for (scan = 0; scan < program.length; ) {
            switch (program[scan]) {
            case EXACTLY:
            int length = program[scan + 2];
            if (length > longest) {
            index = scan;
            longest = length;
            }
            // fall through;
			
            case ANYOF:
            case ANYBUT:
            scan += 3 + program[scan + 2];
            break;
			
            default:
            scan += 2;
            break;
            }
            }
            if (longest > 0) {
            must = new String(program, index + 3, longest);
            }
            }
            */
        }

        internal Match exec(string str, int start, int off)
        {
            if (_ignoreCase)
                str = str.ToLower();
            Match match = new Match();
            match._program = _program;
            // Mark beginning of line for ^ .
            match._str = str;
            match._bol = start;
            match._length = str.Length;
            match._indices = new int[_npar * 2];
            if (_anchored)
            {
                // Simplest case:  anchored match need be tried only once.
                if (match.regtry(off))
                    return match;
            }
            else if (_startChar >= 0)
            {
                // We know what char it must start with.
                while (off < match._length)
                {
                    off = str.IndexOf((System.Char)_startChar, off);
                    if (off < 0)
                        break;
                    if (match.regtry(off))
                        return match;
                    off++;
                }
            }
            else
            {
                // Messy cases:  unanchored match.
                do
                {
                    if (match.regtry(off))
                        return match;
                }
                while (off++ < match._length);
            }
            return null;
        }

        internal class Compiler
        {
            internal char[] _parse;
            internal int _off;
            internal int _npar;
            internal StringBuilder _code;
            internal int _flagp;

            internal const string META = "^$.[()|?+*\\";
            internal const string MULT = "*+?";

            internal const int WORST = 0; // Worst case.
            internal const int HASWIDTH = 1; // Known never to match null string.
            internal const int SIMPLE = 2; // Simple enough to be STAR/PLUS operand.
            internal const int SPSTART = 4; // Starts with * or +.

            /// <summary>
            /// regular expression, i.e. main body or parenthesized thing
            /// Caller must absorb opening parenthesis.
            /// Combining parenthesis handling with the base level of regular expression is a trifle forced, but the need to tie the tails of the branches to what
            /// follows makes it hard to avoid.
            /// </summary>
            /// <param name="paren"></param>
            /// <returns></returns>
            internal int reg(bool paren)
            {
                int netFlags = HASWIDTH;
                int parno = 0;
                int ret = -1;
                if (paren)
                {
                    parno = _npar++;
                    if (_npar >= Regexp.NSUBEXP)
                        throw new ArgumentException("too many ()");
                    ret = regnode((char)(Regexp.OPEN + parno));
                }
                // Pick up the branches, linking them together.
                int br = regbranch();
                if (ret >= 0)
                    regtail(ret, br);
                else
                    ret = br;
                if ((_flagp & HASWIDTH) == 0)
                    netFlags &= ~HASWIDTH;
                netFlags |= (_flagp & SPSTART);
                while (_off < _parse.Length && _parse[_off] == '|')
                {
                    _off++;
                    br = regbranch();
                    regtail(ret, br);
                    if ((_flagp & HASWIDTH) == 0)
                        netFlags &= ~HASWIDTH;
                    netFlags |= (_flagp & SPSTART);
                }
                // Make a closing node, and hook it on the end.
                int ender = regnode((paren) ? (char)(Regexp.CLOSE + parno) : Regexp.END);
                regtail(ret, ender);
                // Hook the tails of the branches to the closing node.
                for (br = ret; br >= 0; br = regnext(br))
                    regoptail(br, ender);
                // Check for proper termination.
                if (paren && (_off >= _parse.Length || _parse[_off++] != ')'))
                    throw new ArgumentException("missing )");
                else if ((paren == false) && (_off < _parse.Length))
                    throw new ArgumentException("unexpected )");
                _flagp = netFlags;
                return ret;
            }

            /// <summary>
            /// regbranch - one alternative of an | operator
            /// Implements the concatenation operator.
            /// </summary>
            /// <returns></returns>
            internal int regbranch()
            {
                int netFlags = WORST; // Tentatively.
                int ret = regnode(Regexp.BRANCH);
                int chain = -1;
                while (_off < _parse.Length && _parse[_off] != '|' && _parse[_off] != ')')
                {
                    int latest = regpiece();
                    netFlags |= _flagp & HASWIDTH;
                    if (chain < 0) // First piece.
                        netFlags |= (_flagp & SPSTART);
                    else
                        regtail(chain, latest);
                    chain = latest;
                }
                if (chain < 0) // Loop ran zero times.
                    regnode(Regexp.NOTHING);
                _flagp = netFlags;
                return ret;
            }

            /// <summary>
            /// something followed by possible [*+?]
            /// Note that the branching code sequences used for ? and the general cases of * and + are somewhat optimized:  they use the same NOTHING node as
            /// both the endmarker for their branch list and the body of the last branch. It might seem that this node could be dispensed with entirely, but the
            /// endmarker role is not redundant.
            /// </summary>
            /// <returns></returns>
            internal int regpiece()
            {
                int netFlags;
                int ret = regatom();
                if (_off >= _parse.Length || !isMult(_parse[_off]))
                    return ret;
                char op = _parse[_off];
                if ((_flagp & HASWIDTH) == 0 && op != '?')
                    throw new ArgumentException("*+ operand could be empty");
                netFlags = (op != '+') ? (WORST | SPSTART) : (WORST | HASWIDTH);
                if (op == '*' && (_flagp & SIMPLE) != 0)
                    reginsert(Regexp.STAR, ret);
                else if (op == '*')
                {
                    // Emit x* as (x&|), where & means "self".
                    reginsert(Regexp.BRANCH, ret); // Either x
                    regoptail(ret, regnode(Regexp.BACK)); // and loop
                    regoptail(ret, ret); // back
                    regtail(ret, regnode(Regexp.BRANCH)); // or
                    regtail(ret, regnode(Regexp.NOTHING)); // null.
                }
                else if (op == '+' && (_flagp & SIMPLE) != 0)
                    reginsert(Regexp.PLUS, ret);
                else if (op == '+')
                {
                    // Emit x+ as x(&|), where & means "self".
                    int next = regnode(Regexp.BRANCH); // Either
                    regtail(ret, next);
                    regtail(regnode(Regexp.BACK), ret); // loop back
                    regtail(next, regnode(Regexp.BRANCH)); // or
                    regtail(ret, regnode(Regexp.NOTHING)); // null.
                }
                else if (op == '?')
                {
                    // Emit x? as (x|)
                    reginsert(Regexp.BRANCH, ret); // Either x
                    regtail(ret, regnode(Regexp.BRANCH)); // or
                    int next = regnode(Regexp.NOTHING); // null.
                    regtail(ret, next);
                    regoptail(ret, next);
                }
                _off++;
                if ((_off < _parse.Length) && isMult(_parse[_off]))
                    throw new ArgumentException("nested *?+");
                _flagp = netFlags;
                return ret;
            }

            /// <summary>
            /// the lowest level
            /// Optimization:  gobbles an entire sequence of ordinary characters so that it can turn them into a single node, which is smaller to store and
            /// faster to run.  Backslashed characters are exceptions, each becoming a separate node; the code is simpler that way and it's not worth fixing.
            /// </summary>
            /// <returns></returns>
            internal int regatom()
            {
                int netFlags = WORST; // Tentatively.
                int ret;
                switch (_parse[_off++])
                {
                    case '^':
                        ret = regnode(Regexp.BOL);
                        break;
                    case '$':
                        ret = regnode(Regexp.EOL);
                        break;
                    case '.':
                        ret = regnode(Regexp.ANY);
                        netFlags |= (HASWIDTH | SIMPLE);
                        break;
                    case '[':
                        {
                            try
                            {
                                if (_parse[_off] == '^')
                                {
                                    ret = regnode(Regexp.ANYBUT);
                                    _off++;
                                }
                                else
                                    ret = regnode(Regexp.ANYOF);
                                int pos = reglen();
                                regc('\x0000');
                                if (_parse[_off] == ']' || _parse[_off] == '-')
                                    regc(_parse[_off++]);
                                while (_parse[_off] != ']')
                                    if (_parse[_off] == '-')
                                    {
                                        _off++;
                                        if (_parse[_off] == ']')
                                            regc('-');
                                        else
                                        {
                                            int start = _parse[_off - 2];
                                            int end = _parse[_off++];
                                            if (start > end)
                                                throw new ArgumentException("invalid [] range");
                                            for (int i = start + 1; i <= end; i++)
                                                regc((char)i);
                                        }
                                    }
                                    else
                                        regc(_parse[_off++]);
                                regset(pos, (char)(reglen() - pos - 1));
                                _off++;
                                netFlags |= HASWIDTH | SIMPLE;
                            }
                            catch (IndexOutOfRangeException) { throw new ArgumentException("missing ]"); }
                            break;
                        }
                    case '(':
                        ret = reg(true);
                        netFlags |= (_flagp & (HASWIDTH | SPSTART));
                        break;
                    case '|':
                    case ')':
                        throw new ArgumentException("internal urp");
                    case '?':
                    case '+':
                    case '*':
                        throw new ArgumentException("?+* follows nothing");
                    case '\\':
                        if (_off >= _parse.Length)
                            throw new ArgumentException("trailing \\");
                        ret = regnode(Regexp.EXACTLY);
                        regc((char)1);
                        regc(_parse[_off++]);
                        netFlags |= HASWIDTH | SIMPLE;
                        break;
                    default:
                        {
                            _off--;
                            int end;
                            for (end = _off; end < _parse.Length; end++)
                                if (META.IndexOf((System.Char)_parse[end]) >= 0)
                                    break;
                            if (end > _off + 1 && end < _parse.Length && isMult(_parse[end]))
                                end--; /* Back off clear of ?+* operand. */
                            netFlags |= HASWIDTH;
                            if (end == _off + 1)
                                netFlags |= SIMPLE;
                            ret = regnode(Regexp.EXACTLY);
                            regc((char)(end - _off));
                            for (; _off < end; _off++)
                                regc(_parse[_off]);
                        }
                        break;
                }
                _flagp = netFlags;
                return ret;
            }

            /// <summary>
            /// emit a node
            /// </summary>
            /// <param name="op"></param>
            /// <returns></returns>
            internal int regnode(char op)
            {
                int ret = _code.Length;
                _code.Append(op);
                _code.Append('\x0000');
                return ret;
            }

            /// <summary>
            /// emit (if appropriate) a byte of code
            /// </summary>
            /// <param name="b"></param>
            internal void regc(char b)
            {
                _code.Append(b);
            }

            internal int reglen()
            {
                return _code.Length;
            }

            internal void regset(int pos, char ch)
            {
                _code[pos] = ch;
            }

            /// <summary>
            /// insert an operator in front of already-emitted operand
            /// Means relocating the operand.
            /// </summary>
            /// <param name="op"></param>
            /// <param name="pos"></param>
            internal void reginsert(char op, int pos)
            {
                char[] tmp = new char[] { op, '\x0000' };
                _code.Insert(pos, tmp);
            }

            /// <summary>
            /// set the next-pointer at the end of a node chain
            /// </summary>
            /// <param name="pos"></param>
            /// <param name="val"></param>
            internal void regtail(int pos, int val)
            {
                // Find last node.
                int scan = pos;
                while (true)
                {
                    int tmp = regnext(scan);
                    if (tmp < 0)
                        break;
                    scan = tmp;
                }
                int offset = (_code[scan] == Regexp.BACK ? scan - val : val - scan);
                _code[scan + 1] = (char)offset;
            }

            /// <summary>
            /// regtail on operand of first argument; nop if operandless
            /// </summary>
            /// <param name="pos"></param>
            /// <param name="val"></param>
            internal void regoptail(int pos, int val)
            {
                if (pos < 0 || _code[pos] != Regexp.BRANCH)
                    return;
                regtail(pos + 2, val);
            }

            /// <summary>
            /// dig the "next" pointer out of a node
            /// </summary>
            /// <param name="pos"></param>
            /// <returns></returns>
            internal int regnext(int pos)
            {
                int offset = _code[pos + 1];
                if (offset == 0)
                    return -1;
                if (_code[pos] == Regexp.BACK)
                    return pos - offset;
                else
                    return pos + offset;
            }

            internal static bool isMult(char ch)
            {
                return (ch == '*' || ch == '+' || ch == '?');
            }
        }

        internal class Match
        {
            internal char[] _program;

            internal string _str;
            internal int _bol;
            internal int _input;
            internal int _length;

            internal int[] _indices;

            internal bool regtry(int off)
            {
                _input = off;
                for (int i = 0; i < _indices.Length; i++)
                    _indices[i] = -1;
                if (regmatch(0))
                {
                    _indices[0] = off;
                    _indices[1] = _input;
                    return true;
                }
                else
                    return false;
            }

            /// <summary>
            /// main matching routine
            /// Conceptually the strategy is simple:  check to see whether the current node matches, call self recursively to see whether the rest matches,
            /// and then act accordingly.  In practice we make some effort to avoid recursion, in particular by going through "ordinary" nodes (that don't
            /// need to know whether the rest of the match failed) by a loop instead of by recursion.
            /// </summary>
            /// <param name="scan"></param>
            /// <returns></returns>
            internal bool regmatch(int scan)
            {
                while (true)
                {
                    int next = regnext(scan);
                    int op = _program[scan];
                    switch (op)
                    {
                        case Regexp.BOL:
                            if (_input != _bol)
                                return false;
                            break;
                        case Regexp.EOL:
                            if (_input != _length)
                                return false;
                            break;
                        case Regexp.ANY:
                            if (_input >= _length)
                                return false;
                            _input++;
                            break;
                        case Regexp.EXACTLY:
                            if (!compare(scan))
                                return false;
                            break;
                        case Regexp.ANYOF:
                            if (_input >= _length)
                                return false;
                            if (!present(scan))
                                return false;
                            _input++;
                            break;
                        case Regexp.ANYBUT:
                            if (_input >= _length)
                                return false;
                            if (present(scan))
                                return false;
                            _input++;
                            break;
                        case Regexp.NOTHING:
                        case Regexp.BACK:
                            break;
                        case Regexp.BRANCH:
                            if (_program[next] != Regexp.BRANCH)
                                next = scan + 2;
                            else
                            {
                                do
                                {
                                    int save = _input;
                                    if (regmatch(scan + 2))
                                        return true;
                                    _input = save;
                                    scan = regnext(scan);
                                }
                                while (scan >= 0 && _program[scan] == Regexp.BRANCH);
                                return false;
                            }
                            break;
                        case Regexp.STAR:
                        case Regexp.PLUS:
                            {
                                // Lookahead to avoid useless match attempts when we know what character comes next.
                                int ch = -1;
                                if (_program[next] == Regexp.EXACTLY)
                                    ch = _program[next + 3];
                                int min = (op == Regexp.STAR) ? 0 : 1;
                                int save = _input;
                                int no = regrepeat(scan + 2);
                                while (no >= min)
                                {
                                    // If it could work, try it.
                                    if (ch < 0 || (_input < _length && _str[_input] == ch))
                                        if (regmatch(next))
                                            return true;
                                    // Couldn't or didn't -- back up.
                                    no--;
                                    _input = save + no;
                                }
                                return false;
                            }
                        case Regexp.END:
                            return true;
                        default:
                            if (op >= Regexp.CLOSE)
                            {
                                int no = op - Regexp.CLOSE;
                                int save = _input;
                                if (regmatch(next))
                                {
                                    // Don't set endp if some later invocation of the same parentheses already has.
                                    if (_indices[no * 2 + 1] <= 0)
                                        _indices[no * 2 + 1] = save;
                                    return true;
                                }
                            }
                            else if (op >= Regexp.OPEN)
                            {
                                int no = op - Regexp.OPEN;
                                int save = _input;
                                if (regmatch(next))
                                {
                                    // Don't set startp if some later invocation of the same parentheses already has.
                                    if (_indices[no * 2] <= 0)
                                        _indices[no * 2] = save;
                                    return true;
                                }
                            }
                            return false;
                    }
                    scan = next;
                }
            }

            internal bool compare(int scan)
            {
                int count = _program[scan + 2];
                if (_input + count > _length)
                    return false;
                int start = scan + 3;
                int end = start + count;
                for (int i = start; i < end; i++)
                    if (_str[_input++] != _program[i])
                        return false;
                return true;
            }

            internal bool present(int scan)
            {
                char ch = _str[_input];
                int count = _program[scan + 2];
                int start = scan + 3;
                int end = start + count;
                for (int i = start; i < end; i++)
                    if (_program[i] == ch)
                        return true;
                return false;
            }

            /// <summary>
            /// repeatedly match something simple, report how many
            /// </summary>
            /// <param name="scan"></param>
            /// <returns></returns>
            internal int regrepeat(int scan)
            {
                int op = _program[scan];
                int count = 0;
                switch (op)
                {
                    case Regexp.ANY:
                        count = _length - _input;
                        _input = _length;
                        break;
                    case Regexp.EXACTLY:
                        // 'g*' matches all the following 'g' characters.
                        char ch = _program[scan + 3];
                        while (_input < _length && _str[_input] == ch)
                        {
                            _input++;
                            count++;
                        }
                        break;
                    case Regexp.ANYOF:
                        while (_input < _length && present(scan))
                        {
                            _input++;
                            count++;
                        }
                        break;
                    case Regexp.ANYBUT:
                        while (_input < _length && !present(scan))
                        {
                            _input++;
                            count++;
                        }
                        break;
                }
                return count;
            }

            /// <summary>
            /// dig the "next" pointer out of a node
            /// </summary>
            /// <param name="scan"></param>
            /// <returns></returns>
            internal int regnext(int scan)
            {
                int offset = _program[scan + 1];
                return (_program[scan] == Regexp.BACK ? scan - offset : scan + offset);
            }
        }
    }
}
