using System;
namespace Core
{
    public class Regsub
    {
        internal Regexp _r;
        internal string _str;
        internal int _ustart;
        internal int _mstart;
        internal int _end;
        internal Regexp.Match _m;

        /// <summary>
        /// Construct a new <code>Regsub</code> that can be used to step 
        /// through the given string, finding each substring that matches
        /// the given regular expression.
        /// <code>Regexp</code> contains two substitution methods,
        /// <code>sub</code> and <code>subAll</code>, that can be used instead
        /// of <code>Regsub</code> if just simple substitutions are being done.
        /// </summary>
        /// <param name="r">
        /// The compiled regular expression.
        /// </param>
        /// <param name="str">
        /// The string to search.
        /// </param>
        /// <seealso cref="Regexp#sub" />
        /// <seealso cref="Regexp#subAll" />
        public Regsub(Regexp r, string str)
        {
            _r = r;
            _str = str;
            _ustart = 0;
            _mstart = -1;
            _end = 0;
        }

        /// <summary>
        /// Searches for the next substring that matches the regular expression.
        /// After calling this method, the caller would call methods like
        /// <code>skipped</code>, <code>matched</code>, etc. to query attributes
        /// of the matched region.
        /// Calling this function again will search for the next match, beginning
        /// at the character just after where the last match ended.
        /// </summary>
        /// <returns>
        /// <code>true</code> if a match was found, <code>false</code>
        /// if there are no more matches.
        /// </returns>
        public bool nextMatch()
        {
            _ustart = _end;
            // Consume one character if the last match didn't consume any characters, to avoid an infinite loop.
            int off = _ustart;
            if (off == _mstart)
            {
                off++;
                if (off >= _str.Length)
                    return false;
            }
            _m = _r.exec(_str, 0, off);
            if (_m == null)
                return false;
            _mstart = _m._indices[0];
            _end = _m._indices[1];
            return true;
        }

        /// <summary>
        /// Returns a substring consisting of all the characters skipped
        /// between the end of the last match (or the start of the original
        /// search string) and the start of this match.
        /// This method can be used extract all the portions of string that
        /// <b>didn't</b> match the regular expression.
        /// </summary>
        /// <returns>
        /// The characters that didn't match.
        /// </returns>
        public string skipped()
        {
            return _str.Substring(_ustart, _mstart - _ustart);
        }

        /// <summary>
        /// Returns a substring consisting of the characters that matched
        /// the entire regular expression during the last call to
        /// <code>nextMatch</code>.  
        /// </summary>
        /// <returns>
        /// The characters that did match.
        /// </returns>
        /// <seealso cref="#submatch" />
        public string matched()
        {
            return _str.Substring(_mstart, _end - _mstart);
        }

        /// <summary>
        /// Returns a substring consisting of the characters that matched the given parenthesized subexpression during the last call to <code>nextMatch</code>.
        /// </summary>
        /// <param name="i">
        /// The index of the parenthesized subexpression.
        /// </param>
        /// <returns>
        /// The characters that matched the subexpression, or <code>null</code> if the given subexpression did not exist or did not match.
        /// </returns>
        public string submatch(int i)
        {
            if (i * 2 + 1 >= _m._indices.Length)
                return null;
            int start = _m._indices[i * 2];
            int end = _m._indices[i * 2 + 1];
            if (start < 0 || end < 0)
                return null;
            return _str.Substring(start, end - start);
        }

        /// <summary>
        /// Returns a substring consisting of all the characters that come after the last match.  As the matches progress, the <code>rest</code>
        /// gets shorter.  When <code>nextMatch</code> returns <code>false</code>, then this method will return the rest of the string that can't be
        /// matched.
        /// </summary>
        /// <returns>
        /// The rest of the characters after the last match.
        /// </returns>
        public string rest()
        {
            return _str.Substring(_end);
        }
    }
}
