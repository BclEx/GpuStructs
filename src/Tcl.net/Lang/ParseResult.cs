#region Foreign-License
/*
Copyright (c) 1997 Cornell University.
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System.Text;
namespace Tcl.Lang
{

    /// <summary> This class stores a single word that's generated inside the Tcl parser
    /// inside the Interp class.
    /// </summary>
    public class ParseResult
    {

        /// <summary> The value of a parse operation. For calls to Interp.intEval(),
        /// this variable is the same as interp.m_result. The ref count
        /// has been incremented, so the user will need to explicitly
        /// invoke release() to drop the ref.
        /// </summary>
        public TclObject Value;

        /// <summary> Points to the next character to be parsed.</summary>
        public int nextIndex;

        /// <summary> Create an empty parsed word.</summary>
        internal ParseResult()
        {
            Value = TclString.NewInstance("");
            Value.Preserve();
        }

        internal ParseResult(string s, int ni)
        {
            Value = TclString.NewInstance(s);
            Value.Preserve();
            nextIndex = ni;
        }

        /// <summary> Assume that the caller has already preserve()'ed the TclObject.</summary>
        internal ParseResult(TclObject o, int ni)
        {
            Value = o;
            nextIndex = ni;
        }

        internal ParseResult(StringBuilder sbuf, int ni)
        {
            Value = TclString.NewInstance(sbuf.ToString());
            Value.Preserve();
            nextIndex = ni;
        }

        public void Release()
        {
            Value.Release();
        }
    }
}
