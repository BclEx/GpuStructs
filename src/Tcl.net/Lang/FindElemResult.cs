#region Foreign-License
/*
Result returned by Util.findElement().

Copyright (c) 1997 Cornell University.
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
namespace Tcl.Lang
{
    /// <summary>
    /// Result returned by Util.findElement().
    /// </summary>
    class FindElemResult
    {
        internal int ElemEnd; // The end of the element in the original string -- the index of the character immediately behind the element.
        internal string Elem; // The element itself.
        internal bool Brace;
        internal int Size;

        internal FindElemResult(int i, string s, int b)
        {
            ElemEnd = i;
            Elem = s;
            Brace = (b != 0);
            Size = s.Length;
        }
    }
}
