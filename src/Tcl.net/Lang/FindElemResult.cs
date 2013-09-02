#region Foreign-License
/*
	Result returned by Util.findElement().

Copyright (c) 1997 Cornell University.
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /*
    * Result returned by Util.findElement().
    */

    class FindElemResult
    {

        /*
        * The end of the element in the original string -- the index of the
        * character immediately behind the element.
        */

        internal int elemEnd;

        /*
        * The element itself.
        */

        internal string elem;
        internal bool brace;
        internal int size;

        internal FindElemResult(int i, string s, int b)
        {
            elemEnd = i;
            elem = s;
            brace = b != 0;
            size = s.Length;
        }
    } // end FindElemResult
}
