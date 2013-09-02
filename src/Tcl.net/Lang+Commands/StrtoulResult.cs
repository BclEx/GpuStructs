#region Foreign-License
/*
	Stores the result of the Util.strtoul() method.

Copyright (c) 1997 Cornell University.
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /// <summary> This class stores the result of the Util.strtoul() method.</summary>

    class StrtoulResult
    {

        // If the conversion is successful, errno = 0;
        //
        // If the number cannot be converted to a valid unsigned 32-bit integer,
        // contains the error code (TCL.INTEGER_RANGE or TCL.INVALID_INTEGER).

        internal int errno;

        // If errno is 0, points to the character right after the number

        internal int Index;

        // If errno is 0, contains the value of the number.

        internal long value;

        internal StrtoulResult(long v, int i, int e)
        {
            value = v;
            Index = i;
            errno = e;
        }
    } // end StrtoulResult
}
