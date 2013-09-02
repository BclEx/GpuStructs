#region Foreign-License
/*
	Interface for creating variable traces.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /*
    * This interface is used to make variable traces. To make a variable
    * trace, write a class that implements the VarTrace and call
    * Interp.traceVar with an instance of that class.
    * 
    */

    public interface VarTrace
    {

        void traceProc(Interp interp, string part1, string part2, TCL.VarFlag flags); // The traceProc may throw a TclException
        // to indicate an error during the trace.
    } // end VarTrace
}
