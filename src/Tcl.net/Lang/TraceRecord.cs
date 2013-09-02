#region Foreign-License
/*
	This class is used internally by CallFrame to store one variable trace.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /// <summary> This class is used internally by CallFrame to store one variable
    /// trace.
    /// </summary>

    class TraceRecord
    {

        /// <summary> Stores info about the conditions under which this trace should be
        /// triggered. Should be a combination of TCL.VarFlag.TRACE_READS, TCL.VarFlag.TRACE_WRITES
        /// or TCL.VarFlag.TRACE_UNSETS.
        /// </summary>

        internal TCL.VarFlag flags;

        /// <summary> Stores the trace procedure to invoke when a trace is fired.</summary>

        internal VarTrace trace;
    } // end TraceRecord
}
