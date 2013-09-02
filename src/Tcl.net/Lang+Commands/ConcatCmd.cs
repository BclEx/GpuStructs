#region Foreign-License
/*
Copyright (c) 1997 Cornell University.
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /// <summary> This class implements the built-in "concat" command in Tcl.</summary>
    class ConcatCmd : Command
    {

        /// <summary> See Tcl user documentation for details.</summary>
        public TCL.CompletionCode cmdProc(Interp interp, TclObject[] argv)
        {
            interp.setResult(Util.concat(1, argv.Length, argv));
            return TCL.CompletionCode.RETURN;
        }
    }
}
