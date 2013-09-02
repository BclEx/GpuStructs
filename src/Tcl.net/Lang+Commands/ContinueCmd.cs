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

    /// <summary> This class implements the built-in "continue" command in Tcl.</summary>

    class ContinueCmd : Command
    {
        /// <summary> This procedure is invoked to process the "continue" Tcl command.
        /// See the user documentation for details on what it does.
        /// </summary>
        /// <exception cref=""> TclException is always thrown.
        /// </exception>

        public TCL.CompletionCode cmdProc(Interp interp, TclObject[] argv)
        {
            if (argv.Length != 1)
            {
                throw new TclNumArgsException(interp, 1, argv, null);
            }
            throw new TclException(interp, null, TCL.CompletionCode.CONTINUE);
        }
    }
}
