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

    /// <summary> This class implements the built-in "break" command in Tcl.</summary>
    class BreakCmd : ICommand
    {
        /// <summary> This procedure is invoked to process the "break" Tcl command.
        /// See the user documentation for details on what it does.
        /// </summary>
        /// <exception cref=""> TclException is always thrown.
        /// </exception>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            if (argv.Length != 1)
            {
                throw new TclNumArgsException(interp, 1, argv, null);
            }
            throw new TclException(interp, null, TCL.CompletionCode.BREAK);
        }
    }
}
