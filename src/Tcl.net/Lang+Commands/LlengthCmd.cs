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

    /// <summary> This class implements the built-in "llength" command in Tcl.</summary>

    class LlengthCmd : ICommand
    {
        /// <summary> See Tcl user documentation for details.</summary>
        /// <exception cref=""> TclException If incorrect number of arguments.
        /// </exception>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            if (argv.Length != 2)
            {
                throw new TclNumArgsException(interp, 1, argv, "list");
            }
            interp.setResult(TclInteger.NewInstance(TclList.getLength(interp, argv[1])));
            return TCL.CompletionCode.RETURN;
        }
    }
}
