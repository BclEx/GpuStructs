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

    /// <summary> This class implements the built-in "exit" command in Tcl.</summary>
    class ExitCmd : ICommand
    {

        /// <summary> See Tcl user documentation for details.</summary>
        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            int code;

            if (argv.Length > 2)
            {
                throw new TclNumArgsException(interp, 1, argv, "?returnCode?");
            }
            if (argv.Length == 2)
            {
                code = TclInteger.get(interp, argv[1]);
            }
            else
            {
                code = 0;
            }
            return TCL.CompletionCode.EXIT;
        }
    }
}
