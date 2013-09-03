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

    /// <summary> This class implements the built-in "catch" command in Tcl.</summary>

    class CatchCmd : ICommand
    {
        /// <summary> This procedure is invoked to process the "catch" Tcl command.
        /// See the user documentation for details on what it does.
        /// 
        /// </summary>
        /// <param name="interp">the current interpreter.
        /// </param>
        /// <param name="argv">command arguments.
        /// </param>
        /// <exception cref=""> TclException if wrong number of arguments.
        /// </exception>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            if (argv.Length != 2 && argv.Length != 3)
            {
                throw new TclNumArgsException(interp, 1, argv, "command ?varName?");
            }

            TclObject result;
            TCL.CompletionCode code = TCL.CompletionCode.OK;

            try
            {
                interp.eval(argv[1], 0);
            }
            catch (TclException e)
            {
                code = e.GetCompletionCode();
            }

            result = interp.GetResult();

            if (argv.Length == 3)
            {
                try
                {
                    interp.SetVar(argv[2], result, 0);
                }
                catch (TclException e)
                {
                    throw new TclException(interp, "couldn't save command result in variable");
                }
            }

            interp.ResetResult();
            interp.setResult(TclInteger.NewInstance((int)code));
            return TCL.CompletionCode.RETURN;
        }
    }
}
