#region Foreign-License
/*
Copyright (c) 1997 Cornell University.
Copyright (c) 1997-1998 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and
redistribution of this file, and for a DISCLAIMER OF ALL
WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /// <summary> This class implements the built-in "incr" command in Tcl.</summary>
    class IncrCmd : ICommand
    {
        /// <summary> This procedure is invoked to process the "incr" Tcl command.
        /// See the user documentation for details on what it does.
        /// </summary>
        /// <exception cref=""> TclException if wrong # of args or increment is not an
        /// integer.
        /// </exception>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] objv)
        {
            int incrAmount;
            TclObject newValue;

            if ((objv.Length != 2) && (objv.Length != 3))
            {
                throw new TclNumArgsException(interp, 1, objv, "varName ?increment?");
            }

            // Calculate the amount to increment by.

            if (objv.Length == 2)
            {
                incrAmount = 1;
            }
            else
            {
                try
                {
                    incrAmount = TclInteger.Get(interp, objv[2]);
                }
                catch (TclException e)
                {
                    interp.AddErrorInfo("\n    (reading increment)");
                    throw;
                }
            }

            // Increment the variable's value.

            newValue = Var.incrVar(interp, objv[1], null, incrAmount, TCL.VarFlag.LEAVE_ERR_MSG);

            // FIXME: we need to look at this exception throwing problem again
            /*
            if (newValue == null) {
            return TCL_ERROR;
            }
            */

            // Set the interpreter's object result to refer to the variable's new
            // value object.

            interp.SetResult(newValue);
            return TCL.CompletionCode.RETURN;
        }
    }
}
