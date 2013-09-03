#region Foreign-License
/*
	Implements the built-in "append" Tcl command.

Copyright (c) 1997 Cornell University.
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /*
    * This class implements the built-in "append" command in Tcl.
    */

    class AppendCmd : ICommand
    {
        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] objv)
        {
            TclObject varValue = null;

            if (objv.Length < 2)
            {
                throw new TclNumArgsException(interp, 1, objv, "varName ?value value ...?");
            }
            else if (objv.Length == 2)
            {
                interp.ResetResult();
                interp.setResult(interp.GetVar(objv[1], 0));
            }
            else
            {
                for (int i = 2; i < objv.Length; i++)
                {
                    varValue = interp.SetVar(objv[1], objv[i], TCL.VarFlag.APPEND_VALUE);
                }

                if (varValue != null)
                {
                    interp.ResetResult();
                    interp.setResult(varValue);
                }
                else
                {
                    interp.ResetResult();
                }
            }
            return TCL.CompletionCode.RETURN;
        }
    } // end AppendCmd
}
