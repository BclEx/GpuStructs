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

    /// <summary> This class implements the built-in "unset" command in Tcl.</summary>

    class UnsetCmd : ICommand
    {
        /// <summary> Tcl_UnsetObjCmd -> UnsetCmd.cmdProc
        /// 
        /// Unsets Tcl variable (s). See Tcl user documentation * for
        /// details.
        /// </summary>
        /// <exception cref=""> TclException If tries to unset a variable that does
        /// not exist.
        /// </exception>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] objv)
        {
            switch (objv.Length)
            {
                case 2:
                    interp.UnsetVar(objv[1], 0);
                    break;
                case 3:
                    for (int i = (objv[1].ToString() != "-nocomplain") ? 1 : 2; i < objv.Length; i++)
                    {
                        Var.UnsetVar(interp, objv[i].ToString(), 0);
                    }
                    break;
                default:
                    if (objv.Length < 2)
                    {
                        throw new TclNumArgsException(interp, 1, objv, "varName ?varName ...?");
                    }
                    break;
            }

            return TCL.CompletionCode.RETURN;
        }
    }
}
