#region Foreign-License
/*
Copyright (c) 1987-1994 The Regents of the University of California.
Copyright (c) 1994-1997 Sun Microsystems, Inc.
Copyright (c) 1998-1999 by Scriptics Corporation.
Copyright (c) 1999      by Moses DeJong.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /// <summary> This class implements the built-in "variable" command in Tcl.</summary>

    class VariableCmd : ICommand
    {
        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] objv)
        {


            string varName;
            int tail, cp;
            Var var, array;
            TclObject varValue;
            int i;

            for (i = 1; i < objv.Length; i = i + 2)
            {
                // Look up each variable in the current namespace context, creating
                // it if necessary.


                varName = objv[i].ToString();
                Var[] result = Var.LookupVar(interp, varName, null, (TCL.VarFlag.NAMESPACE_ONLY | TCL.VarFlag.LEAVE_ERR_MSG), "define", true, false);
                if (result == null)
                {
                    // FIXME:
                    throw new TclException(interp, "");
                }

                var = result[0];
                array = result[1];

                // Mark the variable as a namespace variable and increment its 
                // reference count so that it will persist until its namespace is
                // destroyed or until the variable is unset.

                if ((var.Flags & VarFlags.NAMESPACE_VAR) == 0)
                {
                    var.Flags |= VarFlags.NAMESPACE_VAR;
                    var.RefCount++;
                }

                // If a value was specified, set the variable to that value.
                // Otherwise, if the variable is new, leave it undefined.
                // (If the variable already exists and no value was specified,
                // leave its value unchanged; just create the local link if
                // we're in a Tcl procedure).

                if (i + 1 < objv.Length)
                {
                    // a value was specified
                    varValue = Var.SetVar(interp, objv[i], null, objv[i + 1], (TCL.VarFlag.NAMESPACE_ONLY | TCL.VarFlag.LEAVE_ERR_MSG));

                    if (varValue == null)
                    {
                        // FIXME:
                        throw new TclException(interp, "");
                    }
                }



                // If we are executing inside a Tcl procedure, create a local
                // variable linked to the new namespace variable "varName".

                if ((interp.VarFrame != null) && interp.VarFrame.IsProcCallFrame)
                {

                    // varName might have a scope qualifier, but the name for the
                    // local "link" variable must be the simple name at the tail.
                    //
                    // Locate tail in one pass: drop any prefix after two *or more*
                    // consecutive ":" characters).

                    int len = varName.Length;

                    for (tail = cp = 0; cp < len; )
                    {
                        if (varName[cp++] == ':')
                        {
                            while ((cp < len) && (varName[cp++] == ':'))
                            {
                                tail = cp;
                            }
                        }
                    }

                    // Create a local link "tail" to the variable "varName" in the
                    // current namespace.

                    Var.makeUpvar(interp, null, varName, null, TCL.VarFlag.NAMESPACE_ONLY, varName.Substring(tail), 0);
                }
            }
            return TCL.CompletionCode.RETURN;
        }
    }
}
