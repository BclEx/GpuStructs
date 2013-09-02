#region Foreign-License
/*
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /// <summary> This exception is used to report variable errors in Tcl.</summary>

    class TclVarException : TclException
    {

        /// <summary> Creates an exception with the appropiate Tcl error message to
        /// indicate an error with variable access.
        /// 
        /// </summary>
        /// <param name="interp">currrent interpreter.
        /// </param>
        /// <param name="name1">first part of a variable name.
        /// </param>
        /// <param name="name2">second part of a variable name. May be null.
        /// </param>
        /// <param name="operation">either "read" or "set".
        /// </param>
        /// <param name="reason">a string message to explain why the operation fails..
        /// </param>

        internal TclVarException(Interp interp, string name1, string name2, string operation, string reason)
            : base(TCL.CompletionCode.ERROR)
        {
            if (interp != null)
            {
                interp.resetResult();
                if ((System.Object)name2 == null)
                {
                    interp.setResult("can't " + operation + " \"" + name1 + "\": " + reason);
                }
                else
                {
                    interp.setResult("can't " + operation + " \"" + name1 + "(" + name2 + ")\": " + reason);
                }
            }
        }
    }
}
