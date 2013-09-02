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

    /// <summary> Signals that a unrecoverable run-time error in the interpreter.
    /// Similar to the panic() function in C.
    /// </summary>
    public class TclRuntimeError : System.SystemException
    {
        /// <summary> Constructs a TclRuntimeError with the specified detail
        /// message.
        /// 
        /// </summary>
        /// <param name="s">the detail message.
        /// </param>
        public TclRuntimeError(string s)
            : base(s)
        {
        }
        public TclRuntimeError(string s, Exception inner)
            : base(s, inner)
        {
        }
    }
}
