#region Foreign-License
/*
Interface for Commands that can be added to the Tcl Interpreter.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
namespace Tcl.Lang
{
    /// <summary> The Command interface specifies the method that a new Tcl command must implement.  See the createCommand method of the Interp class
    /// to see how to add a new command to an interperter.
    /// </summary>
    public interface ICommand
    {
        TCL.CompletionCode CmdProc(Interp interp, TclObject[] objv); // Tcl exceptions are thown for Tcl errors.
    }
}
