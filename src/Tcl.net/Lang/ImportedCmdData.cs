#region Foreign-License
/*
	An ImportedCmdData instance is used as the Command implementation
     (the cmd member of the WrappedCommand class).

Copyright (c) 1999 Mo DeJong.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{


    /// <summary> Class which is used as the Command implementation inside a WrappedCommand
    /// that has been imported into another namespace. The cmd member of a Wrapped
    /// command will be set to an instance of this class when a command is imported.
    /// From this ImportedCmdData reference, we can find the "real" command from
    /// another namespace.
    /// </summary>

    class ImportedCmdData : Command, CommandWithDispose
    {
        internal WrappedCommand realCmd; // "Real" command that this imported command
        // refers to.
        internal WrappedCommand self; // Pointer to this imported WrappedCommand. Needed
        // only when deleting it in order to remove
        // it from the real command's linked list of
        // imported commands that refer to it.

        public override string ToString()
        {

            return "ImportedCmd for " + realCmd;
        }

        /// <summary> Called when the command is invoked in the interp.</summary>

        public TCL.CompletionCode cmdProc(Interp interp, TclObject[] objv)
        {
            NamespaceCmd.invokeImportedCmd(interp, this, objv);
            return TCL.CompletionCode.RETURN;
        }

        /// <summary> Called when the command is deleted from the interp.</summary>

        public void disposeCmd()
        {
            NamespaceCmd.deleteImportedCmd(this);
        }
    }
}
