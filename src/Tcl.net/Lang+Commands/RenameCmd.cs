#region Foreign-License
/*
Copyright (c) 1999 Mo DeJong.
Copyright (c) 1997 Cornell University.
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /// <summary> This class implements the built-in "rename" command in Tcl.</summary>

    class RenameCmd : ICommand
    {
        /// <summary>----------------------------------------------------------------------
        /// 
        /// Tcl_RenameObjCmd -> RenameCmd.cmdProc
        /// 
        /// This procedure is invoked to process the "rename" Tcl command.
        /// See the user documentation for details on what it does.
        /// 
        /// Results:
        /// A standard Tcl object result.
        /// 
        /// Side effects:
        /// See the user documentation.
        /// 
        /// ----------------------------------------------------------------------
        /// </summary>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] objv)
        {
            string oldName, newName;

            if (objv.Length != 3)
            {
                throw new TclNumArgsException(interp, 1, objv, "oldName newName");
            }


            oldName = objv[1].ToString();

            newName = objv[2].ToString();

            interp.renameCommand(oldName, newName);
            return TCL.CompletionCode.RETURN;
        }
    }
}
