#region Foreign-License
/*
	This file contains the Jacl implementation of the built-in Tcl "pwd" command.

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
    * This class implements the built-in "pwd" command in Tcl.
    */

    class PwdCmd : ICommand
    {

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            if (argv.Length != 1)
            {
                throw new TclNumArgsException(interp, 1, argv, null);
            }

            // Get the name of the working dir.

            string dirName = interp.getWorkingDir().ToString();

            // Java File Object methods use backslashes on Windows.
            // Convert them to forward slashes before returning the dirName to Tcl.

            if (JACL.PLATFORM == JACL.PLATFORM_WINDOWS)
            {
                dirName = dirName.Replace('\\', '/');
            }

            interp.SetResult(dirName);
            return TCL.CompletionCode.RETURN;
        }
    } // end PwdCmd class
}
