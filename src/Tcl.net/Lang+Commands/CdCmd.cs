#region Foreign-License
/*
	This file contains the Jacl implementation of the built-in Tcl "cd" command.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and
redistribution of this file, and for a DISCLAIMER OF ALL
WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    // This class implements the built-in "cd" command in Tcl.

    class CdCmd : Command
    {

        public TCL.CompletionCode cmdProc(Interp interp, TclObject[] argv)
        {
            string dirName;

            if (argv.Length > 2)
            {
                throw new TclNumArgsException(interp, 1, argv, "?dirName?");
            }

            if (argv.Length == 1)
            {
                dirName = "~";
            }
            else
            {

                dirName = argv[1].ToString();
            }
            if ((JACL.PLATFORM == JACL.PLATFORM_WINDOWS) && (dirName.Length == 2) && (dirName[1] == ':'))
            {
                dirName = dirName + "/";
            }

            // Set the interp's working dir.

            interp.setWorkingDir(dirName);
            return TCL.CompletionCode.RETURN;
        }
    } // end CdCmd class
}
