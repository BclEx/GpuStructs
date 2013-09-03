#region Foreign-License
/*
	Implements the "error" command.

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
    * This class implements the built-in "error" command in Tcl.
    */

    class ErrorCmd : ICommand
    {

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            if (argv.Length < 2 || argv.Length > 4)
            {
                throw new TclNumArgsException(interp, 1, argv, "message ?errorInfo? ?errorCode?");
            }

            if (argv.Length >= 3)
            {

                string errorInfo = argv[2].ToString();

                if (!errorInfo.Equals(""))
                {
                    interp.AddErrorInfo(errorInfo);
                    interp._errAlreadyLogged = true;
                }
            }

            if (argv.Length == 4)
            {
                interp.SetErrorCode(argv[3]);
            }

            interp.setResult(argv[1]);
            throw new TclException(TCL.CompletionCode.ERROR);
        }
    } // end ErrorCmd
}
