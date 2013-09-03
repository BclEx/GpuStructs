#region Foreign-License
/*
	Implements the "source" command.

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
    * This class implements the built-in "source" command in Tcl.
    */

    class SourceCmd : ICommand
    {

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            string fileName = null;
            bool url = false;

            if (argv.Length == 2)
            {

                fileName = argv[1].ToString();
            }
            else if (argv.Length == 3)
            {

                if (argv[1].ToString().Equals("-url"))
                {
                    url = true;

                    fileName = argv[2].ToString();
                }
            }

            if ((System.Object)fileName == null)
            {
                throw new TclNumArgsException(interp, 1, argv, "?-url? fileName");
            }

            try
            {
                if (url)
                {
                    if (fileName.StartsWith("resource:/"))
                    {
                        interp.evalResource(fileName.Substring(9));
                    }
                    else
                    {
                        interp.evalURL(null, fileName);
                    }
                }
                else
                {
                    interp.evalFile(fileName);
                }
            }
            catch (TclException e)
            {
                TCL.CompletionCode code = e.GetCompletionCode();

                if (code == TCL.CompletionCode.RETURN)
                {
                    TCL.CompletionCode realCode = interp.updateReturnInfo();
                    if (realCode != TCL.CompletionCode.OK)
                    {
                        e.setCompletionCode(realCode);
                        throw;
                    }
                }
                else if (code == TCL.CompletionCode.ERROR)
                {
                    /*
                    * Record information telling where the error occurred.
                    */

                    interp.AddErrorInfo("\n    (file line " + interp._errorLine + ")");
                    throw;
                }
                else
                {
                    throw;
                }
            }
            return TCL.CompletionCode.RETURN;
        }
    } // end SourceCmd
}
