#undef DEBUG
#region Foreign-License
/*
	Implements the built-in "set" Tcl command.

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
    * This class implements the built-in "set" command in Tcl.
    */

    class SetCmd : ICommand
    {
        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            bool debug;

            if (argv.Length == 2)
            {
                System.Diagnostics.Debug.WriteLine("getting value of \"" + argv[1].ToString() + "\"");

                interp.setResult(interp.getVar(argv[1], 0));
            }
            else if (argv.Length == 3)
            {
                System.Diagnostics.Debug.WriteLine("setting value of \"" + argv[1].ToString() + "\" to \"" + argv[2].ToString() + "\"");
                interp.setResult(interp.SetVar(argv[1], argv[2], 0));
            }
            else
            {
                throw new TclNumArgsException(interp, 1, argv, "varName ?newValue?");
            }
            return TCL.CompletionCode.RETURN;
        }
    } // end SetCmd
}
