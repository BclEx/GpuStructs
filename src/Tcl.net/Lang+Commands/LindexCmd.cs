#region Foreign-License
/*
	Implements the built-in "lindex" Tcl command.

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
    * This class implements the built-in "lindex" command in Tcl.
    */

    class LindexCmd : ICommand
    {

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            if (argv.Length < 3)
            {
                throw new TclNumArgsException(interp, 1, argv, "list index");
            }

            int size = TclList.getLength(interp, argv[1]);
            int index = Util.getIntForIndex(interp, argv[2], size - 1);
            TclObject element = TclList.index(interp, argv[1], index);

            if (element != null)
            {
                interp.setResult(element);
            }
            else
            {
                interp.ResetResult();
            }
            return TCL.CompletionCode.RETURN;
        }
    } // end 
}
