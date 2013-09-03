#region Foreign-License
/*
	Implements the "update" command.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /*
    * This class implements the built-in "update" command in Tcl.
    */

    class UpdateCmd : ICommand
    {

        /*
        * Valid command options.
        */

        private static readonly string[] validOpts = new string[] { "idletasks" };

        internal const int OPT_IDLETASKS = 0;

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            int flags;

            if (argv.Length == 1)
            {
                flags = TCL.ALL_EVENTS | TCL.DONT_WAIT;
            }
            else if (argv.Length == 2)
            {
                TclIndex.get(interp, argv[1], validOpts, "option", 0);

                /*
                * Since we just have one valid option, if the above call returns
                * without an exception, we've got "idletasks" (or abreviations).
                */

                flags = TCL.IDLE_EVENTS | TCL.DONT_WAIT;
            }
            else
            {
                throw new TclNumArgsException(interp, 1, argv, "?idletasks?");
            }

            while (interp.getNotifier().doOneEvent(flags) != 0)
            {
                /* Empty loop body */
            }

            /*
            * Must clear the interpreter's result because event handlers could
            * have executed commands.
            */

            interp.ResetResult();
            return TCL.CompletionCode.RETURN;
        }
    } // end UpdateCmd
}
