#region Foreign-License
/*
	This file implements the Tcl "vwait" command.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /*
    * This class implements the built-in "vwait" command in Tcl.
    */

    class VwaitCmd : ICommand
    {

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            if (argv.Length != 2)
            {
                throw new TclNumArgsException(interp, 1, argv, "name");
            }

            VwaitTrace trace = new VwaitTrace();
            Var.TraceVar(interp, argv[1], TCL.VarFlag.GLOBAL_ONLY | TCL.VarFlag.TRACE_WRITES | TCL.VarFlag.TRACE_UNSETS, trace);

            int foundEvent = 1;
            while (!trace.done && (foundEvent != 0))
            {
                foundEvent = interp.GetNotifier().doOneEvent(TCL.ALL_EVENTS);
            }

            Var.UntraceVar(interp, argv[1], TCL.VarFlag.GLOBAL_ONLY | TCL.VarFlag.TRACE_WRITES | TCL.VarFlag.TRACE_UNSETS, trace);

            // Clear out the interpreter's result, since it may have been set
            // by event handlers.

            interp.ResetResult();

            if (foundEvent == 0)
            {

                throw new TclException(interp, "can't wait for variable \"" + argv[1] + "\":  would wait forever");
            }
            return TCL.CompletionCode.RETURN;
        }
    } // end VwaitCmd

    class VwaitTrace : VarTrace
    {

        /*
        * TraceCmd.cmdProc continuously watches this variable across calls to
        * doOneEvent(). It returns immediately when done is set to true.
        */

        internal bool done = false;

        public void traceProc(Interp interp, string part1, string part2, TCL.VarFlag flags)
        // Mode flags: Should only be TCL.VarFlag.TRACE_WRITES.
        {
            done = true;
        }
    } // end VwaitTrace
}
