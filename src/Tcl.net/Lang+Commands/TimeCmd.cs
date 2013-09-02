#region Foreign-License
/*
Copyright (c) 1997 Cornell University.
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /// <summary> This class implements the built-in "time" command in Tcl.</summary>

    class TimeCmd : ICommand
    {
        /// <summary> See Tcl user documentation for details.</summary>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            if ((argv.Length < 2) || (argv.Length > 3))
            {
                throw new TclNumArgsException(interp, 1, argv, "script ?count?");
            }

            int count;
            if (argv.Length == 2)
            {
                count = 1;
            }
            else
            {
                count = TclInteger.get(interp, argv[2]);
            }

            long startTime = System.DateTime.Now.Ticks;
            for (int i = 0; i < count; i++)
            {
                interp.eval(argv[1], 0);
            }
            long endTime = System.DateTime.Now.Ticks;
            long uSecs = (((endTime - startTime) / 10) / count);
            if (uSecs == 1)
            {
                interp.setResult(TclString.NewInstance("1 microsecond per iteration"));
            }
            else
            {
                interp.setResult(TclString.NewInstance(uSecs + " microseconds per iteration"));
            }
            return TCL.CompletionCode.RETURN;
        }
    }
}
