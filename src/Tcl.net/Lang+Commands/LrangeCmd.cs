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

    /// <summary> This class implements the built-in "lrange" command in Tcl.</summary>

    class LrangeCmd : ICommand
    {
        /// <summary> See Tcl user documentation for details.</summary>
        /// <exception cref=""> TclException If incorrect number of arguments.
        /// </exception>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            if (argv.Length != 4)
            {
                throw new TclNumArgsException(interp, 1, argv, "list first last");
            }

            int size = TclList.getLength(interp, argv[1]);
            int first;
            int last;

            first = Util.getIntForIndex(interp, argv[2], size - 1);
            last = Util.getIntForIndex(interp, argv[3], size - 1);

            if (last < 0)
            {
                interp.ResetResult();
                return TCL.CompletionCode.RETURN;
            }
            if (first >= size)
            {
                interp.ResetResult();
                return TCL.CompletionCode.RETURN;
            }
            if (first <= 0 && last >= size)
            {
                interp.SetResult(argv[1]);
                return TCL.CompletionCode.RETURN;
            }

            if (first < 0)
            {
                first = 0;
            }
            if (first >= size)
            {
                first = size - 1;
            }
            if (last < 0)
            {
                last = 0;
            }
            if (last >= size)
            {
                last = size - 1;
            }
            if (first > last)
            {
                interp.ResetResult();
                return TCL.CompletionCode.RETURN;
            }

            TclObject list = TclList.NewInstance();

            list.Preserve();
            try
            {
                for (int i = first; i <= last; i++)
                {
                    TclList.Append(interp, list, TclList.index(interp, argv[1], i));
                }
                interp.SetResult(list);
            }
            finally
            {
                list.Release();
            }
            return TCL.CompletionCode.RETURN;
        }
    }
}
