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

    /// <summary> This class implements the built-in "Lset" command in Tcl.</summary>

    class LsetCmd : ICommand
    {
        /// <summary> See Tcl user documentation for details.</summary>
        /// <exception cref=""> TclException If incorrect number of arguments.
        /// </exception>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            if (argv.Length != 4)
            {
                throw new TclNumArgsException(interp, 1, argv, "lset list index element");
            }

            int size = TclList.getLength(interp, argv[1]);
            int index = Util.getIntForIndex(interp, argv[2], size);
            TclObject list = argv[1];
            bool isDuplicate = false;

            // If the list object is unshared we can modify it directly. Otherwise
            // we create a copy to modify: this is "copy on write".

            if (list.Shared)
            {
                list = list.duplicate();
                isDuplicate = true;
            }

            try
            {
                TclObject[] replace = new TclObject[1];
                replace[0] = argv[3];
                TclList.replace(interp, list, index, 1, replace, 0, 0);
                interp.SetResult(list);
            }
            catch (TclException e)
            {
                if (isDuplicate)
                {
                    list.Release();
                }
                throw;
            }
            return TCL.CompletionCode.RETURN;
        }
    }
}
