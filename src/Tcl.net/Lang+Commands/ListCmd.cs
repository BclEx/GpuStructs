#region Foreign-License
/*
Copyright (c) 1997 Cornell University.
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
using System.Text;

namespace Tcl.Lang
{

    /// <summary> This class implements the built-in "list" command in Tcl.</summary>
    class ListCmd : Command
    {

        /// <summary> See Tcl user documentation for details.</summary>
        public TCL.CompletionCode cmdProc(Interp interp, TclObject[] argv)
        {
            TclObject list = TclList.newInstance();

            list.preserve();
            try
            {
                for (int i = 1; i < argv.Length; i++)
                    TclList.append(interp, list, argv[i]);
                interp.setResult(list);
            }
            finally
            {
                list.release();
            }
            return TCL.CompletionCode.RETURN;
        }
    }
}
