#region Foreign-License
/*
Copyright (c) 1997 Cornell University.
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System.Text;

namespace Tcl.Lang
{
    /// <summary>
    /// This class implements the built-in "join" command in Tcl.
    /// </summary>
    class JoinCmd : ICommand
    {

        /// <summary> See Tcl user documentation for details.</summary>
        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            string sep = null;

            if (argv.Length == 2)
            {
                sep = null;
            }
            else if (argv.Length == 3)
            {

                sep = argv[2].ToString();
            }
            else
            {
                throw new TclNumArgsException(interp, 1, argv, "list ?joinString?");
            }
            TclObject list = argv[1];
            int size = TclList.getLength(interp, list);

            if (size == 0)
            {
                interp.ResetResult();
                return TCL.CompletionCode.RETURN;
            }


            StringBuilder sbuf = new StringBuilder(TclList.index(interp, list, 0).ToString());

            for (int i = 1; i < size; i++)
            {
                if ((System.Object)sep == null)
                {
                    sbuf.Append(' ');
                }
                else
                {
                    sbuf.Append(sep);
                }

                sbuf.Append(TclList.index(interp, list, i).ToString());
            }
            interp.SetResult(sbuf.ToString());
            return TCL.CompletionCode.RETURN;
        }
    }
}
