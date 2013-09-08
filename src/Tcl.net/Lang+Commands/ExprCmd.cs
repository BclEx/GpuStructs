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

    /// <summary> This class implements the built-in "expr" command in Tcl.</summary>

    class ExprCmd : ICommand
    {
        /// <summary> Evaluates a Tcl expression. See Tcl user documentation for
        /// details.
        /// </summary>
        /// <exception cref=""> TclException If malformed expression.
        /// </exception>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            if (argv.Length < 2)
            {
                throw new TclNumArgsException(interp, 1, argv, "arg ?arg ...?");
            }

            if (argv.Length == 2)
            {

                interp.SetResult(interp._expr.eval(interp, argv[1].ToString()));
            }
            else
            {
                StringBuilder sbuf = new StringBuilder();

                sbuf.Append(argv[1].ToString());
                for (int i = 2; i < argv.Length; i++)
                {
                    sbuf.Append(' ');

                    sbuf.Append(argv[i].ToString());
                }
                interp.SetResult(interp._expr.eval(interp, sbuf.ToString()));
            }
            return TCL.CompletionCode.RETURN;
        }
    }
}
