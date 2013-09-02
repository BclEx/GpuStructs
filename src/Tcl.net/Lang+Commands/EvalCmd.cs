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

    /// <summary> This class implements the built-in "eval" command in Tcl.</summary>

    class EvalCmd : ICommand
    {
        /// <summary> This procedure is invoked to process the "eval" Tcl command.
        /// See the user documentation for details on what it does.
        /// 
        /// </summary>
        /// <param name="interp">the current interpreter.
        /// </param>
        /// <param name="argv">command arguments.
        /// </param>
        /// <exception cref=""> TclException if script causes error.
        /// </exception>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            if (argv.Length < 2)
            {
                throw new TclNumArgsException(interp, 1, argv, "arg ?arg ...?");
            }

            try
            {
                if (argv.Length == 2)
                {
                    interp.eval(argv[1], 0);
                }
                else
                {
                    string s = Util.concat(1, argv.Length - 1, argv);
                    interp.eval(s, 0);
                }
            }
            catch (TclException e)
            {
                if (e.GetCompletionCode() == TCL.CompletionCode.ERROR)
                {
                    interp.AddErrorInfo("\n    (\"eval\" body line " + interp.errorLine + ")");
                }
                throw;
            }
            return TCL.CompletionCode.RETURN;
        }
    }
}
