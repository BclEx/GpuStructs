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

    /// <summary> This class implements the built-in "while" command in Tcl.</summary>

    class WhileCmd : ICommand
    {
        /// <summary> This procedure is invoked to process the "while" Tcl command.
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
            if (argv.Length != 3)
            {
                throw new TclNumArgsException(interp, 1, argv, "test command");
            }

            string test = argv[1].ToString();
            TclObject command = argv[2];

            {
                while (interp._expr.EvalBoolean(interp, test))
                {
                    try
                    {
                        interp.eval(command, 0);
                    }
                    catch (TclException e)
                    {
                        switch (e.GetCompletionCode())
                        {

                            case TCL.CompletionCode.BREAK:
                                goto loop_brk;


                            case TCL.CompletionCode.CONTINUE:
                                continue;


                            case TCL.CompletionCode.ERROR:
                                interp.AddErrorInfo("\n    (\"while\" body line " + interp._errorLine + ")");
                                throw;


                            default:
                                throw;

                        }
                    }
                }
            }

        loop_brk:
            ;


            interp.ResetResult();
            return TCL.CompletionCode.RETURN;
        }
    }
}
