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

    /// <summary> This class implements the built-in "if" command in Tcl.</summary>
    class IfCmd : ICommand
    {

        /// <summary> See Tcl user documentation for details.</summary>
        /// <exception cref=""> TclException If incorrect number of arguments.
        /// </exception>
        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            int i;
            bool value;

            i = 1;
            while (true)
            {
                /*
                * At this point in the loop, argv and argc refer to an
                * expression to test, either for the main expression or
                * an expression following an "elseif".  The arguments
                * after the expression must be "then" (optional) and a
                * script to execute if the expression is true.
                */

                if (i >= argv.Length)
                {

                    throw new TclException(interp, "wrong # args: no expression after \"" + argv[i - 1] + "\" argument");
                }
                try
                {

                    value = interp._expr.EvalBoolean(interp, argv[i].ToString());
                }
                catch (TclException e)
                {
                    switch (e.GetCompletionCode())
                    {

                        case TCL.CompletionCode.ERROR:
                            interp.AddErrorInfo("\n    (\"if\" test expression)");
                            break;
                    }
                    throw;
                }

                i++;

                if ((i < argv.Length) && (argv[i].ToString().Equals("then")))
                {
                    i++;
                }
                if (i >= argv.Length)
                {

                    throw new TclException(interp, "wrong # args: no script following \"" + argv[i - 1] + "\" argument");
                }
                if (value)
                {
                    try
                    {
                        interp.Eval(argv[i], 0);
                    }
                    catch (TclException e)
                    {
                        switch (e.GetCompletionCode())
                        {

                            case TCL.CompletionCode.ERROR:
                                interp.AddErrorInfo("\n    (\"if\" then script line " + interp._errorLine + ")");
                                break;
                        }
                        throw;
                    }
                    return TCL.CompletionCode.RETURN;
                }

                /*
                * The expression evaluated to false.  Skip the command, then
                * see if there is an "else" or "elseif" clause.
                */

                i++;
                if (i >= argv.Length)
                {
                    interp.ResetResult();
                    return TCL.CompletionCode.RETURN;
                }

                if (argv[i].ToString().Equals("elseif"))
                {
                    i++;
                    continue;
                }
                break;
            }

            /*
            * Couldn't find a "then" or "elseif" clause to execute.
            * Check now for an "else" clause.  We know that there's at
            * least one more argument when we get here.
            */


            if (argv[i].ToString().Equals("else"))
            {
                i++;
                if (i >= argv.Length)
                {
                    throw new TclException(interp, "wrong # args: no script following \"else\" argument");
                }
                else if (i != (argv.Length - 1))
                {
                    throw new TclException(interp, "wrong # args: extra words after \"else\" clause in " + "\"if\" command");
                }
            }
            try
            {
                interp.Eval(argv[i], 0);
            }
            catch (TclException e)
            {
                switch (e.GetCompletionCode())
                {

                    case TCL.CompletionCode.ERROR:
                        interp.AddErrorInfo("\n    (\"if\" else script line " + interp._errorLine + ")");
                        break;
                }
                throw;
            }
            return TCL.CompletionCode.RETURN;
        }
    }
}
