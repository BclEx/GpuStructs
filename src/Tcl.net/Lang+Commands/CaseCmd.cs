#region Foreign-License
/*
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /// <summary> This class implements the built-in "case" command in Tcl.</summary>

    class CaseCmd : ICommand
    {
        /// <summary> Executes a "case" statement. See Tcl user
        /// documentation for details.
        /// 
        /// </summary>
        /// <param name="interp">the current interpreter.
        /// </param>
        /// <param name="argv">command arguments.
        /// </param>
        /// <exception cref=""> TclException If incorrect number of arguments.
        /// </exception>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            if (argv.Length < 3)
            {
                throw new TclNumArgsException(interp, 1, argv, "string ?in? patList body ... ?default body?");
            }

            int i;
            int body;
            TclObject[] caseArgv;
            string inString;


            inString = argv[1].ToString();
            caseArgv = argv;
            body = -1;


            if (argv[2].ToString().Equals("in"))
            {
                i = 3;
            }
            else
            {
                i = 2;
            }

            /*
            * If all of the pattern/command pairs are lumped into a single
            * argument, split them out again.
            */

            if (argv.Length - i == 1)
            {
                caseArgv = TclList.getElements(interp, argv[i]);
                i = 0;
            }

            {
                for (; i < caseArgv.Length; i += 2)
                {
                    int j;

                    if (i == (caseArgv.Length - 1))
                    {
                        throw new TclException(interp, "extra case pattern with no body");
                    }

                    /*
                    * Check for special case of single pattern (no list) with
                    * no backslash sequences.
                    */


                    string caseString = caseArgv[i].ToString();
                    int len = caseString.Length;
                    for (j = 0; j < len; j++)
                    {
                        char c = caseString[j];
                        if (System.Char.IsWhiteSpace(c) || (c == '\\'))
                        {
                            break;
                        }
                    }
                    if (j == len)
                    {
                        if (caseString.Equals("default"))
                        {
                            body = i + 1;
                        }
                        if (Util.StringMatch(inString, caseString))
                        {
                            body = i + 1;
                            goto match_loop_brk;
                        }
                        continue;
                    }

                    /*
                    * Break up pattern lists, then check each of the patterns
                    * in the list.
                    */

                    int numPats = TclList.getLength(interp, caseArgv[i]);
                    for (j = 0; j < numPats; j++)
                    {

                        if (Util.StringMatch(inString, TclList.index(interp, caseArgv[i], j).ToString()))
                        {
                            body = i + 1;
                            goto match_loop_brk;
                        }
                    }
                }
            }

        match_loop_brk:
            ;


            if (body != -1)
            {
                try
                {
                    interp.Eval(caseArgv[body], 0);
                }
                catch (TclException e)
                {
                    if (e.GetCompletionCode() == TCL.CompletionCode.ERROR)
                    {

                        interp.AddErrorInfo("\n    (\"" + caseArgv[body - 1] + "\" arm line " + interp._errorLine + ")");
                    }
                    throw;
                }
            }
            return TCL.CompletionCode.RETURN;
        }
    }
}
