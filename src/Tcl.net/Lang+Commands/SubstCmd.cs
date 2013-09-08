#region Foreign-License
/*
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System.Text;
namespace Tcl.Lang
{

    /// <summary> This class implements the built-in "subst" command in Tcl.</summary>

    class SubstCmd : ICommand
    {
        private static readonly string[] validCmds = new string[] { "-nobackslashes", "-nocommands", "-novariables" };

        internal const int OPT_NOBACKSLASHES = 0;
        internal const int OPT_NOCOMMANDS = 1;
        internal const int OPT_NOVARS = 2;

        /// <summary> This procedure is invoked to process the "subst" Tcl command.
        /// See the user documentation for details on what it does.
        /// 
        /// </summary>
        /// <param name="interp">the current interpreter.
        /// </param>
        /// <param name="argv">command arguments.
        /// </param>
        /// <exception cref=""> TclException if wrong # of args or invalid argument(s).
        /// </exception>

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            int currentObjIndex, len, i;
            int objc = argv.Length - 1;
            bool doBackslashes = true;
            bool doCmds = true;
            bool doVars = true;
            StringBuilder result = new StringBuilder();
            string s;
            char c;

            for (currentObjIndex = 1; currentObjIndex < objc; currentObjIndex++)
            {

                if (!argv[currentObjIndex].ToString().StartsWith("-"))
                {
                    break;
                }
                int opt = TclIndex.Get(interp, argv[currentObjIndex], validCmds, "switch", 0);
                switch (opt)
                {

                    case OPT_NOBACKSLASHES:
                        doBackslashes = false;
                        break;

                    case OPT_NOCOMMANDS:
                        doCmds = false;
                        break;

                    case OPT_NOVARS:
                        doVars = false;
                        break;

                    default:
                        throw new TclException(interp, "SubstCmd.cmdProc: bad option " + opt + " index to cmds");

                }
            }
            if (currentObjIndex != objc)
            {
                throw new TclNumArgsException(interp, currentObjIndex, argv, "?-nobackslashes? ?-nocommands? ?-novariables? string");
            }

            /*
            * Scan through the string one character at a time, performing
            * command, variable, and backslash substitutions.
            */


            s = argv[currentObjIndex].ToString();
            len = s.Length;
            i = 0;
            while (i < len)
            {
                c = s[i];

                if ((c == '[') && doCmds)
                {
                    ParseResult res;
                    try
                    {
                        interp._evalFlags = Parser.TCL_BRACKET_TERM;
                        interp.Eval(s.Substring(i + 1, (len) - (i + 1)));
                        TclObject interp_result = interp.GetResult();
                        interp_result.Preserve();
                        res = new ParseResult(interp_result, i + interp._termOffset);
                    }
                    catch (TclException e)
                    {
                        i = e.errIndex + 1;
                        throw;
                    }
                    i = res.nextIndex + 2;

                    result.Append(res.Value.ToString());
                    res.Release();
                }
                else if (c == '\r')
                {
                    /*
                    * (ToDo) may not be portable on Mac
                    */

                    i++;
                }
                else if ((c == '$') && doVars)
                {
                    ParseResult vres = Parser.parseVar(interp, s.Substring(i, (len) - (i)));
                    i += vres.nextIndex;

                    result.Append(vres.Value.ToString());
                    vres.Release();
                }
                else if ((c == '\\') && doBackslashes)
                {
                    BackSlashResult bs = Tcl.Lang.Interp.backslash(s, i, len);
                    i = bs.NextIndex;
                    if (bs.IsWordSep)
                    {
                        break;
                    }
                    else
                    {
                        result.Append(bs.C);
                    }
                }
                else
                {
                    result.Append(c);
                    i++;
                }
            }

            interp.SetResult(result.ToString());
            return TCL.CompletionCode.RETURN;
        }
    }
}
