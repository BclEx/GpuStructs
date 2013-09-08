#region Foreign-License
/*
	This contains the Jacl implementation of the built-in Tcl "regsub" command.

Copyright (c) 1997-1999 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System.Text;
using Core;
namespace Tcl.Lang
{

    /// <summary> This class implements the built-in "regsub" command in Tcl.</summary>

    class RegsubCmd : ICommand
    {

        private static readonly string[] validOpts = new string[] { "-all", "-nocase", "--" };
        private const int OPT_ALL = 0;
        private const int OPT_NOCASE = 1;
        private const int OPT_LAST = 2;
        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            bool all = false;
            bool nocase = false;

            try
            {
                int i = 1;

                while (argv[i].ToString().StartsWith("-"))
                {
                    int index = TclIndex.Get(interp, argv[i], validOpts, "switch", 0);
                    i++;
                    switch (index)
                    {

                        case OPT_ALL:
                            {
                                all = true;
                                break;
                            }

                        case OPT_NOCASE:
                            {
                                nocase = true;
                                break;
                            }

                        case OPT_LAST:
                            {
                                goto opts_brk;
                            }
                    }
                }

            opts_brk:
                ;


                TclObject exp = argv[i++];

                string inString = argv[i++].ToString();

                string subSpec = argv[i++].ToString();

                string varName = null;
                if (i != argv.Length) varName = argv[i++].ToString();
                if (i != argv.Length)
                {
                    throw new System.IndexOutOfRangeException();
                }

                Regexp r = TclRegexp.compile(interp, exp, nocase);

                int count = 0;
                string result;

                if (all == false)
                {
                    result = r.sub(inString, subSpec);
                    if ((System.Object)result == null)
                    {
                        result = inString;
                    }
                    else
                    {
                        count++;
                    }
                }
                else
                {
                    StringBuilder sb = new StringBuilder();
                    Regsub s = new Regsub(r, inString);
                    while (s.nextMatch())
                    {
                        count++;
                        sb.Append(s.skipped());
                        Regexp.applySubspec(s, subSpec, sb);
                    }
                    sb.Append(s.rest());
                    result = sb.ToString();
                }

                TclObject obj = TclString.NewInstance(result);
                if (varName == null)
                    interp.SetResult(result);
                else
                {
                    try
                    {
                        interp.SetVar(varName, obj, 0);
                    }
                    catch (TclException e)
                    {
                        throw new TclException(interp, "couldn't set variable \"" + varName + "\"");
                    }
                    interp.SetResult(count);
                }
            }
            catch (System.IndexOutOfRangeException e)
            {
                throw new TclNumArgsException(interp, 1, argv, "?switches? exp string subSpec ?varName?");
            }
            return TCL.CompletionCode.RETURN;
        }
    } // end RegsubCmd
}
