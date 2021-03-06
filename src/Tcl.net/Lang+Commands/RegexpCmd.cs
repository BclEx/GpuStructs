#region Foreign-License
/*
	This file contains the Jacl implementation of the built-in Tcl "regexp" command. 

Copyright (c) 1997-1999 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
using Core;
namespace Tcl.Lang
{

    /// <summary> This class implements the built-in "regexp" command in Tcl.</summary>

    class RegexpCmd : ICommand
    {

        private static readonly string[] validOpts = new string[] { "-indices", "-nocase", "--" };
        private const int OPT_INDICES = 0;
        private const int OPT_NOCASE = 1;
        private const int OPT_LAST = 2;
        internal static void Init(Interp interp)
        // Current interpreter. 
        {
            interp.CreateCommand("regexp", new Tcl.Lang.RegexpCmd());
            interp.CreateCommand("regsub", new Tcl.Lang.RegsubCmd());
        }
        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            bool nocase = false;
            bool indices = false;

            try
            {
                int i = 1;

                while (argv[i].ToString().StartsWith("-"))
                {
                    int index = TclIndex.Get(interp, argv[i], validOpts, "switch", 0);
                    i++;
                    switch (index)
                    {

                        case OPT_INDICES:
                            {
                                indices = true;
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


                TclObject exp = TclString.NewInstance(argv[i++].ToString().Replace("\\d", "[0-9]"));

                string inString = argv[i++].ToString();

                int matches = argv.Length - i;

                Regexp r = TclRegexp.compile(interp, exp, nocase);

                int[] args = new int[matches * 2];
                bool matched = r.match(inString, args);
                if (matched)
                {
                    for (int match = 0; i < argv.Length; i++)
                    {
                        TclObject obj;

                        int start = args[match++];
                        int end = args[match++];
                        if (indices)
                        {
                            if (end >= 0)
                            {
                                end--;
                            }
                            obj = TclList.NewInstance();
                            TclList.Append(interp, obj, TclInteger.NewInstance(start));
                            TclList.Append(interp, obj, TclInteger.NewInstance(end));
                        }
                        else
                        {
                            string range = (start >= 0) ? inString.Substring(start, (end) - (start)) : "";
                            obj = TclString.NewInstance(range);
                        }
                        try
                        {

                            interp.SetVar(argv[i].ToString(), obj, 0);
                        }
                        catch (TclException e)
                        {

                            throw new TclException(interp, "couldn't set variable \"" + argv[i] + "\"");
                        }
                    }
                }
                interp.SetResult(matched);
            }
            catch (System.IndexOutOfRangeException e)
            {
                throw new TclNumArgsException(interp, 1, argv, "?switches? exp string ?matchVar? ?subMatchVar subMatchVar ...?");
            }
            return TCL.CompletionCode.RETURN;
        }
    } // end RegexpCmd
}
