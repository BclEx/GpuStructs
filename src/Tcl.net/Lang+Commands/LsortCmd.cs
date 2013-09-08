#region Foreign-License
/*
	The file implements the Tcl "lsort" command.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /*
    * This LsortCmd class implements the Command interface for specifying a new
    * Tcl command.  The Lsort command implements the built-in Tcl command "lsort"
    * which is used to sort Tcl lists.  See user documentation for more details.
    */

    class LsortCmd : ICommand
    {

        /*
        * List of switches that are legal in the lsort command.
        */

        private static readonly string[] validOpts = new string[] { "-ascii", "-command", "-decreasing", "-dictionary", "-increasing", "-index", "-integer", "-real", "-unique" };

        /*
        *----------------------------------------------------------------------
        *
        * cmdProc --
        *
        *	This procedure is invoked as part of the Command interface to 
        *	process the "lsort" Tcl command.  See the user documentation for
        *	details on what it does.
        *
        * Results:
        *	A standard Tcl result.
        *
        * Side effects:
        *	See the user documentation.
        *
        *----------------------------------------------------------------------
        */

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            if (argv.Length < 2)
            {
                throw new TclNumArgsException(interp, 1, argv, "?options? list");
            }

            string command = null;
            int sortMode = QSort.ASCII;
            int sortIndex = -1;
            bool sortIncreasing = true;
            bool unique = false;

            for (int i = 1; i < argv.Length - 1; i++)
            {
                int index = TclIndex.Get(interp, argv[i], validOpts, "option", 0);

                switch (index)
                {

                    case 0:
                        sortMode = QSort.ASCII;
                        break;


                    case 1:
                        if (i == argv.Length - 2)
                        {
                            throw new TclException(interp, "\"-command\" option must be" + " followed by comparison command");
                        }
                        sortMode = QSort.COMMAND;

                        command = argv[i + 1].ToString();
                        i++;
                        break;


                    case 2:
                        sortIncreasing = false;
                        break;


                    case 3:
                        sortMode = QSort.DICTIONARY;
                        break;


                    case 4:
                        sortIncreasing = true;
                        break;


                    case 5:
                        if (i == argv.Length - 2)
                        {
                            throw new TclException(interp, "\"-index\" option must be followed by list index");
                        }
                        sortIndex = Util.getIntForIndex(interp, argv[i + 1], -2);

                        command = argv[i + 1].ToString();
                        i++;
                        break;


                    case 6:
                        sortMode = QSort.INTEGER;
                        break;


                    case 7:
                        sortMode = QSort.REAL;
                        break;

                    case 8:		/* -unique */
                        unique = true;
                        break;
                }
            }

            TclObject list = argv[argv.Length - 1];
            bool isDuplicate = false;

            // If the list object is unshared we can modify it directly. Otherwise
            // we create a copy to modify: this is "copy on write".

            if (list.Shared)
            {
                list = list.duplicate();
                isDuplicate = true;
            }

            try
            {
                TclList.sort(interp, list, sortMode, sortIndex, sortIncreasing, command, unique);
                interp.SetResult(list);
            }
            catch (TclException e)
            {
                if (isDuplicate)
                {
                    list.Release();
                }
                throw;
            }
            return TCL.CompletionCode.RETURN;
        }
    } // LsortCmd
}
