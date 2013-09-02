#undef DEBUG
#region Foreign-License
/*
	This class stores debug information for the interpreter.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /*
    * This class stores debug information for the interpreter.
    */

    public class DebugInfo
    {

        /*
        * The name of the source file that contains code for a given debug
        * stack level. May be null for an unknown source file (if the debug
        * stack is activated by an "eval" command or if the Interp is running
        * in non-debugging mode.)
        */

        internal string fileName;

        /*
        * The beginning line of the current command under execution.
        * 1 means the first line inside a file. 0 means the line number is
        * unknown.
        */

        internal int cmdLine;

        internal DebugInfo(string fname, int line)
        {
            fileName = fname;
            cmdLine = line;
        }
    } // end DebugInfo
}
