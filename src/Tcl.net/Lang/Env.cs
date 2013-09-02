#undef DEBUG
#region Foreign-License
/*
This class is used to create and manage the environment array used by the Tcl interpreter.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
using System.Collections;
using System.Diagnostics;
using System.Security;
namespace Tcl.Lang
{
    /// <summary>
    /// This class manages the environment array for Tcl interpreters.
    /// </summary>
    class Env
    {
        /// <summary>
        /// This method is called to initialize an interpreter with it's  initial values for the env array.
        /// Side effects: The env array in the interpreter is created and populated.
        /// </summary>
        /// <param name="interp"></param>
        internal static void initialize(Interp interp)
        {
            // For a few standrad environment vairables that Tcl users  often assume aways exist (even if they shouldn't), we will
            // try to create those expected variables with the common unix names.
            try { interp.SetVar("env", "HOME", Environment.CurrentDirectory, TCL.VarFlag.GLOBAL_ONLY); }
            catch (TclException) { } // Ignore errors.
            try { interp.SetVar("env", "USER", Environment.UserName, TCL.VarFlag.GLOBAL_ONLY); }
            catch (TclException) { } // Ignore errors.
            // Now we will populate the rest of the env array with the properties recieved from the System classes.  This makes for 
            // a nice shortcut for getting to these useful values.
            try
            {
                for (IDictionaryEnumerator search = Environment.GetEnvironmentVariables().GetEnumerator(); search.MoveNext(); )
                    interp.SetVar("env", search.Key.ToString(), search.Value.ToString(), TCL.VarFlag.GLOBAL_ONLY);
            }
            // We are inside a browser and we can't access the list of property names. That's fine. Life goes on ....
            catch (SecurityException) { }
            // We are inside a browser and we can't access the list of property names. That's fine. Life goes on ....
            catch (Exception e3)
            {
                Debug.WriteLine("Exception while initializing env array");
                Debug.WriteLine(e3);
                Debug.WriteLine(string.Empty);
            }
        }
    }
}
