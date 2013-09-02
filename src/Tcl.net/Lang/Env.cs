#undef DEBUG
#region Foreign-License
/*
	This class is used to create and manage the environment array
	used by the Tcl interpreter.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
using System.Collections;

namespace Tcl.Lang
{

    /// <summary> This class manages the environment array for Tcl interpreters.</summary>

    class Env
    {

        /*
        *----------------------------------------------------------------------
        *
        * initialize --
        *
        *	This method is called to initialize an interpreter with it's 
        *	initial values for the env array.
        *
        * Results:
        *	None.
        *
        * Side effects:
        *	The env array in the interpreter is created and populated.
        *
        *----------------------------------------------------------------------
        */

        internal static void initialize(Interp interp)
        {
            // For a few standrad environment vairables that Tcl users 
            // often assume aways exist (even if they shouldn't), we will
            // try to create those expected variables with the common unix
            // names.

            try
            {
                interp.setVar("env", "HOME", System.Environment.CurrentDirectory, TCL.VarFlag.GLOBAL_ONLY);
            }
            catch (TclException e)
            {
                // Ignore errors.
            }

            try
            {
                interp.setVar("env", "USER", System.Environment.UserName, TCL.VarFlag.GLOBAL_ONLY);
            }
            catch (TclException e)
            {
                // Ignore errors.
            }

            // Now we will populate the rest of the env array with the
            // properties recieved from the System classes.  This makes for 
            // a nice shortcut for getting to these useful values.

            try
            {


                for (IDictionaryEnumerator search = System.Environment.GetEnvironmentVariables().GetEnumerator(); search.MoveNext(); )
                {
                    interp.setVar("env", search.Key.ToString(), search.Value.ToString(), TCL.VarFlag.GLOBAL_ONLY);
                }
            }
            catch (System.Security.SecurityException e2)
            {
                // We are inside a browser and we can't access the list of
                // property names. That's fine. Life goes on ....
            }
            catch (System.Exception e3)
            {
                // We are inside a browser and we can't access the list of
                // property names. That's fine. Life goes on ....

                System.Diagnostics.Debug.WriteLine("Exception while initializing env array");
                System.Diagnostics.Debug.WriteLine(e3);
                System.Diagnostics.Debug.WriteLine("");
            }
        }
    } // end Env
}
