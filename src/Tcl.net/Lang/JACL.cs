#region Foreign-License
/*
	This class stores all the Jacl-specific package protected constants.
	The exact values should match those in tcl.h.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /*
    * This class holds all the Jacl-specific package protected constants.
    */

    public class JACL
    {

        /*
        * Platform constants.  PLATFORM is not final because we may change it for
        * testing purposes only.
        */

        public const int PLATFORM_UNIX = 0;
        public const int PLATFORM_WINDOWS = 1;
        public const int PLATFORM_MAC = 2;
        public static int PLATFORM;
        static JACL()
        {
            PLATFORM = Util.ActualPlatform;
        }
    } // end JACL class
}
