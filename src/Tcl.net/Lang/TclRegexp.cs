#region Foreign-License
/*
Copyright (c) 1999 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
using Core;
namespace Tcl.Lang
{
    public class TclRegexp
    {
        private TclRegexp() { }

        public static Regexp compile(Interp interp, TclObject exp, bool nocase)
        {
            try
            {
                return new Regexp(exp.ToString(), nocase);
            }
            catch (ArgumentException e)
            {
                string msg = e.Message;
                if (msg.Equals("missing )"))
                {
                    msg = "unmatched ()";
                }
                else if (msg.Equals("missing ]"))
                {
                    msg = "unmatched []";
                }
                msg = "couldn't compile regular expression pattern: " + msg;
                throw new TclException(interp, msg);
            }
        }
    }
}
