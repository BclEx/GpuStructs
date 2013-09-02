#region Foreign-License
/*
Copyright (c) 1997 Cornell University.
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
namespace Tcl.Lang
{
    class BackSlashResult
    {
        internal char C;
        internal int NextIndex;
        internal bool IsWordSep;
        internal BackSlashResult(char c, int w)
        {
            C = c;
            NextIndex = w;
            IsWordSep = false;
        }
        internal BackSlashResult(char c, int w, bool b)
        {
            C = c;
            NextIndex = w;
            IsWordSep = b;
        }
    }
}
