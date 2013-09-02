#region Foreign-License
/*
The API for registering named data objects in the Tcl interpreter.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
namespace Tcl.Lang
{
    /// <summary> This interface is the API for registering named data objects in the
    /// Tcl interpreter.
    /// </summary>
    public interface IAssocData
    {
        void Dispose(Interp interp); // The interpreter in which this AssocData instance is registered in.
    }
}
