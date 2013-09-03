#region Foreign-License
/*
	This file contains the abstract class declaration for the
	internal representations of TclObjects.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
namespace Tcl.Lang
{
    /// <summary>
    /// This is the interface for implementing internal representation of Tcl objects.  A class that implements InternalRep should define the following:
    /// (1) the two abstract methods specified in this base class: dispose(), duplicate()
    /// (2) The method toString()
    /// (3) class method(s) newInstance() if appropriate
    /// (4) class method set<Type>FromAny() if appropriate
    /// (5) class method get() if appropriate
    /// </summary>
    public interface IInternalRep
    {
        void Dispose();
        IInternalRep Duplicate();
    }
}
