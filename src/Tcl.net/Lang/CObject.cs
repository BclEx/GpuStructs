#region Foreign-License
/*
A stub class that represents objects created by the NativeTcl interpreter.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
namespace Tcl.Lang
{
    /// <summary>
    /// This is a stub class used in Jacl to represent objects created in the Tcl Blend interpreter. Actually CObjects will never appear inside
    /// Jacl. However, since TclObject (which is shared between the Tcl Blend and Jacl implementations) makes some references to CObject, we include
    /// a stub class here to make the compiler happy.
    /// 
    /// None of the methods in this implementation will ever be called.
    /// </summary>
    class CObject : InternalRep
    {
        public long CObjectPtr;

        public void Dispose()
        {
            throw new TclRuntimeError("This shouldn't be called");
        }

        public InternalRep Duplicate()
        {
            throw new TclRuntimeError("This shouldn't be called");
        }

        internal void MakeReference(TclObject tobj)
        {
            throw new TclRuntimeError("This shouldn't be called");
        }

        public override string ToString()
        {
            throw new TclRuntimeError("This shouldn't be called");
        }

        public void DecrRefCount() { }
        public void IncrRefCount() { }
    }
}
