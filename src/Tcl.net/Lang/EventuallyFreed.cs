#region Foreign-License
/*
	This class makes sure that certain objects
	aren't disposed when there are nested procedures that
	depend on their existence.

Copyright (c) 1991-1994 The Regents of the University of California.
Copyright (c) 1994-1998 Sun Microsystems, Inc.
Copyright (c) 2000 Christian Krone.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    public abstract class EventuallyFreed
    {

        // Number of preserve() calls in effect for this object.

        internal int refCount = 0;

        // True means dispose() was called while a preserve()
        // call was in effect, so the object must be disposed
        // when refCount becomes zero.

        internal bool mustFree = false;

        // Procedure to call to dispose.

        public abstract void EventuallyDispose();
        internal void preserve()
        {
            // Just increment its reference count.

            refCount++;
        }
        internal void release()
        {
            refCount--;
            if (refCount == 0)
            {

                if (mustFree)
                {
                    Dispose();
                }
            }
        }
        public void Dispose()
        {
            // See if there is a reference for this pointer.  If so, set its
            // "mustFree" flag (the flag had better not be set already!).

            if (refCount >= 1)
            {
                if (mustFree)
                {
                    throw new TclRuntimeError("eventuallyDispose() called twice");
                }
                mustFree = true;
                return;
            }

            // No reference for this block.  Free it now.

            EventuallyDispose();
        }
    } // end EventuallyFreed
}
