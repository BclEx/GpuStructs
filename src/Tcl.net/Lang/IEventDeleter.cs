#region Foreign-License
/*
Interface for deleting events in the notifier's event queue.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
namespace Tcl.Lang
{
    // This is the interface for deleting events in the notifier's event queue. It's used together with the Notifier.deleteEvents() method.
    public interface IEventDeleter
    {
        /// <summary>
        /// This method is called once for each event in the event queue. It returns true for all events that should be deleted and false for events that should remain in the queue.
        /// If this method determines that an event should be removed, it should perform appropriate clean up on the event object.
        /// Side effects: After this method returns 1, the event will be removed from the event queue and will not be processed.
        /// </summary>
        /// <param name="evt"></param>
        /// <returns>
        /// true means evt should be removed from the event queue.
        /// </returns>
        bool DeleteEvent(TclEvent evt); // Check whether this event should be removed.
    }
}
