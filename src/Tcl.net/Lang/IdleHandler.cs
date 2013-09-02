#region Foreign-License
/*
	The API for defining idle event handler.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System.Threading;
namespace Tcl.Lang
{
    /// <summary>
    /// This abstract class is used to define idle handlers.
    /// </summary>
    public abstract class IdleHandler
    {
        private Notifier _notifier; // Back pointer to the notifier that will fire this idle.
        private bool _isCancelled; // isCancelled True if the cancel() method has been called.
        internal int Generation; // Used to distinguish older idle handlers from recently-created ones.

        public IdleHandler(Notifier notifier)
        {
            _notifier = notifier;
            _isCancelled = false;
            lock (_notifier)
            {
                _notifier.IdleList.Add(this);
                Generation = _notifier.IdleGeneration;
                if (Thread.CurrentThread != _notifier.PrimaryThread)
                    Monitor.PulseAll(_notifier);
            }
        }

        public void Cancel()
        {
            lock (this)
            {
                if (_isCancelled)
                    return;
                _isCancelled = true;
                lock (_notifier)
                    for (int i = 0; i < _notifier.IdleList.Count; i++)
                        if (_notifier.IdleList[i] == this)
                        {
                            _notifier.IdleList.RemoveAt(i);
                            // We can return now because the same idle handler can be registered only once in the list of idles.
                            return;
                        }
            }
        }

        internal int Invoke()
        {
            lock (this)
            {
                // The idle handler may be cancelled after it was registered in the notifier. Check the isCancelled field to make sure it's not cancelled.
                if (!_isCancelled)
                {
                    ProcessIdleEvent();
                    return 1;
                }
                return 0;
            }
        }

        abstract public void ProcessIdleEvent();
    }
}
