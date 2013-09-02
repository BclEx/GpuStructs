#region Foreign-License
/*
	The API for defining idle event handler.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /*
    * This abstract class is used to define idle handlers.
    */

    public abstract class IdleHandler
    {

        /*
        * Back pointer to the notifier that will fire this idle.
        */

        internal Notifier notifier;

        /*
        * True if the cancel() method has been called.
        */

        internal bool isCancelled;

        /*
        * Used to distinguish older idle handlers from recently-created ones.
        */

        internal int generation;

        public IdleHandler(Notifier n)
        {
            notifier = (Notifier)n;
            isCancelled = false;

            lock (notifier)
            {
                notifier.idleList.Add(this);
                generation = notifier.idleGeneration;
                if (System.Threading.Thread.CurrentThread != notifier.primaryThread)
                {
                    System.Threading.Monitor.PulseAll(notifier);
                }
            }
        }
        public void cancel()
        {
            lock (this)
            {
                if (isCancelled)
                {
                    return;
                }

                isCancelled = true;

                lock (notifier)
                {
                    for (int i = 0; i < notifier.idleList.Count; i++)
                    {
                        if (notifier.idleList[i] == this)
                        {
                            notifier.idleList.RemoveAt(i);

                            /*
                            * We can return now because the same idle handler can
                            * be registered only once in the list of idles.
                            */

                            return;
                        }
                    }
                }
            }
        }
        internal int invoke()
        {
            lock (this)
            {
                /*
                * The idle handler may be cancelled after it was registered in
                * the notifier. Check the isCancelled field to make sure it's not
                * cancelled.
                */

                if (!isCancelled)
                {
                    processIdleEvent();
                    return 1;
                }
                else
                {
                    return 0;
                }
            }
        }
        abstract public void processIdleEvent();
    } // end IdleHandler
}
