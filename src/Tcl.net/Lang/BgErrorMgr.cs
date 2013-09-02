#region Foreign-License
/*
    This class manages the background errors for a Tcl interpreter.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey
 * 
See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
using System.Collections;
using System.IO;

namespace Tcl.Lang
{
    /// <summary>
    /// This class manages the background errors for a Tcl interpreter. It stores the error information about the interpreter and use an idle
    /// handler to report the error when the notifier is idle.
    /// </summary>
    class BgErrorMgr : IAssocData
    {
        internal Interp Interp; // We manage the background errors in this interp instance.
        internal TclObject BgerrorCmdObj; // A TclObject for invoking the "bgerror" command. We use a TclObject instead of a String so that we don't need to look up the command every time.
        internal ArrayList Errors = new ArrayList(10); // A list of the pending background error handlers.

        internal BgErrorMgr(Interp i)
        {
            Interp = i;
            BgerrorCmdObj = TclString.newInstance("bgerror");
            BgerrorCmdObj.preserve();

            Errors = new ArrayList(10);
        }

        internal void addBgError()
        {
            BgError bgErr = new BgError(this, Interp.getNotifier());

            // The addErrorInfo() call below (with an empty string) ensures that errorInfo gets properly set.  It's needed in
            // cases where the error came from a utility procedure like Interp.getVar() instead of Interp.eval(); in these cases
            // errorInfo still won't have been set when this procedure is called.
            Interp.addErrorInfo("");

            bgErr.errorMsg = Interp.getResult();
            bgErr.errorInfo = null;
            try { bgErr.errorInfo = Interp.getVar("errorInfo", null, TCL.VarFlag.GLOBAL_ONLY); }
            catch (TclException) { } // Do nothing if var does not exist.

            bgErr.errorCode = null;
            try { bgErr.errorCode = Interp.getVar("errorCode", null, TCL.VarFlag.GLOBAL_ONLY); }
            catch (TclException) { } // Do nothing if var does not exist.

            bgErr.errorMsg.preserve();
            bgErr.errorInfo.preserve();
            bgErr.errorCode.preserve();

            Errors.Add(bgErr);
        }

        // The interpreter in which this AssocData instance is registered in.
        public void Dispose(Interp interp)
        {
            for (int i = Errors.Count - 1; i >= 0; i--)
            {
                BgError bgErr = (BgError)Errors[i];
                Errors.RemoveAt(i);
                bgErr.cancel();

                bgErr.errorMsg.release();
                bgErr.errorMsg = null;
                bgErr.errorInfo.release();
                bgErr.errorInfo = null;
                bgErr.errorCode.release();
                bgErr.errorCode = null;
            }

            BgerrorCmdObj.release();
            BgerrorCmdObj = null;
        }
        internal class BgError : IdleHandler
        {
            private void InitBlock(BgErrorMgr enclosingInstance)
            {
                this.enclosingInstance = enclosingInstance;
            }
            private BgErrorMgr enclosingInstance;
            public BgErrorMgr Enclosing_Instance
            {
                get
                {
                    return enclosingInstance;
                }

            }

            /*
            * The interp's result, errorCode and errorInfo when the bgerror happened.
            */

            internal TclObject errorMsg;
            internal TclObject errorCode;
            internal TclObject errorInfo;

            internal BgError(BgErrorMgr enclosingInstance, Notifier n)
                : base(n)
            {
                InitBlock(enclosingInstance);
            }
            public override void processIdleEvent()
            {

                // During the execution of this method, elements may be removed from
                // the errors list (because a TCL.CompletionCode.BREAK was returned by the bgerror
                // command, or because the interp was deleted). We remove this
                // BgError instance from the list first so that this instance won't
                // be deleted twice.

                SupportClass.VectorRemoveElement(Enclosing_Instance.Errors, this);

                // Restore important state variables to what they were at
                // the time the error occurred.

                try
                {
                    Enclosing_Instance.Interp.setVar("errorInfo", null, errorInfo, TCL.VarFlag.GLOBAL_ONLY);
                }
                catch (TclException e)
                {

                    // Ignore any TclException's, possibly caused by variable traces on
                    // the errorInfo variable. This is compatible with the behavior of
                    // the Tcl C API.
                }

                try
                {
                    Enclosing_Instance.Interp.setVar("errorCode", null, errorCode, TCL.VarFlag.GLOBAL_ONLY);
                }
                catch (TclException e)
                {

                    // Ignore any TclException's, possibly caused by variable traces on
                    // the errorCode variable. This is compatible with the behavior of
                    // the Tcl C API.
                }

                // Make sure, that the interpreter will surive the invocation
                // of the bgerror command.

                Enclosing_Instance.Interp.preserve();

                try
                {

                    // Invoke the bgerror command.

                    TclObject[] argv = new TclObject[2];
                    argv[0] = Enclosing_Instance.BgerrorCmdObj;
                    argv[1] = errorMsg;

                    Parser.evalObjv(Enclosing_Instance.Interp, argv, 0, TCL.EVAL_GLOBAL);
                }
                catch (TclException e)
                {
                    switch (e.getCompletionCode())
                    {

                        case TCL.CompletionCode.ERROR:
                            try
                            {
                                Channel chan = TclIO.getStdChannel(StdChannel.STDERR);

                                if (Enclosing_Instance.Interp.getResult().ToString().Equals("\"bgerror\" is an invalid command name or ambiguous abbreviation"))
                                {
                                    chan.write(Enclosing_Instance.Interp, errorInfo);
                                    chan.write(Enclosing_Instance.Interp, "\n");
                                }
                                else
                                {
                                    chan.write(Enclosing_Instance.Interp, "bgerror failed to handle background error.\n");
                                    chan.write(Enclosing_Instance.Interp, "    Original error: ");
                                    chan.write(Enclosing_Instance.Interp, errorMsg);
                                    chan.write(Enclosing_Instance.Interp, "\n");
                                    chan.write(Enclosing_Instance.Interp, "    Error in bgerror: ");
                                    chan.write(Enclosing_Instance.Interp, Enclosing_Instance.Interp.getResult());
                                    chan.write(Enclosing_Instance.Interp, "\n");
                                }
                                chan.flush(Enclosing_Instance.Interp);
                            }
                            catch (TclException e1)
                            {

                                // Ignore.
                            }
                            catch (IOException e2)
                            {

                                // Ignore, too.
                            }
                            break;


                        case TCL.CompletionCode.BREAK:

                            for (int i = Enclosing_Instance.Errors.Count - 1; i >= 0; i--)
                            {
                                BgError bgErr = (BgError)Enclosing_Instance.Errors[i];
                                Enclosing_Instance.Errors.RemoveAt(i);
                                bgErr.cancel();

                                bgErr.errorMsg.release();
                                bgErr.errorMsg = null;
                                bgErr.errorInfo.release();
                                bgErr.errorInfo = null;
                                bgErr.errorCode.release();
                                bgErr.errorCode = null;
                            }
                            break;
                    }
                }

                Enclosing_Instance.Interp.release();

                errorMsg.release();
                errorMsg = null;
                errorInfo.release();
                errorInfo = null;
                errorCode.release();
                errorCode = null;
            }
        } // end BgErrorMgr.BgError
    } // end BgErrorMgr
}
