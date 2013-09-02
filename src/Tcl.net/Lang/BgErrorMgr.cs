#region Foreign-License
/*
    This class manages the background errors for a Tcl interpreter.

Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey
 * 
See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
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
        private Interp Interp; // We manage the background errors in this interp instance.
        private TclObject BgerrorCmdObj; // A TclObject for invoking the "bgerror" command. We use a TclObject instead of a String so that we don't need to look up the command every time.
        private ArrayList Errors = new ArrayList(10); // A list of the pending background error handlers.

        internal BgErrorMgr(Interp i)
        {
            Interp = i;
            BgerrorCmdObj = TclString.NewInstance("bgerror");
            BgerrorCmdObj.Preserve();
            Errors = new ArrayList(10);
        }

        internal void AddBgError()
        {
            BgError bgError = new BgError(this, Interp.getNotifier());
            // The addErrorInfo() call below (with an empty string) ensures that errorInfo gets properly set.  It's needed in
            // cases where the error came from a utility procedure like Interp.getVar() instead of Interp.eval(); in these cases
            // errorInfo still won't have been set when this procedure is called.
            Interp.AddErrorInfo("");
            bgError.ErrorMsg = Interp.GetResult();
            bgError.ErrorInfo = null;
            try { bgError.ErrorInfo = Interp.getVar("errorInfo", null, TCL.VarFlag.GLOBAL_ONLY); }
            catch (TclException) { } // Do nothing if var does not exist.
            bgError.ErrorCode = null;
            try { bgError.ErrorCode = Interp.getVar("errorCode", null, TCL.VarFlag.GLOBAL_ONLY); }
            catch (TclException) { } // Do nothing if var does not exist.
            bgError.ErrorMsg.Preserve();
            bgError.ErrorInfo.Preserve();
            bgError.ErrorCode.Preserve();
            Errors.Add(bgError);
        }

        // The interpreter in which this AssocData instance is registered in.
        public void Dispose(Interp interp)
        {
            for (int i = Errors.Count - 1; i >= 0; i--)
            {
                BgError bgError = (BgError)Errors[i];
                Errors.RemoveAt(i);
                bgError.Cancel();
                bgError.ErrorMsg.Release(); bgError.ErrorMsg = null;
                bgError.ErrorInfo.Release(); bgError.ErrorInfo = null;
                bgError.ErrorCode.Release(); bgError.ErrorCode = null;
            }
            BgerrorCmdObj.Release(); BgerrorCmdObj = null;
        }

        internal class BgError : IdleHandler
        {
            private BgErrorMgr EnclosingInstance;

            // The interp's result, errorCode and errorInfo when the bgerror happened.
            internal TclObject ErrorMsg;
            internal TclObject ErrorCode;
            internal TclObject ErrorInfo;

            internal BgError(BgErrorMgr enclosingInstance, Notifier n)
                : base(n)
            {
                EnclosingInstance = enclosingInstance;
            }

            public override void ProcessIdleEvent()
            {
                // During the execution of this method, elements may be removed from the errors list (because a TCL.CompletionCode.BREAK was returned by the bgerror
                // command, or because the interp was deleted). We remove this BgError instance from the list first so that this instance won't
                // be deleted twice.
                SupportClass.VectorRemoveElement(EnclosingInstance.Errors, this);
                // Restore important state variables to what they were at the time the error occurred.
                try { EnclosingInstance.Interp.SetVar("errorInfo", null, ErrorInfo, TCL.VarFlag.GLOBAL_ONLY); }
                // Ignore any TclException's, possibly caused by variable traces on the errorInfo variable. This is compatible with the behavior of the Tcl C API.
                catch (TclException) { }
                try { EnclosingInstance.Interp.SetVar("errorCode", null, ErrorCode, TCL.VarFlag.GLOBAL_ONLY); }
                // Ignore any TclException's, possibly caused by variable traces on the errorCode variable. This is compatible with the behavior of the Tcl C API.
                catch (TclException) { }
                // Make sure, that the interpreter will surive the invocation of the bgerror command.
                EnclosingInstance.Interp.preserve();
                try
                {
                    // Invoke the bgerror command.
                    TclObject[] argv = new TclObject[2];
                    argv[0] = EnclosingInstance.BgerrorCmdObj;
                    argv[1] = ErrorMsg;
                    Parser.EvalObjv(EnclosingInstance.Interp, argv, 0, TCL.EVAL_GLOBAL);
                }
                catch (TclException e)
                {
                    switch (e.GetCompletionCode())
                    {
                        case TCL.CompletionCode.ERROR:
                            try
                            {
                                Channel chan = TclIO.GetStdChannel(StdChannel.STDERR);
                                var interp = EnclosingInstance.Interp;
                                if (EnclosingInstance.Interp.GetResult().ToString().Equals("\"bgerror\" is an invalid command name or ambiguous abbreviation"))
                                {
                                    chan.Write(interp, ErrorInfo);
                                    chan.Write(interp, "\n");
                                }
                                else
                                {
                                    chan.Write(interp, "bgerror failed to handle background error.\n");
                                    chan.Write(interp, "    Original error: ");
                                    chan.Write(interp, ErrorMsg);
                                    chan.Write(interp, "\n");
                                    chan.Write(interp, "    Error in bgerror: ");
                                    chan.Write(interp, EnclosingInstance.Interp.GetResult());
                                    chan.Write(interp, "\n");
                                }
                                chan.Flush(EnclosingInstance.Interp);
                            }
                            catch (TclException) { } // Ignore.
                            catch (IOException) { } // Ignore, too.
                            break;
                        case TCL.CompletionCode.BREAK:
                            for (int i = EnclosingInstance.Errors.Count - 1; i >= 0; i--)
                            {
                                BgError bgError = (BgError)EnclosingInstance.Errors[i];
                                EnclosingInstance.Errors.RemoveAt(i);
                                bgError.Cancel();
                                bgError.ErrorMsg.Release(); bgError.ErrorMsg = null;
                                bgError.ErrorInfo.Release(); bgError.ErrorInfo = null;
                                bgError.ErrorCode.Release(); bgError.ErrorCode = null;
                            }
                            break;
                    }
                }
                EnclosingInstance.Interp.release();
                ErrorMsg.Release(); ErrorMsg = null;
                ErrorInfo.Release(); ErrorInfo = null;
                ErrorCode.Release(); ErrorCode = null;
            }
        }
    }
}
