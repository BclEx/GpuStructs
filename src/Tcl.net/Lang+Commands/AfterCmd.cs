#region Foreign-License
/*
    Implements the built-in "after" Tcl command.

Copyright (c) 1997 Cornell University.
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
using System.Collections;
using System.Threading;

namespace Tcl.Lang
{
    /// <summary>
    /// This class implements the built-in "after" command in Tcl.
    /// </summary>
    class AfterCmd : ICommand
    {
        internal AfterAssocData _assocData = null; // The list of handler are stored as AssocData in the interp.
        private static readonly string[] _validOpts = new string[] { "cancel", "idle", "info" }; // Valid command options.
        internal const int OPT_CANCEL = 0;
        internal const int OPT_IDLE = 1;
        internal const int OPT_INFO = 2;

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            int i;
            Notifier notifier = (Notifier)interp.GetNotifier();
            Object info;
            if (_assocData == null)
            {
                // Create the "after" information associated for this interpreter, if it doesn't already exist.
                _assocData = (AfterAssocData)interp.GetAssocData("tclAfter");
                if (_assocData == null)
                {
                    _assocData = new AfterAssocData(this);
                    interp.SetAssocData("tclAfter", _assocData);
                }
            }
            if (argv.Length < 2)
                throw new TclNumArgsException(interp, 1, argv, "option ?arg arg ...?");
            // First lets see if the command was passed a number as the first argument.
            bool isNumber = false;
            int ms = 0;
            if (argv[1].InternalRep is TclInteger)
            {
                ms = TclInteger.Get(interp, argv[1]);
                isNumber = true;
            }
            else
            {
                string s = argv[1].ToString();
                if (s.Length > 0 && char.IsDigit(s[0]))
                {
                    ms = TclInteger.Get(interp, argv[1]);
                    isNumber = true;
                }
            }
            if (isNumber)
            {
                if (ms < 0)
                    ms = 0;
                if (argv.Length == 2)
                {
                    // Sleep for at least the given milliseconds and return.
                    long endTime = DateTime.Now.Ticks / 10000 + ms;
                    while (true)
                    {
                        try
                        {
                            Thread.Sleep(ms);
                            return TCL.CompletionCode.RETURN;
                        }
                        catch (ThreadInterruptedException e)
                        {
                            // We got interrupted. Sleep again if we havn't slept long enough yet.
                            long sysTime = System.DateTime.Now.Ticks / 10000;
                            if (sysTime >= endTime)
                                return TCL.CompletionCode.RETURN;
                            ms = (int)(endTime - sysTime);
                            continue;
                        }
                    }
                }
                TclObject cmd = GetCmdObject(argv);
                cmd.Preserve();
                _assocData.LastAfterId++;
                TimerInfo timerInfo = new TimerInfo(this, notifier, ms);
                timerInfo.Interp = interp;
                timerInfo.Command = cmd;
                timerInfo.Id = _assocData.LastAfterId;
                _assocData.Handlers.Add(timerInfo);
                interp.SetResult("after#" + timerInfo.Id);
                return TCL.CompletionCode.RETURN;
            }

            // If it's not a number it must be a subcommand.
            int index;
            try { index = TclIndex.Get(interp, argv[1], _validOpts, "option", 0); }
            catch (TclException e) { throw new TclException(interp, "bad argument \"" + argv[1] + "\": must be cancel, idle, info, or a number"); }

            switch (index)
            {
                case OPT_CANCEL:
                    if (argv.Length < 3)
                        throw new TclNumArgsException(interp, 2, argv, "id|command");
                    TclObject arg = GetCmdObject(argv);
                    arg.Preserve();
                    // Search the timer/idle handler by id or by command.
                    info = null;
                    for (i = 0; i < _assocData.Handlers.Count; i++)
                    {
                        Object obj = _assocData.Handlers[i];
                        if (obj is TimerInfo)
                        {
                            TclObject cmd = ((TimerInfo)obj).Command;
                            if (cmd == arg || cmd.ToString().Equals(arg.ToString()))
                            {
                                info = obj;
                                break;
                            }
                        }
                        else
                        {
                            TclObject cmd = ((IdleInfo)obj).Command;
                            if (cmd == arg || cmd.ToString().Equals(arg.ToString()))
                            {
                                info = obj;
                                break;
                            }
                        }
                    }
                    if (info == null)
                        info = GetAfterEvent(arg.ToString());
                    arg.Release();
                    // Cancel the handler.
                    if (info != null)
                    {
                        if (info is TimerInfo)
                        {
                            ((TimerInfo)info).Cancel();
                            ((TimerInfo)info).Command.Release();
                        }
                        else
                        {
                            ((IdleInfo)info).Cancel();
                            ((IdleInfo)info).Command.Release();
                        }
                        SupportClass.VectorRemoveElement(_assocData.Handlers, info);
                    }
                    break;
                case OPT_IDLE:
                    if (argv.Length < 3)
                        throw new TclNumArgsException(interp, 2, argv, "script script ...");
                    TclObject cmd2 = GetCmdObject(argv);
                    cmd2.Preserve();
                    _assocData.LastAfterId++;
                    IdleInfo idleInfo = new IdleInfo(this, notifier);
                    idleInfo.Interp = interp;
                    idleInfo.Command = cmd2;
                    idleInfo.Id = _assocData.LastAfterId;
                    _assocData.Handlers.Add(idleInfo);
                    interp.SetResult("after#" + idleInfo.Id);
                    break;
                case OPT_INFO:
                    if (argv.Length == 2)
                    {
                        // No id is given. Return a list of current after id's.
                        TclObject list = TclList.NewInstance();
                        for (i = 0; i < _assocData.Handlers.Count; i++)
                        {
                            int id;
                            Object obj = _assocData.Handlers[i];
                            if (obj is TimerInfo)
                                id = ((TimerInfo)obj).Id;
                            else
                                id = ((IdleInfo)obj).Id;
                            TclList.Append(interp, list, TclString.NewInstance("after#" + id));
                        }
                        interp.ResetResult();
                        interp.SetResult(list);
                        return TCL.CompletionCode.RETURN;
                    }
                    if (argv.Length != 3)
                        throw new TclNumArgsException(interp, 2, argv, "?id?");
                    // Return command and type of the given after id.
                    info = GetAfterEvent(argv[2].ToString());
                    if (info == null)
                        throw new TclException(interp, "event \"" + argv[2] + "\" doesn't exist");
                    TclObject list2 = TclList.NewInstance();
                    TclList.Append(interp, list2, ((info is TimerInfo) ? ((TimerInfo)info).Command : ((IdleInfo)info).Command));
                    TclList.Append(interp, list2, TclString.NewInstance((info is TimerInfo) ? "timer" : "idle"));
                    interp.ResetResult();
                    interp.SetResult(list2);
                    break;
            }
            return TCL.CompletionCode.RETURN;
        }

        private TclObject GetCmdObject(TclObject[] argv) // Argument list passed to the "after" command.
        {
            if (argv.Length == 3)
                return argv[2];
            else
            {
                TclObject cmd = TclString.NewInstance(Util.concat(2, argv.Length - 1, argv));
                return cmd;
            }
        }

        private Object GetAfterEvent(string inString) // Textual identifier for after event, such as "after#6".
        {
            if (!inString.StartsWith("after#"))
                return null;
            StrtoulResult res = Util.Strtoul(inString, 6, 10);
            if (res.errno != 0)
                return null;
            for (int i = 0; i < _assocData.Handlers.Count; i++)
            {
                Object obj = _assocData.Handlers[i];
                if (obj is TimerInfo)
                {
                    if (((TimerInfo)obj).Id == res.value)
                        return obj;
                }
                else
                {
                    if (((IdleInfo)obj).Id == res.value)
                        return obj;
                }
            }
            return null;
        }

        internal class AfterAssocData : IAssocData
        {
            public AfterAssocData(AfterCmd enclosingInstance)
            {
                EnclosingInstance = enclosingInstance;
                Handlers = new ArrayList(10);
            }

            public AfterCmd EnclosingInstance;
            internal ArrayList Handlers; // The set of handlers created but not yet fired.
            internal int LastAfterId = 0; // Timer identifier of most recently created timer.	

            public void Dispose(Interp interp) // The interpreter in which this AssocData instance is registered in.
            {
                for (int i = EnclosingInstance._assocData.Handlers.Count - 1; i >= 0; i--)
                {
                    Object info = EnclosingInstance._assocData.Handlers[i];
                    EnclosingInstance._assocData.Handlers.RemoveAt(i);
                    if (info is TimerInfo)
                    {
                        ((TimerInfo)info).Cancel();
                        ((TimerInfo)info).Command.Release();
                    }
                    else
                    {
                        ((IdleInfo)info).Cancel();
                        ((IdleInfo)info).Command.Release();
                    }
                }
                EnclosingInstance._assocData = null;
            }
        }

        internal class TimerInfo : TimerHandler
        {
            public AfterCmd EnclosingInstance;
            internal Interp Interp; // Interpreter in which the script should be executed.
            internal TclObject Command; // Command to execute when the timer fires.
            internal int Id; // Integer identifier for command;  used to cancel it.

            internal TimerInfo(AfterCmd enclosingInstance, Notifier n, int milliseconds)
                : base(n, milliseconds)
            {
                EnclosingInstance = enclosingInstance;
            }

            public override void ProcessTimerEvent()
            {
                try
                {
                    SupportClass.VectorRemoveElement(EnclosingInstance._assocData.Handlers, this);
                    Interp.Eval(Command, TCL.EVAL_GLOBAL);
                }
                catch (TclException) { Interp.AddErrorInfo("\n    (\"after\" script)"); Interp.BackgroundError(); }
                finally { Command.Release(); Command = null; }
            }
        }

        internal class IdleInfo : IdleHandler
        {
            public AfterCmd EnclosingInstance;
            internal Interp Interp; // Interpreter in which the script should be executed.
            internal TclObject Command; // Command to execute when the idle event fires.
            internal int Id; // Integer identifier for command;  used to cancel it.

            internal IdleInfo(AfterCmd enclosingInstance, Notifier n)
                : base(n)
            {
                EnclosingInstance = enclosingInstance;
            }

            public override void ProcessIdleEvent()
            {
                try
                {
                    SupportClass.VectorRemoveElement(EnclosingInstance._assocData.Handlers, this);
                    Interp.Eval(Command, TCL.EVAL_GLOBAL);
                }
                catch (TclException e) { Interp.AddErrorInfo("\n    (\"after\" script)"); Interp.BackgroundError(); }
                finally { Command.Release(); Command = null; }
            }
        }
    }
}
