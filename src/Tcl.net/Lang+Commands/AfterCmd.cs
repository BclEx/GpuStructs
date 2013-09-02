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

namespace Tcl.Lang
{

    /*
    * This class implements the built-in "after" command in Tcl.
    */

    class AfterCmd : ICommand
    {

        /*
        * The list of handler are stored as AssocData in the interp.
        */

        internal AfterAssocData assocData = null;

        /*
        * Valid command options.
        */

        private static readonly string[] validOpts = new string[] { "cancel", "idle", "info" };

        internal const int OPT_CANCEL = 0;
        internal const int OPT_IDLE = 1;
        internal const int OPT_INFO = 2;

        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] argv)
        {
            int i;
            Notifier notifier = (Notifier)interp.getNotifier();
            Object info;

            if (assocData == null)
            {
                /*
                * Create the "after" information associated for this
                * interpreter, if it doesn't already exist.
                */

                assocData = (AfterAssocData)interp.getAssocData("tclAfter");
                if (assocData == null)
                {
                    assocData = new AfterAssocData(this);
                    interp.setAssocData("tclAfter", assocData);
                }
            }

            if (argv.Length < 2)
            {
                throw new TclNumArgsException(interp, 1, argv, "option ?arg arg ...?");
            }

            /*
            * First lets see if the command was passed a number as the first argument.
            */

            bool isNumber = false;
            int ms = 0;

            if (argv[1].InternalRep is TclInteger)
            {
                ms = TclInteger.get(interp, argv[1]);
                isNumber = true;
            }
            else
            {
                string s = argv[1].ToString();
                if ((s.Length > 0) && (System.Char.IsDigit(s[0])))
                {
                    ms = TclInteger.get(interp, argv[1]);
                    isNumber = true;
                }
            }

            if (isNumber)
            {
                if (ms < 0)
                {
                    ms = 0;
                }
                if (argv.Length == 2)
                {
                    /*
                    * Sleep for at least the given milliseconds and return.
                    */

                    long endTime = System.DateTime.Now.Ticks / 10000 + ms;
                    while (true)
                    {
                        try
                        {
                            System.Threading.Thread.Sleep(ms);
                            return TCL.CompletionCode.RETURN;
                        }
                        catch (System.Threading.ThreadInterruptedException e)
                        {
                            /*
                            * We got interrupted. Sleep again if we havn't slept
                            * long enough yet.
                            */

                            long sysTime = System.DateTime.Now.Ticks / 10000;
                            if (sysTime >= endTime)
                            {
                                return TCL.CompletionCode.RETURN;
                            }
                            ms = (int)(endTime - sysTime);
                            continue;
                        }
                    }
                }

                TclObject cmd = getCmdObject(argv);
                cmd.Preserve();

                assocData.lastAfterId++;
                TimerInfo timerInfo = new TimerInfo(this, notifier, ms);
                timerInfo.interp = interp;
                timerInfo.command = cmd;
                timerInfo.id = assocData.lastAfterId;

                assocData.handlers.Add(timerInfo);

                interp.setResult("after#" + timerInfo.id);

                return TCL.CompletionCode.RETURN;
            }

            /*
            * If it's not a number it must be a subcommand.
            */

            int index;

            try
            {
                index = TclIndex.get(interp, argv[1], validOpts, "option", 0);
            }
            catch (TclException e)
            {
                throw new TclException(interp, "bad argument \"" + argv[1] + "\": must be cancel, idle, info, or a number");
            }

            switch (index)
            {

                case OPT_CANCEL:
                    if (argv.Length < 3)
                    {
                        throw new TclNumArgsException(interp, 2, argv, "id|command");
                    }

                    TclObject arg = getCmdObject(argv);
                    arg.Preserve();

                    /*
                    * Search the timer/idle handler by id or by command.
                    */

                    info = null;
                    for (i = 0; i < assocData.handlers.Count; i++)
                    {
                        Object obj = assocData.handlers[i];
                        if (obj is TimerInfo)
                        {
                            TclObject cmd = ((TimerInfo)obj).command;

                            if ((cmd == arg) || cmd.ToString().Equals(arg.ToString()))
                            {
                                info = obj;
                                break;
                            }
                        }
                        else
                        {
                            TclObject cmd = ((IdleInfo)obj).command;

                            if ((cmd == arg) || cmd.ToString().Equals(arg.ToString()))
                            {
                                info = obj;
                                break;
                            }
                        }
                    }
                    if (info == null)
                    {

                        info = getAfterEvent(arg.ToString());
                    }
                    arg.Release();

                    /*
                    * Cancel the handler.
                    */

                    if (info != null)
                    {
                        if (info is TimerInfo)
                        {
                            ((TimerInfo)info).cancel();
                            ((TimerInfo)info).command.Release();
                        }
                        else
                        {
                            ((IdleInfo)info).Cancel();
                            ((IdleInfo)info).command.Release();
                        }

                        SupportClass.VectorRemoveElement(assocData.handlers, info);
                    }
                    break;


                case OPT_IDLE:
                    if (argv.Length < 3)
                    {
                        throw new TclNumArgsException(interp, 2, argv, "script script ...");
                    }

                    TclObject cmd2 = getCmdObject(argv);
                    cmd2.Preserve();
                    assocData.lastAfterId++;

                    IdleInfo idleInfo = new IdleInfo(this, notifier);
                    idleInfo.interp = interp;
                    idleInfo.command = cmd2;
                    idleInfo.id = assocData.lastAfterId;

                    assocData.handlers.Add(idleInfo);

                    interp.setResult("after#" + idleInfo.id);
                    break;


                case OPT_INFO:
                    if (argv.Length == 2)
                    {
                        /*
                        * No id is given. Return a list of current after id's.
                        */

                        TclObject list = TclList.NewInstance();
                        for (i = 0; i < assocData.handlers.Count; i++)
                        {
                            int id;
                            Object obj = assocData.handlers[i];
                            if (obj is TimerInfo)
                            {
                                id = ((TimerInfo)obj).id;
                            }
                            else
                            {
                                id = ((IdleInfo)obj).id;
                            }
                            TclList.Append(interp, list, TclString.NewInstance("after#" + id));
                        }
                        interp.resetResult();
                        interp.setResult(list);
                        return TCL.CompletionCode.RETURN;
                    }
                    if (argv.Length != 3)
                    {
                        throw new TclNumArgsException(interp, 2, argv, "?id?");
                    }

                    /*
                    * Return command and type of the given after id.
                    */


                    info = getAfterEvent(argv[2].ToString());
                    if (info == null)
                    {

                        throw new TclException(interp, "event \"" + argv[2] + "\" doesn't exist");
                    }
                    TclObject list2 = TclList.NewInstance();
                    TclList.Append(interp, list2, ((info is TimerInfo) ? ((TimerInfo)info).command : ((IdleInfo)info).command));
                    TclList.Append(interp, list2, TclString.NewInstance((info is TimerInfo) ? "timer" : "idle"));

                    interp.resetResult();
                    interp.setResult(list2);
                    break;
            }
            return TCL.CompletionCode.RETURN;
        }
        private TclObject getCmdObject(TclObject[] argv)
        // Argument list passed to the "after" command.
        {
            if (argv.Length == 3)
            {
                return argv[2];
            }
            else
            {
                TclObject cmd = TclString.NewInstance(Util.concat(2, argv.Length - 1, argv));
                return cmd;
            }
        }
        private Object getAfterEvent(string inString)
        // Textual identifier for after event, such
        // as "after#6".
        {
            if (!inString.StartsWith("after#"))
            {
                return null;
            }

            StrtoulResult res = Util.Strtoul(inString, 6, 10);
            if (res.errno != 0)
            {
                return null;
            }

            for (int i = 0; i < assocData.handlers.Count; i++)
            {
                Object obj = assocData.handlers[i];
                if (obj is TimerInfo)
                {
                    if (((TimerInfo)obj).id == res.value)
                    {
                        return obj;
                    }
                }
                else
                {
                    if (((IdleInfo)obj).id == res.value)
                    {
                        return obj;
                    }
                }
            }

            return null;
        }
        internal class AfterAssocData : IAssocData
        {
            public AfterAssocData(AfterCmd enclosingInstance)
            {
                InitBlock(enclosingInstance);
            }
            private void InitBlock(AfterCmd enclosingInstance)
            {
                this.enclosingInstance = enclosingInstance;
                handlers = new ArrayList(10);
            }
            private AfterCmd enclosingInstance;
            public AfterCmd Enclosing_Instance
            {
                get
                {
                    return enclosingInstance;
                }

            }

            /*
            * The set of handlers created but not yet fired.
            */

            internal ArrayList handlers;

            /*
            * Timer identifier of most recently created timer.	
            */

            internal int lastAfterId = 0;

            public void Dispose(Interp interp)
            // The interpreter in which this AssocData
            // instance is registered in.
            {
                for (int i = Enclosing_Instance.assocData.handlers.Count - 1; i >= 0; i--)
                {
                    Object info = Enclosing_Instance.assocData.handlers[i];
                    Enclosing_Instance.assocData.handlers.RemoveAt(i);
                    if (info is TimerInfo)
                    {
                        ((TimerInfo)info).cancel();
                        ((TimerInfo)info).command.Release();
                    }
                    else
                    {
                        ((IdleInfo)info).Cancel();
                        ((IdleInfo)info).command.Release();
                    }
                }
                Enclosing_Instance.assocData = null;
            }
        } // end AfterCmd.AfterAssocData

        internal class TimerInfo : TimerHandler
        {
            private void InitBlock(AfterCmd enclosingInstance)
            {
                this.enclosingInstance = enclosingInstance;
            }
            private AfterCmd enclosingInstance;
            public AfterCmd Enclosing_Instance
            {
                get
                {
                    return enclosingInstance;
                }

            }

            /*
            * Interpreter in which the script should be executed.
            */

            internal Interp interp;

            /*
            * Command to execute when the timer fires.
            */

            internal TclObject command;

            /*
            * Integer identifier for command;  used to cancel it.
            */

            internal int id;

            internal TimerInfo(AfterCmd enclosingInstance, Notifier n, int milliseconds)
                : base(n, milliseconds)
            {
                InitBlock(enclosingInstance);
            }
            public override void processTimerEvent()
            {
                try
                {
                    SupportClass.VectorRemoveElement(Enclosing_Instance.assocData.handlers, this);
                    interp.eval(command, TCL.EVAL_GLOBAL);
                }
                catch (TclException e)
                {
                    interp.AddErrorInfo("\n    (\"after\" script)");
                    interp.backgroundError();
                }
                finally
                {
                    command.Release();
                    command = null;
                }
            }
        } // end AfterCmd.AfterInfo

        internal class IdleInfo : IdleHandler
        {
            private void InitBlock(AfterCmd enclosingInstance)
            {
                this.enclosingInstance = enclosingInstance;
            }
            private AfterCmd enclosingInstance;
            public AfterCmd Enclosing_Instance
            {
                get
                {
                    return enclosingInstance;
                }

            }

            /*
            * Interpreter in which the script should be executed.
            */

            internal Interp interp;

            /*
            * Command to execute when the idle event fires.
            */

            internal TclObject command;

            /*
            * Integer identifier for command;  used to cancel it.
            */

            internal int id;

            internal IdleInfo(AfterCmd enclosingInstance, Notifier n)
                : base(n)
            {
                InitBlock(enclosingInstance);
            }
            public override void ProcessIdleEvent()
            {
                try
                {
                    SupportClass.VectorRemoveElement(Enclosing_Instance.assocData.handlers, this);
                    interp.eval(command, TCL.EVAL_GLOBAL);
                }
                catch (TclException e)
                {
                    interp.AddErrorInfo("\n    (\"after\" script)");
                    interp.backgroundError();
                }
                finally
                {
                    command.Release();
                    command = null;
                }
            }
        } // end AfterCmd.AfterInfo
    } // end AfterCmd
}
