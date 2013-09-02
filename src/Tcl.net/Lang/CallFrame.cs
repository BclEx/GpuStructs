#region Foreign-License
/*
Copyright (c) 1997 Cornell University.
Copyright (c) 1997-1998 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
using System.Collections;
using System.Text;
namespace Tcl.Lang
{
    /// <summary>
    /// This class implements a frame in the call stack.
    /// This class can be overridden to define new variable scoping rules for the Tcl interpreter.
    /// </summary>
    public class CallFrame
    {
        internal ArrayList VarNames
        {
            get
            {
                // FIXME : need to port Tcl 8.1 implementation here
                ArrayList vector = new ArrayList(10);
                if (VarTable == null)
                    return vector;
                for (IEnumerator e1 = VarTable.Values.GetEnumerator(); e1.MoveNext(); )
                {
                    Var v = (Var)e1.Current;
                    if (!v.IsVarUndefined())
                        vector.Add(v.HashKey);
                }
                return vector;
            }

        }

        /// <returns>
        /// an Vector the names of the (defined) local variables in this CallFrame (excluding upvar's)
        /// </returns>
        internal ArrayList LocalVarNames
        {
            get
            {
                ArrayList vector = new ArrayList(10);
                if (VarTable == null)
                    return vector;
                for (IEnumerator e1 = VarTable.Values.GetEnumerator(); e1.MoveNext(); )
                {
                    Var v = (Var)e1.Current;
                    if (!v.IsVarUndefined() && !v.isVarLink())
                        vector.Add(v.HashKey);
                }
                return vector;
            }
        }

        protected internal Interp Interp; // The interpreter associated with this call frame.
        internal NamespaceCmd.Namespace NS; // The Namespace this CallFrame is executing in. Used to resolve commands and global variables.
        // If true, the frame was pushed to execute a Tcl procedure and may have local vars. If false, the frame was pushed to execute
        // a namespace command and var references are treated as references to namespace vars; varTable is ignored.
        internal bool IsProcCallFrame;
        internal TclObject[] Objv; // Stores the arguments of the procedure associated with this CallFrame. Is null for global level.
        protected internal CallFrame Caller; // Value of interp.frame when this procedure was invoked (i.e. next in stack of all active procedures).
        // Value of interp.varFrame when this procedure was invoked (i.e. determines variable scoping within caller; same as
        // caller unless an "uplevel" command or something equivalent was active in the caller).
        protected internal CallFrame CallerVar;
        protected internal int Level; // Level of recursion. = 0 for the global level.
        protected internal Hashtable VarTable; // Stores the variables of this CallFrame.

        /// <summary>
        /// Creates a CallFrame for the global variables.</summary>
        /// <param name="interp">
        /// current interpreter.
        /// </param>
        internal CallFrame(Interp i)
        {
            Interp = i;
            NS = i.GlobalNs;
            VarTable = new Hashtable();
            Caller = null;
            CallerVar = null;
            Objv = null;
            Level = 0;
            IsProcCallFrame = true;
        }

        /// <summary>
        /// Creates a CallFrame. It changes the following variables:
        /// <ul>
        /// <li>this.caller</li>
        /// <li>this.callerVar</li>
        /// <li>interp.frame</li>
        /// <li>interp.varFrame</li>
        /// </ul>
        /// </summary>
        /// <param name="i">
        /// current interpreter.
        /// </param>
        /// <param name="proc">
        /// the procedure to invoke in this call frame.
        /// </param>
        /// <param name="objv">
        /// the arguments to the procedure.
        /// </param>
        /// <exception cref="">
        /// TclException if error occurs in parameter bindings.
        /// </exception>
        internal CallFrame(Interp i, Procedure proc, TclObject[] objv)
            : this(i)
        {
            try { Chain(proc, objv); }
            catch (TclException)
            {
                Dispose();
                throw;
            }
        }

        /// <summary>
        /// Chain this frame into the call frame stack and binds the parameters values to the formal parameters of the procedure.
        /// </summary>
        /// <param name="proc">
        /// the procedure.
        /// </param>
        /// <param name="proc">
        /// argv the parameter values.
        /// </param>
        /// <exception cref="">
        /// TclException if wrong number of arguments.
        /// </exception>
        internal void Chain(Procedure proc, TclObject[] objv)
        {
            // FIXME: double check this ns thing in case where proc is renamed to different ns.
            NS = proc.NS;
            Objv = objv;
            // FIXME : quick level hack : fix later
            Level = (Interp.VarFrame == null ? 1 : Interp.VarFrame.Level + 1);
            Caller = Interp.Frame;
            CallerVar = Interp.VarFrame;
            Interp.Frame = this;
            Interp.VarFrame = this;
            // parameter bindings
            int numArgs = proc.ArgList.Length;
            if (!proc.isVarArgs && objv.Length - 1 > numArgs)
                WrongNumProcArgs(objv[0], proc);
            for (int i = 0, j = 1; i < numArgs; i++, j++)
            {
                // Handle the special case of the last formal being "args".  When it occurs, assign it a list consisting of
                // all the remaining actual arguments.
                TclObject varName = proc.ArgList[i][0];
                TclObject value = null;
                if (i == (numArgs - 1) && proc.isVarArgs)
                {
                    value = TclList.NewInstance();
                    value.Preserve();
                    for (int k = j; k < objv.Length; k++)
                        TclList.Append(Interp, value, objv[k]);
                    Interp.SetVar(varName, value, 0);
                    value.Release();
                }
                else
                {
                    if (j < objv.Length)
                        value = objv[j];
                    else if (proc.ArgList[i][1] != null)
                        value = proc.ArgList[i][1];
                    else
                        WrongNumProcArgs(objv[0], proc);
                    Interp.SetVar(varName, value, 0);
                }
            }
        }

        private string WrongNumProcArgs(TclObject name, Procedure proc)
        {
            StringBuilder sbuf = new StringBuilder(200);
            sbuf.Append("wrong # args: should be \"");
            sbuf.Append(name.ToString());
            for (int i = 0; i < proc.ArgList.Length; i++)
            {
                TclObject arg = proc.ArgList[i][0];
                TclObject def = proc.ArgList[i][1];
                sbuf.Append(" ");
                if (def != null)
                    sbuf.Append("?");
                sbuf.Append(arg.ToString());
                if (def != null)
                    sbuf.Append("?");
            }
            sbuf.Append("\"");
            throw new TclException(Interp, sbuf.ToString());
        }

        /// <param name="name">
        /// the name of the variable.
        /// </param>
        /// <returns>
        /// true if a variable exists and is defined inside this CallFrame, false otherwise
        /// </returns>
        internal static bool exists(Interp interp, string name)
        {
            try
            {
                Var[] result = Var.LookupVar(interp, name, null, 0, "lookup", false, false);
                if (result == null)
                    return false;
                if (result[0].IsVarUndefined())
                    return false;
                return true;
            }
            catch (TclException e) { throw new TclRuntimeError("unexpected TclException: " + e.Message, e); }
        }

        /// <summary>
        /// Tcl_GetFrame -> getFrame
        /// 
        /// Given a description of a procedure frame, such as the first argument to an "uplevel" or "upvar" command, locate the
        /// call frame for the appropriate level of procedure.
        /// 
        /// The return value is 1 if string was either a number or a number preceded by "#" and it specified a valid frame. 0 is returned
        /// if string isn't one of the two things above (in this case, the lookup acts as if string were "1"). The frameArr[0] reference
        /// will be filled by the reference of the desired frame (unless an error occurs, in which case it isn't modified).
        /// </summary>
        /// <param name="string">
        /// a string that specifies the level.
        /// </param>
        /// <returns>
        /// an Vector the names of the (defined) variables in this CallFrame.
        /// </returns>
        /// <exception cref="">
        /// TclException if s is a valid level specifier but refers to a bad level that doesn't exist.
        /// </exception>
        internal static int GetFrame(Interp interp, string inString, CallFrame[] frameArr)
        {
            // Parse string to figure out which level number to go to.
            int level;
            int result = 1;
            int curLevel = (interp.VarFrame == null ? 0 : interp.VarFrame.Level);
            if (inString.Length > 0 && inString[0] == '#')
            {
                level = Util.getInt(interp, inString.Substring(1));
                if (level < 0)
                    throw new TclException(interp, "bad level \"" + inString + "\"");
            }
            else if (inString.Length > 0 && Char.IsDigit(inString[0]))
            {
                level = Util.getInt(interp, inString);
                level = curLevel - level;
            }
            else
            {
                level = curLevel - 1;
                result = 0;
            }
            // FIXME: is this a bad comment from some other proc?
            // Figure out which frame to use, and modify the interpreter so its variables come from that frame.
            CallFrame frame;
            if (level == 0)
                frame = null;
            else
            {
                for (frame = interp.VarFrame; frame != null; frame = frame.CallerVar)
                    if (frame.Level == level)
                        break;
                if (frame == null)
                    throw new TclException(interp, "bad level \"" + inString + "\"");
            }
            frameArr[0] = frame;
            return result;
        }


        /// <summary>
        /// This method is called when this CallFrame is no longer needed. Removes the reference of this object from the interpreter so
        /// that this object can be garbage collected.
        ///
        /// For this procedure to work correctly, it must not be possible for any of the variable in the table to be accessed from Tcl
        /// commands (e.g. from trace procedures).
        /// </summary>
        protected internal void Dispose()
        {
            // Unchain this frame from the call stack.
            Interp.Frame = Caller;
            Interp.VarFrame = CallerVar;
            Caller = null;
            CallerVar = null;
            if (VarTable != null)
            {
                Var.DeleteVars(Interp, VarTable);
                VarTable.Clear(); VarTable = null;
            }
        }
    }
}
