#region Foreign-License
/*
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
using System.Collections;

namespace Tcl.Lang
{
    /// <summary>
    /// This class implements the built-in "array" command in Tcl.
    /// </summary>
    class ArrayCmd : ICommand
    {
        internal static Type _procClass = null;
        private static readonly string[] _validCmds = new string[] { "anymore", "donesearch", "exists", "get", "names", "nextelement", "set", "size", "startsearch", "unset" };
        internal const int OPT_ANYMORE = 0;
        internal const int OPT_DONESEARCH = 1;
        internal const int OPT_EXISTS = 2;
        internal const int OPT_GET = 3;
        internal const int OPT_NAMES = 4;
        internal const int OPT_NEXTELEMENT = 5;
        internal const int OPT_SET = 6;
        internal const int OPT_SIZE = 7;
        internal const int OPT_STARTSEARCH = 8;
        internal const int OPT_UNSET = 9;

        /// <summary>
        /// This procedure is invoked to process the "array" Tcl command. See the user documentation for details on what it does.
        /// </summary>
        public TCL.CompletionCode CmdProc(Interp interp, TclObject[] objv)
        {
            Var var = null, array = null;
            bool notArray = false;
            string varName, msg;
            int index;
            if (objv.Length < 3)
                throw new TclNumArgsException(interp, 1, objv, "option arrayName ?arg ...?");
            index = TclIndex.Get(interp, objv[1], _validCmds, "option", 0);
            // Locate the array variable (and it better be an array).
            varName = objv[2].ToString();
            Var[] retArray = Var.LookupVar(interp, varName, null, 0, null, false, false);
            // Assign the values returned in the array
            if (retArray != null)
            {
                var = retArray[0];
                array = retArray[1];
            }
            if (var == null || !var.IsVarArray() || var.IsVarUndefined())
                notArray = true;
            // Special array trace used to keep the env array in sync for array names, array get, etc.
            if (var != null && var.Traces != null)
            {
                msg = Var.CallTraces(interp, array, var, varName, null, (TCL.VarFlag.LEAVE_ERR_MSG | TCL.VarFlag.NAMESPACE_ONLY | TCL.VarFlag.GLOBAL_ONLY | TCL.VarFlag.TRACE_ARRAY));
                if ((object)msg != null)
                    throw new TclVarException(interp, varName, null, "trace array", msg);
            }

            switch (index)
            {
                case OPT_ANYMORE:
                    {
                        if (objv.Length != 4)
                            throw new TclNumArgsException(interp, 2, objv, "arrayName searchId");
                        if (notArray)
                            ErrorNotArray(interp, objv[2].ToString());
                        if (var.SidVec == null)
                            ErrorIllegalSearchId(interp, objv[2].ToString(), objv[3].ToString());
                        SearchId e = var.getSearch(objv[3].ToString());
                        if (e == null)
                            ErrorIllegalSearchId(interp, objv[2].ToString(), objv[3].ToString());
                        if (e.HasMore)
                            interp.SetResult("1");
                        else
                            interp.SetResult("0");
                        break;
                    }
                case OPT_DONESEARCH:
                    {
                        if (objv.Length != 4)
                            throw new TclNumArgsException(interp, 2, objv, "arrayName searchId");
                        if (notArray)
                            ErrorNotArray(interp, objv[2].ToString());
                        bool rmOK = true;
                        if (var.SidVec != null)
                            rmOK = var.RemoveSearch(objv[3].ToString());
                        if (var.SidVec == null || !rmOK)
                            ErrorIllegalSearchId(interp, objv[2].ToString(), objv[3].ToString());
                        break;
                    }
                case OPT_EXISTS:
                    {
                        if (objv.Length != 3)
                            throw new TclNumArgsException(interp, 2, objv, "arrayName");
                        interp.SetResult(!notArray);
                        break;
                    }
                case OPT_GET:
                    {
                        // Due to the differences in the hashtable implementation from the Tcl core and Java, the output will be rearranged.
                        // This is not a negative side effect, however, test results will differ.
                        if (objv.Length != 3 && objv.Length != 4)
                            throw new TclNumArgsException(interp, 2, objv, "arrayName ?pattern?");
                        if (notArray)
                            return TCL.CompletionCode.RETURN;
                        string pattern = null;
                        if (objv.Length == 4)
                            pattern = objv[3].ToString();
                        Hashtable table = (Hashtable)var._value;
                        TclObject tobj = TclList.NewInstance();
                        string arrayName = objv[2].ToString();
                        string key, strValue;
                        Var var2;
                        // Go through each key in the hash table.  If there is a pattern, test for a match.  Each valid key and its value 
                        // is written into sbuf, which is returned.
                        // FIXME : do we need to port over the 8.1 code for this loop?
                        for (IDictionaryEnumerator e = table.GetEnumerator(); e.MoveNext(); )
                        {
                            key = ((string)e.Key);
                            var2 = (Var)e.Value;
                            if (var2.IsVarUndefined())
                                continue;
                            if ((object)pattern != null && !Util.StringMatch(key, pattern))
                                continue;
                            strValue = interp.GetVar(arrayName, key, 0).ToString();
                            TclList.Append(interp, tobj, TclString.NewInstance(key));
                            TclList.Append(interp, tobj, TclString.NewInstance(strValue));
                        }
                        interp.SetResult(tobj);
                        break;
                    }
                case OPT_NAMES:
                    {
                        if ((objv.Length != 3) && (objv.Length != 4))
                            throw new TclNumArgsException(interp, 2, objv, "arrayName ?pattern?");
                        if (notArray)
                            return TCL.CompletionCode.RETURN;
                        string pattern = null;
                        if (objv.Length == 4)
                            pattern = objv[3].ToString();
                        Hashtable table = (Hashtable)var._value;
                        TclObject tobj = TclList.NewInstance();
                        string key;
                        // Go through each key in the hash table.  If there is a pattern, test for a match. Each valid key and its value 
                        // is written into sbuf, which is returned.
                        for (IDictionaryEnumerator e = table.GetEnumerator(); e.MoveNext(); )
                        {
                            key = (string)e.Key;
                            Var elem = (Var)e.Value;
                            if (!elem.IsVarUndefined())
                            {
                                if ((System.Object)pattern != null)
                                    if (!Util.StringMatch(key, pattern))
                                        continue;
                                TclList.Append(interp, tobj, TclString.NewInstance(key));
                            }
                        }
                        interp.SetResult(tobj);
                        break;
                    }
                case OPT_NEXTELEMENT:
                    {
                        if (objv.Length != 4)
                            throw new TclNumArgsException(interp, 2, objv, "arrayName searchId");
                        if (notArray)
                            ErrorNotArray(interp, objv[2].ToString());
                        if (var.SidVec == null)
                            ErrorIllegalSearchId(interp, objv[2].ToString(), objv[3].ToString());
                        SearchId e = var.getSearch(objv[3].ToString());
                        if (e == null)
                            ErrorIllegalSearchId(interp, objv[2].ToString(), objv[3].ToString());
                        if (e.HasMore)
                        {
                            Hashtable table = (Hashtable)var._value;
                            DictionaryEntry entry = e.nextEntry();
                            string key = (string)entry.Key;
                            Var elem = (Var)entry.Value;
                            if ((elem.Flags & VarFlags.UNDEFINED) == 0)
                                interp.SetResult(key);
                            else
                                interp.SetResult(string.Empty);
                        }
                        break;
                    }
                case OPT_SET:
                    {
                        if (objv.Length != 4)
                            throw new TclNumArgsException(interp, 2, objv, "arrayName list");
                        int size = TclList.getLength(interp, objv[3]);
                        if (size % 2 != 0)
                            throw new TclException(interp, "list must have an even number of elements");
                        string name1 = objv[2].ToString();
                        string name2, strValue;
                        // Set each of the array variable names in the interp
                        for (int i = 0; i < size; i++)
                        {
                            name2 = TclList.index(interp, objv[3], i++).ToString();
                            strValue = TclList.index(interp, objv[3], i).ToString();
                            interp.SetVar(name1, name2, TclString.NewInstance(strValue), 0);
                        }
                        break;
                    }

                case OPT_SIZE:
                    {
                        if (objv.Length != 3)
                            throw new TclNumArgsException(interp, 2, objv, "arrayName");
                        if (notArray)
                            interp.SetResult(0);
                        else
                        {
                            Hashtable table = (Hashtable)var._value;
                            int size = 0;
                            for (IDictionaryEnumerator e = table.GetEnumerator(); e.MoveNext(); )
                            {
                                Var elem = (Var)e.Value;
                                if ((elem.Flags & VarFlags.UNDEFINED) == 0)
                                    size++;
                            }
                            interp.SetResult(size);
                        }
                        break;
                    }
                case OPT_STARTSEARCH:
                    {
                        if (objv.Length != 3)
                            throw new TclNumArgsException(interp, 2, objv, "arrayName");
                        if (notArray)
                            ErrorNotArray(interp, objv[2].ToString());
                        if (var.SidVec == null)
                            var.SidVec = new ArrayList(10);
                        // Create a SearchId Object:
                        // To create a new SearchId object, a unique string identifier needs to be composed and we need to
                        // create an Enumeration of the array keys.  The unique string identifier is created from three strings:
                        //     "s-"   is the default prefix
                        //     "i"    is a unique number that is 1+ the greatest SearchId index currently on the ArrayVar.
                        //     "name" is the name of the array
                        // Once the SearchId string is created we construct a new SearchId object using the string and the
                        // Enumeration.  From now on the string is used to uniquely identify the SearchId object.
                        int i = var.NextIndex;
                        string s = "s-" + i + "-" + objv[2].ToString();
                        IDictionaryEnumerator e = ((Hashtable)var._value).GetEnumerator();
                        var.SidVec.Add(new SearchId(e, s, i));
                        interp.SetResult(s);
                        break;
                    }
                case OPT_UNSET:
                    {
                        string pattern;
                        string name;
                        if (objv.Length != 3 && objv.Length != 4)
                            throw new TclNumArgsException(interp, 2, objv, "arrayName ?pattern?");
                        if (notArray) //Ignot this error -- errorNotArray(interp, objv[2].ToString());
                            break;
                        if (objv.Length == 3) // When no pattern is given, just unset the whole array
                            interp.UnsetVar(objv[2], 0);
                        else
                        {
                            pattern = objv[3].ToString();
                            Hashtable table = (Hashtable)(((Hashtable)var._value).Clone());
                            for (IDictionaryEnumerator e = table.GetEnumerator(); e.MoveNext(); )
                            {
                                name = (string)e.Key;
                                Var elem = (Var)e.Value;
                                if (var.IsVarUndefined())
                                    continue;
                                if (Util.StringMatch(name, pattern))
                                    interp.UnsetVar(varName, name, 0);
                            }
                        }
                        break;
                    }
            }
            return TCL.CompletionCode.RETURN;
        }

        /// <summary>
        /// Error meassage thrown when an invalid identifier is used to access an array.
        /// </summary>
        /// <param name="interp">
        /// currrent interpreter.
        /// </param>
        /// <param name="String">
        /// var is the string representation of the variable that was passed in.
        /// </param>
        private static void ErrorNotArray(Interp interp, string var)
        {
            throw new TclException(interp, "\"" + var + "\" isn't an array");
        }

        /// <summary>
        /// Error message thrown when an invalid SearchId is used.  The string used to reference the SearchId is parced to determine
        /// the reason for the failure. 
        /// </summary>
        /// <param name="interp">
        /// currrent interpreter.
        /// </param>
        /// <param name="String">
        /// sid is the string represenation of the SearchId that was passed in.
        /// </param>
        internal static void ErrorIllegalSearchId(Interp interp, string varName, string sid)
        {
            int val = ValidSearchId(sid.ToCharArray(), varName);
            if (val == 1)
                throw new TclException(interp, "couldn't find search \"" + sid + "\"");
            else if (val == 0)
                throw new TclException(interp, "illegal search identifier \"" + sid + "\"");
            else
                throw new TclException(interp, "search identifier \"" + sid + "\" isn't for variable \"" + varName + "\"");
        }

        /// <summary>
        /// A valid SearchId is represented by the format s-#-arrayName.  If the SearchId string does not match this format than it is illegal,
        /// else we cannot find it.  This method is used by the ErrorIllegalSearchId method to determine the type of error message.
        /// </summary>
        /// <param name="char">
        /// pattern[] is the string use dto identify the SearchId
        /// </param>
        /// <returns>
        /// 1 if its a valid searchID; 0 if it is not a valid searchId, but it is for the array, -1 if it is not a valid searchId and NOT for the array.
        /// </returns>
        private static int ValidSearchId(char[] pattern, string varName)
        {
            int i;
            if (pattern[0] != 's' || pattern[1] != '-' || pattern[2] < '0' || pattern[2] > '9')
                return 0;
            for (i = 3; (i < pattern.Length && pattern[i] != '-'); i++)
                if (pattern[i] < '0' || pattern[i] > '9')
                    return 0;
            if (++i >= pattern.Length)
                return 0;
            if (varName.Equals(new string(pattern, i, (pattern.Length - i))))
                return 1;
            else
                return -1;
        }
    }
}
