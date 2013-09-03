#region Foreign-License
/*
Copyright (c) 1997 Sun Microsystems, Inc.
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
    /// Flag bits for variables. The first three (SCALAR, ARRAY, and LINK) are mutually exclusive and give the "type" of the variable.
    /// UNDEFINED is independent of the variable's type. 
    /// 
    /// SCALAR - 1 means this is a scalar variable and not an array or link. The value field points to the variable's value, a Tcl object.
    /// ARRAY -			1 means this is an array variable rather than a scalar variable or link. The table field points to the array's hashtable for its elements.
    /// LINK -			1 means this Var structure contains a reference to another Var structure that either has the real value or is itself another LINK pointer. Variables like this come about through "upvar" and "global" commands, or through references to variables in enclosing namespaces.
    /// UNDEFINED -		1 means that the variable is in the process of being deleted. An undefined variable logically does not exist and survives only while it has a trace, or if it is a global variable currently being used by some procedure.
    /// IN_HASHTABLE -		1 means this variable is in a hashtable. 0 if a local variable that was assigned a slot in a procedure frame by	the compiler so the Var storage is part of the call frame.
    /// TRACE_ACTIVE -		1 means that trace processing is currently underway for a read or write access, so new read or write accesses should not cause trace procedures to be called and the variable can't be deleted.
    /// ARRAY_ELEMENT -		1 means that this variable is an array element, so it is not legal for it to be an array itself (the ARRAY flag had better not be set).
    /// NAMESPACE_VAR -		1 means that this variable was declared as a namespace variable. This flag ensures it persists until its namespace is destroyed or until the variable is unset; it will persist even if it has not been initialized and is marked undefined. The variable's refCount is incremented to reflect the "reference" from its namespace.
    /// </summary>
    [Flags]
    public enum VarFlags : int
    {
        SCALAR = 0x1,
        ARRAY = 0x2,
        LINK = 0x4,
        UNDEFINED = 0x8,
        IN_HASHTABLE = 0x10,
        TRACE_ACTIVE = 0x20,
        ARRAY_ELEMENT = 0x40,
        NAMESPACE_VAR = 0x80,
        EXT_LINK_INT = 0x100,
        EXT_LINK_DOUBLE = 0x200,
        EXT_LINK_BOOLEAN = 0x400,
        EXT_LINK_STRING = 0x800,
        EXT_LINK_WIDE_INT = 0x1000,
        EXT_LINK = 0x10000,
        EXT_LINK_READ_ONLY = 0x20000,
    }

    /// <summary>
    /// Implements variables in Tcl. The Var class encapsulates most of the functionality of the methods in generic/tclVar.c and the structure TCL.Tcl_Var from the C version.
    /// </summary>
    public class Var
    {
        /// <summary>
        /// Used by ArrayCmd to create a unique searchId string.  If the sidVec Vector is empty then simply return 1.  Else return 1 
        /// plus the SearchId.index value of the last Object in the vector.
        /// </summary>
        /// <param name="">
        /// None
        /// </param>
        /// <returns>
        /// The int value for unique SearchId string.
        /// </returns>
        protected internal int NextIndex
        {
            get
            {
                lock (this)
                {
                    if (_sidVec.Count == 0)
                        return 1;
                    SearchId sid = (SearchId)SupportClass.VectorLastElement(_sidVec);
                    return (sid.Index + 1);
                }
            }
        }

        // Methods to read various flag bits of variables.
        internal bool IsVarScalar() { return ((_flags & VarFlags.SCALAR) != 0); }
        internal bool IsVarLink() { return ((_flags & VarFlags.LINK) != 0); }
        internal bool IsVarArray() { return ((_flags & VarFlags.ARRAY) != 0); }
        internal bool IsVarUndefined() { return ((_flags & VarFlags.UNDEFINED) != 0); }
        internal bool IsVarArrayElement() { return ((_flags & VarFlags.ARRAY_ELEMENT) != 0); }
        // Methods to ensure that various flag bits are set properly for variables.
        internal void SetVarScalar() { _flags = (_flags & ~(VarFlags.ARRAY | VarFlags.LINK)) | VarFlags.SCALAR; }
        internal void SetVarArray() { _flags = (_flags & ~(VarFlags.SCALAR | VarFlags.LINK)) | VarFlags.ARRAY; }
        internal void SetVarLink() { _flags = (_flags & ~(VarFlags.SCALAR | VarFlags.ARRAY)) | VarFlags.LINK; }
        internal void SetVarArrayElement() { _flags = (_flags & ~VarFlags.ARRAY) | VarFlags.ARRAY_ELEMENT; }
        internal void SetVarUndefined() { _flags |= VarFlags.UNDEFINED; }
        internal void ClearVarUndefined() { _flags &= ~VarFlags.UNDEFINED; }

        /// <summary>
        /// Stores the "value" of the variable. It stored different information depending on the type of the variable: <ul>
        /// <li>Scalar variable - (TclObject) value is the object stored in the variable.</li>
        /// <li>Array variable - (Hashtable) value is the hashtable that stores all the elements.</li>
        /// <li> Upvar (Link) - (Var) value is the variable associated by this upvar.</li>
        /// </ul>
        /// </summary>
        internal Object _value;
        internal ArrayList _traces; // Vector that holds the traces that were placed in this Var
        internal ArrayList _sidVec;

        /// <summary>
        /// Miscellaneous bits of information about variable.
        /// </summary>
        /// <seealso cref="Var#SCALAR" />
        /// <seealso cref="Var#ARRAY" />
        /// <seealso cref="Var#LINK" />
        /// <seealso cref="Var#UNDEFINED" />
        /// <seealso cref="Var#IN_HASHTABLE" />
        /// <seealso cref="Var#TRACE_ACTIVE" />
        /// <seealso cref="Var#ARRAY_ELEMENT" />
        /// <seealso cref="Var#NAMESPACE_VAR" />
        internal VarFlags _flags;

        /// <summary>
        /// If variable is in a hashtable, either the hash table entry that refers to this
        /// variable or null if the variable has been detached from its hash table (e.g. an
        /// array is deleted, but some of its elements are still referred to in
        /// upvars). null if the variable is not in a hashtable. This is used to delete an
        /// variable from its hashtable if it is no longer needed.
        /// </summary>
        internal Hashtable _table;

        /// <summary>
        /// The key under which this variable is stored in the hash table.
        /// </summary>
        internal string HashKey;

        /// <summary>
        /// Counts number of active uses of this variable, not including its entry in the
        /// call frame or the hash table: 1 for each additional variable whose link points
        /// here, 1 for each nested trace active on variable, and 1 if the variable is a 
        /// namespace variable. This record can't be deleted until refCount becomes 0.
        /// </summary>
        internal int RefCount;

        /// <summary>
        /// Reference to the namespace that contains this variable or null if the variable is
        /// a local variable in a Tcl procedure.
        /// </summary>
        internal NamespaceCmd.Namespace _ns;

        public class Ext_GETSET
        {
            string _name = "";
            int _integer = 0;                   // Internal integer value
            StringBuilder _stringBuilder = null;   // Internal string value
            
            public Ext_GETSET(string name)
            {
                _integer = 0;
                _stringBuilder = new StringBuilder(500);
                _name = name;
            }

            public int iValue
            {
                get { return _integer; }
                set { _integer = value; }
            }
            public string sValue
            {
                get { return _stringBuilder.ToString(); }
                set
                {
                    _stringBuilder.Length = 0;
                    _stringBuilder.Append(value);
                }
            }

            public void Append(byte[] append)
            {
                _stringBuilder.Append(Encoding.UTF8.GetString(append, 0, append.Length));
            }

            public void Append(string append)
            {
                _stringBuilder.Append(append);
            }

            public void Trim()
            {
                _stringBuilder = new StringBuilder(_stringBuilder.ToString().Trim());
            }

            public int Length
            {
                get { return _stringBuilder.Length; }
            }
        }

        /// <summary>
        /// Reference to the object the allows getting & setting the sqlite3 linked variable
        /// </summary>
        internal object ext_getset;

        internal TclObject Ext_Get()
        {
            TclObject to;
            if ((_flags & VarFlags.EXT_LINK_READ_ONLY) != 0 && (_flags & VarFlags.EXT_LINK_INT) != 0)
                if (ext_getset.GetType().Name == "Int32")
                    to = TclInteger.NewInstance((Int32)ext_getset);
                else
                    to = TclInteger.NewInstance(((Ext_GETSET)ext_getset).iValue);
            else if ((_flags & VarFlags.EXT_LINK_INT) != 0)
            {
                if (ext_getset.GetType().Name == "Int32")
                    to = TclInteger.NewInstance((Int32)ext_getset);
                else
                    to = TclInteger.NewInstance(((Ext_GETSET)ext_getset).iValue);
            }
            else
                to = TclString.NewInstance(((Ext_GETSET)ext_getset).sValue);
            to.Preserve();
            return to;
        }

        internal void Ext_Set(TclObject to)
        {
            if ((_flags & VarFlags.EXT_LINK_READ_ONLY) == 0)
            {
                if ((_flags & VarFlags.EXT_LINK_INT) != 0)
                    ((Ext_GETSET)ext_getset).iValue = Convert.ToInt32(to.ToString());
                else
                    if ((_flags & VarFlags.EXT_LINK_STRING) != 0)
                        ((Ext_GETSET)ext_getset).sValue = to.ToString();
                    else
                        ((Ext_GETSET)ext_getset).sValue = to.ToString();
            }
        }

        internal bool isSQLITE3_Link()
        {
            return ((_flags & VarFlags.EXT_LINK) != 0);
        }


        /// <summary> NewVar -> Var
        /// 
        /// Construct a variable and initialize its fields.
        /// </summary>

        internal Var()
        {
            _value = null;
            //name     = null; // Like hashKey in Jacl
            _ns = null;
            HashKey = null; // Like hPtr in the C implementation
            _table = null; // Like hPtr in the C implementation
            RefCount = 0;
            _traces = null;
            //search   = null;
            _sidVec = null; // Like search in the C implementation
            _flags = (VarFlags.SCALAR | VarFlags.UNDEFINED | VarFlags.IN_HASHTABLE);
        }

        /// <summary> Used to create a String that describes this variable
        /// 
        /// </summary>

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(_ns);
            if (sb.Length == 2)
            {
                // It is in the global namespace
                sb.Append(HashKey);
            }
            else
            {
                // It is not in the global namespaces
                sb.Append("::");
                sb.Append(HashKey);
            }
            return sb.ToString();
        }

        /// <summary> Find the SearchId that in the sidVec Vector that is equal the 
        /// unique String s and returns the enumeration associated with
        /// that SearchId.
        /// 
        /// </summary>
        /// <param name="s">String that ia a unique identifier for a SearchId object
        /// </param>
        /// <returns> Enumeration if a match is found else null.
        /// </returns>

        protected internal SearchId getSearch(string s)
        {
            SearchId sid;
            for (int i = 0; i < _sidVec.Count; i++)
            {
                sid = (SearchId)_sidVec[i];
                if (sid.equals(s))
                {
                    return sid;
                }
            }
            return null;
        }


        /// <summary> Find the SearchId object in the sidVec Vector and remove it.
        /// 
        /// </summary>
        /// <param name="sid">String that ia a unique identifier for a SearchId object.
        /// </param>

        protected internal bool removeSearch(string sid)
        {
            SearchId curSid;

            for (int i = 0; i < _sidVec.Count; i++)
            {
                curSid = (SearchId)_sidVec[i];
                if (curSid.equals(sid))
                {
                    _sidVec.RemoveAt(i);
                    return true;
                }
            }
            return false;
        }





        // End of the instance method for the Var class, the rest of the methods
        // are Var related methods ported from the code in generic/tclVar.c



        // The strings below are used to indicate what went wrong when a
        // variable access is denied.

        internal const string noSuchVar = "no such variable";
        internal const string isArray = "variable is array";
        internal const string needArray = "variable isn't array";
        internal const string noSuchElement = "no such element in array";
        internal const string danglingElement = "upvar refers to element in deleted array";
        internal const string danglingVar = "upvar refers to variable in deleted namespace";
        internal const string badNamespace = "parent namespace doesn't exist";
        internal const string missingName = "missing variable name";



        /// <summary> TclLookupVar -> lookupVar
        /// 
        /// This procedure is used by virtually all of the variable
        /// code to locate a variable given its name(s).
        /// 
        /// </summary>
        /// <param name="part1">if part2 isn't NULL, this is the name of an array.
        /// Otherwise, this is a full variable name that could include 
        /// a parenthesized array elemnt or a scalar.
        /// </param>
        /// <param name="part2">Name of an element within array, or null.
        /// </param>
        /// <param name="flags">Only the TCL.VarFlag.GLOBAL_ONLY bit matters.
        /// </param>
        /// <param name="msg">Verb to use in error messages, e.g.  "read" or "set".
        /// </param>
        /// <param name="create">OR'ed combination of CRT_PART1 and CRT_PART2.
        /// Tells which entries to create if they don't already exist.
        /// </param>
        /// <param name="throwException">true if an exception should be throw if the
        /// variable cannot be found.
        /// </param>
        /// <returns> a two element array. a[0] is the variable indicated by
        /// part1 and part2, or null if the variable couldn't be
        /// found and throwException is false.
        /// <p>
        /// If the variable is found, a[1] is the array that
        /// contains the variable (or null if the variable is a scalar).
        /// If the variable can't be found and either createPart1 or
        /// createPart2 are true, a new as-yet-undefined (VAR_UNDEFINED)
        /// variable instance is created, entered into a hash
        /// table, and returned.
        /// Note: it's possible that var.value of the returned variable
        /// may be null (variable undefined), even if createPart1 or createPart2
        /// are true (these only cause the hash table entry or array to be created).
        /// For example, the variable might be a global that has been unset but
        /// is still referenced by a procedure, or a variable that has been unset
        /// but it only being kept in existence by a trace.
        /// </returns>
        /// <exception cref=""> TclException if the variable cannot be found and 
        /// throwException is true.
        /// 
        /// </exception>

        internal static Var[] LookupVar(Interp interp, string part1, string part2, TCL.VarFlag flags, string msg, bool createPart1, bool createPart2)
        {
            CallFrame varFrame = interp.VarFrame;
            // Reference to the procedure call frame whose
            // variables are currently in use. Same as
            // the current procedure's frame, if any,
            // unless an "uplevel" is executing.
            Hashtable table; //  to the hashtable, if any, in which
            // to look up the variable.
            Var var; // Used to search for global names.
            string elName; // Name of array element or null.
            int openParen;
            // If this procedure parses a name into
            // array and index, these point to the
            // parens around the index.  Otherwise they
            // are -1. These are needed to restore
            // the parens after parsing the name.
            NamespaceCmd.Namespace varNs, cxtNs;
            int p;
            int i, result;

            var = null;
            openParen = -1;
            varNs = null; // set non-null if a nonlocal variable

            // Parse part1 into array name and index.
            // Always check if part1 is an array element name and allow it only if
            // part2 is not given.   
            // (if one does not care about creating array elements that can't be used
            // from tcl, and prefer slightly better performance, one can put
            // the following in an   if (part2 == null) { ... } block and remove
            // the part2's test and error reporting  or move that code in array set)
            elName = part2;
            int len = part1.Length;
            for (p = 0; p < len; p++)
            {
                if (part1[p] == '(')
                {
                    openParen = p;
                    p = len - 1;
                    if (part1[p] == ')')
                    {
                        if ((System.Object)part2 != null)
                        {
                            if ((flags & TCL.VarFlag.LEAVE_ERR_MSG) != 0)
                            {
                                throw new TclVarException(interp, part1, part2, msg, needArray);
                            }
                            return null;
                        }
                        elName = part1.Substring(openParen + 1, (len - 1) - (openParen + 1));
                        part2 = elName; // same as elName, only used in error reporting
                        part1 = part1.Substring(0, (openParen) - (0));
                    }
                    break;
                }
            }



            // If this namespace has a variable resolver, then give it first
            // crack at the variable resolution.  It may return a Var
            // value, it may signal to continue onward, or it may signal
            // an error.

            if (((flags & TCL.VarFlag.GLOBAL_ONLY) != 0) || (interp.VarFrame == null))
            {
                cxtNs = interp.GlobalNS;
            }
            else
            {
                cxtNs = interp.VarFrame.NS;
            }

            if (cxtNs.resolver != null || interp._resolvers != null)
            {
                try
                {
                    if (cxtNs.resolver != null)
                    {
                        var = cxtNs.resolver.resolveVar(interp, part1, cxtNs, flags);
                    }
                    else
                    {
                        var = null;
                    }

                    if (var == null && interp._resolvers != null)
                    {
                        IEnumerator enum_Renamed = interp._resolvers.GetEnumerator();
                        foreach (Interp.ResolverScheme res in interp._resolvers)
                        {
                            var = res.resolver.resolveVar(interp, part1, cxtNs, flags);
                            if (var != null)
                                break;
                        }
                    }
                }
                catch (TclException e)
                {
                    var = null;
                }
            }

            // Look up part1. Look it up as either a namespace variable or as a
            // local variable in a procedure call frame (varFrame).
            // Interpret part1 as a namespace variable if:
            //    1) so requested by a TCL.VarFlag.GLOBAL_ONLY or TCL.VarFlag.NAMESPACE_ONLY flag,
            //    2) there is no active frame (we're at the global :: scope),
            //    3) the active frame was pushed to define the namespace context
            //       for a "namespace eval" or "namespace inscope" command,
            //    4) the name has namespace qualifiers ("::"s).
            // Otherwise, if part1 is a local variable, search first in the
            // frame's array of compiler-allocated local variables, then in its
            // hashtable for runtime-created local variables.
            //
            // If createPart1 and the variable isn't found, create the variable and,
            // if necessary, create varFrame's local var hashtable.

            if (((flags & (TCL.VarFlag.GLOBAL_ONLY | TCL.VarFlag.NAMESPACE_ONLY)) != 0) || (varFrame == null) || !varFrame.IsProcCallFrame || (part1.IndexOf("::") != -1))
            {
                string tail;

                // Don't pass TCL.VarFlag.LEAVE_ERR_MSG, we may yet create the variable,
                // or otherwise generate our own error!

                var = NamespaceCmd.findNamespaceVar(interp, part1, null, flags & ~TCL.VarFlag.LEAVE_ERR_MSG);
                if (var == null)
                {
                    if (createPart1)
                    {
                        // var wasn't found so create it

                        // Java does not support passing an address so we pass
                        // an array of size 1 and then assign arr[0] to the value
                        NamespaceCmd.Namespace[] varNsArr = new NamespaceCmd.Namespace[1];
                        NamespaceCmd.Namespace[] dummyArr = new NamespaceCmd.Namespace[1];
                        string[] tailArr = new string[1];

                        NamespaceCmd.getNamespaceForQualName(interp, part1, null, flags, varNsArr, dummyArr, dummyArr, tailArr);

                        // Get the values out of the arrays!
                        varNs = varNsArr[0];
                        tail = tailArr[0];

                        if (varNs == null)
                        {
                            if ((flags & TCL.VarFlag.LEAVE_ERR_MSG) != 0)
                            {
                                throw new TclVarException(interp, part1, part2, msg, badNamespace);
                            }
                            return null;
                        }
                        if ((System.Object)tail == null)
                        {
                            if ((flags & TCL.VarFlag.LEAVE_ERR_MSG) != 0)
                            {
                                throw new TclVarException(interp, part1, part2, msg, missingName);
                            }
                            return null;
                        }
                        var = new Var();
                        varNs.varTable.Add(tail, var);

                        // There is no hPtr member in Jacl, The hPtr combines the table
                        // and the key used in a table lookup.
                        var.HashKey = tail;
                        var._table = varNs.varTable;

                        var._ns = varNs;
                    }
                    else
                    {
                        // var wasn't found and not to create it
                        if ((flags & TCL.VarFlag.LEAVE_ERR_MSG) != 0)
                        {
                            throw new TclVarException(interp, part1, part2, msg, noSuchVar);
                        }
                        return null;
                    }
                }
            }
            else
            {
                // local var: look in frame varFrame
                // removed code block that searches for local compiled vars

                if (var == null)
                {
                    // look in the frame's var hash table
                    table = varFrame.VarTable;
                    if (createPart1)
                    {
                        if (table == null)
                        {
                            table = new Hashtable();
                            varFrame.VarTable = table;
                        }
                        var = (Var)table[part1];
                        if (var == null)
                        {
                            // we are adding a new entry
                            var = new Var();
                            SupportClass.PutElement(table, part1, var);

                            // There is no hPtr member in Jacl, The hPtr combines
                            // the table and the key used in a table lookup.
                            var.HashKey = part1;
                            var._table = table;

                            var._ns = null; // a local variable
                        }
                    }
                    else
                    {
                        if (table != null)
                        {
                            var = (Var)table[part1];
                        }
                        if (var == null)
                        {
                            if ((flags & TCL.VarFlag.LEAVE_ERR_MSG) != 0)
                            {
                                throw new TclVarException(interp, part1, part2, msg, noSuchVar);
                            }
                            return null;
                        }
                    }
                }
            }

            // If var is a link variable, we have a reference to some variable
            // that was created through an "upvar" or "global" command. Traverse
            // through any links until we find the referenced variable.

            while (var.IsVarLink())
            {
                var = (Var)var._value;
            }

            // If we're not dealing with an array element, return var.

            if ((System.Object)elName == null)
            {
                var ret = new Var[2];
                ret[0] = var;
                ret[1] = null;
                return ret;
            }

            // We're dealing with an array element. Make sure the variable is an
            // array and look up the element (create the element if desired).

            if (var.IsVarUndefined() && !var.IsVarArrayElement())
            {
                if (!createPart1)
                {
                    if ((flags & TCL.VarFlag.LEAVE_ERR_MSG) != 0)
                    {
                        throw new TclVarException(interp, part1, part2, msg, noSuchVar);
                    }
                    return null;
                }

                // Make sure we are not resurrecting a namespace variable from a
                // deleted namespace!

                if (((var._flags & VarFlags.IN_HASHTABLE) != 0) && (var._table == null))
                {
                    if ((flags & TCL.VarFlag.LEAVE_ERR_MSG) != 0)
                    {
                        throw new TclVarException(interp, part1, part2, msg, danglingVar);
                    }
                    return null;
                }

                var.SetVarArray();
                var.ClearVarUndefined();
                var._value = new Hashtable();
            }
            else if (!var.IsVarArray())
            {
                if ((flags & TCL.VarFlag.LEAVE_ERR_MSG) != 0)
                {
                    throw new TclVarException(interp, part1, part2, msg, needArray);
                }
                return null;
            }

            Var arrayVar = var;
            Hashtable arrayTable = (Hashtable)var._value;
            if (createPart2)
            {
                Var searchvar = (Var)arrayTable[elName];

                if (searchvar == null)
                {
                    // new entry
                    if (var._sidVec != null)
                    {
                        deleteSearches(var);
                    }

                    var = new Var();
                    SupportClass.PutElement(arrayTable, elName, var);

                    // There is no hPtr member in Jacl, The hPtr combines the table
                    // and the key used in a table lookup.
                    var.HashKey = elName;
                    var._table = arrayTable;

                    var._ns = varNs;
                    var.SetVarArrayElement();
                }
                else
                {
                    var = searchvar;
                }
            }
            else
            {
                var = (Var)arrayTable[elName];
                if (var == null)
                {
                    if ((flags & TCL.VarFlag.LEAVE_ERR_MSG) != 0)
                    {
                        throw new TclVarException(interp, part1, part2, msg, noSuchElement);
                    }
                    return null;
                }
            }

            var ret2 = new Var[2];
            ret2[0] = var; // The Var in the array
            ret2[1] = arrayVar; // The array (Hashtable) Var
            return ret2;
        }


        /// <summary> Query the value of a variable whose name is stored in a Tcl object.
        /// 
        /// </summary>
        /// <param name="interp">the interp that holds the variable
        /// </param>
        /// <param name="nameObj">name of the variable.
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method.
        /// </param>
        /// <returns> the value of the variable.
        /// </returns>

        internal static TclObject GetVar(Interp interp, TclObject nameObj, TCL.VarFlag flags)
        {

            return GetVar(interp, nameObj.ToString(), null, flags);
        }

        /// <summary> Query the value of a variable.
        /// 
        /// </summary>
        /// <param name="interp">the interp that holds the variable
        /// </param>
        /// <param name="name">name of the variable.
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method.
        /// </param>
        /// <returns> the value of the variable.
        /// </returns>

        internal static TclObject GetVar(Interp interp, string name, TCL.VarFlag flags)
        {
            return GetVar(interp, name, null, flags);
        }

        /// <summary> Tcl_ObjGetVar2 -> getVar
        /// 
        /// Query the value of a variable.
        /// 
        /// </summary>
        /// <param name="interp">the interp that holds the variable
        /// </param>
        /// <param name="part1">1st part of the variable name.
        /// </param>
        /// <param name="part2">2nd part of the variable name.
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method.
        /// </param>
        /// <returns> the value of the variable.
        /// </returns>

        internal static TclObject GetVar(Interp interp, TclObject part1Obj, TclObject part2Obj, TCL.VarFlag flags)
        {
            string part1, part2;


            part1 = part1Obj.ToString();

            if (part2Obj != null)
            {

                part2 = part2Obj.ToString();
            }
            else
            {
                part2 = null;
            }

            return GetVar(interp, part1, part2, flags);
        }

        /// <summary> TCL.Tcl_GetVar2Ex -> getVar
        /// 
        /// Query the value of a variable, given a two-part name consisting
        /// of array name and element within array.
        /// 
        /// </summary>
        /// <param name="interp">the interp that holds the variable
        /// </param>
        /// <param name="part1">1st part of the variable name.
        /// </param>
        /// <param name="part2">2nd part of the variable name.
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method.
        /// </param>
        /// <returns> the value of the variable.
        /// </returns>

        internal static TclObject GetVar(Interp interp, string part1, string part2, TCL.VarFlag flags)
        {
            Var[] result = LookupVar(interp, part1, part2, flags, "read", false, true);

            if (result == null)
            {
                // lookupVar() returns null only if TCL.VarFlag.LEAVE_ERR_MSG is
                // not part of the flags argument, return null in this case.

                return null;
            }

            Var var = result[0];
            Var array = result[1];

            try
            {
                // Invoke any traces that have been set for the variable.

                if ((var._traces != null) || ((array != null) && (array._traces != null)))
                {
                    string msg = callTraces(interp, array, var, part1, part2, (flags & (TCL.VarFlag.NAMESPACE_ONLY | TCL.VarFlag.GLOBAL_ONLY)) | TCL.VarFlag.TRACE_READS);
                    if ((System.Object)msg != null)
                    {
                        if ((flags & TCL.VarFlag.LEAVE_ERR_MSG) != 0)
                        {
                            throw new TclVarException(interp, part1, part2, "read", msg);
                        }
                        return null;
                    }
                }

                if (var.IsVarScalar() && !var.IsVarUndefined())
                {
                    return (TclObject)var._value;
                }

                if (var.isSQLITE3_Link())
                    return var.Ext_Get();

                if ((flags & TCL.VarFlag.LEAVE_ERR_MSG) != 0)
                {
                    string msg;
                    if (var.IsVarUndefined() && (array != null) && !array.IsVarUndefined())
                    {
                        msg = noSuchElement;
                    }
                    else if (var.IsVarArray())
                    {
                        msg = isArray;
                    }
                    else
                    {
                        msg = noSuchVar;
                    }
                    throw new TclVarException(interp, part1, part2, "read", msg);
                }
            }
            finally
            {
                // If the variable doesn't exist anymore and no-one's using it,
                // then free up the relevant structures and hash table entries.

                if (var.IsVarUndefined())
                {
                    cleanupVar(var, array);
                }
            }

            return null;
        }

        /// <summary> Set a variable whose name is stored in a Tcl object.
        /// 
        /// </summary>
        /// <param name="interp">the interp that holds the variable
        /// </param>
        /// <param name="nameObj">name of the variable.
        /// </param>
        /// <param name="value">the new value for the variable
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method.
        /// </param>

        internal static TclObject SetVar(Interp interp, TclObject nameObj, TclObject value, TCL.VarFlag flags)
        {

            return SetVar(interp, nameObj.ToString(), null, value, flags);
        }

        /// <summary> Set a variable.
        /// 
        /// </summary>
        /// <param name="interp">the interp that holds the variable
        /// </param>
        /// <param name="name">name of the variable.
        /// </param>
        /// <param name="value">the new value for the variable
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method
        /// </param>

        internal static TclObject SetVar(Interp interp, string name, TclObject value, TCL.VarFlag flags)
        {
            return SetVar(interp, name, null, value, flags);
        }

        /// <summary> Tcl_ObjSetVar2 -> setVar
        /// 
        /// Set the value of a variable.
        /// 
        /// </summary>
        /// <param name="interp">the interp that holds the variable
        /// </param>
        /// <param name="part1">1st part of the variable name.
        /// </param>
        /// <param name="part2">2nd part of the variable name.
        /// </param>
        /// <param name="newValue">the new value for the variable
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method
        /// </param>

        internal static TclObject SetVar(Interp interp, TclObject part1Obj, TclObject part2Obj, TclObject newValue, TCL.VarFlag flags)
        {
            string part1, part2;


            part1 = part1Obj.ToString();

            if (part2Obj != null)
            {

                part2 = part2Obj.ToString();
            }
            else
            {
                part2 = null;
            }

            return SetVar(interp, part1, part2, newValue, flags);
        }


        /// <summary> TCL.Tcl_SetVar2Ex -> setVar
        /// 
        /// Given a two-part variable name, which may refer either to a scalar
        /// variable or an element of an array, change the value of the variable
        /// to a new Tcl object value. If the named scalar or array or element
        /// doesn't exist then create one.
        /// 
        /// </summary>
        /// <param name="interp">the interp that holds the variable
        /// </param>
        /// <param name="part1">1st part of the variable name.
        /// </param>
        /// <param name="part2">2nd part of the variable name.
        /// </param>
        /// <param name="newValue">the new value for the variable
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method
        /// 
        /// Returns a pointer to the TclObject holding the new value of the
        /// variable. If the write operation was disallowed because an array was
        /// expected but not found (or vice versa), then null is returned; if
        /// the TCL.VarFlag.LEAVE_ERR_MSG flag is set, then an exception will be raised.
        /// Note that the returned object may not be the same one referenced
        /// by newValue because variable traces may modify the variable's value.
        /// The value of the given variable is set. If either the array or the
        /// entry didn't exist then a new variable is created.
        /// 
        /// The reference count is decremented for any old value of the variable
        /// and incremented for its new value. If the new value for the variable
        /// is not the same one referenced by newValue (perhaps as a result
        /// of a variable trace), then newValue's ref count is left unchanged
        /// by TCL.Tcl_SetVar2Ex. newValue's ref count is also left unchanged if
        /// we are appending it as a string value: that is, if "flags" includes
        /// TCL.VarFlag.APPEND_VALUE but not TCL.VarFlag.LIST_ELEMENT.
        /// 
        /// The reference count for the returned object is _not_ incremented: if
        /// you want to keep a reference to the object you must increment its
        /// ref count yourself.
        /// </param>

        internal static TclObject SetVar(Interp interp, string part1, string part2, TclObject newValue, TCL.VarFlag flags)
        {
            Var var;
            Var array;
            TclObject oldValue;
            string bytes;

            Var[] result = LookupVar(interp, part1, part2, flags, "set", true, true);
            if (result == null)
            {
                return null;
            }

            var = result[0];
            array = result[1];

            // If the variable is in a hashtable and its table field is null, then we
            // may have an upvar to an array element where the array was deleted
            // or an upvar to a namespace variable whose namespace was deleted.
            // Generate an error (allowing the variable to be reset would screw up
            // our storage allocation and is meaningless anyway).

            if (((var._flags & VarFlags.IN_HASHTABLE) != 0) && (var._table == null))
            {
                if ((flags & TCL.VarFlag.LEAVE_ERR_MSG) != 0)
                {
                    if (var.IsVarArrayElement())
                    {
                        throw new TclVarException(interp, part1, part2, "set", danglingElement);
                    }
                    else
                    {
                        throw new TclVarException(interp, part1, part2, "set", danglingVar);
                    }
                }
                return null;
            }

            // It's an error to try to set an array variable itself.

            if (var.IsVarArray() && !var.IsVarUndefined())
            {
                if ((flags & TCL.VarFlag.LEAVE_ERR_MSG) != 0)
                {
                    throw new TclVarException(interp, part1, part2, "set", isArray);
                }
                return null;
            }


            // At this point, if we were appending, we used to call read traces: we
            // treated append as a read-modify-write. However, it seemed unlikely to
            // us that a real program would be interested in such reads being done
            // during a set operation.

            // Set the variable's new value. If appending, append the new value to
            // the variable, either as a list element or as a string. Also, if
            // appending, then if the variable's old value is unshared we can modify
            // it directly, otherwise we must create a new copy to modify: this is
            // "copy on write".

            try
            {
                if (var.isSQLITE3_Link())
                {
                    var.Ext_Set(newValue);
                    return var.Ext_Get();
                }
                else
                {
                    oldValue = (TclObject)var._value;

                    if ((flags & TCL.VarFlag.APPEND_VALUE) != 0)
                    {
                        if (var.IsVarUndefined() && (oldValue != null))
                        {
                            oldValue.Release(); // discard old value
                            var._value = null;
                            oldValue = null;
                        }
                        if ((flags & TCL.VarFlag.LIST_ELEMENT) != 0)
                        {
                            // append list element
                            if (oldValue == null)
                            {
                                oldValue = TclList.NewInstance();
                                var._value = oldValue;
                                oldValue.Preserve(); // since var is referenced
                            }
                            else if (oldValue.Shared)
                            {
                                // append to copy
                                var._value = oldValue.duplicate();
                                oldValue.Release();
                                oldValue = (TclObject)var._value;
                                oldValue.Preserve(); // since var is referenced
                            }
                            TclList.Append(interp, oldValue, newValue);
                        }
                        else
                        {
                            // append string
                            // We append newValuePtr's bytes but don't change its ref count.


                            bytes = newValue.ToString();
                            if (oldValue == null)
                            {
                                var._value = TclString.NewInstance(bytes);
                                ((TclObject)var._value).Preserve();
                            }
                            else
                            {
                                if (oldValue.Shared)
                                {
                                    // append to copy
                                    var._value = oldValue.duplicate();
                                    oldValue.Release();
                                    oldValue = (TclObject)var._value;
                                    oldValue.Preserve(); // since var is referenced
                                }
                                TclString.append(oldValue, newValue);
                            }
                        }
                    }
                    else
                    {
                        if ((flags & TCL.VarFlag.LIST_ELEMENT) != 0)
                        {
                            // set var to list element
                            int listFlags;

                            // We set the variable to the result of converting newValue's
                            // string rep to a list element. We do not change newValue's
                            // ref count.

                            if (oldValue != null)
                            {
                                oldValue.Release(); // discard old value
                            }

                            bytes = newValue.ToString();
                            listFlags = Util.scanElement(interp, bytes);
                            oldValue = TclString.NewInstance(Util.convertElement(bytes, listFlags));
                            var._value = oldValue;
                            ((TclObject)var._value).Preserve();
                        }
                        else if (newValue != oldValue)
                        {
                            var._value = newValue.duplicate();
                            ((TclObject)var._value).Preserve(); // var is another ref
                            if (oldValue != null)
                            {
                                oldValue.Release(); // discard old value
                            }
                        }
                    }
                    var.SetVarScalar();
                    var.ClearVarUndefined();
                    if (array != null)
                    {
                        array.ClearVarUndefined();
                    }

                    // Invoke any write traces for the variable.

                    if ((var._traces != null) || ((array != null) && (array._traces != null)))
                    {

                        string msg = callTraces(interp, array, var, part1, part2, (flags & (TCL.VarFlag.GLOBAL_ONLY | TCL.VarFlag.NAMESPACE_ONLY)) | TCL.VarFlag.TRACE_WRITES);
                        if ((System.Object)msg != null)
                        {
                            if ((flags & TCL.VarFlag.LEAVE_ERR_MSG) != 0)
                            {
                                throw new TclVarException(interp, part1, part2, "set", msg);
                            }
                            return null; // Same as "goto cleanup" in C verison
                        }
                    }

                    // Return the variable's value unless the variable was changed in some
                    // gross way by a trace (e.g. it was unset and then recreated as an
                    // array).

                    if (var.IsVarScalar() && !var.IsVarUndefined())
                    {
                        return (TclObject)var._value;
                    }

                    // A trace changed the value in some gross way. Return an empty string
                    // object.

                    return TclString.NewInstance("");
                }
            }
            finally
            {
                // If the variable doesn't exist anymore and no-one's using it,
                // then free up the relevant structures and hash table entries.

                if (var.IsVarUndefined())
                {
                    cleanupVar(var, array);
                }
            }
        }


        /// <summary>  TclIncrVar2 -> incrVar
        /// 
        /// Given a two-part variable name, which may refer either to a scalar
        /// variable or an element of an array, increment the Tcl object value
        /// of the variable by a specified amount.
        /// 
        /// </summary>
        /// <param name="part1">1st part of the variable name.
        /// </param>
        /// <param name="part2">2nd part of the variable name.
        /// </param>
        /// <param name="incrAmount">Amount to be added to variable.
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method
        /// 
        /// Results:
        /// Returns a reference to the TclObject holding the new value of the
        /// variable. If the specified variable doesn't exist, or there is a
        /// clash in array usage, or an error occurs while executing variable
        /// traces, then a TclException will be raised.
        /// 
        /// Side effects:
        /// The value of the given variable is incremented by the specified
        /// amount. If either the array or the entry didn't exist then a new
        /// variable is created. The ref count for the returned object is _not_
        /// incremented to reflect the returned reference; if you want to keep a
        /// reference to the object you must increment its ref count yourself.
        /// 
        /// ----------------------------------------------------------------------
        /// </param>

        internal static TclObject incrVar(Interp interp, TclObject part1, TclObject part2, int incrAmount, TCL.VarFlag flags)
        {
            TclObject varValue = null;
            bool createdNewObj; // Set to true if var's value object is shared
            // so we must increment a copy (i.e. copy
            // on write).
            int i;
            bool err;

            // There are two possible error conditions that depend on the setting of
            // TCL.VarFlag.LEAVE_ERR_MSG. an exception could be raised or null could be returned
            err = false;
            try
            {
                varValue = GetVar(interp, part1, part2, flags);
            }
            catch (TclException e)
            {
                err = true;
                throw;
            }
            finally
            {
                // FIXME : is this the correct way to catch the error?
                if (err || varValue == null)
                    interp.AddErrorInfo("\n    (reading value of variable to increment)");
            }


            // Increment the variable's value. If the object is unshared we can
            // modify it directly, otherwise we must create a new copy to modify:
            // this is "copy on write". Then free the variable's old string
            // representation, if any, since it will no longer be valid.

            createdNewObj = false;
            if (varValue.Shared)
            {
                varValue = varValue.duplicate();
                createdNewObj = true;
            }

            try
            {
                i = TclInteger.get(interp, varValue);
            }
            catch (TclException e)
            {
                if (createdNewObj)
                {
                    varValue.Release(); // free unneeded copy
                }
                throw;
            }

            TclInteger.set(varValue, (i + incrAmount));

            // Store the variable's new value and run any write traces.

            return SetVar(interp, part1, part2, varValue, flags);
        }

        /// <summary> Unset a variable whose name is stored in a Tcl object.
        /// 
        /// </summary>
        /// <param name="nameObj">name of the variable.
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method.
        /// </param>

        internal static void UnsetVar(Interp interp, TclObject nameObj, TCL.VarFlag flags)
        {

            UnsetVar(interp, nameObj.ToString(), null, flags);
        }

        /// <summary> Unset a variable.
        /// 
        /// </summary>
        /// <param name="name">name of the variable.
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method.
        /// </param>

        internal static void UnsetVar(Interp interp, string name, TCL.VarFlag flags)
        {
            UnsetVar(interp, name, null, flags);
        }

        /// <summary> TCL.Tcl_UnsetVar2 -> unsetVar
        /// 
        /// Unset a variable, given a two-part name consisting of array
        /// name and element within array.
        /// 
        /// </summary>
        /// <param name="part1">1st part of the variable name.
        /// </param>
        /// <param name="part2">2nd part of the variable name.
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method.
        /// 
        /// If part1 and part2 indicate a local or global variable in interp,
        /// it is deleted.  If part1 is an array name and part2 is null, then
        /// the whole array is deleted.
        /// 
        /// </param>

        internal static void UnsetVar(Interp interp, string part1, string part2, TCL.VarFlag flags)
        {
            Var dummyVar;
            Var var;
            Var array;
            //ActiveVarTrace active;
            TclObject obj;
            TCL.CompletionCode result;

            // FIXME : what about the null return vs exception thing here?
            Var[] lookup_result = LookupVar(interp, part1, part2, flags, "unset", false, false);
            if (lookup_result == null)
            {
                if ((flags & TCL.VarFlag.LEAVE_ERR_MSG) != 0)
                    throw new TclRuntimeError("unexpected null reference");
                else
                    return;
            }

            var = lookup_result[0];
            array = lookup_result[1];

            result = (var.IsVarUndefined() ? TCL.CompletionCode.ERROR : TCL.CompletionCode.OK);

            if ((array != null) && (array._sidVec != null))
            {
                deleteSearches(array);
            }


            // The code below is tricky, because of the possibility that
            // a trace procedure might try to access a variable being
            // deleted. To handle this situation gracefully, do things
            // in three steps:
            // 1. Copy the contents of the variable to a dummy variable
            //    structure, and mark the original Var structure as undefined.
            // 2. Invoke traces and clean up the variable, using the dummy copy.
            // 3. If at the end of this the original variable is still
            //    undefined and has no outstanding references, then delete
            //	  it (but it could have gotten recreated by a trace).

            dummyVar = new Var();
            //FIXME: Var class really should implement clone to make a bit copy. 
            dummyVar._value = var._value;
            dummyVar._traces = var._traces;
            dummyVar._flags = var._flags;
            dummyVar.HashKey = var.HashKey;
            dummyVar._table = var._table;
            dummyVar.RefCount = var.RefCount;
            dummyVar._ns = var._ns;

            var.SetVarUndefined();
            var.SetVarScalar();
            var._value = null; // dummyVar points to any value object
            var._traces = null;
            var._sidVec = null;

            // Call trace procedures for the variable being deleted. Then delete
            // its traces. Be sure to abort any other traces for the variable
            // that are still pending. Special tricks:
            // 1. We need to increment var's refCount around this: CallTraces
            //    will use dummyVar so it won't increment var's refCount itself.
            // 2. Turn off the TRACE_ACTIVE flag in dummyVar: we want to
            //    call unset traces even if other traces are pending.

            if ((dummyVar._traces != null) || ((array != null) && (array._traces != null)))
            {
                var.RefCount++;
                dummyVar._flags &= ~VarFlags.TRACE_ACTIVE;
                callTraces(interp, array, dummyVar, part1, part2, (flags & (TCL.VarFlag.GLOBAL_ONLY | TCL.VarFlag.NAMESPACE_ONLY)) | TCL.VarFlag.TRACE_UNSETS);

                dummyVar._traces = null;

                // Active trace stuff is not part of Jacl's interp

                var.RefCount--;
            }

            // If the variable is an array, delete all of its elements. This must be
            // done after calling the traces on the array, above (that's the way
            // traces are defined). If it is a scalar, "discard" its object
            // (decrement the ref count of its object, if any).

            if (dummyVar.IsVarArray() && !dummyVar.IsVarUndefined())
            {
                deleteArray(interp, part1, dummyVar, (flags & (TCL.VarFlag.GLOBAL_ONLY | TCL.VarFlag.NAMESPACE_ONLY)) | TCL.VarFlag.TRACE_UNSETS);
            }
            if (dummyVar.IsVarScalar() && (dummyVar._value != null))
            {
                obj = (TclObject)dummyVar._value;
                obj.Release();
                dummyVar._value = null;
            }

            // If the variable was a namespace variable, decrement its reference count.

            if ((var._flags & VarFlags.NAMESPACE_VAR) != 0)
            {
                var._flags &= ~VarFlags.NAMESPACE_VAR;
                var.RefCount--;
            }

            // Finally, if the variable is truly not in use then free up its Var
            // structure and remove it from its hash table, if any. The ref count of
            // its value object, if any, was decremented above.

            cleanupVar(var, array);

            // It's an error to unset an undefined variable.

            if (result != TCL.CompletionCode.OK)
            {
                if ((flags & TCL.VarFlag.LEAVE_ERR_MSG) != 0)
                {
                    throw new TclVarException(interp, part1, part2, "unset", ((array == null) ? noSuchVar : noSuchElement));
                }
            }
        }


        /// <summary> Trace a variable whose name is stored in a Tcl object.
        /// 
        /// </summary>
        /// <param name="nameObj">name of the variable.
        /// </param>
        /// <param name="trace">the trace to add.
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method.
        /// </param>

        internal static void TraceVar(Interp interp, TclObject nameObj, TCL.VarFlag flags, VarTrace proc)
        {

            TraceVar(interp, nameObj.ToString(), null, flags, proc);
        }

        /// <summary> Trace a variable.
        /// 
        /// </summary>
        /// <param name="name">name of the variable.
        /// </param>
        /// <param name="trace">the trace to add.
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method.
        /// </param>

        internal static void TraceVar(Interp interp, string name, TCL.VarFlag flags, VarTrace proc)
        {
            TraceVar(interp, name, null, flags, proc);
        }

        /// <summary> TCL.Tcl_TraceVar2 -> traceVar
        /// 
        /// Trace a variable, given a two-part name consisting of array
        /// name and element within array.
        /// 
        /// </summary>
        /// <param name="part1">1st part of the variable name.
        /// </param>
        /// <param name="part2">2nd part of the variable name.
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method.
        /// </param>
        /// <param name="trace">the trace to comand to add.
        /// </param>

        internal static void TraceVar(Interp interp, string part1, string part2, TCL.VarFlag flags, VarTrace proc)
        {
            Var[] result;
            Var var, array;

            // FIXME: what about the exception problem here?
            result = LookupVar(interp, part1, part2, (flags | TCL.VarFlag.LEAVE_ERR_MSG), "trace", true, true);
            if (result == null)
            {
                throw new TclException(interp, "");
            }

            var = result[0];
            array = result[1];

            // Set up trace information.

            if (var._traces == null)
            {
                var._traces = new ArrayList(10);
            }

            var rec = new TraceRecord();
            rec.trace = proc;
            rec.flags = flags & (TCL.VarFlag.TRACE_READS | TCL.VarFlag.TRACE_WRITES | TCL.VarFlag.TRACE_UNSETS | TCL.VarFlag.TRACE_ARRAY);

            var._traces.Insert(0, rec);


            // FIXME: is this needed ?? It was in Jacl but not 8.1

            /*
            // When inserting a trace for an array on an UNDEFINED variable,
            // the search IDs for that array are reset.

            if (array != null && var.isVarUndefined()) {
            array.sidVec = null;
            }
            */
        }


        /// <summary> Untrace a variable whose name is stored in a Tcl object.
        /// 
        /// </summary>
        /// <param name="nameObj">name of the variable.
        /// </param>
        /// <param name="trace">the trace to delete.
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method.
        /// </param>

        internal static void UntraceVar(Interp interp, TclObject nameObj, TCL.VarFlag flags, VarTrace proc)
        {

            UntraceVar(interp, nameObj.ToString(), null, flags, proc);
        }

        /// <summary> Untrace a variable.
        /// 
        /// </summary>
        /// <param name="name">name of the variable.
        /// </param>
        /// <param name="trace">the trace to delete.
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method.
        /// </param>

        internal static void UntraceVar(Interp interp, string name, TCL.VarFlag flags, VarTrace proc)
        {
            UntraceVar(interp, name, null, flags, proc);
        }

        /// <summary> TCL.Tcl_UntraceVar2 -> untraceVar
        /// 
        /// Untrace a variable, given a two-part name consisting of array
        /// name and element within array. This will Remove a
        /// previously-created trace for a variable.
        /// 
        /// </summary>
        /// <param name="interp">Interpreter containing variable.
        /// </param>
        /// <param name="part1">1st part of the variable name.
        /// </param>
        /// <param name="part2">2nd part of the variable name.
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method.
        /// </param>
        /// <param name="proc">the trace to delete.
        /// </param>

        internal static void UntraceVar(Interp interp, string part1, string part2, TCL.VarFlag flags, VarTrace proc)
        {
            Var[] result = null;
            Var var;

            try
            {
                result = LookupVar(interp, part1, part2, flags & (TCL.VarFlag.GLOBAL_ONLY | TCL.VarFlag.NAMESPACE_ONLY), null, false, false);
                if (result == null)
                {
                    return;
                }
            }
            catch (TclException e)
            {
                // FIXME: check for problems in exception in lookupVar

                // We have set throwException argument to false in the
                // lookupVar() call, so an exception should never be
                // thrown.

                throw new TclRuntimeError("unexpected TclException: " + e.Message, e);
            }

            var = result[0];

            if (var._traces != null)
            {
                int len = var._traces.Count;
                for (int i = 0; i < len; i++)
                {
                    TraceRecord rec = (TraceRecord)var._traces[i];
                    if (rec.trace == proc)
                    {
                        var._traces.RemoveAt(i);
                        break;
                    }
                }
            }

            // If this is the last trace on the variable, and the variable is
            // unset and unused, then free up the variable.

            if (var.IsVarUndefined())
            {
                cleanupVar(var, null);
            }
        }

        /// <summary> TCL.Tcl_VarTraceInfo -> getTraces
        /// 
        /// </summary>
        /// <param name="interp">Interpreter containing variable.
        /// </param>
        /// <param name="name">name of the variable.
        /// </param>
        /// <param name="flags">flags that control the actions of this method.
        /// </param>
        /// <returns> the Vector of traces of a variable.
        /// </returns>

        static protected internal ArrayList getTraces(Interp interp, string name, TCL.VarFlag flags)
        {
            return getTraces(interp, name, null, flags);
        }

        /// <summary> TCL.Tcl_VarTraceInfo2 -> getTraces
        /// 
        /// </summary>
        /// <returns> the list of traces of a variable.
        /// 
        /// </returns>
        /// <param name="interp">Interpreter containing variable.
        /// </param>
        /// <param name="part1">1st part of the variable name.
        /// </param>
        /// <param name="part2">2nd part of the variable name (can be null).
        /// </param>
        /// <param name="flags">misc flags that control the actions of this method.
        /// </param>

        static protected internal ArrayList getTraces(Interp interp, string part1, string part2, TCL.VarFlag flags)
        {
            Var[] result;

            result = LookupVar(interp, part1, part2, flags & (TCL.VarFlag.GLOBAL_ONLY | TCL.VarFlag.NAMESPACE_ONLY), null, false, false);

            if (result == null)
            {
                return null;
            }

            return result[0]._traces;
        }


        /// <summary> MakeUpvar -> makeUpvar
        /// 
        /// Create a reference of a variable in otherFrame in the current
        /// CallFrame, given a two-part name consisting of array name and
        /// element within array.
        /// 
        /// </summary>
        /// <param name="interp">Interp containing the variables
        /// </param>
        /// <param name="frame">CallFrame containing "other" variable.
        /// null means use global context.
        /// </param>
        /// <param name="otherP1">the 1st part name of the variable in the "other" frame.
        /// </param>
        /// <param name="otherP2">the 2nd part name of the variable in the "other" frame.
        /// </param>
        /// <param name="otherFlags">the flags for scaope of "other" variable
        /// </param>
        /// <param name="myName">Name of scalar variable which will refer to otherP1/otherP2.
        /// </param>
        /// <param name="myFlags">only the TCL.VarFlag.GLOBAL_ONLY bit matters, 
        /// indicating the scope of myName.
        /// </param>
        /// <exception cref=""> TclException if the upvar cannot be created. 
        /// </exception>

        protected internal static void makeUpvar(Interp interp, CallFrame frame, string otherP1, string otherP2, TCL.VarFlag otherFlags, string myName, TCL.VarFlag myFlags)
        {
            Var other, var, array;
            Var[] result;
            CallFrame varFrame;
            CallFrame savedFrame = null;
            Hashtable table;
            NamespaceCmd.Namespace ns, altNs;
            string tail;
            bool newvar = false;

            // Find "other" in "frame". If not looking up other in just the
            // current namespace, temporarily replace the current var frame
            // pointer in the interpreter in order to use TclLookupVar.

            if ((otherFlags & TCL.VarFlag.NAMESPACE_ONLY) == 0)
            {
                savedFrame = interp.VarFrame;
                interp.VarFrame = frame;
            }
            result = LookupVar(interp, otherP1, otherP2, (otherFlags | TCL.VarFlag.LEAVE_ERR_MSG), "access", true, true);

            if ((otherFlags & TCL.VarFlag.NAMESPACE_ONLY) == 0)
            {
                interp.VarFrame = savedFrame;
            }

            other = result[0];
            array = result[1];

            if (other == null)
            {
                // FIXME : leave error message thing again
                throw new TclRuntimeError("unexpected null reference");
            }

            // Now create a hashtable entry for "myName". Create it as either a
            // namespace variable or as a local variable in a procedure call
            // frame. Interpret myName as a namespace variable if:
            //    1) so requested by a TCL.VarFlag.GLOBAL_ONLY or TCL.VarFlag.NAMESPACE_ONLY flag,
            //    2) there is no active frame (we're at the global :: scope),
            //    3) the active frame was pushed to define the namespace context
            //       for a "namespace eval" or "namespace inscope" command,
            //    4) the name has namespace qualifiers ("::"s).
            // If creating myName in the active procedure, look in its
            // hashtable for runtime-created local variables. Create that
            // procedure's local variable hashtable if necessary.

            varFrame = interp.VarFrame;
            if (((myFlags & (TCL.VarFlag.GLOBAL_ONLY | TCL.VarFlag.NAMESPACE_ONLY)) != 0) || (varFrame == null) || !varFrame.IsProcCallFrame || (myName.IndexOf("::") != -1))
            {

                // Java does not support passing an address so we pass
                // an array of size 1 and then assign arr[0] to the value
                NamespaceCmd.Namespace[] nsArr = new NamespaceCmd.Namespace[1];
                NamespaceCmd.Namespace[] altNsArr = new NamespaceCmd.Namespace[1];
                NamespaceCmd.Namespace[] dummyNsArr = new NamespaceCmd.Namespace[1];
                string[] tailArr = new string[1];

                NamespaceCmd.getNamespaceForQualName(interp, myName, null, myFlags, nsArr, altNsArr, dummyNsArr, tailArr);

                // Get the values out of the arrays!
                ns = nsArr[0];
                altNs = altNsArr[0];
                tail = tailArr[0];

                if (ns == null)
                {
                    ns = altNs;
                }
                if (ns == null)
                {
                    throw new TclException(interp, "bad variable name \"" + myName + "\": unknown namespace");
                }

                // Check that we are not trying to create a namespace var linked to
                // a local variable in a procedure. If we allowed this, the local
                // variable in the shorter-lived procedure frame could go away
                // leaving the namespace var's reference invalid.

                if ((((System.Object)otherP2 != null) ? array._ns : other._ns) == null)
                {
                    throw new TclException(interp, "bad variable name \"" + myName + "\": upvar won't create namespace variable that refers to procedure variable");
                }

                // AKT var = (Var) ns.varTable.get(tail);
                var = (Var)ns.varTable[tail];
                if (var == null)
                {
                    // we are adding a new entry
                    newvar = true;
                    var = new Var();
                    // ATK ns.varTable.put(tail, var);
                    ns.varTable.Add(tail, var);

                    // There is no hPtr member in Jacl, The hPtr combines the table
                    // and the key used in a table lookup.
                    var.HashKey = tail;
                    var._table = ns.varTable;

                    var._ns = ns;
                }
            }
            else
            {
                // Skip Compiled Local stuff
                var = null;
                if (var == null)
                {
                    // look in frame's local var hashtable
                    table = varFrame.VarTable;
                    if (table == null)
                    {
                        table = new Hashtable();
                        varFrame.VarTable = table;
                    }

                    var = (Var)table[myName];
                    if (var == null)
                    {
                        // we are adding a new entry
                        newvar = true;
                        var = new Var();
                        SupportClass.PutElement(table, myName, var);

                        // There is no hPtr member in Jacl, The hPtr combines the table
                        // and the key used in a table lookup.
                        var.HashKey = myName;
                        var._table = table;

                        var._ns = varFrame.NS;
                    }
                }
            }

            if (!newvar)
            {
                // The variable already exists. Make sure this variable "var"
                // isn't the same as "other" (avoid circular links). Also, if
                // it's not an upvar then it's an error. If it is an upvar, then
                // just disconnect it from the thing it currently refers to.

                if (var == other)
                {
                    throw new TclException(interp, "can't upvar from variable to itself");
                }
                if (var.IsVarLink())
                {
                    Var link = (Var)var._value;
                    if (link == other)
                    {
                        return;
                    }
                    link.RefCount--;
                    if (link.IsVarUndefined())
                    {
                        cleanupVar(link, null);
                    }
                }
                else if (!var.IsVarUndefined())
                {
                    throw new TclException(interp, "variable \"" + myName + "\" already exists");
                }
                else if (var._traces != null)
                {
                    throw new TclException(interp, "variable \"" + myName + "\" has traces: can't use for upvar");
                }
            }

            var.SetVarLink();
            var.ClearVarUndefined();
            var._value = other;
            other.RefCount++;
            return;
        }

        /*
        *----------------------------------------------------------------------
        *
        * TCL.Tcl_GetVariableFullName -> getVariableFullName
        *
        *  Given a Var token returned by NamespaceCmd.FindNamespaceVar, this
        *	procedure appends to an object the namespace variable's full
        *	name, qualified by a sequence of parent namespace names.
        *
        * Results:
        *  None.
        *
        * Side effects:
        *  The variable's fully-qualified name is returned.
        *
        *----------------------------------------------------------------------
        */

        internal static string getVariableFullName(Interp interp, Var var)
        {
            StringBuilder buff = new StringBuilder();

            // Add the full name of the containing namespace (if any), followed by
            // the "::" separator, then the variable name.

            if (var != null)
            {
                if (!var.IsVarArrayElement())
                {
                    if (var._ns != null)
                    {
                        buff.Append(var._ns.fullName);
                        if (var._ns != interp.GlobalNS)
                        {
                            buff.Append("::");
                        }
                    }
                    // Jacl's Var class does not include the "name" member
                    // We use the "hashKey" member which is equivalent

                    if ((System.Object)var.HashKey != null)
                    {
                        buff.Append(var.HashKey);
                    }
                }
            }

            return buff.ToString();
        }

        /// <summary> CallTraces -> callTraces
        /// 
        /// This procedure is invoked to find and invoke relevant
        /// trace procedures associated with a particular operation on
        /// a variable.  This procedure invokes traces both on the
        /// variable and on its containing array (where relevant).
        /// 
        /// </summary>
        /// <param name="interp">Interpreter containing variable.
        /// </param>
        /// <param name="array">array variable that contains the variable, or null
        /// if the variable isn't an element of an array.
        /// </param>
        /// <param name="var">Variable whose traces are to be invoked.
        /// </param>
        /// <param name="part1">the first part of a variable name.
        /// </param>
        /// <param name="part2">the second part of a variable name.
        /// </param>
        /// <param name="flags">Flags to pass to trace procedures: indicates
        /// what's happening to variable, plus other stuff like
        /// TCL.VarFlag.GLOBAL_ONLY, TCL.VarFlag.NAMESPACE_ONLY, and TCL.VarFlag.INTERP_DESTROYED.
        /// </param>
        /// <returns> null if no trace procedures were invoked, or
        /// if all the invoked trace procedures returned successfully.
        /// The return value is non-null if a trace procedure returned an
        /// error (in this case no more trace procedures were invoked
        /// after the error was returned). In this case the return value
        /// is a pointer to a string describing the error.
        /// </returns>

        static protected internal string callTraces(Interp interp, Var array, Var var, string part1, string part2, TCL.VarFlag flags)
        {
            TclObject oldResult;
            int i;

            // If there are already similar trace procedures active for the
            // variable, don't call them again.

            if ((var._flags & VarFlags.TRACE_ACTIVE) != 0)
            {
                return null;
            }
            var._flags |= VarFlags.TRACE_ACTIVE;
            var.RefCount++;

            // If the variable name hasn't been parsed into array name and
            // element, do it here.  If there really is an array element,
            // make a copy of the original name so that nulls can be
            // inserted into it to separate the names (can't modify the name
            // string in place, because the string might get used by the
            // callbacks we invoke).

            // FIXME : come up with parsing code to use for all situations!
            if ((System.Object)part2 == null)
            {
                int len = part1.Length;

                if (len > 0)
                {
                    if (part1[len - 1] == ')')
                    {
                        for (i = 0; i < len - 1; i++)
                        {
                            if (part1[i] == '(')
                            {
                                break;
                            }
                        }
                        if (i < len - 1)
                        {
                            if (i < len - 2)
                            {
                                part2 = part1.Substring(i + 1, (len - 1) - (i + 1));
                                part1 = part1.Substring(0, (i) - (0));
                            }
                        }
                    }
                }
            }

            oldResult = interp.GetResult();
            oldResult.Preserve();
            interp.ResetResult();

            try
            {
                // Invoke traces on the array containing the variable, if relevant.

                if (array != null)
                {
                    array.RefCount++;
                }
                if ((array != null) && (array._traces != null))
                {
                    for (i = 0; (array._traces != null) && (i < array._traces.Count); i++)
                    {
                        TraceRecord rec = (TraceRecord)array._traces[i];
                        if ((rec.flags & flags) != 0)
                        {
                            try
                            {
                                rec.trace.traceProc(interp, part1, part2, flags);
                            }
                            catch (TclException e)
                            {
                                if ((flags & TCL.VarFlag.TRACE_UNSETS) == 0)
                                {

                                    return interp.GetResult().ToString();
                                }
                            }
                        }
                    }
                }

                // Invoke traces on the variable itself.

                if ((flags & TCL.VarFlag.TRACE_UNSETS) != 0)
                {
                    flags |= TCL.VarFlag.TRACE_DESTROYED;
                }

                for (i = 0; (var._traces != null) && (i < var._traces.Count); i++)
                {
                    TraceRecord rec = (TraceRecord)var._traces[i];
                    if ((rec.flags & flags) != 0)
                    {
                        try
                        {
                            rec.trace.traceProc(interp, part1, part2, flags);
                        }
                        catch (TclException e)
                        {
                            if ((flags & TCL.VarFlag.TRACE_UNSETS) == 0)
                            {

                                return interp.GetResult().ToString();
                            }
                        }
                    }
                }

                return null;
            }
            finally
            {
                if (array != null)
                {
                    array.RefCount--;
                }
                var._flags &= ~VarFlags.TRACE_ACTIVE;
                var.RefCount--;

                interp.setResult(oldResult);
                oldResult.Release();
            }
        }

        /// <summary> DeleteSearches -> deleteSearches
        /// 
        /// This procedure is called to free up all of the searches
        /// associated with an array variable.
        /// 
        /// </summary>
        /// <param name="interp">Interpreter containing array.
        /// </param>
        /// <param name="arrayVar">the array variable to delete searches from.
        /// </param>

        static protected internal void deleteSearches(Var arrayVar)
        // Variable whose searches are to be deleted.
        {
            arrayVar._sidVec = null;
        }

        /// <summary> TclDeleteVars -> deleteVars
        /// 
        /// This procedure is called to recycle all the storage space
        /// associated with a table of variables. For this procedure
        /// to work correctly, it must not be possible for any of the
        /// variables in the table to be accessed from Tcl commands
        /// (e.g. from trace procedures).
        /// 
        /// </summary>
        /// <param name="interp">Interpreter containing array.
        /// </param>
        /// <param name="table">Hashtbale that holds the Vars to delete
        /// </param>

        static protected internal void DeleteVars(Interp interp, Hashtable table)
        {
            IEnumerator search;
            string hashKey;
            Var var;
            Var link;
            TCL.VarFlag flags;
            //ActiveVarTrace active;
            TclObject obj;
            NamespaceCmd.Namespace currNs = NamespaceCmd.getCurrentNamespace(interp);

            // Determine what flags to pass to the trace callback procedures.

            flags = TCL.VarFlag.TRACE_UNSETS;
            if (table == interp.GlobalNS.varTable)
            {
                flags |= (TCL.VarFlag.INTERP_DESTROYED | TCL.VarFlag.GLOBAL_ONLY);
            }
            else if (table == currNs.varTable)
            {
                flags |= TCL.VarFlag.NAMESPACE_ONLY;
            }


            for (search = table.Values.GetEnumerator(); search.MoveNext(); )
            {
                var = (Var)search.Current;

                // For global/upvar variables referenced in procedures, decrement
                // the reference count on the variable referred to, and free
                // the referenced variable if it's no longer needed. Don't delete
                // the hash entry for the other variable if it's in the same table
                // as us: this will happen automatically later on.

                if (var.IsVarLink())
                {
                    link = (Var)var._value;
                    link.RefCount--;
                    if ((link.RefCount == 0) && link.IsVarUndefined() && (link._traces == null) && ((link._flags & VarFlags.IN_HASHTABLE) != 0))
                    {

                        if ((System.Object)link.HashKey == null)
                        {
                            var._value = null; // Drops reference to the link Var
                        }
                        else if (link._table != table)
                        {
                            SupportClass.HashtableRemove(link._table, link.HashKey);
                            link._table = null; // Drops the link var's table reference
                            var._value = null; // Drops reference to the link Var
                        }
                    }
                }

                // free up the variable's space (no need to free the hash entry
                // here, unless we're dealing with a global variable: the
                // hash entries will be deleted automatically when the whole
                // table is deleted). Note that we give callTraces the variable's
                // fully-qualified name so that any called trace procedures can
                // refer to these variables being deleted.

                if (var._traces != null)
                {
                    string fullname = getVariableFullName(interp, var);

                    callTraces(interp, null, var, fullname, null, flags);

                    // The var.traces = null statement later will drop all the
                    // references to the traces which will free them up
                }

                if (var.IsVarArray())
                {
                    deleteArray(interp, var.HashKey, var, flags);
                    var._value = null;
                }
                if (var.IsVarScalar() && (var._value != null))
                {
                    obj = (TclObject)var._value;
                    obj.Release();
                    var._value = null;
                }

                // There is no hPtr member in Jacl, The hPtr combines the table
                // and the key used in a table lookup.
                var.HashKey = null;
                var._table = null;
                var._traces = null;
                var.SetVarUndefined();
                var.SetVarScalar();

                // If the variable was a namespace variable, decrement its 
                // reference count. We are in the process of destroying its
                // namespace so that namespace will no longer "refer" to the
                // variable.

                if ((var._flags & VarFlags.NAMESPACE_VAR) != 0)
                {
                    var._flags &= ~VarFlags.NAMESPACE_VAR;
                    var.RefCount--;
                }

                // Recycle the variable's memory space if there aren't any upvar's
                // pointing to it. If there are upvars to this variable, then the
                // variable will get freed when the last upvar goes away.

                if (var.RefCount == 0)
                {
                    // When we drop the last reference it will be freeded
                }
            }
            table.Clear();
        }


        /// <summary> DeleteArray -> deleteArray
        /// 
        /// This procedure is called to free up everything in an array
        /// variable.  It's the caller's responsibility to make sure
        /// that the array is no longer accessible before this procedure
        /// is called.
        /// 
        /// </summary>
        /// <param name="interp">Interpreter containing array.
        /// </param>
        /// <param name="arrayName">name of array (used for trace callbacks).
        /// </param>
        /// <param name="var">the array variable to delete.
        /// </param>
        /// <param name="flags">Flags to pass to CallTraces.
        /// </param>

        static protected internal void deleteArray(Interp interp, string arrayName, Var var, TCL.VarFlag flags)
        {
            IEnumerator search;
            Var el;
            TclObject obj;

            deleteSearches(var);
            Hashtable table = (Hashtable)var._value;

            Var dummyVar;
            for (search = table.Values.GetEnumerator(); search.MoveNext(); )
            {
                el = (Var)search.Current;

                if (el.IsVarScalar() && (el._value != null))
                {
                    obj = (TclObject)el._value;
                    obj.Release();
                    el._value = null;
                }

                string tmpkey = (string)el.HashKey;
                // There is no hPtr member in Jacl, The hPtr combines the table
                // and the key used in a table lookup.
                el.HashKey = null;
                el._table = null;
                if (el._traces != null)
                {
                    el._flags &= ~VarFlags.TRACE_ACTIVE;
                    // FIXME : Old Jacl impl passed a dummy var to callTraces, should we?
                    callTraces(interp, null, el, arrayName, tmpkey, flags);
                    el._traces = null;
                    // Active trace stuff is not part of Jacl
                }
                el.SetVarUndefined();
                el.SetVarScalar();
                if (el.RefCount == 0)
                {
                    // We are no longer using the element
                    // element Vars are IN_HASHTABLE
                }
            }
            ((Hashtable)var._value).Clear();
            var._value = null;
        }


        /// <summary> CleanupVar -> cleanupVar
        /// 
        /// This procedure is called when it looks like it may be OK
        /// to free up the variable's record and hash table entry, and
        /// those of its containing parent.  It's called, for example,
        /// when a trace on a variable deletes the variable.
        /// 
        /// </summary>
        /// <param name="var">variable that may be a candidate for being expunged.
        /// </param>
        /// <param name="array">Array that contains the variable, or NULL if this
        /// variable isn't an array element.
        /// </param>

        static protected internal void cleanupVar(Var var, Var array)
        {
            if (var.IsVarUndefined() && (var.RefCount == 0) && (var._traces == null) && ((var._flags & VarFlags.IN_HASHTABLE) != 0))
            {
                if (var._table != null)
                {
                    SupportClass.HashtableRemove(var._table, var.HashKey);
                    var._table = null;
                    var.HashKey = null;
                }
            }
            if (array != null)
            {
                if (array.IsVarUndefined() && (array.RefCount == 0) && (array._traces == null) && ((array._flags & VarFlags.IN_HASHTABLE) != 0))
                {
                    if (array._table != null)
                    {
                        SupportClass.HashtableRemove(array._table, array.HashKey);
                        array._table = null;
                        array.HashKey = null;
                    }
                }
            }
        }
    } // End of Var class
}
