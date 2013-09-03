#region Foreign-License
/*
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{

    /// <summary> This class implements the object type in Tcl.</summary>

    public class TclObj : IInternalRep
    {
        /// <summary> Internal representation of a object value.</summary>
        private object value;

        /// <summary> Construct a TclObj representation with the given object value.</summary>
        private TclObj(object o)
        {
            value = o;
        }

        /// <summary> Returns a dupilcate of the current object.</summary>
        /// <param name="obj">the TclObject that contains this internalRep.
        /// </param>
        public IInternalRep Duplicate()
        {
            return new TclObj(value);
        }

        /// <summary> Implement this no-op for the InternalRep interface.</summary>

        public void Dispose()
        {
            value = null;
        }

        /// <summary> Called to query the string representation of the Tcl object. This
        /// method is called only by TclObject.toString() when
        /// TclObject.stringRep is null.
        /// 
        /// </summary>
        /// <returns> the string representation of the Tcl object.
        /// </returns>
        public override string ToString()
        {
            return value.ToString();
        }

        /// <summary> Tcl_NewIntObj -> TclObj.newInstance
        /// 
        /// Creates a new instance of a TclObject with a TclObj internal
        /// representation.
        /// 
        /// </summary>
        /// <param name="b">initial value of the object object.
        /// </param>
        /// <returns> the TclObject with the given object value.
        /// </returns>

        public static TclObject newInstance(object o)
        {
            return new TclObject(new TclObj(o));
        }


        /// <summary> Changes the object value of the object.
        /// 
        /// </summary>
        /// <param name="interp">current interpreter.
        /// </param>
        /// <param name="tobj">the object to operate on.
        /// @paran i the new object value.
        /// </param>
        public static void set(TclObject tobj, object o)
        {
            tobj.invalidateStringRep();
            IInternalRep rep = tobj.InternalRep;
            TclObj tint;

            if (rep is TclObj)
            {
                tint = (TclObj)rep;
                tint.value = o;
            }
            else
            {
                tobj.InternalRep = new TclObj(o);
            }
        }
    }
}
