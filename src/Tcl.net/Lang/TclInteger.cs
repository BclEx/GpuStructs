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

    /// <summary> This class implements the integer object type in Tcl.</summary>

    public class TclInteger : IInternalRep
    {
        /// <summary> Internal representation of a integer value.</summary>
        private int value;

        /// <summary> Construct a TclInteger representation with the given integer value.</summary>
        private TclInteger(int i)
        {
            value = i;

        }

        /// <summary> Construct a TclInteger representation with the initial value taken
        /// from the given string.
        /// 
        /// </summary>
        /// <param name="interp">current interpreter.
        /// </param>
        /// <param name="str">string rep of the integer.
        /// </param>
        /// <exception cref=""> TclException if the string is not a well-formed Tcl integer
        /// value.
        /// </exception>
        private TclInteger(Interp interp, string str)
        {
            value = Util.getInt(interp, str);
        }

        /// <summary> Returns a dupilcate of the current object.</summary>
        /// <param name="obj">the TclObject that contains this internalRep.
        /// </param>
        public IInternalRep Duplicate()
        {
            return new TclInteger(value);
        }

        /// <summary> Implement this no-op for the InternalRep interface.</summary>

        public void Dispose()
        {
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

        /// <summary> TCL.Tcl_NewIntObj -> TclInteger.newInstance
        /// 
        /// Creates a new instance of a TclObject with a TclInteger internal
        /// representation.
        /// 
        /// </summary>
        /// <param name="b">initial value of the integer object.
        /// </param>
        /// <returns> the TclObject with the given integer value.
        /// </returns>

        public static TclObject NewInstance(int i)
        {
            return new TclObject(new TclInteger(i));
        }

        /// <summary> SetIntFromAny -> TclInteger.setIntegerFromAny
        /// 
        /// Called to convert the other object's internal rep to this type.
        /// 
        /// </summary>
        /// <param name="interp">current interpreter.
        /// </param>
        /// <param name="forIndex">true if this methid is called by getForIndex.
        /// </param>
        /// <param name="tobj">the TclObject to convert to use the
        /// representation provided by this class.
        /// </param>

        private static void setIntegerFromAny(Interp interp, TclObject tobj)
        {
            IInternalRep rep = tobj.InternalRep;

            if (rep is TclInteger)
            {
                // Do nothing.
            }
            else if (rep is TclBoolean)
            {
                bool b = TclBoolean.get(interp, tobj);
                if (b)
                {
                    tobj.InternalRep = new TclInteger(1);
                }
                else
                {
                    tobj.InternalRep = new TclInteger(0);
                }
            }
            else
            {
                // (ToDo) other short-cuts
                tobj.InternalRep = new TclInteger(interp, tobj.ToString());
            }
        }

        /// <summary> TCL.Tcl_GetIntFromObj -> TclInteger.get
        /// 
        /// Returns the integer value of the object.
        /// 
        /// </summary>
        /// <param name="interp">current interpreter.
        /// </param>
        /// <param name="tobj">the object to operate on.
        /// </param>
        /// <returns> the integer value of the object.
        /// </returns>

        public static int get(Interp interp, TclObject tobj)
        {
            setIntegerFromAny(interp, tobj);
            TclInteger tint = (TclInteger)tobj.InternalRep;
            return tint.value;
        }

        /// <summary> Changes the integer value of the object.
        /// 
        /// </summary>
        /// <param name="interp">current interpreter.
        /// </param>
        /// <param name="tobj">the object to operate on.
        /// @paran i the new integer value.
        /// </param>
        public static void set(TclObject tobj, int i)
        {
            tobj.invalidateStringRep();
            IInternalRep rep = tobj.InternalRep;
            TclInteger tint;

            if (rep is TclInteger)
            {
                tint = (TclInteger)rep;
                tint.value = i;
            }
            else
            {
                tobj.InternalRep = new TclInteger(i);
            }
        }
    }
}
