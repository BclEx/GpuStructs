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

    /// <summary> This class implements the boolean object type in Tcl.</summary>

    public class TclBoolean : IInternalRep
    {
        /// <summary> Internal representation of a boolean value.</summary>
        private bool value;

        /// <summary> Construct a TclBoolean representation with the given boolean value.
        /// 
        /// </summary>
        /// <param name="b">initial boolean value.
        /// </param>
        private TclBoolean(bool b)
        {
            value = b;
        }

        /// <summary> Construct a TclBoolean representation with the initial value taken
        /// from the given string.
        /// 
        /// </summary>
        /// <param name="interp">current interpreter.
        /// </param>
        /// <exception cref=""> TclException if the string is not a well-formed Tcl boolean
        /// value.
        /// </exception>
        private TclBoolean(Interp interp, string str)
        {
            value = Util.GetBoolean(interp, str);
        }

        /// <summary> Returns a dupilcate of the current object.
        /// 
        /// </summary>
        /// <param name="tobj">the TclObject that contains this ObjType.
        /// </param>
        public IInternalRep Duplicate()
        {
            return new TclBoolean(value);
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
            if (value)
            {
                return "1";
            }
            else
            {
                return "0";
            }
        }

        /// <summary> Creates a new instance of a TclObject with a TclBoolean internal
        /// representation.
        /// 
        /// </summary>
        /// <param name="b">initial value of the boolean object.
        /// </param>
        /// <returns> the TclObject with the given boolean value.
        /// </returns>

        public static TclObject newInstance(bool b)
        {
            return new TclObject(new TclBoolean(b));
        }

        /// <summary> Called to convert the other object's internal rep to boolean.
        /// 
        /// </summary>
        /// <param name="interp">current interpreter.
        /// </param>
        /// <param name="tobj">the TclObject to convert to use the
        /// representation provided by this class.
        /// </param>
        private static void setBooleanFromAny(Interp interp, TclObject tobj)
        {
            IInternalRep rep = tobj.InternalRep;

            if (rep is TclBoolean)
            {
                /*
                * Do nothing.
                */
            }
            else if (rep is TclInteger)
            {
                int i = TclInteger.Get(interp, tobj);
                tobj.InternalRep = new TclBoolean(i != 0);
            }
            else
            {
                /*
                * (ToDo) other short-cuts
                */
                tobj.InternalRep = new TclBoolean(interp, tobj.ToString());
            }
        }

        /// <summary> Returns the value of the object as an boolean.
        /// 
        /// </summary>
        /// <param name="interp">current interpreter.
        /// </param>
        /// <param name="tobj">the TclObject to use as an boolean.
        /// </param>
        /// <returns> the boolean value of the object.
        /// </returns>
        /// <exception cref=""> TclException if the object cannot be converted into a
        /// boolean.
        /// </exception>
        public static bool get(Interp interp, TclObject tobj)
        {
            setBooleanFromAny(interp, tobj);
            TclBoolean tbool = (TclBoolean)(tobj.InternalRep);
            return tbool.value;
        }
    }
}
