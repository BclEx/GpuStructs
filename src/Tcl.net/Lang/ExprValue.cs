#region Foreign-License
/*
Copyright (c) 1997 Cornell University.
Copyright (c) 1997 Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
using System;
namespace Tcl.Lang
{
    /// <summary>
    /// Describes an expression value, which can be either an integer (the usual case), a double-precision floating-point value, or a string.
    /// A given number has only one value at a time.
    /// </summary>
    class ExprValue
    {
        internal const int ERROR = 0;
        internal const int INT = 1;
        internal const int DOUBLE = 2;
        internal const int STRING = 3;

        internal long IntValue; // Integer value, if any.
        internal double DoubleValue; // Floating-point value, if any.
        internal string StringValue; // Used to hold a string value, if any.
        internal int Type; // Type of value: INT, DOUBLE, or STRING.

        internal ExprValue()
        {
            Type = ERROR;
        }
        internal ExprValue(long i)
        {
            IntValue = i;
            Type = INT;
        }
        internal ExprValue(double d)
        {
            DoubleValue = d;
            Type = DOUBLE;
        }
        internal ExprValue(string s)
        {
            StringValue = s;
            Type = STRING;
        }
    }
}
