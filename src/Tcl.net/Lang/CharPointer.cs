#region Foreign-License
/*
Used in the Parser, this class implements the functionality of a C character pointer.  CharPointers referencing the same
script share a reference to one array, while maintaining there own current index into the array.

Copyright (c) 1997 by Sun Microsystems, Inc.
Copyright (c) 2012 Sky Morey

See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
*/
#endregion
namespace Tcl.Lang
{
    public class CharPointer
    {
        public char[] _array; // A string of characters.
        public int _index; // The current index into the array.

        internal CharPointer()
        {
            _array = null;
            _index = -1;
        }
        internal CharPointer(CharPointer c)
        {
            _array = c._array;
            _index = c._index;
        }
        public CharPointer(string str)
        {
            int len = str.Length;
            _array = new char[len + 1];
            SupportClass.GetCharsFromString(str, 0, len, ref _array, 0);
            _array[len] = '\x0000';
            _index = 0;
        }

        internal char CharAt()
        {
            return (_array[_index]);
        }
        internal char CharAt(int x)
        {
            return (_array[_index + x]);
        }

        public int Length()
        {
            return (_array.Length - 1);
        }

        public override string ToString()
        {
            return new string(_array, 0, _array.Length - 1);
        }
    }
}
