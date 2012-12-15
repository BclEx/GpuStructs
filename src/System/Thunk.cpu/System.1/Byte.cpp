#include "..\System.h"
using namespace System;

Type Byte::MyType = { (__typeCtor)null, L"byte", (__typeBoxer)&Byte::_box, (__typeUnboxer)&Byte::_unbox, (char*)default(byte) };

// boxing/unboxing
Object *Byte::_box(byte value)
{
	Byte *box = null; //gcnew Byte();
	box->_value = value;
	return (Object *)box;
}

byte Byte::_unbox(Object *value)
{
	return ((Byte*)value)->_value;
}
