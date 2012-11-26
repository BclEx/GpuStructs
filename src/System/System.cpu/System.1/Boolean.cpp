#include "..\System.h"
using namespace System;

//Type Boolean::MyType = { (__typeCtor)null, L"bool", (__typeBoxer)&Boolean::_box, (__typeUnboxer)&Boolean::_unbox, (char*)default(bool) };
//
//// boxing/unboxing
//Object *BooleanType::_box(bool value)
//{
//	Boolean *box = null; //gcnew Boolean();
//	box->_value = value;
//	return (Object *)box;
//}
//
//bool BooleanType::_unbox(Object *value)
//{
//	return ((Boolean*)value)->_value;
//}
