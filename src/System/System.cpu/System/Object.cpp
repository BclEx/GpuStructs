#include "..\System.h"
using namespace System;

//Type Object::MyType = { (__typeCtor)null, L"object", (__typeBoxer)nullptr, (__typeUnboxer)nullptr, nullptr };

//Object::Object(Object *parent)
//{
//	__parent = parent;
//	__next = nullptr;
//	if (parent != null) parent->__next = this;
//	//__typeBinder2(t,__getBase(t,Tbase),typeof(Tbase));
//	__interfaces = nullptr;
//}

// boxing/unboxing
Object *ObjectType::_box(object value)
{
	return (Object *)value;
}

object ObjectType::_unbox(Object *value)
{
	return value;
}

///// <summary>
///// _getInstanceByType
///// </summary>
//Object *Object::GetInstanceByType(Object *t, Type *type, bool throwOnNull, void **_ref)
//{
//	Type *tType = t->__type;
//	// rebase t if interface, can not search on interfaces
//	t = ((tType->__flags & TypeFlags::Interface) == TypeFlags::Interface ? t : t->__parent);
//	// types already match, return
//	if (tType = type)
//	{
//		if (_ref != null)
//			*_ref = t;
//		return t;
//	}
//	Object *p = t->__parent;
//	Object *t2;
//	// interface
//	if ((type->__flags & TypeFlags::Interface) == TypeFlags::Interface)
//	{
//		// search descendants
//		for (; t != null; t = t->__next)
//		{
//			for (t2 = (Object *)t->__interfaces; t2 != null; t2 = t2->__next)
//				if (t2->__type == type)
//				{
//					if (_ref != null)
//						*_ref = t;
//					return t2;
//				}
//		}
//		// search ancestors
//		if (p != null)
//			for (; p != null; p = p->__parent)
//				for (t2 = (Object *)p->__interfaces; t2 != null; t2 = t2->__next)
//					if (t2->__type == type)
//					{
//						if (_ref != null)
//							*_ref = p;
//						return t2;
//					}
//	} else {
//		// search descendants
//		for (; t != null; t = t->__next)
//		{
//			if (t->__type == type)
//			{
//				if (_ref != null)
//					*_ref = t;
//				return t;
//			}
//		}
//		// search ancestors
//		if (p != null)
//			for (; p != null; p = p->__parent)
//				if (p->__type == type)
//				{
//					if (_ref != null)
//						*_ref = p;
//					return p;
//				}
//	}
//	if (!throwOnNull)
//		return (Object *)null;
//	throw(UnableToFindInstance);
//}
