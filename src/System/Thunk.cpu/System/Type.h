#ifndef __System_Type_H__
#define __System_Type_H_
#include "..\Thunk.h"
namespace System {
	typedef Object *(*__typeBoxer)(void *value);
	typedef void *(*__typeUnboxer)(Object *value);
	enum TypeFlags 
	{
		None = 0,
		Interface = 1,
		Value = 2,
	};
	/// <summary>
	/// Type
	/// </summary>
	public_ class Type
	{
	//public:
	//	Type(string name,  __typeBoxer boxer,  __typeUnboxer unboxer, char *default, TypeFlags flags)
	//		: Name(name) , __boxer(boxer), __unboxer(unboxer), __default(default), __flags(flags) { }
	public:
		string Name;
		static Type *FirstType(Type *t, Type *type, bool throwOnNull, void **_ref);
	private:
		__typeBoxer __boxer;
		__typeUnboxer __unboxer;
		char *__default;
		TypeFlags __flags;
		//
		Type *__parent;
		Type *__next;
		void *__interfaces;
	};
}

///<summary>
/// Implement_Type
///</summary>
//#define Implement_ValueType(T,Tbase,typeInterfaces,...) \

//class T##Type {
//	//{ __VA_ARGS__ }
//	Type = { (__typeCtor)null, L#T, (__typeBoxer)&##T##::_box, (__typeUnboxer)&##T##::_unbox, (char*)default(T) };
//};

//#define Implement_RefType(T,Tbase,typeInterfaces,...) \
//	void *_##T##TypeCtor(System::Object *p, T *t) \
//{ \
//	printf("%s __ctor(0x%08x, 0x%08x)\n", #T, p, t); \
//{ __VA_ARGS__ } \
//	return t; \
//} \
//	Type T::MyType = { (__typeCtor)null, L#T, (__typeBoxer)&Object::_box, (__typeUnboxer)&Object::_unbox, (char*)null };

#endif /* __System_Type_H__ */
