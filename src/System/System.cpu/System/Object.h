#ifndef __System_Type_H__
#define __System_Type_H_
#include "..\Thunk.h"
namespace System {
	class Type;
	/// <summary>
	/// Object
	/// </summary>
	class Object;
	class ObjectType {
		// boxing/unboxing
		static Object *_box(object value);
		static object _unbox(Object *value);
	};

	public_ class Object
	{
	private:
		Type *__type;
	public:
		inline Type *GetType() { return __type; }
		/*static Object *GetInstanceByType(Object *t, Type *type, bool throwOnNull, void **_ref);*/
	};
}
#endif /* __System_Type_H__ */
