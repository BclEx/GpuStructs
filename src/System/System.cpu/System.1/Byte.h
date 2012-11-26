#ifndef __System_Byte_H__
#define __System_Byte_H__
#include "..\System.h"
using namespace System;

// default
#define __defaultValueTable_bool false

namespace System {
	/// <summary>
	/// Boolean
	/// </summary>
	public_ class Byte : public Object
	{
	public:
		static Type MyType;
	private:
		byte _value;
		// boxing/unboxing
		static Object *_box(byte value);
		static byte _unbox(Object *value);
	};
}
#endif /* __System_Byte_H__ */
