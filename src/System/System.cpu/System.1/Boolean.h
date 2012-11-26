#ifndef __System_Boolean_H__
#define __System_Boolean_H__
#include "..\System.h"
using namespace System;

// default
#define __defaultValueTable_byte 0

namespace System {
	/// <summary>
	/// Boolean
	/// </summary>
	//Define_ValueType(Boolean, Object,)

	class BooleanType : public ObjectType {
	};

	public_ class Boolean : public Object
	{	
	private:
		bool _value;
	};
}
#endif /* __System_Boolean_H__ */
