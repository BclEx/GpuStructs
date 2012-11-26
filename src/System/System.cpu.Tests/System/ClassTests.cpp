using namespace System;
using namespace Xunit;

/*
///<summary>
/// A
///</summary>
public class(A,System_Object,classInterface(System_IDisposable),
);
Declare_DefaultCtor(public,A)
Implement_Type(A,System_Object,typeInterface(System_IDisposable))
Implement_DefaultCtor(A,System_Object)

///<summary>
/// B
///</summary>
public class(B,A,,
);
Declare_DefaultCtor(public,B)
Implement_Type(B,A,)
Implement_DefaultCtor(B,A)

///<summary>
/// C
///</summary>
private enum {
_CVIndex_Test,
_CVIndexNext,
};
public classV(C,B,,
virtual(C,void,Test);
,
);
Declare_DefaultCtor(public,C)
Implement_TypeV(C,B,)
Implement_DefaultCtor(C,B)

///<summary>
/// D
///</summary>
public class(D,C,,
);
Declare_DefaultCtor(public,D)
Implement_Type(D,C,)
Implement_DefaultCtor(D,C)


private_ void EDispose()
{
printf("test");
}

///<summary>
/// E
///</summary>
public class(E,D,classInterface(System_IDisposable),
);
Declare_DefaultCtor(public,E)
Implement_Type(E,D,typeInterface(System_IDisposable),
System_ObjectVector *vtable = t->System_IDisposable.__vtable;
vtable[_System_IDisposableVIndex_Dispose] = (System_ObjectVector)&EDispose;
)
Implement_DefaultCtor(E,D)

*/

namespace System
{
	/// <summary>
	/// ClassTests
	/// </summary>
	public ref class ClassTests
	{
	public:

		/// <summary>
		/// Object
		/// </summary>
		[Fact]
		void Object()
		{
			/*
			int_ index = 0;
			void *x;
			E *obj = new(,E);
			vcall(System_IDisposable,obj,Dispose);
			x = as(obj,System_Object);
			delete(obj);
			*/
		}

	};
}