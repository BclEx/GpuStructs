using namespace System;
using namespace Xunit;

/*
///<summary>
/// ClassA class
///</summary>
public class(ClassA,System_Object,,
public char *Name;
);
Declare_DefaultCtor(public,ClassA)
Implement_Type(ClassA,System_Object,)
Implement_DefaultCtor(ClassA,System_Object)

///<summary>
/// ClassB class
///</summary>
public class(ClassB,ClassA,classInterface(System_IDisposable),
public char *Name;
);
Declare_DefaultCtor(public,ClassB)
Implement_Type(ClassB,ClassA,typeInterface(System_IDisposable))
Implement_DefaultCtor(ClassB,ClassA)
*/

namespace System
{
	/// <summary>
	/// ObjectTests
	/// </summary>
	public ref class ObjectTests
	{
	public:

		/// <summary>
		/// NewClass
		/// </summary>
		[Fact]
		void NewClass()
		{
			////char *name;
			//ClassA *classA = new(,ClassA);
			////name = vcall(System_Object,sample,GetHashCode);
			////printf("name: %d %s\n", name);
			//delete(classA);
		}

		/// <summary>
		/// VCall
		/// </summary>
		[Fact]
		void VCall()
		{
			////char *name;
			//ClassA *sample = new(,ClassA);
			////name = vcall(System_Object,sample,GetHashCode)((System_Object *)sample);
			////printf("name: %d %s\n", name);
			//delete(sample);
		}

		[Fact]
		void Constructor_Valid_NotNull()
		{
		}

		[Fact]
		void Equals_OneObject_EqualsTrue()
		{
		}

		[Fact]
		void Equals2_OneObject_EqualsTrue()
		{
		}

		[Fact]
		void GetHashCode_OneObject_NotEqualsZero()
		{
		}

		[Fact]
		void GetType_OneObject_SameType()
		{
		}

		[Fact]
		void ReferenceEquals_OneObject_EqualsTrue()
		{
		}
	};
}