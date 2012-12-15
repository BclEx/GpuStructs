using namespace Xunit;
using namespace System;

/*
Declare_Collections_IEqualityComparer(System_String, string)
Declare_Collections_EqualityComparer(System_String, string)

//#region Sample

public class(SampleEqualityComparer,Collections_EqualityComparer(System_String),,
);
Declare_DefaultCtor(public,SampleEqualityComparer)


private_ bool Equals(Collections_EqualityComparer(System_String) *t, string x, string y)
{
return (x == y);
}

private_ int_ GetHashCode(Collections_EqualityComparer(System_String) *t, string obj)
{
return 1;
}
Implement_Type(SampleEqualityComparer,Collections_EqualityComparer(System_String),,
System_ObjectVector *vtable = t->Collections_EqualityComparer(System_String).__vtable;
vtable[_Collections_EqualityComparer_System_StringVIndex_Equals] = (System_ObjectVector)&space(SampleEqualityComparer,Equals);
vtable[_Collections_EqualityComparer_System_StringVIndex_GetHashCode] = (System_ObjectVector)&space(SampleEqualityComparer,GetHashCode);
)
Implement_DefaultCtor(SampleEqualityComparer,Collections_EqualityComparer(System_String))

//#endregion Sample
*/

namespace System { namespace Collections {
	/// <summary>
	/// EqualityComparerTests
	/// </summary>
	public ref class EqualityComparerTests
	{
	public:
		/*
		[Fact]
		void Constructor_Valid_NotNull()
		{
			SampleEqualityComparer *comparer = new(,SampleEqualityComparer);
			Assert::NotNull(comparer);
		}

		[Fact]
		void Equals_Valid_EqualsTrue()
		{
			SampleEqualityComparer *comparer = new(,SampleEqualityComparer);
			Assert::True(vcallT(Collections_EqualityComparer,System_String,comparer,Equals, L"test", L"test"));
		}

		[Fact]
		void GetHashCode_Valid_NotEqualZero()
		{
			SampleEqualityComparer *comparer = new(,SampleEqualityComparer);
			NUnit_Assert_iAreNotEqual(1, vcallT(Collections_EqualityComparer,System_String,comparer,GetHashCode, L"test"));
		}

		[Fact]
		void getDefault_Valid_NotNull()
		{
			Assert::NotNull(getsT(Collections_EqualityComparer,System_String,Default));
		}
		*/
	};
}}