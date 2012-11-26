using namespace Xunit;
using namespace System;

/*
Declare_Collections_IComparer(System_String, string)
Declare_Collections_Comparer(System_String, string)

//#region Sample

public class(SampleComparer,Collections_Comparer(System_String),,
);
Declare_DefaultCtor(public,SampleComparer)

private_ int_ Compare(Collections_Comparer(System_String) *t, string x, string y)
{
return 0;
}
Implement_Type(SampleComparer,Collections_Comparer(System_String),,
System_ObjectVector *vtable = t->Collections_Comparer(System_String).__vtable;
vtable[_Collections_Comparer_System_StringVIndex_Compare] = (System_ObjectVector)&space(SampleComparer,Compare);
)
Implement_DefaultCtor(SampleComparer,Collections_Comparer(System_String))

//#endregion Sample
*/

namespace System { namespace Collections {
	/// <summary>
	/// ComparerTests
	/// </summary>
	public ref class ComparerTests
	{
	public:
		/*
		[Fact]
		void Constructor_Valid_NotNull()
		{
		SampleComparer *comparer = new(,SampleComparer);
		Assert::NotNull(comparer);
		}

		[Fact]
		void Compare_Valid_EqualsZero()
		{
		SampleComparer *comparer = new(,SampleComparer);
		Assert::Equal(0, vcallT(Collections_Comparer,System_String,comparer,Compare, L"test", L"test"));
		}

		[Fact]
		void getDefault_Valid_NotNull()
		{
		Assert::NotNull(getsT(Collections_Comparer,System_String,Default));
		}
		*/
	};
}}