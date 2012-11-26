using namespace Xunit;
using namespace System;

/*
Declare_Collections_IEnumerator(System_Int32, int_)
Declare_Collections_IEnumerable(System_Int32, int_)
Declare_Collections_ICollection(System_Int32, int_)

array_Declare_GetEnumerable(System_Int32, int_)
Declare_System_Array(System_Int32, int_)
Declare_System_Action(System_Int32, int_)
Declare_System_Predicate(System_Int32, int_)
Declare_System_Comparison(System_Int32, int_)
Declare_Collections_IComparer(System_Int32, int_)
Declare_Collections_List(System_Int32, int_)

Declare_System_ArrayXSZArrayEnumerator(System_Int32, int_)
public bool MoveNext(System_ArrayXSZArrayEnumerator(System_Int32) *t)
{
printf("1a: %d, b: %d\n", t->_index, t->_endIndex);
if (t->_index < t->_endIndex)
{
t->_index++;
return (t->_index < t->_endIndex);
}
return false;
}
*/

namespace System { namespace Collections {
	/// <summary>
	/// ICollectionTests
	/// </summary>
	public ref class ICollectionTests
	{
	public:
		[Fact]
		void Add_TwoElementList_CountEqualsThree()
		{
			/*
			Collections_IEnumerable(System_Int32) *collection;
			Collections_IEnumerator(System_Int32) *enumerator;
			localArray(3, int_ ints[] = { 0, 1 });
			collection = array_getEnumerable(System_Int32, ints);
			enumerator = vcallT(Collections_IEnumerable,System_Int32,collection,GetEnumerator);
			enumerator->__vtable[0] = (System_ObjectVector)&MoveNext;
			while (vcallT(Collections_IEnumerator,System_Int32,enumerator,MoveNext) == true)
			printf("%d\n", getVcallT(Collections_IEnumerator,System_Int32,enumerator,Current));
			//b = array_getEnumerator(System_Int32, a);
			//collection = new(2,Collections_List(System_Int32), array_getEnumerable(System_Int32,ints));
			//vcallT(Collections_ICollection,System_Int32,collection,Add, 2);
			//NUnit_Assert_iAreEqual(3, getVcallT(Collections_ICollection,System_Int32,collection,Count));
			*/
		}

		[Fact]
		void Clear_TwoElementList_CountEqualsZero()
		{
		}

		[Fact]
		void Contains_TwoElementList_EqualsTrue()
		{
		}

		[Fact]
		void CopyTo_TwoElementList_CountEqualsTrue()
		{
		}

		[Fact]
		void Remove_TwoElementList_CountEqualsOne()
		{
		}

		[Fact]
		void getCount_TwoElementList_EqualsTwo()
		{
		}

		[Fact]
		void getReadOnly_TwoElementList_EqualsFalse()
		{
		}

	};
}}


