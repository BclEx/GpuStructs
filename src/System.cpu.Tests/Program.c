#include <stdio.h>
#include "..\ThunkC\ThunkC_\System.h"
#include "NUnit\NUnitModule.h"

//+ Forward-declare
// collections
public static_ void space(Collections_ComparerFixture,RunTests)();
public static_ void space(Collections_EqualityComparerFixture,RunTests)();
public static_ void space(Collections_ICollectionFixture,RunTests)();
public static_ void space(Collections_IComparerFixture,RunTests)();
public static_ void space(Collections_IEnumerableFixture,RunTests)();
public static_ void space(Collections_IEnumeratorFixture,RunTests)();
public static_ void space(Collections_IEqualityComparerFixture,RunTests)();
// collections.1
public static_ void space(Collections_Dictionary_KeyCollectionFixture,RunTests)();
public static_ void space(Collections_Dictionary_ValueCollectionFixture,RunTests)();
public static_ void space(Collections_DictionaryFixture,RunTests)();
public static_ void space(Collections_EqualityComparerFixture,RunTests)();
public static_ void space(Collections_IDictionaryFixture,RunTests)();
public static_ void space(Collections_IListFixture,RunTests)();
public static_ void space(Collections_KeyValuePairFixture,RunTests)();
public static_ void space(Collections_ListFixture,RunTests)();
// collections.2
public static_ void space(Collections_LinkedListFixture,RunTests)();
public static_ void space(Collections_LinkedListNodeFixture,RunTests)();
public static_ void space(Collections_SortedDictionaryFixture,RunTests)();
// system
public static_ void space(System_ObjectFixture,RunTests)();
//
public static_ void space(ClassFixture,RunTests)();
public static_ void space(ThunkFixture,RunTests)();

///<summary>
/// InitModules
///</summary>
public void InitModules()
{
	NUnit_cctor();
}

///<summary>
/// main
///</summary>
void main()
{
	InitModules();
	//
	Test2(space(ClassFixture,RunTests));
	Test2(space(ThunkFixture,RunTests));
	// system
	Test2(space(System_ObjectFixture,RunTests));
	// collections
	Test2(space(Collections_ComparerFixture,RunTests));
	Test2(space(Collections_EqualityComparerFixture,RunTests));
	Test2(space(Collections_ICollectionFixture,RunTests));
	Test2(space(Collections_IComparerFixture,RunTests));
	Test2(space(Collections_IEnumerableFixture,RunTests));
	Test2(space(Collections_IEnumeratorFixture,RunTests));
	Test2(space(Collections_IEqualityComparerFixture,RunTests));
	// collections.1
	Test2(space(Collections_Dictionary_KeyCollectionFixture,RunTests));
	Test2(space(Collections_Dictionary_ValueCollectionFixture,RunTests));
	Test2(space(Collections_DictionaryFixture,RunTests));
	Test2(space(Collections_EqualityComparerFixture,RunTests));
	Test2(space(Collections_IDictionaryFixture,RunTests));
	Test2(space(Collections_IListFixture,RunTests));
	Test2(space(Collections_KeyValuePairFixture,RunTests));
	Test2(space(Collections_ListFixture,RunTests));
	// collections.2
	Test2(space(Collections_LinkedListFixture,RunTests));
	Test2(space(Collections_LinkedListNodeFixture,RunTests));
	Test2(space(Collections_SortedDictionaryFixture,RunTests));
	//
	printf("done\n");
}
