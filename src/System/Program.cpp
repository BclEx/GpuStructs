#include <cstdio>
extern void TestHeap();
extern void TestSort();
extern void TestList();
extern void TestString();

void main()
{
	TestHeap();
	TestSort();
	TestList();
	TestString();
	//
	printf("End.");
	char c; scanf_s("%c", &c);
}
