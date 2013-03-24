#include <stdio.h>
#include <string.h>
using namespace System;
using namespace Xunit;

extern void simpleTest_host();

namespace Tests
{
	public ref class SimpleTest
	{
	public:
		[Fact]
		void Launces_cuda_device()
		{
			simpleTest_host();
			Assert::Equal(1, 1);
		}
	};
}
