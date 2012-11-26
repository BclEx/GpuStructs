using namespace System;
using namespace Xunit;
//Declare_System_IComparer(string, char*)
namespace System
{
	/// <summary>
	/// ThunkTests
	/// </summary>
	public ref class ThunkTests
	{
	public:
		/// <summary>
		/// Types::Boxing and Unboxing
		/// </summary>
		[Fact]
		void Box()
		{
			//box();
		}
		[Fact]
		void Unbox()
		{
			//unbox();
		}

		/*
		///<summary>
		/// Types::Reference Types
		///</summary>
		//class
		//interface

		///<summary>
		/// Types::Types Reference Tables::Default Values Table
		///</summary>
		[Fact]
		void Types_TypesReferenceTables_DefaultValuesTable()
		{
		Assert::Equal(false, default_(bool));
		Assert::Equal(0, default_(byte));
		Assert::Equal('\0', default_(char));
		Assert::Equal(0, default_(decimal));
		Assert::Equal(0, default_(double));
		Assert::Equal(0, default_(float));
		Assert::Equal(0, default_(int_));
		Assert::Equal(0, default_(long_));
		Assert::Equal(0, default_(sbyte));
		Assert::Equal(0, default_(short_));
		Assert::Equal(0, default_(ulong));
		Assert::Equal(0, default_(ushort));
		Assert::Equal(nullptr, default_(string));
		}

		///<summary>
		/// Statements::Selection Statements
		///</summary>
		[Fact]
		void Statements_SelectionStatements()
		{
		//foreach_in(x);
		}

		///<summary>
		/// Statements::Exception Handling Statements
		///</summary>
		[Fact]
		void Statements_ExceptionHandlingStatements()
		{
		//throw(Error, L"Param");
		//throw_;
		//+ try_catch
		int_ index = 0;
		try_catch(
		index++;
		,
		index++;
		);
		if (index != 1) { throw(AssertFailed, L"try_catch"); }
		//+ try_finally
		index = 0;
		try_finally(
		index++;
		,
		index++;
		);
		if (index != 2) { throw(AssertFailed, L"try_finally"); }
		//+ try_catch_finally
		index = 0;
		try_catch_finally(
		index++;
		,
		index++;
		,
		index++;
		);
		if (index != 2) { throw(AssertFailed, L"try_catch_finally"); }
		}

		///<summary>
		/// DisposableObject class
		///</summary>
		private_ void Dispose(System_IDisposable *t) { }
		public class(DisposableObject,System_Object,classInterface(System_IDisposable),
		);
		Declare_DefaultCtor(public,DisposableObject)
		Implement_Type(DisposableObject,System_Object,typeInterface(System_IDisposable),
		System_ObjectVector *vtable = t->System_IDisposable.__vtable;
		vtable[_System_IDisposableVIndex_Dispose] = (System_ObjectVector)&Dispose;
		)
		Implement_DefaultCtor(DisposableObject,System_Object)

		///<summary>
		/// Namespaces
		///</summary>
		[Fact]
		void Namespaces()
		{
		//+ using
		int_ index = 0;
		using(obj,DisposableObject *obj = new(,DisposableObject),
		index++;
		,delete(obj));
		if (index != 1) { throw(AssertFailed, L"using"); }
		////+ using
		//index = 0;
		//obj = null;
		//using(,obj,DisposableObject,
		//	index++;
		//);
		//delete(obj);
		//if (index != 1) { throw(AssertFailed, L"using"); }
		}
		*/

		/// <summary>
		/// Operator Keywords
		/// </summary>
		[Fact]
		void OperatorKeywords()
		{
		}


		/// <summary>
		/// Access Keywords
		/// </summary>
		[Fact]
		void AccessKeywords()
		{
		}

		/// <summary>
		/// ThunkC Keywords
		/// </summary>
		[Fact]
		void ThunkCKeywords()
		{
		}


		/// <summary>
		/// ThunkC Keywords::Class Extentions
		/// </summary>
		[Fact]
		void ThunkCKeywords_ClassExtentions()
		{
		}

	};
}
