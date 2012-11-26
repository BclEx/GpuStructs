# C# Keywords
Definition: [http://msdn.microsoft.com/en-us/library/ba0a1yw2(VS.71).aspx](http://msdn.microsoft.com/en-us/library/ba0a1yw2(VS.71).aspx)

----------

Keywords are derived from the C# spec and extened to support thunking. They have been listed below:

_Legend_  
^ - native/not-defined  
\! - not supported  
\* - different/new functionaly  
% - sinked  
[] - new keyword



## Types
* **void**^ - When used as the return type for a method, void specifies that the method does not return a value.

Types::Value Types

* **bool**^ - The bool keyword is an alias of System.Boolean. It is used to declare variables to store the Boolean values,  true and false.
* **byte** - The byte keyword denotes an integral type that stores values as indicated in the following table.
* **char**^ - The char keyword is used to declare a Unicode character in the range indicated in the following table. Unicode characters are 16-bit characters used to represent most of the known written languages throughout the world.
* **decimal**\! - The decimal keyword denotes a 128-bit data type. Compared to floating-point types, the decimal type has a greater precision and a smaller range, which makes it suitable for financial and monetary calculations. The approximate range and precision for the decimal type are shown in the following table.
* **double**^ - The double keyword denotes a simple type that stores 64-bit floating-point values. The following table shows the precision and approximate range for the double type.
* **enum**^ - The enum keyword is used to declare an enumeration, a distinct type consisting of a set of named constants called the enumerator list. Every enumeration type has an underlying type, which can be any integral type except char.
* **float**^ - The float keyword denotes a simple type that stores 32-bit floating-point values. The following table shows the precision and approximate range for the float type.
* **int\_** - The int_ keyword denotes an integral type that stores values according to the size and range shown in the following table.
* **long\_** - The long_ keyword denotes an integral type that stores values according to the size and range shown in the following table.
* **sbyte** - The sbyte keyword denotes an integral type that stores values according to the size and range shown in the following table.
* **short\_** - The short keyword denotes an integral data type that stores values according to the size and range shown in the following table.
* **struct**^ - A struct type is a value type that can contain constructors, constants, fields, methods, properties, indexers, operators, events, and nested types.
* **uint** - The uint keyword denotes an integral type that stores values according to the size and range shown in the following table.
* **ulong** - The ulong keyword denotes an integral type that stores values according to the size and range shown in the following table.
* **ushort** - The ushort keyword denotes an integral data type that stores values according to the size and range shown in the following table.

Types::Boxing and Unboxing

* **box**\* - Boxing is an implicit conversion of a  value type to the type object or to any interface type implemented by this value type. Boxing a value of a value allocates an object instance and copies the value into the new object.
* **unbox**\* - Unboxing is an explicit conversion from the type object to a  value type or from an interface type to a value type that implements the interface.

Types::Reference Types

* **class**^ - Classes are declared using the keyword class.
* **delegate**\! - A delegate declaration defines a reference type that can be used to encapsulate a method with a specific signature. A delegate instance encapsulates a static or an instance method. Delegates are roughly similar to function pointers in C++; however, delegates are type-safe and secure.
* **interface**\* - An interface defines a contract. A class or struct that implements an interface must adhere to its contract.
* **object** - The object type is an alias for System.Object in the .NET Framework. You can assign values of any type to variables of type object.
* **string**\#, **stringA**\# - The string type represents a string of Unicode characters. string is an alias for System.String in the .NET Framework.
* **[_ctor]**\* - xxxx.
* **[baseCtor]**\* - xxxx.
* **[thisCtor]**\* - xxxx.
* **[inherit]**\* - xxxx.
* **[inherit2]**\* - xxxx.
* **[inherit3]**\* - xxxx.
* **[inherit4]**\* - xxxx.

Types::Types Reference Tables

* **default** - xxxx.
* *Build-in Types Table*
* *Integral Types Table*
* *Floating-Point Types Table*
* *Default Values Table*
* *Value Types Table*
* *Implicit Numeric Conversions Table*
* *Explicit Numeric Conversions Table*
* *Formatting Numeric Results Table*



## Modifiers
* **abstract**\! - Use the abstract modifier in a method or property declaration to indicate that the method or property does not contain implementation.
* **const**^ - The const keyword is used to modify a declaration of a field or local variable. It specifies that the value of the field or the local variable cannot be modified. A constant declaration introduces one or more constants of a given type.
* **event**\! - Specifies an event.
* **extern**^ - Use the extern modifier in a method declaration to indicate that the method is implemented externally. A common use of the extern modifier is with the DllImport attribute.
* **override**\! - Use the override modifier to modify a method, a property, an indexer, or an event. An override method provides a new implementation of a member inherited from a base class. The method overridden by an override declaration is known as the overridden base method. The overridden base method must have the same signature as the override method.
* **readonly**% - The readonly keyword is a modifier that you can use on fields. When a field declaration includes a readonly modifier, assignments to the fields introduced by the declaration can only occur as part of the declaration or in a constructor in the same class.
* **[readonly_]**# - ThunkC extension to define C-style const on field definitions
* **sealed**\! - A sealed class cannot be inherited. It is an error to use a sealed class as a base class. Use the sealed modifier in a class declaration to prevent inheritance of the class.
* **[static_]**% - Use the static modifier to declare a static member, which belongs to the type itself rather than to a specific object. The static modifier can be used with fields, methods, properties, operators, events and constructors, but cannot be used with indexers, destructors, or types.
* **unsafe**! - The unsafe keyword denotes an unsafe context, which is required for any operation involving pointers.
* **virtual**\! - The virtual keyword is used to modify a method or property declaration, in which case the method or the property is called a virtual member. The implementation of a virtual member can be changed by an  overriding member in a derived class.
* **volatile**^ - The volatile keyword indicates that a field can be modified in the program by something such as the operating system, the hardware, or a concurrently executing thread.

Modifiers::Access Modifiers

* **public**% - Access is not restricted.
* **protected**% - Access is limited to the containing class or types derived from the containing class.
* **internal**% - Access is limited to the current assembly.
protected internal% - Access is limited to the current assembly or types derived from the containing class.
* **private**% - Access is limited to the containing type.
* **[private_]**\* - ThunkC extension to define C-style static for private method definitions.
* **[keyword_]**% - designates definition a new keyword.



## Statements
* **fixed**\! - Prevents relocation of a variable by the garbage collector.
* **locked**\! - The lock keyword marks a statement block as a critical section by obtaining the mutual-exclusion lock for a given object, executing a statement, and then releasing the lock.

Statements::Selection Statements

* **if-else**^ - The if statement selects a statement for execution based on the value of a Boolean expression.
* **switch**^ - The switch statement is a control statement that handles multiple selections by passing control to one of the case statements within its body. 

Statements::Iteration Statements

* **do**^ - The do statement executes a statement or a block of statements repeatedly until a specified expression evaluates to false.
* **for**^ - The for loop executes a statement or a block of statements repeatedly until a specified expression evaluates to false.
* **foreach**, **in**\! - The foreach statement repeats a group of embedded statements for each element in an array or an object collection. The foreach statement is used to iterate through the collection to get the desired information, but should not be used to change the contents of the collection to avoid unpredictable side effects.
* **while**^ - The while statement executes a statement or a block of statements until a specified expression evaluates to false.

Statements::Jump Statements

* **break**^ - The break statement terminates the closest enclosing loop or switch statement in which it appears. Control is passed to the statement that follows the terminated statement, if any.
* **continue**^ - The continue statement passes control to the next iteration of the enclosing iteration statement in which it appears.
* **goto**^ - The goto statement transfers the program control directly to a labeled statement.
* **return**^ - The return statement terminates execution of the method in which it appears and returns control to the calling method. It can also return the value of the optional expression. If the method is of the type void, the return statement can be omitted.

Statements::Exception Handling Statements

* **throw**\* - The throw statement is used to signal the occurrence of an anomalous situation (exception) during the program execution.
* **[throw_]**\* - The throw statement without an argument.
* **[try\_catch]**%\* - The try-catch statement consists of a try block followed by one or more catch clauses, which specify handlers for different exceptions.
* **[try\_finally]**%\* - The finally block is useful for cleaning up any resources allocated in the  try block. Control is always passed to the finally block regardless of how the try block exits.
* **[try\_catch\_finally]**%\* - A common usage of catch and finally together is to obtain and use resources in a try block, deal with exceptional circumstances in a catch block, and release the resources in the finally block.

Statements::Checked and Unchecked

* **checked**\! - The checked keyword is used to control the overflow-checking context for integral-type arithmetic operations and conversions.
* **unchecked**\! - The unchecked keyword is used to control the overflow-checking context for integral-type arithmetic operations and conversions.


## Method Parameters
* **params**\* - The params keyword lets you specify a  method parameter that takes an argument where the number of arguments is variable.
* **ref**% - The ref  method parameter keyword on a method parameter causes a method to refer to the same variable that was passed into the method. Any changes made to the parameter in the method will be reflected in that variable when control passes back to the calling method.
* **out**% - The out  method parameter keyword on a method parameter causes a method to refer to the same variable that was passed into the method. Any changes made to the parameter in the method will be reflected in that variable when control passes back to the calling method.



## Namespaces
* **namespace**\! - The namespace keyword is used to declare a scope. This namespace scope lets you organize code and gives you a way to create globally-unique types.
* **using**\* - The using keyword has two major uses.
* **[using_]**\* - The using keyword has two major uses.
* **[using2_]**\* - The using keyword has two major uses.
* **[bindT]**\* - Namespaces a symbol.
* **[space]**\* - Namespaces a symbol.
* **[space_T]**\* - Namespaces a symbol (generic method).
* **[spaceT]**\* - Namespaces a symbol (generic type).
* **[spaceT_T]**\* - Namespaces a symbol (generic type)(generic method).



## Operator Keywords
* **as**\* - The as operator is used to perform conversions between compatible types.
* **is**\* - The is operator is used to check whether the run-time type of an object is compatible with a given type.
* **[_new]**\* - ThunkC extension to allocate space using malloc.
* **[_newz]**\* - ThunkC extension to allocate space using zalloc(0).
* **[_newzZ]**\* - ThunkC extension to allocate space using zalloc(z).
* **new**\* - In C#, the new keyword can be used as an operator or as a modifier. ThunkC extension to instantiate a class using malloc(0).
* **[newz]**\* - ThunkC extension to instantiate a class using zalloc(0).
* **[newzZ]**\* - ThunkC extension to instantiate a class using zalloc(z).
* **[delete]**\* - ThunkC extension to de-allocate a class;
* **[deletez]**\* - ThunkC extension to de-allocate a zalloc(z) class;
* **sizeof**^ - The sizeof operator is used to obtain the size in bytes for a  value type.
* **typeof**\* - The typeof operator is used to obtain the System.Type object for a type.
* **true** - In C#, the true keyword can be used as an overloaded operator or as a literal.
* **false** - In C#, the false keyword can be used as an overloaded operator or as a literal.
* **stackalloc**\! - Allocates a block of memory on the stack.



## Conversion Keywords
* **explicit**\! - The explicit keyword is used to declare an explicit user-defined type conversion operator ( 6.4.4 User-defined explicit conversions).
* **implicit**\! - The implicit keyword is used to declare an implicit user-defined type conversion operator ( 6.4.3 User-defined implicit conversions).
operator! - The operator keyword is used to declare an operator in a class or struct declaration.



## Access Keywords
* **base**\* - The base keyword is used to access members of the base class from within a derived class.
* **this**\* - The this keyword refers to the current instance of the class. Static member functions do not have a this pointer. The this keyword can be used to access members from within constructors, instance methods, and instance accessors.



## Literal Keywords
* **null** - The null keyword is a literal that represents a null reference, one that does not refer to any object. null is the default value of reference-type variables.



*The following keywords are new keywords to support ThunkC's thunking functionality:*

## ThunkC Keywords
* **Array** - ThunkC extension to imply an array.
* **newArray** - ThunkC extension to allocate an array.
* **freeArray** - ThunkC extension to de-allocate an array.

ThunkC Keywords::Class Extentions

* **vcall** - ThunkC extension to perform a virtual call.
* **vcallT** - ThunkC extension to perform a virtual call (generic type).
* **get** - ThunkC getter property accessor.
* **gets** - ThunkC getter property accessor (static type).
* **getT** - ThunkC getter property accessor (generic type).
* **getsT** - ThunkC getter property accessor (static type)(generic type).
* **getIndexor** - ThunkC indexor getter property accessor.
* **getIndexor2** - ThunkC indexor getter property accessor.
* **set** - ThunkC setter property accessor.
* **sets** - ThunkC setter property accessor (static type).
* **setT** - ThunkC setter property accessor (generic type).
* **setsT** - ThunkC setter property accessor (static type)(generic type).
* **setIndexor** - ThunkC indexor setter property accessor.
* **setIndexor2** - ThunkC indexor setter property accessor.


# More

C has no variable or access modifiers.
Marker keywords are masked so meta information can still be applied to code with out behavior effects.

**Existing or ignored keyword implementations**  
ThunkC's keywords where derived from C# keywords. Some C# keywords are implemented, some are not supported and will generate compiler errors, some are ignored and allow description of intended use, the remaining are slightly changed or new keywords and support thunking functionality. A few C# keywords collide with C keywords or have duel functionaly, these have been postfixed with the underscore(_) character.

* Normal implemented keywords Implemented keywords are normal.

* Non-supported keywords. These keywords will generate a compiler error like the example below:

`The following code segment:  
/// <summary>  
/// Description of my function  
/// </summary>  
private virtual void Foo()  
{  
}  
Would result in a compiler error of:  
NOTSUPPORTED_KEYWORD_virtual_`

* Ignored keywords permit meta type information to describe intended use, and will be ignored by the compiler. See examples below.

> The public, static and out keywords:
> /// <summary>
> /// Description of my function
> /// </summary>
> public static_ bool TryFoo(out int_ *value)
> {
>    *value = 0;
>    return true;
> }
> Which will be seen by the compiler as:
> bool TryFoo(int_ *value)
> {
>    *value = 0;
>    return true;
> }
> 

* Some keywords have slightly changed implementations such as the try-catch, try-finally, try-catch-finally, and using keyword. See examples below.

> The c# try-catch-finally code segment of:
> DisposeableObject obj = new DisposeableObject(parameter);
> try
> {
>    obj.DangerousMethod();
> }
> catch
> {
>    throw;
> {
> finally
> {
>    if (obj != null)
>    {
>       obj.Dispose();
>    }
> }
> Would be converted to:
> DisposeableObject *obj = new(,DisposeableObject,parameter);
> try_catch_finally(
>    obj.DangerousMethod();
> ,
>    throw_;
> ,
>    if (obj != null)
>    {
>       obj.Dispose();
>    }
> )
> 
> The c# using statement.
> using (DisposeableObject obj = new DisposeableObject(parameter))
> {
>    obj.DangerousMethod();
> }
> Would be converted to:
> DisposeableObject *obj;
> using2_(
>    obj.DangerousMethod();
> ,obj,DisposeableObject,parameter);
> 

# Namespacing

In C all symbols share a public scope, so name collisions are high. The space, spaceT, space_T and spaceT_T keywords namespace symbols to prevent collisions.

* The space(namespace, name) keyword emits a general spaced symbol by called name, by namespace.
> The namespaced method:
> public void System_Array::Clear(Array array) { }
> Would be converted to:
> public void space(System_Array,Clear)(Array array) { }

* The space_T(namespace, name, T) keyword emits a general spaced symbol by called name, by namespace for a generic method. If the generic has multiple generic parameters then fuse them using the ## keyword.

> The namespaced generic method:
> public void System_Array::Clear<string>(Array array) { }
> Would be converted to:
> public void space_T(System_Array,Clear,string)(Array array) { }

* The spaceT(namespace, T, name) keyword emits a general spaced symbol by called name, by namespace for a generic type. If the generic has multiple generic parameters then fuse them using the ## keyword.

> The namespaced generic method:
> public void System_Array<string, string>::Clear(Array array) { }
> Would be converted to:
> public void spaceT(System_Array,string##string,Clear)(Array array) { }

* The spaceT_T(namespace, T, name, T2) keyword emits a general spaced symbol by called name, by namespace for a generic method of a generic type. If the generic has multiple generic parameters then fuse them by using the ## keyword.

> The namespaced generic method:
> public void System_Array<string>::Clear<string>(Array array) { }
> Would be converted to:
> public void spaceT(System_Array,string,Clear,string)(Array array) { }


# Classes

You can create an instance of a class using the new keyword and since there is no gargage collector when your done with free it with the delete keyword. If you have multiple ctors pass the ctor number as the first parameter otherwise it blank. The second parameter is for the type name, the remaining variaic parameters are passed to the ctor. classes can be instanciatd with the newz and newzZ keywords also which uses zalloc to allocate in zones.

class methods both instance and static are call through the class methods passing the instance as the first parameter for instance methods. virtual methods are also callable but by using the vcall and vcallT keywords. see VCalls and Interfaces.

> example:
> Simple *simple = new(,Simple);
> space(Simple,DoSomething)(simple);
> delete(simple);

> example:
> SimpleGeneric(string) *simpleGeneric = new(2,SimpleGeneric(string),"Param");
> spaceT(Simple,string,DoSomething)(simpleGeneric);
> delete(simpleGeneric);


defining Classes are a little more complicated.

> A simple c# class of:
> ///<summary>
> /// Simple class
> ///</summary>
> public class Simple
> {
>    public string Field;
> }
> Would be converted to:
> ///<summary>
> /// Simple class
> ///</summary>
> public class(Simple,System_Object,,
> 	public char *Field;
> );
> Implement_Type(Simple,)
> 

# VCalls and Interfaces

NEED TEXT


# Generics

NEED TEXT
