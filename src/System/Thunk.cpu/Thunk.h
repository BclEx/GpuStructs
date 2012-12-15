//[names] http://wiki.songbirdnest.com/index.php?title=Developer/Articles/Style_Manual/C//C%2B%2B_Headers_and_Object_Definition
//[c keywords] http://gcc.gnu.org/onlinedocs/cpp/index.html#Top http://tigcc.ticalc.org/doc/keywords.html
//[c macros] http://tigcc.ticalc.org/doc/cpp.html
//[c variable types] http://en.wikipedia.org/wiki/C_variable_types_and_declarations
//[c reserved identifiers] http://web.archive.org/web/20040209031039/http://oakroadsystems.com/tech/c-predef.htm

#ifdef _MSC_VER
#pragma once
#endif
#ifndef __System_Thunk_H__
#define __System_Thunk_H__

#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#define notSupported(keyword) NOTSUPPORTED_KEYWORD_##keyword##_;
#define notImplemented(keyword) NOTIMPLEMENTED_KEYWORD_##keyword##_;

//## Types
//typedef void^

// Types::Values Types
//typedef bool^
typedef unsigned char byte;
//typedef char^
typedef long int decimal;
//typedef double^
//typedef enum^
//typedef float^
typedef long int int_;
typedef long long int long_;
typedef signed char sbyte;
typedef short int short_;
//typedef struct^
typedef unsigned long int uint;
typedef unsigned long long int ulong;
typedef unsigned short int ushort;

// Types::Boxing and Unboxing
/// <summary>
/// box
/// </summary>
#define __box(T,value) T##::_box(value)
#define __boxT(T,value) typeof(T)->__boxer((int)value)
/// <summary>
/// unbox
/// </summary>
#define __unbox(T,value) T##_unbox((System::Object*)value)
#define __unboxT(T,value) (T)typeof(T)->__unboxer((System::Object*)value)

// Types::Reference Types
//typedef class* [see object.core.h]
#define delegate notSupported(delegate)
//typedef interface* [see object.core.h]
typedef void* object;
typedef wchar_t* string;
typedef char* stringA;

// Types::Types Reference Tables::Default Values Table
///<summary>
/// default
///</summary>
#define default(T) (T)__defaultValueTable_##T
#define defaultT(T,Ti) (Ti)(unsigned char*)typeof(T)->__default

//## Modifiers
#define abstract_
#define abstract virtual
#define abstractT virtualT
//const^
#define event notSupported(event)
//extern^
#define override notSupported(override)
#define readonly
//#define readonly_ const
#define sealed notSupported(sealed)
#define static_
#define unsafe notSupported(unsafe)
#define virtual_
//volatile^

// Modifiers::Access Modifiers
#define public_
#define protected_
#define internal_
#define private_


//## Statements
#define fixed notSupported(fixed)
#define locked notSupported(locked)

// Statements::Selection Statements
//if-else^
//switch^
//+ Statements::Iteration Statements
//do^
//for^
#define foreach notSupported(foreach)
#define in notSupported(in)
//while^std^

// Statements::Jump  Statements
//break
//continue^
//goto^
//return^

// Statements::Exception Handling Statements
///<summary>
/// throw
///</summary>
#define throw_(type,...) { printf("%s\n%s-%d\n", #type, __FILE__, __LINE__); scanf_s("s"); exit(1); }
#define try_catch(try, catch) {try}
#define try_finally(try, finally) {try} {finally}
#define try_catch_finally(try, catch, finally) {try} {finally}

// Statements::Checked and Unchecked
#define checked notSupported(checked)
#define unchecked notSupported(unchecked)


//## Method Parameters
///<summary>
/// params
///</summary>
#define params
#define ref
#define out


//## Namespaces
//typedef namespace^
///<summary>
/// using
///</summary>
#define using(t,pre,body,post) {pre;body{vcall(System_IDisposable,t,Dispose);}post;}
#define using_() notImplemented(using)
///<summary>
/// bind+space
///</summary>
#define bindT(name,T) name##_##T


//## Operator Keywords
///<summary>
/// as/is
///</summary>
#define as(t,T) (T *)Object::GetInstanceByType((System::Object *)t,typeof(T),false,(void**)null)
#define is(t,T) (Object::GetInstanceByType((System::Object *)t,typeof(T),false,(void**)null)!=null)
///<summary>
/// new/_new/delete
///</summary>
extern void *__lastObject;

//#define new(n,T,...) newh(n,T,__VA_ARGS__)
//#define asNew(T,n,T2,...) (T*)_getInstanceByType((System_Object *)bindT(T2,ctor##n)((T2*)__typeBinder(alloch(T2),typeof(T2)),__VA_ARGS__),typeof(T),false,(void**)null)
//#define alloc(T) alloch(T)
//#define delete(t) deleteh(t)
//#define gcnew(n,T,...) (T*)bindT(T,ctor##n)((T*)__typeBinder(gcalloc(T),typeof(T)),__VA_ARGS__);
//#define newh(n,T,...) (T*)bindT(T,ctor##n)((T*)__typeBinder(alloch(T),typeof(T)),__VA_ARGS__);
//#define news(n,T,...) (T*)bindT(T,ctor##n)((T*)__typeBinder(allocs(T),typeof(T)),__VA_ARGS__);
//#define gcalloc(T) malloc(sizeof(T))
//#define alloch(T) malloc(sizeof(T))
//#define allocs(T) _alloca(sizeof(T))
//#define deleteh(t) free(t)

//sizeof^
///<summary>
/// typeof
///</summary>
#define typeof(T) (&##T##::MyType)
//true^
//false^
#define stackalloc notSupported(stackalloc)


//## Conversion Keywords
#define explicit notSupported(explicit)
#define implicit notSupported(implicit)
#define operator notSupported(operator)


//## Access Keywords
///<summary>
/// base
///</summary>
//base^
//this^


//## Literal Keywords
#define null nullptr


//## ThunkC Keywords::Class Extentions

///<summary>
/// get/set
///</summary>
#define get(T,t,property) T##_get##property(t)
#define gets_(T,property) T##_get##property()
#define getT(T,T2,t,property) T##_##T2##_get##property(t)
#define getsT(T,T2,property) T##_##T2##_get##property()
#define getIndex(T,t,...) T##_getIndex(t,__VA_ARGS__)
#define getIndexT(T,T2,t,...) T##_##T2##_getIndex(t,__VA_ARGS__)
#define set(T,t,property,value) T##_set##property(t, value)
#define sets(T,property,value) T##_set##property(value)
#define setT(T,T2,t,property,value) T##_##T2##_set##property(t,value)
#define setsT(T,T2,property,value) T##_##T2##_set##property(value)
#define setIndex(T,t,value,...) T##_setIndex(t,value,__VA_ARGS__)
#define setIndexT(T,T2,t,value,...) T##_##T2##_setIndex(t,value,__VA_ARGS__)

#endif /* __System_Thunk_H__ */

// Includes
//#include "System\Zalloc.h"
#include "System\Object.h"
#include "System\Type.h"
//#include "System\Array.Core.h"
//#include "System\Enum.Core.h"
//#include "System\Struct.Core.h"
//#include "System\String.Core.h"


