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
#define __box(Ti,value) ___box##Ti(value)
#define __boxT(Ti,value) typeof(Ti)->__boxer((int)value)
/// <summary>
/// unbox
/// </summary>
#define __unbox(Ti,value) ___unbox##Ti((System_Object*)value)
#define __unboxT(Ti,value) (Ti)typeof(Ti)->__unboxer((System_Object*)value)

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
#define default(Ti) (Ti)__defaultValueTable_##Ti
#define defaultT(T,Ti) (Ti)(unsigned char*)typeof(T)->__default

//## Modifiers
#define abstract_
#define abstract virtual
#define abstractT virtualT
//typedef const^
#define event notSupported(event)
//typedef extern^
#define override notSupported(override)
#define readonly
//?#define readonly_ const
#define sealed notSupported(sealed)
#define static_
#define unsafe notSupported(unsafe)
///<summary>
/// virtual
///</summary>
#define virtual_
#define virtual(T,iReturn,method,...) typedef iReturn (*space(T,method)##Delegate)(T *t,__VA_ARGS__);
#define virtualT(T,T2,iReturn,method,...) typedef iReturn (*spaceT(T,T2,method)##Delegate)(T##_##T2 *t,__VA_ARGS__);
#define getVirtual(T,iValue,property) typedef iValue (*space(T,get##property)##Delegate)(T *t);
#define getVirtualT(T,T2,iValue,property) typedef iValue (*spaceT(T,T2,get##property)##Delegate)(T##_##T2 *t);
#define getIndexVirtual(T,iValue,...) typedef iValue (*space(T,getIndex)##Delegate)(T *t,__VA_ARGS__);
#define getIndexVirtualT(T,T2,iValue,...) typedef iValue (*spaceT(T,T2,getIndex)##Delegate)(T##_##T2 *t,__VA_ARGS__);
#define setVirtual(T,iValue,property) typedef void (*space(T,set##property)##Delegate)(T *t,iValue value);
#define setVirtualT(T,T2,iValue,property) typedef void (*spaceT(T,T2,set##property)##Delegate)(T##_##T2 *t,iValue value);
#define setIndexVirtual(T,iValue,...) typedef void (*space(T,setIndex)##Delegate)(T *t,iValue value,__VA_ARGS__);
#define setIndexVirtualT(T,T2,iValue,...) typedef void (*spaceT(T,T2,setIndex)##Delegate)(T##_##T2 *t,iValue value,__VA_ARGS__);
//typedef volatile^

// Modifiers::Access Modifiers
#define public
#define protected
#define internal
#define private
#define private_ static
#define keyword_


//## Statements
#define fixed notSupported(fixed)
#define locked notSupported(locked)

// Statements::Selection Statements
//if-else^std^
//switch^std^
//+ Statements::Iteration Statements
//do^std^
//for^std^
#define foreach notSupported(foreach)
#define in notSupported(in)
//while^std^

// Statements::Jump  Statements
//break^std^
//continue^std^
//goto^std^
//return^std^

// Statements::Exception Handling Statements
///<summary>
/// throw
///</summary>
#define throw(type,...) { printf("%s\n%s-%d\n", #type, __FILE__, __LINE__); scanf_s("s"); exit(1); }
#define throw_ throw(Rethrow);
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
#define namespace notSupported(namespace)
///<summary>
/// using
///</summary>
#define using(t,pre,body,post) {pre;body{vcall(System_IDisposable,t,Dispose);}post;}
#define using_() notImplemented(using)
///<summary>
/// bind+space
///</summary>
#define bindT(name,T) name##_##T
#define space(namespace,name) namespace##_##name
#define space_T(namespace,name,T) namespace##_##name##_##T
#define spaceT(namespace,T,name) namespace##_##T##_##name
#define spaceT_T(namespace,T,name,T2) namespace##_##T##_##name##_##T2


//## Operator Keywords
///<summary>
/// as/is
///</summary>
#define as(t,T) (T *)_getInstanceByType((System_Object *)t,typeof(T),false,(void**)null)
#define is(t,T) (_getInstanceByType((System_Object *)t,typeof(T),false,(void**)null)!=null)
///<summary>
/// new/_new/delete
///</summary>
extern void *__lastObject;

#define new(n,T,...) newh(n,T,__VA_ARGS__)
#define asNew(T,n,T2,...) (T*)_getInstanceByType((System_Object *)bindT(T2,ctor##n)((T2*)__typeBinder(alloch(T2),typeof(T2)),__VA_ARGS__),typeof(T),false,(void**)null)
#define alloc(T) alloch(T)
#define delete(t) deleteh(t)
//
#define gcnew(n,T,...) (T*)bindT(T,ctor##n)((T*)__typeBinder(gcalloc(T),typeof(T)),__VA_ARGS__);
#define newh(n,T,...) (T*)bindT(T,ctor##n)((T*)__typeBinder(alloch(T),typeof(T)),__VA_ARGS__);
#define news(n,T,...) (T*)bindT(T,ctor##n)((T*)__typeBinder(allocs(T),typeof(T)),__VA_ARGS__);
#define newz(n,T,...) (T*)bindT(T,ctor##n)((T*)__typeBinder(allocz(T),typeof(T)),__VA_ARGS__);
#define newzZ(n,z,T,...) (T*)bindT(T,ctor##n)((T*)__typeBinder(alloczZ(z,T),typeof(T)),__VA_ARGS__);
#define gcalloc(T) malloc(sizeof(T))
#define alloch(T) malloc(sizeof(T))
#define allocs(T) _alloca(sizeof(T))
#define allocz(T) zalloc(sizeof(T))
#define alloczZ(z,T) zalloc2(sizeof(T),z)
#define deleteh(t) free(t)
#define deletez(t) zfree(t)

//typedef sizeof^
///<summary>
/// typeof
///</summary>
#define typeof(T) (&_##T##Type)
//true
//false
#define stackalloc notSupported(stackalloc)


//## Conversion Keywords
#define explicit notSupported(explicit)
#define implicit notSupported(implicit)
#define operator notSupported(operator)


//## Access Keywords
///<summary>
/// base
///</summary>
#define base t
#define this t


//## Literal Keywords
#define null ((void*)0)
#define nullT(Ti) ((Ti)0)


//## ThunkC Keywords::Class Extentions
///<summary>
/// vcall
///</summary>
#define vcall(T,t,method,...) (*( (T##_##method##Delegate)__vcallBinder((System_Object*)t,typeof(T),(unsigned long)&(((T*)0)->__vtable),_##T##VIndex_##method,&__lastObject) )) ((T*)__lastObject,__VA_ARGS__)
#define vcallT(T,T2,t,method,...) (*( (T##_##T2##_##method##Delegate)__vcallBinder((System_Object*)t,typeof(T##_##T2),(unsigned long)&(((T##_##T2*)0)->__vtable),_##T##_##T2##VIndex_##method,&__lastObject) )) ((T##_##T2*)__lastObject,__VA_ARGS__)
#define getVcall(T,t,method,...) (*( (T##_get##method##Delegate)__vcallBinder((System_Object*)t,typeof(T),(unsigned long)&(((T*)0)->__vtable),_##T##VIndex_get##method,&__lastObject) )) ((T*)__lastObject,__VA_ARGS__)
#define getVcallT(T,T2,t,method,...) (*( (T##_##T2##_get##method##Delegate)__vcallBinder((System_Object*)t,typeof(T##_##T2),(unsigned long)&(((T##_##T2*)0)->__vtable),_##T##_##T2##VIndex_get##method,&__lastObject) )) ((T##_##T2*)__lastObject,__VA_ARGS__)
#define getIndexVcall(T,t,...) (*( (T##_getIndexDelegate)__vcallBinder((System_Object*)t,typeof(T),(unsigned long)&(((T*)0)->__vtable),_##T##VIndex_getIndex,&__lastObject) )) ((T*)__lastObject,__VA_ARGS__)
#define getIndexVcallT(T,T2,t,...) (*( (T##_##T2##_getIndexDelegate)__vcallBinder((System_Object*)t,typeof(T##_##T2),(unsigned long)&(((T##_##T2*)0)->__vtable),_##T##_##T2##VIndex_getIndex,&__lastObject) )) ((T##_##T2*)__lastObject,__VA_ARGS__)
#define setVcall(T,t,method,...) (*( (T##_set##method##Delegate)__vcallBinder((System_Object*)t,typeof(T),(unsigned long)&(((T*)0)->__vtable),_##T##VIndex_set##method,&__lastObject) )) ((T*)__lastObject,__VA_ARGS__)
#define setVcallT(T,T2,t,method,...) (*( (T##_##T2##_set##method##Delegate)__vcallBinder((System_Object*)t,typeof(T##_##T2),(unsigned long)&(((T##_##T2*)0)->__vtable),_##T##_##T2##VIndex_set##method,&__lastObject) )) ((T##_##T2*)__lastObject,__VA_ARGS__)
#define setIndexVcall(T,t,...) (*( (T##_setIndexDelegate)__vcallBinder((System_Object*)t,typeof(T),(unsigned long)&(((T*)0)->__vtable),_##T##VIndex_setIndex,&__lastObject) )) ((T*)__lastObject,__VA_ARGS__)
#define setIndexVcallT(T,T2,t,...) (*( (T##_##T2##_setIndexDelegate)__vcallBinder((System_Object*)t,typeof(T##_##T2),(unsigned long)&(((T##_##T2*)0)->__vtable),_##T##_##T2##VIndex_setIndex,&__lastObject) )) ((T##_##T2*)__lastObject,__VA_ARGS__)


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


//+ Includes
//#include "System\Zalloc.h"
//#include "System\Object.Core.h"
//#include "System\Type.Core.h"
//#include "System\Array.Core.h"
//#include "System\Enum.Core.h"
//#include "System\Struct.Core.h"
//#include "System\String.Core.h"

#endif /* __System_Thunk_H__ */
