#ifndef __SYSTEM_SYSTEM_DEFINES_H__
#define __SYSTEM_SYSTEM_DEFINES_H__
namespace Sys {

#define FORMAT_PRINTF(x)

#define PATHSEPARATOR_STR "\\"
#define PATHSEPARATOR_CHAR '\\'
#define NEWLINE "\r\n"

#ifndef WIN32 // we should never rely on this define in our code. this is here so dodgy external libraries don't get confused
#define WIN32
#endif

#define likely(x) (x)
#define unlikely(x) (x)

	// A macro to disallow the copy constructor and operator= functions
	// NOTE: The macro contains "private:" so all members defined after it will be private until public: or protected: is specified.
#define DISALLOW_COPY_AND_ASSIGN(Type) \
private: \
	Type(const Type &); \
	void operator=(const Type &);

	// Setup for /analyze code analysis, which we currently only have on the 360, but we may get later for win32 if we buy the higher end vc++ licenses.
	// Even with VS2010 ultmate, /analyze only works for x86, not x64
	// Also note the __analysis_assume macro in sys_assert.h relates to code analysis.
	// This header should be included even by job code that doesn't reference the bulk of the codebase, so it is the best place for analyze pragmas.

	// disable some /analyze warnings here
	//#pragma warning(disable: 6255)	// warning C6255: _alloca indicates failure by raising a stack overflow exception. Consider using _malloca instead. (Note: _malloca requires _freea.)
	//#pragma warning(disable: 6262)	// warning C6262: Function uses '36924' bytes of stack: exceeds /analyze:stacksize'32768'. Consider moving some data to heap
	//#pragma warning(disable: 6326)	// warning C6326: Potential comparison of a constant with another constant
	//#pragma warning(disable: 6031)	//  warning C6031: Return value ignored
	// this warning fires whenever you have two calls to new in a function, but we assume new never fails, so it is not relevant for us
	//#pragma warning(disable: 6211)	// warning C6211: Leaking memory 'staticModel' due to an exception. Consider using a local catch block to clean up memory
	// we want to fix all these at some point...
#pragma warning(disable: 6246)	// warning C6246: Local declaration of 'es' hides declaration of the same name in outer scope. For additional information, see previous declaration at line '969' of 'w:\tech5\rage\game\ai\fsm\fsm_combat.cpp': Lines: 969
#pragma warning(disable: 6244)	// warning C6244: Local declaration of 'viewList' hides previous declaration at line '67' of 'w:\tech5\engine\renderer\rendertools.cpp'
	// win32 needs this, but 360 doesn't
#pragma warning(disable: 6540)	// warning C6540: The use of attribute annotations on this function will invalidate all of its existing __declspec annotations [D:\tech5\engine\engine-10.vcxproj]
#pragma warning(disable: 4996) // sky: stncpy unsafe

	// checking format strings catches a LOT of errors
	//#include <CodeAnalysis\SourceAnnotations.h>
	//#define VERIFY_FORMAT_STRING [SA_FormatString(Style="printf")]
#define VERIFY_FORMAT_STRING

	// We need to inform the compiler that Error() and FatalError() will never return, so any conditions that leeds to them being called are
	// guaranteed to be false in the following code
#define NO_RETURN __declspec(noreturn)

	// I don't want to disable "warning C6031: Return value ignored" from /analyze but there are several cases with sprintf where we pre-initialized the variables
	// being scanned into, so we truly don't care if they weren't all scanned. Rather than littering #pragma statements around these cases, we can assign the
	// return value to this, which means we have considered the issue and decided that it doesn't require action.
	// The volatile qualifier is to prevent:PVS-Studio warnings like:
	// False	2	4214	V519	The 'ignoredReturnValue' object is assigned values twice successively. Perhaps this is a mistake. Check lines: 545, 547.	Rage	collisionmodelmanager_debug.cpp	547	False
	extern volatile int ignoredReturnValue;
}
#endif /* __SYSTEM_SYSTEM_DEFINES_H__ */
