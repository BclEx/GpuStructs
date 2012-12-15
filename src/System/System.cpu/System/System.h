#ifndef __SYSTEM_SYSTEM_H__
#define __SYSTEM_SYSTEM_H__
#include "..\Thunk.h"
namespace System {

	/// <summary>
	/// Exception
	/// </summary>
	class Exception
	{
	public:
		static const int MAX_ERROR_LEN = 2048;
		Exception(const char *text = "") { strncpy( _error, text, MAX_ERROR_LEN ); }
		// this really, really should be a const function, but it's referenced too many places to change right now
		const char *GetError() { return _error; }
	protected:
		// if GetError() were correctly const this would be named GetError(), too
		char *GetErrorBuffer() { return _error; }
		int GetErrorBufferSize() { return MAX_ERROR_LEN; }
	private:
		friend class FatalException;
		static char _error[MAX_ERROR_LEN];
	};

	/// <summary>
	/// FatalException
	/// </summary>
	class FatalException
	{
	public:
		static const int MAX_ERROR_LEN = 2048;
		FatalException(const char *text = "") { strncpy( Exception::_error, text, MAX_ERROR_LEN );  }
		// this really, really should be a const function, but it's referenced too many places to change right now
		const char *GetError() { return Exception::_error; }	
	protected:
		// if GetError() were correctly const this would be named GetError(), too
		char *GetErrorBuffer() { return Exception::_error; }	
		int GetErrorBufferSize() { return MAX_ERROR_LEN; }
	};

}
#endif /* __SYSTEM_SYSTEM_H__ */
