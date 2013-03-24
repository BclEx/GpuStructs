#include <cstdarg>
#include <cstring>

namespace Sys {
	/// <summary>
	/// Exception
	/// </summary>
	static const int Exception_MAX_ERROR_LEN = 2048;
	__device__ static char Exception_error[Exception_MAX_ERROR_LEN];
	class Exception
	{
	public:
		__device__ Exception(const char *text = "") { strncpy(Exception_error, text, Exception_MAX_ERROR_LEN); }
		__device__ const char *GetError() const { return Exception_error; }
		//protected:
		//	__device__ char *GetErrorBuffer() { return Exception_error; }
		//	__device__ int GetErrorBufferSize() { return Exception_MAX_ERROR_LEN; }
	};

	/// <summary>
	/// FatalException
	/// </summary>
	static const int FatalException_MAX_ERROR_LEN = 2048;
	__device__ static char FatalException_error[FatalException_MAX_ERROR_LEN];
	class FatalException
	{
	public:
		__device__ FatalException(const char *text = "") { strncpy(FatalException_error, text, Exception_MAX_ERROR_LEN);  }
		__device__ const char *GetError() const { return FatalException_error; }	
		//protected:
		//	__device__ char *GetErrorBuffer() { return FatalException_error; }	
		//	__device__ int GetErrorBufferSize() { return FatalException_MAX_ERROR_LEN; }
	};

	/// <summary>
	/// NetworkLoadException
	/// </summary>
	class NetworkLoadException : public Exception
	{
	public:
		__device__ NetworkLoadException(const char *text = "") : Exception(text) { }
	};
}

__global__ void foo()
{

}
