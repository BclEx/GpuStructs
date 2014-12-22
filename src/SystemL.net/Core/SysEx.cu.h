﻿// os.h
namespace Core
{
#pragma region Log & Trace

#ifdef _DEBUG
	extern bool OSTrace;
	extern bool IOTrace;
#ifndef __CUDACC__
	__device__ inline static void SysEx_LOG(RC rc, const char *fmt, ...) { }
	__device__ inline static void SysEx_OSTRACE(const char *fmt, ...) { }
	__device__ inline static void SysEx_IOTRACE(const char *fmt, ...) { }
#else
	__device__ inline static void SysEx_LOG(RC rc, const char *fmt) { }
	template <typename T1> __device__ inline static void SysEx_LOG(RC rc, const char *fmt, T1 arg1) { }
	template <typename T1, typename T2> __device__ inline static void SysEx_LOG(RC rc, const char *fmt, T1 arg1, T2 arg2) { }
	template <typename T1, typename T2, typename T3> __device__ inline static void SysEx_LOG(RC rc, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { }
	template <typename T1, typename T2, typename T3, typename T4> __device__ inline static void SysEx_LOG(RC rc, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { }
	__device__ inline static void SysEx_OSTRACE(const char *fmt) { }
	template <typename T1> __device__ inline static void SysEx_OSTRACE(const char *fmt, T1 arg1) { }
	template <typename T1, typename T2> __device__ inline static void SysEx_OSTRACE(const char *fmt, T1 arg1, T2 arg2) { }
	template <typename T1, typename T2, typename T3> __device__ inline static void SysEx_OSTRACE(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { }
	template <typename T1, typename T2, typename T3, typename T4> __device__ inline static void SysEx_OSTRACE(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { }
	__device__ inline static void SysEx_IOTRACE(const char *fmt) { }
	template <typename T1> __device__ inline static void SysEx_IOTRACE(const char *fmt, T1 arg1) { }
	template <typename T1, typename T2> __device__ inline static void SysEx_IOTRACE(const char *fmt, T1 arg1, T2 arg2) { }
	template <typename T1, typename T2, typename T3> __device__ inline static void SysEx_IOTRACE(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { }
	template <typename T1, typename T2, typename T3, typename T4> __device__ inline static void SysEx_IOTRACE(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { }
#endif
#else
#define SysEx_LOG(X, ...) ((void)0)
#define SysEx_OSTRACE(X, ...) ((void)0)
#define SysEx_IOTRACE(X, ...) ((void)0)
#endif

#pragma endregion

#define SysEx_VERSION_NUMBER 3007016

	class TagBase
	{
	public:
		MutexEx Mutex;		// Connection mutex 
		bool MallocFailed;	// True if we have seen a malloc failure
		RC ErrCode;			// Most recent error code (RC_*)
		int ErrMask;		// & result codes with this before returning
	};

	class SysEx
	{
	public:
		__device__ static RC Initialize();
		__device__ static void Shutdown();
#ifdef ENABLE_8_3_NAMES
		__device__ static void FileSuffix3(const char *baseFilename, char *z)
#else
		__device__ inline static void FileSuffix3(const char *baseFilename, char *z) { }
#endif
		// random
		__device__ static void PutRandom(int length, void *buffer);
		//		// utf
		//		__device__ static uint32 Utf8Read(const unsigned char **z);
		//		__device__ static int Utf8CharLen(const char *z, int bytes);
		//#if defined(TEST) && defined(_DEBUG)
		//		__device__ static int Utf8To8(unsigned char *z);
		//#endif
		//		__device__ static int Utf16ByteLen(const void *z, int chars);
		//#if defined(TEST)
		//		__device__ static void UtfSelfTest();
		//#endif
	};

#if _DEBUG
	__device__ inline static RC CORRUPT_BKPT_(int line)
	{
		SysEx_LOG(RC_CORRUPT, "database corruption at line %d of [%.10s]", line, "src");
		return RC_CORRUPT;
	}
	__device__ inline static RC MISUSE_BKPT_(int line)
	{
		SysEx_LOG(RC_MISUSE, "misuse at line %d of [%.10s]", line, "src");
		return RC_MISUSE;
	}
	__device__ inline static RC CANTOPEN_BKPT_(int line)
	{
		SysEx_LOG(RC_CANTOPEN, "cannot open file at line %d of [%.10s]", line, "src");
		return RC_CANTOPEN;
	}
#define SysEx_CORRUPT_BKPT CORRUPT_BKPT_(__LINE__)
#define SysEx_MISUSE_BKPT MISUSE_BKPT_(__LINE__)
#define SysEx_CANTOPEN_BKPT CANTOPEN_BKPT_(__LINE__)
#else
#define SysEx_CORRUPT_BKPT RC_CORRUPT
#define SysEx_MISUSE_BKPT RC_MISUSE
#define SysEx_CANTOPEN_BKPT RC_CANTOPEN
#endif

}
