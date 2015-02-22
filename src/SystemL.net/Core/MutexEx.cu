// mutex.c
#include "Core.cu.h"

namespace Core
{
	MutexEx MutexEx::Empty;
	MutexEx MutexEx::NotEmpty;

	__device__ RC MutexEx::Init()
	{ 
		Empty.Tag = nullptr;
		NotEmpty.Tag = (void *)1;
		return RC_OK;
	}

	__device__ RC MutexEx::End()
	{
		return RC_OK;
	}
}
