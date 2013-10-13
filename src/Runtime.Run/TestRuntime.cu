#include <Runtime.cu.h>

__global__ static void runtimeExample0(void *r)
{
	_runtimeSetHeap(r);
	_assert(false);
}

__global__ static void runtimeExample1(void *r)
{
	_runtimeSetHeap(r);
	_printf("t0\n");
	_printf("t1 %s\n", "1");
	_printf("t2 %s %d\n", "1", 2);
	_printf("t3 %s %d %d\n", "1", 2, 3);
	_printf("t4 %s %d %d %d\n", "1", 2, 3, 4);
	_printf("t5 %s %d %d %d %d\n", "1", 2, 3, 4, 5);
	_printf("t6 %s %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6);
	_printf("t7 %s %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7);
	_printf("t8 %s %d %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7, 8);
	_printf("t9 %s %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, "9");
	_printf("ta %s %d %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A");
}

__global__ static void runtimeExample2(void *r)
{
	_runtimeSetHeap(r);
	char buf[100];
	__snprintf(buf, sizeof(buf), "t0\n");
	__snprintf(buf, sizeof(buf), "t1 %s\n", "1");
	__snprintf(buf, sizeof(buf), "t2 %s %d\n", "1", 2);
	__snprintf(buf, sizeof(buf), "t3 %s %d %d\n", "1", 2, 3);
	__snprintf(buf, sizeof(buf), "t4 %s %d %d %d\n", "1", 2, 3, 4);
	__snprintf(buf, sizeof(buf), "t5 %s %d %d %d %d\n", "1", 2, 3, 4, 5);
	__snprintf(buf, sizeof(buf), "t6 %s %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6);
	__snprintf(buf, sizeof(buf), "t7 %s %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7);
	__snprintf(buf, sizeof(buf), "t8 %s %d %d %d %d %d %d %d\n", "1", 2, 3, 4, 5, 6, 7, 8);
	__snprintf(buf, sizeof(buf), "t9 %s %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, "9");
	__snprintf(buf, sizeof(buf), "ta %s %d %d %d %d %d %d %d %d %s\n", "1", 2, 3, 4, 5, 6, 7, 8, 9, "A");
}

__global__ static void runtimeExample3(void *r)
{
	_runtimeSetHeap(r);
	_throw("t0\n");
	_throw("t1 %s\n", "1");
	_throw("t2 %s %d\n", "1", 2);
	_throw("t3 %s %d %d\n", "1", 2, 3);
	_throw("t4 %s %d %d %d\n", "1", 2, 3, 4);
}

__global__ static void runtimeExample4(void *r)
{
	_runtimeSetHeap(r);
	char a0 = __toupper('a'); char a0n = __toupper('A'); _assert(a0 == 'A' || a0n == 'A');
	bool a1 = _isspace('a'); bool a1n = _isspace(' '); _assert(!a1 && a1n);
	bool a2 = _isalnum('a'); bool a2n = _isalnum('1'); _assert(a2 && a2n);
	bool a3 = _isalpha('a'); bool a3n = _isalpha('A'); _assert(a3 && a3n);
	bool a4 = _isdigit('a'); bool a4n = _isdigit('1'); _assert(!a4 && a4n);
	bool a5 = _isxdigit('a'); bool a5n = _isxdigit('A'); _assert(!a5 && !a5n);
	char a6 = __tolower('a'); char a6n = __tolower('A'); _assert(a6 == 'a' && a6n == 'a');
}

__global__ static void runtimeExample5(void *r)
{
	_runtimeSetHeap(r);
	array_t<char> name = "SkyM";
	name = "ScottP";
	char *a0 = name;
	size_t a0l = name.length;
}

__global__ static void runtimeExample6(void *r)
{
	_runtimeSetHeap(r);
	char buf[100];
	int a0 = _strcmp("Test", "Test"); _assert(!a0);
	int a1 = _strncmp("Tesa", "Tesb", 3); _assert(!a1);
	_memcpy(buf, "Test", 4);
	_memset(buf, 0, sizeof(buf));
	int a2 = _memcmp("Test", "Test", 4); _assert(!a2);
	int a3 = _strlen30("Test"); _assert(a3 == 4);
}

void __runtimeExample(cudaRuntimeHost &r)
{
	cudaRuntimeSetHeap(r.heap);
	runtimeExample0<<<1, 1>>>(r.heap); cudaRuntimeExecute(r);
	runtimeExample1<<<1, 1>>>(r.heap); cudaRuntimeExecute(r);
	runtimeExample2<<<1, 1>>>(r.heap); cudaRuntimeExecute(r);
	runtimeExample3<<<1, 1>>>(r.heap); cudaRuntimeExecute(r);
	runtimeExample4<<<1, 1>>>(r.heap); cudaRuntimeExecute(r);
	runtimeExample5<<<1, 1>>>(r.heap); cudaRuntimeExecute(r);
	runtimeExample6<<<1, 1>>>(r.heap); cudaRuntimeExecute(r);
}