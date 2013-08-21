#include <cassert>

__global__ void simpleTest(int N)
{
    //int gtid = blockIdx.x*blockDim.x + threadIdx.x ;
    //thread whose id > N will print assertion failed error msg
    //assert(gtid < N);
}

void simpleTest_host()
{
	simpleTest<<<1, 1>>>(5);
}