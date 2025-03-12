#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction
{
    namespace Naive
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // TODO: __global__
        __global__ void kernNaiveScan(int n, int *odata, const int *idata, int k)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }

            int pos = 1 << (k - 1);
            if (index >= pos)
            {
                odata[index] = idata[index] + idata[index - pos];
            }
            else
            {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata)
        {
            timer().startGpuTimer();
            // TODO
            int logn = (int)ceil(log2((double)n));
            int size = 1 << logn;

            int *dev_in;
            int *dev_out;
            cudaMalloc((void **)&dev_in, size * sizeof(int));
            cudaMalloc((void **)&dev_out, size * sizeof(int));

            cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemset(dev_in + n, 0, sizeof(int) * (size - n));

            int blockSize = 128;
            dim3 blocksPerGrid((size + blockSize - 1) / blockSize);
            // 3  1  7  0  4  1  6  3
            //      k: 1 ~ 3
            // k = 1, 1
            // 3  4  8  7  4  5  7  9       take n-1: 6 as example
            // k = 2, 2
            // 3  4 11 11 12 12 11 14       11 = 7 + 4 = (6 + 1 + 4 + 0)
            // k = 3, 4
            // 3  4 11 11 15 16 22 25       22 = 11 + 11 = 11 + 8 + 3 = 11 + (7 + 1 + 3) 
            for (int k = 1; k <= logn; k++)
            {
                kernNaiveScan<<<blocksPerGrid, blockSize>>>(size, dev_out, dev_in, k);
                std::swap(dev_in, dev_out);
            }

            cudaMemcpy(odata, dev_in, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_in);
            cudaFree(dev_out);

            timer().endGpuTimer();
        }
    }
}
