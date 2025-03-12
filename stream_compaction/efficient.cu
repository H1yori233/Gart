#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction
{
    namespace Efficient
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        __global__ void kernUpSweep(int n, int *tree, int count)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= count)
            {
                return;
            }

            // The node we actually want to use
            // count = 4 -> index: 3 ~ 7
            // 0=1+2, 1=3+4 2=5+6, 3=7+8 4=8+9
            // 2n+1 2n+2
            index += count - 1;
            int double_index = index << 1;
            tree[index] = tree[double_index + 1] + tree[double_index + 2];
        }

        __global__ void kernDownSweep(int n, int *tree, int count)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= count)
            {
                return;
            }
            // The node we actually want to use
            index += count - 1;
            int double_index = index << 1;
            int temp = tree[double_index + 1];
            tree[double_index + 1] = tree[index];
            tree[double_index + 2] = tree[index] + temp;
        }

        __global__ void kernFlag(int n, int *odata, const int *idata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }

            if (idata[index] > 0)
            {
                odata[index] = 1;
            }
            else
            {
                odata[index] = 0;
            }
        }

        __global__ void kernScatter(int n, int *odata, const int *idata,
                                    const int *scan_result)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }

            if (idata[index] > 0 && scan_result[index] >= 0)
            {
                odata[scan_result[index]] = idata[index];
            }
        }

        void scan(int n, int *odata, const int *idata)
        {
            timer().startGpuTimer();
            // TODO
            int logn = (int)ceil(log2((double)n));
            int tree_size = (2 << logn) - 1;
            int leaf_offset = (1 << logn) - 1;

            int *dev_tree;
            cudaMalloc((void **)&dev_tree, sizeof(int) * tree_size);
            checkCUDAErrorWithLine("malloc dev_tree failed!");
            cudaMemset(dev_tree, 0, sizeof(int) * tree_size);
            checkCUDAErrorWithLine("memset full tree failed!");
            cudaMemcpy(dev_tree + leaf_offset, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("memcpy dev_tree failed!");

            // Up Sweep
            for (int k = logn - 1; k >= 0; k--)
            {
                // 16
                // 31 15 -> 8 4 2 1
                int count = 1 << k;
                int blockSize = min(count, 128);
                dim3 gridSize((count + blockSize - 1) / blockSize);
                kernUpSweep<<<gridSize, blockSize>>>(tree_size, dev_tree, count);
                checkCUDAErrorWithLine("kernUpSweep failed!");
            }

            // Down Sweep
            int first = 0;
            int final_add;
            cudaMemcpy(&final_add, dev_tree, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(dev_tree, &first, sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("set root to zero failed!");

            // Thread Num: 1, 2, 4, ...
            // offset: 0, 1, 3...
            for (int k = 0; k < logn; k++)
            {
                int count = 1 << k;
                int blockSize = min(count, 128);
                dim3 gridSize((count + blockSize - 1) / blockSize);
                kernDownSweep<<<gridSize, blockSize>>>(tree_size, dev_tree, count);
                checkCUDAErrorWithLine("kernDownSweep failed!");
            }

            // cudaMemcpy(odata, dev_tree + leaf_offset, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_tree + leaf_offset + 1, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
            odata[n - 1] = final_add;
            checkCUDAErrorWithLine("copy dev_tree to output failed!");
            cudaFree(dev_tree);
            checkCUDAErrorWithLine("free dev_tree failed!");

            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata)
        {
            timer().startGpuTimer();

            // TODO
            int count = -1;

            int *dev_in;
            int *dev_out;
            int *dev_temp;
            int *dev_scan_result;

            cudaMalloc((void **)&dev_in, sizeof(int) * n);
            checkCUDAErrorWithLine("malloc dev_in failed!");
            cudaMalloc((void **)&dev_out, sizeof(int) * n);
            checkCUDAErrorWithLine("malloc dev_out failed!");
            cudaMalloc((void **)&dev_temp, sizeof(int) * n);
            checkCUDAErrorWithLine("malloc dev_temp failed!");
            cudaMalloc((void **)&dev_scan_result, sizeof(int) * n);
            checkCUDAErrorWithLine("malloc dev_scan_result failed!");

            cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("memcpy dev_in failed!");

            int blockSize = 128;
            dim3 gridSize((n + blockSize - 1) / blockSize);
            kernFlag<<<gridSize, blockSize>>>(n, dev_temp, dev_in);
            checkCUDAErrorWithLine("kernFlag failed!");

            // scan(n, scan_result, temp)
            int logn = (int)ceil(log2((double)n));
            int tree_size = (2 << logn) - 1;
            int leaf_offset = (1 << logn) - 1;

            int *dev_tree;
            cudaMalloc((void **)&dev_tree, sizeof(int) * tree_size);
            checkCUDAErrorWithLine("malloc dev_tree failed!");
            cudaMemset(dev_tree, 0, sizeof(int) * tree_size);
            checkCUDAErrorWithLine("memset full tree failed!");
            cudaMemcpy(dev_tree + leaf_offset, dev_temp, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            checkCUDAErrorWithLine("memcpy dev_tree failed!");

            // Up Sweep
            for (int k = logn - 1; k >= 0; k--)
            {
                int count = 1 << k;
                int blockSize = min(count, 128);
                gridSize = dim3((count + blockSize - 1) / blockSize);
                kernUpSweep<<<gridSize, blockSize>>>(tree_size, dev_tree, count);
                checkCUDAErrorWithLine("kernUpSweep failed!");
            }

            // Down Sweep
            cudaMemcpy(&count, dev_tree, sizeof(int), cudaMemcpyDeviceToHost);
            int first = 0;
            cudaMemcpy(dev_tree, &first, sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("set root to zero failed!");

            for (int k = 0; k < logn; k++)
            {
                int count = 1 << k;
                int blockSize = min(count, 128);
                gridSize = dim3((count + blockSize - 1) / blockSize);
                kernDownSweep<<<gridSize, blockSize>>>(tree_size, dev_tree, count);
                checkCUDAErrorWithLine("kernDownSweep failed!");
            }
            cudaMemcpy(dev_scan_result, dev_tree + leaf_offset, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            checkCUDAErrorWithLine("copy dev_tree to output failed!");
            cudaFree(dev_tree);
            checkCUDAErrorWithLine("free dev_tree failed!");

            // Scatter
            checkCUDAErrorWithLine("copy count failed!");
            gridSize = dim3((n + blockSize - 1) / blockSize);
            kernScatter<<<gridSize, blockSize>>>(n, dev_out, dev_in, dev_scan_result);
            checkCUDAErrorWithLine("kernScatter failed!");

            cudaMemcpy(odata, dev_out, sizeof(int) * count, cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("copy dev_out failed!");

            cudaFree(dev_in);
            checkCUDAErrorWithLine("free dev_in failed!");
            cudaFree(dev_out);
            checkCUDAErrorWithLine("free dev_out failed!");
            cudaFree(dev_scan_result);
            checkCUDAErrorWithLine("free dev_scan_result failed!");

            timer().endGpuTimer();
            return count;
        }
    }
}
