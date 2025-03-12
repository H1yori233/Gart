#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction
{
    namespace CPU
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata)
        {
            timer().startCpuTimer();
            // TODO
            odata[0] = idata[0];
            for (int i = 1; i < n; i++)
            {
                odata[i] = odata[i - 1] + idata[i];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata)
        {
            timer().startCpuTimer();
            // TODO
            int count = 0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] > 0)
                {
                    odata[count++] = idata[i];
                }
            }

            timer().endCpuTimer();
            // return -1;
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata)
        {   
            timer().startCpuTimer();

            // TODO
            int count = -1;
            int *temp = (int *)malloc(sizeof(int) * n);
            int *scan_result = (int *)malloc(sizeof(int) * n);

            for (int i = 0; i < n; i++)
            {
                temp[i] = idata[i] > 0 ? 1 : 0;
            }
            
            // scan(n, scan_result, temp);
            scan_result[0] = 0;
            for (int i = 1; i < n; i++)
            {
                scan_result[i] = scan_result[i - 1] + temp[i - 1];
            }
            
            count = 0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] > 0)
                {
                    count++;
                    odata[scan_result[i]] = idata[i];
                }
            }

            free(temp);
            free(scan_result);

            timer().endCpuTimer();
            return count;
        }
    }
}
