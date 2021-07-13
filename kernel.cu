#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define dx 0.01
#define dy 0.01
#define rho 8800
#define C 381
#define lambda 384.0
#define tau 0.01
#define BLOCK_SIZE 16

__global__ void __laplas__(float *T,float *T_old, const int n, const int height)
{
        double at = lambda / (rho * C);

        int iA = n * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;

        if(blockDim.y * blockIdx.y + threadIdx.y>0 && blockDim.x * blockIdx.x + threadIdx.x>0 && blockDim.y * blockIdx.y + threadIdx.y < height && blockDim.x * blockIdx.x + threadIdx.x < n)
                T[iA] = T_old[iA] + (tau / (dx * dx)) * at * (T_old[n * (blockDim.y * blockIdx.y + threadIdx.y-1) + blockDim.x * blockIdx.x + threadIdx.x] - 2 * T_old[iA] + T_old[n * (blockDim.y * blockIdx.y + threadIdx.y+1) + blockDim.x * blockIdx.x + threadIdx.x]) + (tau / (dy * dy)) * at * (T_old[n * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x-1] - 2 * T_old[iA] + T_old[n * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x+1]);
}

extern "C" void gpu(int index, int numDev, int n, int height, float *T, float *T_old)
{
        cudaSetDevice(numDev);
        if (index == 0)
        {
                int count;
                unsigned int flag;
                int device;
                cudaGetDevice(&device);
                cudaGetDeviceCount(&count);
                cudaGetDeviceFlags(&flag);
                printf("set device %d\n", numDev);
                printf("device %d\n", device);
                printf("device flag %d\n", flag);
                printf("device count %d\n", count);
        }
        size_t size = (height+1) * (n+1) * sizeof(float);

        float *dev_T = NULL;
        cudaMalloc((void **)&dev_T, size);
        float *dev_T_old = NULL;
        cudaMalloc((void **)&dev_T_old, size);

        cudaMemcpy( dev_T, T, size, cudaMemcpyHostToDevice );
        cudaMemcpy( dev_T_old, T_old, size, cudaMemcpyHostToDevice );

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 dimGrid(n/BLOCK_SIZE, height/BLOCK_SIZE, 1);

        __laplas__<<<dimGrid, dimBlock>>>(dev_T,dev_T_old, n, height);
        cudaDeviceSynchronize();

        cudaMemcpy(T, dev_T, size, cudaMemcpyDeviceToHost);

        cudaFree(dev_T);
        cudaFree(dev_T_old);
}
