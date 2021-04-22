#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "IUnityInterface.h"
#include "DebugDLL.h"
#include <cufft.h>
#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
std::unique_ptr<int[]> getAArray(const unsigned int size);
std::unique_ptr<int[]> getBArray(const unsigned int size);


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

inline static bool debugError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        DebugDLL::ss << msg << " : " << error;
        DebugDLL::log(&DebugDLL::ss.str(), Color::Red);
        return true;
    }
    return false;
}

inline static bool debugResult(cufftResult result, const char* msg) {
    if (result != cudaSuccess) {
        DebugDLL::ss << msg << " : " << result;
        DebugDLL::log(&DebugDLL::ss.str(), Color::Red);
        return true;
    }
    return false;
}


extern "C"
{
    UNITY_INTERFACE_EXPORT int cudaTest(float* data, int size)
    {
        DebugDLL::clear();

        const unsigned int arraySize = 1024;

        cufftHandle plan;
        cufftComplex* complexHostData;
        cufftComplex* complexDeviceData;
        
        cufftResult result;
        cudaError_t error;

        // init host data
        complexHostData = (cufftComplex*)malloc(sizeof(cufftComplex) * size);
        for (int i = 0; i < size; i++)
            complexHostData[i] = make_cuFloatComplex(data[i], 0);


        // create device data
        error = cudaMalloc((void**)&complexDeviceData, sizeof(cufftComplex) * size);
        if (debugError(error, "Unable to cudaMalloc complexData")) {
            goto CUDA_MALLOC_ERROR;
        }

        result = cufftPlan1d(&plan, size, CUFFT_C2C, 1);
        if (debugResult(result, "cufftPlan1d Failed")) {
            goto CUFFT_PLAN_ERROR;
        }

        error = cudaMemcpy((void*)complexDeviceData, complexHostData, sizeof(cufftComplex) * size, cudaMemcpyHostToDevice);
        if (debugError(error, "cudaMemcpy Host => Device failed")) {
            goto CPY_TO_DEVICE_ERR;
        }

        cufftResult result = cufftExecC2C(plan, complexDeviceData, complexDeviceData, CUFFT_FORWARD);
        if (debugResult(result, "cufftExecC2R failed")) {
            goto CUFFT_EXEC_ERR;
        }

        error = cudaDeviceSynchronize();
        if (debugError(error, "cudaDeviceSynchronize failed")) {
            goto DEVICE_SYNCH_ERR;
        }

        error = cudaMemcpy((void*)complexHostData, complexDeviceData, sizeof(cufftComplex) * size, cudaMemcpyDeviceToHost);
        if (debugError(error, "cudaMemcpy Device => Host failed")) {
            goto CPY_TO_HOST_ERR;
        }


        DebugDLL::ss << "Device Copy Success! Value[777777]: {" << complexHostData[777777].x << "," << complexHostData[777777].y << "}";
        DebugDLL::log(&DebugDLL::ss.str(), Color::Blue);

        
        // ...
        result = cufftDestroy(plan);
        if (debugResult(result, "cufftDestroy failed")) {
            return -7;
        }

        error = cudaFree(complexDeviceData);
        if (debugError(error, "cudaFree failed")) {
            return -8;
        }

        free(complexHostData);

        /*
        auto a = getAArray(arraySize);
        auto b = getBArray(arraySize);

        int c[arraySize] = {0};

       
        // Add vectors in parallel
        cudaError_t cudaStatus = addWithCuda(c, a.get(), b.get(), arraySize);

        if (cudaStatus != cudaSuccess) {
            DebugDLL::ss << "addWithCuda failed!";
            DebugDLL::log(&DebugDLL::ss.str(), Color::Red);
            return -4;
        }

       
        DebugDLL::ss << "data[1000000]: " << data[1000000];
        DebugDLL::log(&DebugDLL::ss.str(), Color::Blue);
        */

        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        //cudaError_t cudaStatus = cudaDeviceReset();
        //if (cudaStatus != cudaSuccess) {
        //    DebugDLL::ss << "cudaDeviceReset failed!";
        //    DebugDLL::log(&DebugDLL::ss.str(), Color::Red);
        //    return -5;
        //}


        return 0;

    CUDA_MALLOC_ERROR:
        free(complexHostData);
        return -1;
    CUFFT_PLAN_ERROR:
        error = cudaFree(complexDeviceData);
        if (debugError(error, "cudaFree failed")) {
            return -8;
        }
        free(complexHostData);
        return -2;
    CPY_TO_DEVICE_ERR:
        result = cufftDestroy(plan);
        if (debugResult(result, "cufftDestroy failed")) {
            return -7;
        }
        free(complexHostData);
        return -3;
    CUFFT_EXEC_ERR:
        result = cufftDestroy(plan);
        if (debugResult(result, "cufftDestroy failed")) {
            return -7;
        }
        free(complexHostData);
        return -4;
    DEVICE_SYNCH_ERR:
        result = cufftDestroy(plan);
        if (debugResult(result, "cufftDestroy failed")) {
            return -7;
        }
        free(complexHostData);
        return -5;
    CPY_TO_HOST_ERR:
        result = cufftDestroy(plan);
        if (debugResult(result, "cufftDestroy failed")) {
            return -7;
        }
        free(complexHostData);
        return -6;
    }
}


// Helper function for using CUDA to add vectors in parallel
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    dim3 gridDimensions(2);
    dim3 blockDimensions(size / gridDimensions.x);
    addKernel<<<gridDimensions, blockDimensions>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

std::unique_ptr<int[]> getAArray(const unsigned int size) {
    auto a = std::unique_ptr<int[]>(new int[size]);
    for (int i = 0, int n = 1; i < size; i++, n++) {
        a[i] = n * n;
    }
    return a;
}

std::unique_ptr<int[]> getBArray(const unsigned int size) {
    auto a = std::unique_ptr<int[]>(new int[size]);
    for (int i = 0, int n = 1; i < size; i++, n++) {
        a[i] = n;
    }
    return a;
}