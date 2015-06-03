//
//  cpuComputing.cpp
//  MultiplyMatrixOpenCL
//
//  Created by Антон Звонарёв on 21.01.15.
//  Copyright (c) 2015 Антон Звонарёв. All rights reserved.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>
#include <stdlib.h>
#include <time.h>

#include "kernel.cl.h"
#include "computing.h"

using namespace std;

int cpuComputing(float** A, float** B, float** C, const int DATA_SIZE)
{
    const int sizeNameDevice = 128;
    
    clock_t cpuTime = 0;
    double cpuTempTime = 0;
    char nameCpuDevice[sizeNameDevice];
    
    float* iMatrix1Buf = new float[DATA_SIZE * DATA_SIZE];
    float* iMatrix2Buf = new float[DATA_SIZE * DATA_SIZE];
    float* oMatrixHostBuf = new float[DATA_SIZE * DATA_SIZE];
    
    int k = 0;
    for (int i = 0; i < DATA_SIZE; i++)
    {
        for(int j = 0; j < DATA_SIZE; j++)
        {
            iMatrix1Buf[k] = (cl_float)A[i][j];
            iMatrix2Buf[k] = (cl_float)B[i][j];
            k++;
        }
    }
    
    dispatch_queue_t queueCpuDevices = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    if(queueCpuDevices == NULL)
    {
        cout << "Не найдено ни одного процессора" << endl;
        return 1;
    }
    
    cl_device_id cpuDevices = gcl_get_device_id_with_dispatch_queue(queueCpuDevices);
    clGetDeviceInfo(cpuDevices, CL_DEVICE_NAME, sizeNameDevice, nameCpuDevice, NULL);
    
    cout << "Имя устройства: " << nameCpuDevice <<endl;

    
    void* memMatrix1  = gcl_malloc(sizeof(cl_float) * DATA_SIZE * DATA_SIZE, iMatrix1Buf, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* memMatrix2  = gcl_malloc(sizeof(cl_float) * DATA_SIZE * DATA_SIZE, iMatrix2Buf, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* memMatrixOutHost = gcl_malloc(sizeof(cl_float) * DATA_SIZE * DATA_SIZE, NULL, CL_MEM_WRITE_ONLY);
    
    //Вычисления
    try
    {
        cpuTime = clock();
        
        dispatch_sync(queueCpuDevices,
                      ^{
                          size_t wgs;
                          size_t size = (size_t)DATA_SIZE;
                          gcl_get_kernel_block_workgroup_info(multiplyMatrix_kernel, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs, NULL);
                          cl_ndrange range = {
                              2,
                              {0, 0},
                              {8192, size, 0},
                              {wgs, 1, 5}
                          };
                          multiplyMatrix_kernel(&range,(cl_float*)memMatrix1, (cl_float*)memMatrix2, (cl_float*)memMatrixOutHost, DATA_SIZE);
                          gcl_memcpy(oMatrixHostBuf, memMatrixOutHost, sizeof(cl_float) * DATA_SIZE * DATA_SIZE);
                          
                      });
        
        cpuTime = clock() - cpuTime;
        cpuTempTime = (double)cpuTime / 1000;
    }
    catch (exception ex)
    {
        cout << "Не хватает мощности процессора для вычислений \n" << &ex << endl;
    }
    
    cout << "Вычисления на процессоре закончены за: " << cpuTempTime << endl;

    //Преобразование к виду матрицы
    k = 0;
    for (int i = 0; i < DATA_SIZE; i++)
    {
        for(int j = 0; j < DATA_SIZE; j++)
        {
            C[i][j] = oMatrixHostBuf[k];
            k++;
        }
    }

    //Очистка памяти
    delete [] iMatrix1Buf;
    delete [] iMatrix2Buf;
    delete [] oMatrixHostBuf;
    
    return 0;
}


//
//ВИДЕОКАРТА
//
int gpuComputing(float** A, float** B, float** C, const int DATA_SIZE)
{
    const int sizeNameDevice = 128;
    
    clock_t gpuTime = 0;
    double gpuTempTime = 0;
    char nameGpuDevice[sizeNameDevice];
    
    float* iMatrix1Buf = new float[DATA_SIZE * DATA_SIZE];
    float* iMatrix2Buf = new float[DATA_SIZE * DATA_SIZE];
    float* oMatrixBuf = new float[DATA_SIZE * DATA_SIZE];
    
    int k = 0;
    for (int i = 0; i < DATA_SIZE; i++)
    {
        for(int j = 0; j < DATA_SIZE; j++)
        {
            iMatrix1Buf[k] = (cl_float)A[i][j];
            iMatrix2Buf[k] = (cl_float)B[i][j];
            k++;
        }
    }
    
    dispatch_queue_t queueGpuDevices = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    if(queueGpuDevices == NULL)
    {
        cout << "Не найдено ни одной видеокарты" << endl;
        return 1;
    }
    
    cl_device_id gpuDevices = gcl_get_device_id_with_dispatch_queue(queueGpuDevices);
    clGetDeviceInfo(gpuDevices, CL_DEVICE_NAME, sizeNameDevice, nameGpuDevice, NULL);
    
    cout << "Имя устройства: " << nameGpuDevice <<endl;
    
    
    void* memMatrix1  = gcl_malloc(sizeof(cl_float) * DATA_SIZE * DATA_SIZE, iMatrix1Buf, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* memMatrix2  = gcl_malloc(sizeof(cl_float) * DATA_SIZE * DATA_SIZE, iMatrix2Buf, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* memMatrixOut = gcl_malloc(sizeof(cl_float) * DATA_SIZE * DATA_SIZE, NULL, CL_MEM_WRITE_ONLY);
    
    //Вычисления
    try
    {
        gpuTime = clock();
        
        dispatch_sync(queueGpuDevices,
                      ^{
                          size_t wgs;
                          size_t size = (size_t)DATA_SIZE;
                          gcl_get_kernel_block_workgroup_info(multiplyMatrix_kernel, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs, NULL);
                          cl_ndrange range = {
                              2,
                              {0, 0},
                              {8192, size, 0},
                              {wgs, 1, 5}
                          };
                          multiplyMatrix_kernel(&range,(cl_float*)memMatrix1, (cl_float*)memMatrix2, (cl_float*)memMatrixOut, DATA_SIZE);
                          gcl_memcpy(oMatrixBuf, memMatrixOut, sizeof(cl_float) * DATA_SIZE * DATA_SIZE);
                          
                      });
        
        gpuTime = clock() - gpuTime;
        gpuTempTime = (double)gpuTime / 1000;
    }
    catch (exception ex)
    {
        cout << "Не хватает мощности видеокарты для вычислений \n" << &ex << endl;
    }
    
    cout << "Вычисления на видеокарте закончены за: " << gpuTempTime << endl;
    
    //Преобразование к виду матрицы
    k = 0;
    for (int i = 0; i < DATA_SIZE; i++)
    {
        for(int j = 0; j < DATA_SIZE; j++)
        {
            C[i][j] = oMatrixBuf[k];
            k++;
        }
    }
    
    //Очистка памяти
    delete [] iMatrix1Buf;
    delete [] iMatrix2Buf;
    delete [] oMatrixBuf;
    
    return 0;

}