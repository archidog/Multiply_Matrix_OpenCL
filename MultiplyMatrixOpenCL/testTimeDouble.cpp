//
//  testTimeDouble.cpp
//  MultiplyMatrixOpenCL
//
//  Created by Антон Звонарёв on 14.02.15.
//  Copyright (c) 2015 Антон Звонарёв. All rights reserved.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>
#include <stdlib.h>
#include <time.h>


#include "kernel.cl.h"
#include "testTimeDouble.h"
#include "matrixOperation.h"

using namespace std;

int testTimeDouble(const int DATA_SIZE, const int testCount)
{
    cout << "Начало тестирования скорости выполнения.."<<endl;
    
    ofstream gpuresult;
    gpuresult.open("/Users/account_96/testTime/gpuBig.txt", ios_base::out | ios_base::trunc);
    
    ofstream cpuresult;
    cpuresult.open("/Users/account_96/testTime/cpuBig.txt", ios_base::out | ios_base::trunc);
    
    
    
    
    
    const int sizeNameDevice = 128;
    
    clock_t gpuTime = 0;
    clock_t cpuTime = 0;
    double gpuTempTime = 0;
    double cpuTempTime = 0;
    char nameGpuDevice[sizeNameDevice];
    char nameCpuDevice[sizeNameDevice];
    
    vector<double> deltaResult;
    
    float** iMatrix1 = new float*[DATA_SIZE];
    float** iMatrix2 = new float*[DATA_SIZE];
    float** oMatrix = new float*[DATA_SIZE];
    float** oMatrixHost = new float*[DATA_SIZE];
    
    for (int i = 0; i < DATA_SIZE; i++)
    {
        iMatrix1[i] = new float[DATA_SIZE];
        iMatrix2[i] = new float[DATA_SIZE];
        oMatrix[i] = new float[DATA_SIZE];
        oMatrixHost[i] = new float[DATA_SIZE];
    }
    
    float* iMatrix1Buf = new float[DATA_SIZE * DATA_SIZE];
    float* iMatrix2Buf = new float[DATA_SIZE * DATA_SIZE];
    float* oMatrixBuf = new float[DATA_SIZE * DATA_SIZE];
    float* oMatrixHostBuf = new float[DATA_SIZE * DATA_SIZE];
    
    
    cout << "Инициализация входных данных..." << endl;
    
    //Инициализация входных данных
    makeMatrix(iMatrix1, DATA_SIZE, &bigDoubleRnd);
    makeMatrix(iMatrix2, DATA_SIZE, &bigDoubleRnd);
    makeMatrix(oMatrixHost, DATA_SIZE, &null);
    makeMatrix(oMatrix, DATA_SIZE, &null);
    
    
    int k = 0;
    for (int i = 0; i < DATA_SIZE; i++)
    {
        for(int j = 0; j < DATA_SIZE; j++)
        {
            iMatrix1Buf[k] = (cl_float)iMatrix1[i][j];
            iMatrix2Buf[k] = (cl_float)iMatrix2[i][j];
            k++;
        }
        
    }
    
    cout << "Создание очередей..." << endl;
    //Очередь
    dispatch_queue_t queueGpuDevices = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    if(queueGpuDevices == NULL)
    {
        cout << "Не найдено ни одного графического устройства" << endl;
        return 1;
    }
    
    dispatch_queue_t queueCpuDevices = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    if(queueCpuDevices == NULL)
    {
        cout << "Не найдено ни одного процессора" << endl;
        return 1;
    }
    
    cout << "Инициализация устройств..." << endl;
    //Вывод названий устройств на экран
    cl_device_id gpuDevices = gcl_get_device_id_with_dispatch_queue(queueGpuDevices);
    clGetDeviceInfo(gpuDevices, CL_DEVICE_NAME, sizeNameDevice, nameGpuDevice, NULL);
    cl_device_id cpuDevices = gcl_get_device_id_with_dispatch_queue(queueCpuDevices);
    clGetDeviceInfo(cpuDevices, CL_DEVICE_NAME, sizeNameDevice, nameCpuDevice, NULL);
    
    cout << "Имя видеокарты: " << nameGpuDevice <<endl;
    cout << "Имя процессора: " << nameCpuDevice <<endl;
    cout<<endl;
    
    //Создание копий для передачи в kernel
    
    void* memMatrix11  = gcl_malloc(sizeof(cl_float) * DATA_SIZE * DATA_SIZE, iMatrix1Buf, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* memMatrix22  = gcl_malloc(sizeof(cl_float) * DATA_SIZE * DATA_SIZE, iMatrix2Buf, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    
    cout << "Старт тестовых вычислений: " << endl;
    //Начало вычислений
    for(int i = 0; i < testCount; i++)
    {
        void* memMatrixOut = gcl_malloc(sizeof(cl_float) * DATA_SIZE * DATA_SIZE, NULL, CL_MEM_WRITE_ONLY);
        void* memMatrixOutHost = gcl_malloc(sizeof(cl_float) * DATA_SIZE * DATA_SIZE, NULL, CL_MEM_WRITE_ONLY);
        
        if(i % 1 == i)
        {
            //Процессор
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
                                  multiplyMatrix_kernel(&range,(cl_float*)memMatrix11, (cl_float*)memMatrix22, (cl_float*)memMatrixOutHost, DATA_SIZE);
                                  gcl_memcpy(oMatrixHostBuf, memMatrixOutHost, sizeof(cl_float) * DATA_SIZE * DATA_SIZE);
                                  
                              });
                
                cpuTime = clock() - cpuTime;
                cpuTempTime = (double)cpuTime / 1000;
                
                cout << i+1 << " Процессор: Done! За:  " << cpuTempTime << endl;
                
            }
            catch (exception ex)
            {
                cout << "Не хватает мощности процессора для вычислений \n" << &ex << endl;
            }
            
            
            //Видеокарта
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
                                  multiplyMatrix_kernel(&range,(cl_float*)memMatrix11, (cl_float*)memMatrix22, (cl_float*)memMatrixOut, DATA_SIZE);
                                  gcl_memcpy(oMatrixBuf, memMatrixOut, sizeof(cl_float) * DATA_SIZE * DATA_SIZE);
                              });
                
                gpuTime = clock() - gpuTime;
                gpuTempTime = (double)gpuTime / 1000;
                
                cout << i+1 << " Видеокарта: Done! За: " << gpuTempTime << endl;
            }
            catch(exception ex)
            {
                cout << "Не хватает мощности видеокарты для вычистений \n " << &ex << endl;
            }
            
        }//endif
        else
        {
            
            //Видеокарта
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
                                  multiplyMatrix_kernel(&range,(cl_float*)memMatrix11, (cl_float*)memMatrix22, (cl_float*)memMatrixOut, DATA_SIZE);
                                  gcl_memcpy(oMatrixBuf, memMatrixOut, sizeof(cl_float) * DATA_SIZE * DATA_SIZE);
                              });
                
                gpuTime = clock() - gpuTime;
                gpuTempTime = (double)gpuTime / 1000;
                
                cout << i+1 << " Видеокарта: Done! За: " << gpuTempTime << endl;
            }
            catch(exception ex)
            {
                cout << "Не хватает мощности видеокарты для вычистений \n " << &ex << endl;
            }
            
            //Процессор
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
                                  multiplyMatrix_kernel(&range,(cl_float*)memMatrix11, (cl_float*)memMatrix22, (cl_float*)memMatrixOutHost, DATA_SIZE);
                                  gcl_memcpy(oMatrixHostBuf, memMatrixOutHost, sizeof(cl_float) * DATA_SIZE * DATA_SIZE);
                                  
                              });
                
                cpuTime = clock() - cpuTime;
                cpuTempTime = (double)cpuTime / 1000;
                
                cout << i+1 << " Процессор: Done! За:  " << cpuTempTime << endl;
                
            }
            catch (exception ex)
            {
                cout << "Не хватает мощности процессора для вычислений \n" << &ex << endl;
            }
            
            
            
        }//endelse
        
        gpuresult << gpuTempTime << endl;
        cpuresult << cpuTempTime << endl;
        if(gpuTempTime < cpuTempTime)
        {
            cout << "   Видеокарта быстрее на: " << cpuTempTime - gpuTempTime << endl;
            deltaResult.push_back(cpuTempTime - gpuTempTime);
        }
        else
        {
            if(gpuTempTime > cpuTempTime)
            {
                cout << "   Процессор быстрее на: " << gpuTempTime - cpuTempTime << endl;
                deltaResult.push_back(gpuTempTime - cpuTempTime);
            }
            else
            {
                cout << "Вычисления были проведены на одной скорости" << endl;
                deltaResult.push_back(cpuTempTime - gpuTempTime);
            }
            
        }
        cout<<endl;
    }
    
    
    k = 0;
    for (int i = 0; i < DATA_SIZE; i++)
    {
        for(int j = 0; j < DATA_SIZE; j++)
        {
            oMatrixHost[i][j] = oMatrixHostBuf[k];
            oMatrix[i][j] = oMatrixBuf[k];
            k++;
        }
        
    }
    /*cout << "Входные данные: " << endl;
    printMatrix((float**)iMatrix1, DATA_SIZE);
    cout << endl;
    printMatrix((float**)iMatrix2, DATA_SIZE);
    cout << endl;
    cout << "Результат процессора: " << endl;
    printMatrix((float**)oMatrixHost, DATA_SIZE);
    cout << endl;
    cout << "Результат видеокарты: " << endl;
    printMatrix((float**)oMatrix, DATA_SIZE);
    cout << endl;*/
    
    
    
    
    
    
    /*
     if(countTimeCpu > countTimeGpu)
     cout << "На вашем устройстве лучше производить вычисления на процессоре" << endl;
     if(countTimeCpu < countTimeGpu)
     cout << "На вашем устройстве лучше производить вычисления на видеокарте" << endl;
     if(countTimeCpu == countTimeGpu)
     cout << "На вашем устройстве производительость видеокарты и процессора одинаковая" << endl;
     cout << endl;
     */
    
    cout << "Тестирование скорости вычисления завершено успешно" << endl;
    
    
    
    
    //Очистка памяти
    gpuresult.close();
    cpuresult.close();
    
    ofstream deltaresult;
    deltaresult.open("/Users/account_96/testTime/deltaBig.txt", ios_base::out | ios_base::trunc);
    
    
    for(int i = 0; i < testCount; i++)
        deltaresult << deltaResult[i] << endl;
    deltaresult.close();
    
    
    for(int i = 0; i < DATA_SIZE; i++)
    {
        delete [] iMatrix1[i];
        delete [] iMatrix2[i];
        delete [] oMatrix[i];
        delete [] oMatrixHost[i];
    }
    delete [] iMatrix1;
    delete [] iMatrix1Buf;
    delete [] iMatrix2;
    delete [] iMatrix2Buf;
    delete [] oMatrix;
    delete [] oMatrixBuf;
    delete [] oMatrixHost;
    
    return 0;
}
