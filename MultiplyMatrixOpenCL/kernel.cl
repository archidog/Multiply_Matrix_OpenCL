//
//  kernel.cl
//  MultiplyMatrixOpenCL
//
//  Created by Антон Звонарёв on 13.01.15.
//  Copyright (c) 2015 Антон Звонарёв. All rights reserved.
//

__kernel void multiplyMatrix(__global float *iMatrix1, __global float *iMatrix2, __global float *oMatrix, int count)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if(i < count && j < count)
    {
        for(int k = 0; k < count; k++)
        {
            oMatrix[i * count + j] += iMatrix1[i * count + k] * iMatrix2[k * count + j];
        }
    }
    
}