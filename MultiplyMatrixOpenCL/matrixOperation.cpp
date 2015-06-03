//
//  matrixOperation.cpp
//  MultiplyMatrixOpenCL
//
//  Created by Антон Звонарёв on 21.01.15.
//  Copyright (c) 2015 Антон Звонарёв. All rights reserved.
//

#include "matrixOperation.h"
#include <iostream>
#include <math.h>

using namespace std;

void makeMatrix(float** A, const int N, float (*p)())
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            A[i][j] = p();
        }
    }
}

void printMatrix(float** A, const int N)
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            cout<<A[i][j]<<"  ";
        }
        cout<<endl;
    }
}

float rnd()
{
    return rand() % 10;
}
float bigDoubleRnd()
{
    return sqrt(rand() % 10 + 1);
}
float null()
{
    return 0;
}