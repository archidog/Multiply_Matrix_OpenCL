//
//  cpuComputing.h
//  MultiplyMatrixOpenCL
//
//  Created by Антон Звонарёв on 21.01.15.
//  Copyright (c) 2015 Антон Звонарёв. All rights reserved.
//

int cpuComputing(float** A, float** B, float** C, const int DATA_SIZE);
int gpuComputing(float** A, float** B, float** C, const int DATA_SIZE);