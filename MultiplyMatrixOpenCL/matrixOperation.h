//
//  matrixOperation.h
//  MultiplyMatrixOpenCL
//
//  Created by Антон Звонарёв on 21.01.15.
//  Copyright (c) 2015 Антон Звонарёв. All rights reserved.
//

void makeMatrix(float** A, const int N, float (*p)()); //Заполнение матрицы
void printMatrix(float** A, const int N); //Вывод матрицы
float bigDoubleRnd(); //Параметр больших дробных значений
float rnd(); //Парметр рандомных значений
float null(); //Параметр нулей