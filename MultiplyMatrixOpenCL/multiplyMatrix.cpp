//
//  main.cpp
//  MultiplyMatrixOpenCL
//
//  Created by Антон Звонарёв on 13.01.15.
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

#include "computing.h"
#include "matrixOperation.h"

using namespace std;



int main()
{
    srand((unsigned int)time(NULL));
    char next;
    //Объявление переменных
    
    
    
    
    cout << "Демонстрация вычислений с использованием возможностей OpenCL" << endl;
    cout << "Вычислений небольшой точности." << endl;
    cout << "Произведем умножение матрицы небольшого рамера (20x20), но с большими дробными числами." << endl;
    testTimeDouble(20, 700);
    cout << "Продолжить _ "; cin >> next;

    cout << "Произведем умножение матрицы большого рамера рамера(100x100), с большими дробными числами." << endl;
    testTimeDouble(50, 700);
    

    
    return 0;
}
