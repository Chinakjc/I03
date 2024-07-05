#ifndef __BINARY__
#define __BINARY__

#include <fstream>
#include <iostream>
#include <cstring>

std::string genName(int n)
{
    std::string s = "data_";
    s += std::to_string(n);
    s += ".bin";
    return s;
}

void writeVector(const char *fileName, double *u, int n)
{    
    std::cerr << fileName << std::endl;
    std::ofstream f(fileName, std::ios::binary);
    f.write((char*) &n, sizeof(int));
    f.write((char*) u, n*sizeof(double));
}

void readVector(const char *fileName, double * &u, int &n)
{    
    std::ifstream f(fileName, std::ios::binary);
    f.read((char*) &n, sizeof(int));

    u = new double[n];
    memset(u, 0, n*sizeof(double));

    f.read((char*) u, n*sizeof(double));
}

#endif
