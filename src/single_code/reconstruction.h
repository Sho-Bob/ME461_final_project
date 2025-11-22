#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include <vector>

void reconstruct_WENOJS(const std::vector<double>& u, std::vector<double>& uL,  // uL[i] = u_{i-1/2}^L (left state at interface i)
    std::vector<double>& uR,  // uR[i] = u_{i-1/2}^R (right state at interface i)
    int N,int ibd);


    