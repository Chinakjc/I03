#ifndef __ITERATION_HXX
#define __ITERATION_HXX

#include "values.hxx"

void iteration(
    Values & v, Values & u, double dt, int n[3],
    int imin, int imax, 
    int jmin, int jmax,
    int kmin, int kmax);

#endif
