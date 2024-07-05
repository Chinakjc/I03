#ifndef _USER_CUH
#define _USER_CUH

__device__
double cond_ini(double x, double y, double z);

__device__
double cond_lim(double x, double y, double z);

__device__
double force(double x, double y, double z);

#endif
