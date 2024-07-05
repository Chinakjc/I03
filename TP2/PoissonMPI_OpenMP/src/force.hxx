#include <cmath>

#ifndef __FORCE_HXX
#define __FORCE_HXX

inline double force(double x, double y, double z, double t)
{
  return sin(x - 0.5) * cos(y - 0.5) * exp(-z * z);
}

#endif
