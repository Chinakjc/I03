#ifndef __COND_INI_HXX
#define __COND_INI_HXX

inline double cond_ini(double x, double y, double z)
{
  x -= 0.5;
  y -= 0.5;
  z -= 0.5;
  if (x * x + y * y + z * z < 0.1)
    return 1.0;
  else
    return 0.0;
}

#endif

