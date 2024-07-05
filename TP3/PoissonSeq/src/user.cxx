#include "user.hxx"
#include <cmath>

double cond_ini(double x, double y, double z)
{
  x -= 0.5;
  y -= 0.5;
  z -= 0.5;
  if (x * x + y * y + z * z < 0.1)
    return 1.0;
  else
    return 0.0;
}

double cond_lim(double x, double y, double z)
{
  return 0.0;
}

double force(double x, double y, double z, double t)
{
  return sin(x - 0.5) * cos(y - 0.5) * exp(-z * z);
}
