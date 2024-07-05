#include "values.hxx"

double variation
 (const Values & u, const Values & v, 
  double *& diff, double *&diffPartial, int n);

void allocVariationData(double *& diff, int n,
  double *& diffPartial, int nPartial);

void freeVariationData(double *& diff, 
  double *& diffPartial);
