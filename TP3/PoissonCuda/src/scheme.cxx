#include "scheme.hxx"
#include "parameters.hxx"
#include "version.hxx"
#include <cmath>

#include <sstream>
#include <iomanip>

#include "iteration.hxx"
#include "variation.hxx"
#include "timer.hxx"
#include "dim.hxx"


Scheme::Scheme(Parameters &P) :
    codeName(version), m_P(P), m_u(P), m_v(P)  {

  m_t = 0.0;
  m_duv = 0.0;

  double lx[3];
  int i;
  for (i=0; i<3; i++) {
    m_n[i] = m_P.n(i);
    m_dx[i] = m_P.dx(i);
    m_xmin[i] = m_P.xmin(i);
    lx[i] = 1.0/(m_dx[i]*m_dx[i]);
  }

  m_dt = m_P.dt();

  setDims(m_n, m_xmin, m_dx, lx);

  diff = NULL;
  partialDiff = NULL;
}

Scheme::~Scheme()
{
  freeVariationData(diff, partialDiff);
}

double Scheme::present()
{
  return m_t;
}

void Scheme::iteration()
{
  m_duv = iteration_domaine(
      m_P.imin(0), m_P.imax(0),
      m_P.imin(1), m_P.imax(1),
      m_P.imin(2), m_P.imax(2));

  m_t += m_dt;
  m_u.swap(m_v);
  m_u.synchronized(false);
}

double Scheme::iteration_domaine(int imin, int imax,
                                 int jmin, int jmax,
                                 int kmin, int kmax)
{
  ::iteration(m_v, m_u, m_dt, m_n,
                   imin, imax, jmin, jmax, kmin, kmax);

  m_v.synchronized(false);
  return ::variation(m_u, m_v, 
                          diff, partialDiff, 
                          m_n[0]*m_n[1]*m_n[2]);
}

const Values & Scheme::getOutput()
{
  return m_u;
}

void Scheme::setInput(const Values & u)
{
  m_u = u;
  m_v = u;
}
