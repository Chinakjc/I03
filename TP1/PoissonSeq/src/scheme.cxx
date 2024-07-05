#include "scheme.hxx"
#include "parameters.hxx"
#include "version.hxx"
#include "force.hxx"
#include <cmath>

#include <sstream>
#include <iomanip>

Scheme::Scheme(Parameters &P) :
    codeName(version), m_P(P), m_u(P), m_v(P)  {

  m_t = 0.0;
  m_duv = 0.0;

  int i;
  for (i = 0; i < 3; i++)
  {
    m_n[i] = m_P.n(i);
    m_dx[i] = m_P.dx(i);
    m_xmin[i] = m_P.xmin(i);
  }

  m_dt = m_P.dt();
}

Scheme::~Scheme()
{
}

double Scheme::present()
{
  return m_t;
}

bool Scheme::iteration()
{

  m_duv = iteration_domaine(
      m_P.imin(0), m_P.imax(0),
      m_P.imin(1), m_P.imax(1),
      m_P.imin(2), m_P.imax(2));

  m_t += m_dt;
  m_u.swap(m_v);

  return true;
}

double Scheme::iteration_domaine(int imin, int imax,
                                 int jmin, int jmax,
                                 int kmin, int kmax)
{
  double lam_x = 1 / (m_dx[0] * m_dx[0]);
  double lam_y = 1 / (m_dx[1] * m_dx[1]);
  double lam_z = 1 / (m_dx[2] * m_dx[2]);
  double xmin = m_xmin[0];
  double ymin = m_xmin[1];
  double zmin = m_xmin[2];
  int i, j, k;
  double du, du1, du2, du_sum = 0.0;
  
  double x, y, z;

  for (i = imin; i <= imax; i++)
    for (j = jmin; j <= jmax; j++)
      for (k = kmin; k <= kmax; k++) {

        du1 = (-2 * m_u(i, j, k) + m_u(i + 1, j, k) + m_u(i - 1, j, k)) * lam_x
            + (-2 * m_u(i, j, k) + m_u(i, j + 1, k) + m_u(i, j - 1, k)) * lam_y
            + (-2 * m_u(i, j, k) + m_u(i, j, k + 1) + m_u(i, j, k - 1)) * lam_z;

        x = xmin + i * m_dx[0];
        y = ymin + j * m_dx[1];
        z = zmin + k * m_dx[2];
        du2 = force(x, y, z, m_t);

        du = m_dt * (du1 + du2);
        m_v(i, j, k) = m_u(i, j, k) + du;

        du_sum += du > 0 ? du : -du;
      }

  return du_sum;
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

