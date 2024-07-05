#include "scheme.hxx"
#include "parameters.hxx"
#include "version.hxx"
#include "force.hxx"

#include <cmath>
#include <mpi.h>
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
  double du, du1, du2, du_sum_local = 0.0, du_sum;
  
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
        du_sum_local += du > 0 ? du : -du;
      }

  MPI_Allreduce(&du_sum_local, &du_sum, 1, MPI_DOUBLE, MPI_SUM, m_P.comm());
  return du_sum;
}

void Scheme::synchronize()
{

  for (int idim=0; idim<3; idim++) {

    int jdim = (idim+1)%3;
    int kdim = (jdim+1)%3;

    int omin = m_P.imin(idim), omax = m_P.imax(idim);
    int pmin = m_P.imin(jdim), pmax = m_P.imax(jdim);
    int qmin = m_P.imin(kdim), qmax = m_P.imax(kdim);
    int k, p, q, m = (pmax-pmin+1)*(qmax-qmin+1);

    int i[3];
    std::vector<double> bufferIn(m), bufferOut(m);
    MPI_Status status;

    int voisin = m_P.neighbour(2*idim);
    if (voisin >=0) {
      i[idim] = omin;
      for (k=0, p=pmin; p<=pmax; p++)
        for (q=qmin; q<=qmax; q++, k++) {
          i[jdim] = p; i[kdim] = q;
          bufferOut[k] = m_u(i[0], i[1], i[2]);
        }
  
      MPI_Sendrecv(bufferOut.data(), m, MPI_DOUBLE, voisin, 0,
                   bufferIn.data(),  m, MPI_DOUBLE, voisin, 0,
                   m_P.comm(), &status);

      i[idim] = omin-1;
      for (k=0, p=pmin; p<=pmax; p++)
        for (q=qmin; q<=qmax; q++, k++) {
          i[jdim] = p; i[kdim] = q;
          m_u(i[0], i[1], i[2]) = bufferIn[k];
        }
    }

    voisin = m_P.neighbour(2*idim+1);
    if (voisin >=0) {
      i[idim] = omax;
      for (k=0, p=pmin; p<=pmax; p++)
        for (q=qmin; q<=qmax; q++, k++) {
          i[jdim] = p; i[kdim] = q;
          bufferOut[k] = m_u(i[0], i[1], i[2]);
        }

      MPI_Sendrecv(bufferOut.data(), m, MPI_DOUBLE, voisin, 0,
                   bufferIn.data(),  m, MPI_DOUBLE, voisin, 0,
                   m_P.comm(), &status);

      i[idim] = omax+1;
      for (k=0, p=pmin; p<=pmax; p++)
        for (q=qmin; q<=qmax; q++, k++) {
          i[jdim] = p; i[kdim] = q;
          m_u(i[0], i[1], i[2]) = bufferIn[k];
        }
    }
  }
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

