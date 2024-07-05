#include "values.hxx"
#include "os.hxx"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include "cond_ini.hxx"

Values::Values(Parameters & prm) : m_p(prm)
{
  int i;
  nn = 1;
  for (i=0; i<3; i++) {
    m_imin[i] = m_p.imin(i);
    m_imax[i] = m_p.imax(i);
    m_n[i] = m_imax[i] - m_imin[i] + 3;
    m_imin_global[i] = m_p.imin_global(i);
    m_imax_global[i] = m_p.imax_global(i);
    m_dx[i] = m_p.dx(i);
    m_xmin[i] =  m_p.xmin(i);
    m_xmax[i] =  m_p.xmax(i);
    nn *= m_n[i];
  }
  
  n1 = m_n[2];      // nombre de points dans la premiere direction
  n2 = m_n[1] * n1; // nombre de points dans le plan des 2 premieres directions
  m_u = new double[nn];
}

Values::~Values()
{
  delete [] m_u;
}

void Values::init()
{
  int i, j, k;
  double x, y, z;

  for (i=m_imin[0]; i<=m_imax[0]; i++)
    for (j=m_imin[1]; j<=m_imax[1]; j++)
      for (k=m_imin[2]; k<=m_imax[2]; k++) {
        x = (m_imin_global[0] + i)*m_dx[0];
        y = (m_imin_global[1] + j)*m_dx[1]; 
        z = (m_imin_global[2] + k)*m_dx[2];
        operator()(i,j,k) = cond_ini(x, y, z);
      }
}

void Values::boundaries()
{
  int i[3];
  double x[3];
  for (int idim=0; idim<3; idim++) {

    int jdim = (idim+1)%3;
    int kdim = (idim+2)%3;

    int omin = m_imin[idim]-1, omax = m_imax[idim]+1;
    int pmin = m_imin[jdim]-1, pmax = m_imax[jdim]+1;
    int qmin = m_imin[kdim]-1, qmax = m_imax[kdim]+1;

    int p, q;

    if (m_p.neighbour(2 * idim) < 0)
    {
      i[idim] = omin;
      x[idim] = m_xmin[idim];
      for (p=pmin; p<=pmax; p++)
        for (q=qmin; q<=qmax; q++) {
          i[jdim] = p;
          i[kdim] = q;
          x[jdim] = (m_imin_global[jdim] + p)*m_dx[jdim];
          x[kdim] = (m_imin_global[kdim] + q)*m_dx[kdim];
          operator()(i[0], i[1], i[2]) = cond_ini(x[0], x[1], x[2]);
        }
    }

    if (m_p.neighbour(2 * idim + 1) < 0)
    {
      i[idim] = omax;
      x[idim] = m_xmax[idim];
      for (p=pmin; p<=pmax; p++)
        for (q=qmin; q<=qmax; q++) {
          i[jdim] = p;
          i[kdim] = q;
          x[jdim] = (m_imin_global[jdim] + p)*m_dx[jdim];
          x[kdim] = (m_imin_global[kdim] + q)*m_dx[kdim];
          operator()(i[0], i[1], i[2]) = cond_ini(x[0], x[1], x[2]);
        }
    }
  }
}

std::ostream & operator<< (std::ostream & f, const Values & v)
{
  v.print(f);
  return f;
}

void Values::print(std::ostream & f) const
{
  int i, j, k;
  int imin = m_imin[0], jmin = m_imin[1], kmin = m_imin[2];
  int imax = m_imax[0], jmax = m_imax[1], kmax = m_imax[2];

  for (k=kmin; k<=kmax; k++) {
    f << "k = " << k << std::endl;
    for (j=jmin; j<=jmax; j++) {
      for (i=imin; i<=imax; i++) {
        f << " " << std::setw(12) << std::setprecision(6) << operator()(i,j,k);
      }
      f << std::endl;
    }
    f << std::endl;
  }
}

template<typename T>
void swap(T & a, T & b)
{
  T t = a;
  a = b;
  b = t;
}

void Values::swap(Values & other)
{
  ::swap(m_u, other.m_u);
  int i;
  for (i=0; i<3; i++) {
    ::swap(m_n[i], other.m_n[i]);
    ::swap(m_imin[i], other.m_imin[i]);
    ::swap(m_imax[i], other.m_imax[i]);
    ::swap(m_imin_global[i], other.m_imin_global[i]);
    ::swap(m_imax_global[i], other.m_imax_global[i]);
    ::swap(m_dx[i], other.m_dx[i]);
    ::swap(m_xmin[i], other.m_xmin[i]);
    ::swap(m_xmax[i], other.m_xmax[i]);
  }
  ::swap(nn, other.nn);
  ::swap(n1, other.n1);
  ::swap(n2, other.n2);
}

void Values::plot(int order) const
{
  int ghost = 0;
  std::ostringstream s;
  int i, j, k;
  int imin = m_imin[0] - ghost, jmin = m_imin[1] - ghost, kmin = m_imin[2] - ghost;
  int imax = m_imax[0] + ghost, jmax = m_imax[1] + ghost, kmax = m_imax[2] + ghost;

  int rank = m_p.rank();
  int size = m_p.size();

  s << m_p.resultPath() << "/" << size;
  mkdir_p(s.str().c_str());
  int *send_data = new int[6];
  int *recv_data = m_p.rank() == 0 ? new int[6 * m_p.size()] : nullptr;

  send_data[0] = m_imin_global[0] - ghost;
  send_data[1] = m_imax_global[0] + ghost;
  send_data[2] = m_imin_global[1] - ghost;
  send_data[3] = m_imax_global[1] + ghost;
  send_data[4] = m_imin_global[2] - ghost;
  send_data[5] = m_imax_global[2] + ghost;
  MPI_Gather(send_data, 6, MPI_INT, recv_data, 6, MPI_INT, 0, MPI_COMM_WORLD);

  if (m_p.rank() == 0)
  {
    s << "/plot_" << order << ".pvtr";
    std::ofstream fp(s.str().c_str());

    fp << "<?xml version=\"1.0\"?>\n";
    fp << "<VTKFile type=\"PRectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
       << "<PRectilinearGrid WholeExtent=\""
       << 1 - ghost << " " << m_p.n_global(0) + ghost << " "
       << 1 - ghost << " " << m_p.n_global(1) + ghost << " "
       << 1 - ghost << " " << m_p.n_global(2) + ghost << "\" "
       << "GhostLevel=\"0\">\n";
    for (int i = 0; i < m_p.size(); i++)
      fp << "<Piece Extent=\""
         << recv_data[0 + i * 6] << " "
         << recv_data[1 + i * 6] << " "
         << recv_data[2 + i * 6] << " "
         << recv_data[3 + i * 6] << " "
         << recv_data[4 + i * 6] << " "
         << recv_data[5 + i * 6] << "\""
         << " Source=\"plot_" << order << "_" << i << ".vtr\"/>\n";
    fp << "<PPointData Scalars=\"values\">\n"
       << "  <PDataArray type=\"Float64\" Name=\"values\" format=\"ascii\"/>\n"
       << "  <PDataArray type=\"Int32\" Name=\"process\" format=\"ascii\"/>\n"
       << "</PPointData>\n";
    fp << "<PCoordinates>\n"
       << "  <PDataArray type=\"Float64\" Name=\"X\" format=\"ascii\"/>\n"
       << "  <PDataArray type=\"Float64\" Name=\"Y\" format=\"ascii\"/>\n"
       << "  <PDataArray type=\"Float64\" Name=\"Z\" format=\"ascii\"/>\n"
       << "</PCoordinates>\n";
    fp << "</PRectilinearGrid>\n"
       << "</VTKFile>\n"
       << std::endl;
  }

  s << m_p.resultPath() << "/" << size;
  s << "/plot_" << order << "_" << rank << ".vtr";
  std::ofstream f(s.str().c_str());

  f << "<?xml version=\"1.0\"?>\n";
  f << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    << "<RectilinearGrid WholeExtent=\""
    << m_imin_global[0] - ghost << " " << m_imax_global[0] + ghost << " "
    << m_imin_global[1] - ghost << " " << m_imax_global[1] + ghost << " "
    << m_imin_global[2] - ghost << " " << m_imax_global[2] + ghost
    << "\">\n"
    << "<Piece Extent=\""
    << m_imin_global[0] - ghost << " " << m_imax_global[0] + ghost << " "
    << m_imin_global[1] - ghost << " " << m_imax_global[1] + ghost << " "
    << m_imin_global[2] - ghost << " " << m_imax_global[2] + ghost
    << "\">\n";

  f << "<PointData Scalars=\"values\">\n";
  f << "  <DataArray type=\"Float64\" Name=\"values\" format=\"ascii\">\n";

  for (k = kmin; k <= kmax; k++)
  {
    for (j = jmin; j <= jmax; j++)
    {
      for (i = imin; i <= imax; i++)
        f << " " << std::setw(12) << std::setprecision(6) << operator()(i, j, k);
      f << "\n";
    }
    f << "\n";
  }
  f << " </DataArray>\n";

  f << "  <DataArray type=\"Int32\" Name=\"process\" format=\"ascii\">\n";

  for (k = kmin; k <= kmax; k++)
    for (j = jmin; j <= jmax; j++)
    {
      for (i = imin; i <= imax; i++)
        f << " " << rank;
      f << "\n";
    }
  f << " </DataArray>\n";

  f << "</PointData>\n";

  f << " <Coordinates>\n";

  for (k = 0; k < 3; k++)
  {
    f << "   <DataArray type=\"Float64\" Name=\"" << char('X' + k) << "\""
      << " format=\"ascii\">";

    int imin = m_imin[k] - ghost;
    int imax = m_imax[k] + ghost;
    double x0 = m_xmin[k], dx = m_dx[k];
    for (i = imin; i <= imax; i++)
      f << " " << x0 + i * dx;
    f << " </DataArray>\n";
  }
  f << " </Coordinates>\n";

  f << "</Piece>\n"
    << "</RectilinearGrid>\n"
    << "</VTKFile>\n"
    << std::endl;
}

void Values::operator=(const Values &other)
{
  int i;

  for (i=0; i<3; i++) {
    m_n[i] = other.m_n[i];
    m_imin[i] = other.m_imin[i];
    m_imax[i] = other.m_imax[i];
    m_dx[i] = other.m_dx[i];
    m_xmin[i] = other.m_xmin[i];
    m_xmax[i] = other.m_xmax[i];
  }
  nn = other.nn;
  n1 = other.n1;
  n2 = other.n2;
  std::memcpy(m_u, other.m_u, sizeof(double) * nn);
}
