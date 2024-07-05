const char kPathSeparator =
#ifdef _WIN32
'\\';
#else
'/';
#endif

#include "values.hxx"
#include "memmove.hxx"
#include "os.hxx"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include "timer.hxx"

Values::Values(Parameters & prm) : m_p(prm)
{
  int i;

  nn= 1;
  for (i=0; i<3; i++) {
    m_imin[i] = m_p.imin(i);
    m_imax[i] = m_p.imax(i);
    m_n[i] = m_imax[i] - m_imin[i] + 3;
    m_dx[i] = m_p.dx(i);
    m_xmin[i] = m_p.xmin(i);
    m_xmax[i] = m_p.xmax(i);
    nn *= m_n[i];
  }

  n1 = m_n[0];      // nombre de points dans la premiere direction
  n2 = m_n[1] * n1; // nombre de points dans le plan des 2 premieres directions

  d_u = allocate(nn);

  h_u = new double[nn];
  h_synchronized = false;
}


Values::~Values()
{
  delete [] h_u;
  deallocate(d_u);
}

void Values::zero()
{
  ::zero(d_u, nn);
  h_synchronized = false;
}

void Values::init()
{
  ::init(d_u, m_n);
  h_synchronized = false;
}

void Values::boundaries()
{
  ::boundaries(d_u, m_n, m_imin, m_imax);
  h_synchronized = false;
}

std::ostream & operator<< (std::ostream & f, const Values & v)
{
  v.print(f);
  return f;
}

void Values::print(std::ostream & f) const
{
  int i, j, k;
  int imin = m_imin[0]-1, jmin = m_imin[1]-1, kmin = m_imin[2]-1;
  int imax = m_imax[0]+1, jmax = m_imax[1]+1, kmax = m_imax[2]+1;
    
  if (!h_synchronized) {
    copyDeviceToHost(h_u, d_u, nn);
    h_synchronized = true;
  }

  for (k=kmin; k<=kmax; k++) {
    for (j=jmin; j<=jmax; j++) {
      for (i=imin; i<=imax; i++) {
        f << " " << operator()(i,j,k);
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
  ::swap(d_u, other.d_u);
  ::swap(h_u, other.h_u);
  ::swap(h_synchronized, other.h_synchronized);
  int i;
  for (i=0; i<3; i++) {
    ::swap(m_imin[i], other.m_imin[i]);
    ::swap(m_imax[i], other.m_imax[i]);
    ::swap(m_n[i], other.m_n[i]);
    ::swap(m_dx[i], other.m_dx[i]);
    ::swap(m_xmin[i], other.m_xmin[i]);
    ::swap(m_xmax[i], other.m_xmax[i]);
  }
  ::swap(n1, other.n1);
  ::swap(n2, other.n2);
}

void Values::plot(int order) const {

  if (!h_synchronized) {
    copyDeviceToHost(h_u, d_u, nn);
    h_synchronized = true;
  }

  std::ostringstream s;
  int i, j, k;
  int imin = m_imin[0]-1, jmin = m_imin[1]-1, kmin = m_imin[2]-1;
  int imax = m_imax[0]+1, jmax = m_imax[1]+1, kmax = m_imax[2]+1;

  s << m_p.resultPath();
  mkdir_p(s.str().c_str());

  s << kPathSeparator << "plot_" << std::setw(5) << std::setfill('0') << order << ".vtr";
  std::ofstream f(s.str().c_str());

  f << "<?xml version=\"1.0\"?>\n";
  f << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    << "<RectilinearGrid WholeExtent=\""
    << imin << " " << imax  << " "
    << jmin << " " << jmax  << " "
    << kmin << " " << kmax
    << "\">\n"
    << "<Piece Extent=\""
    << imin << " " << imax  << " "
    << jmin << " " << jmax  << " "
    << kmin << " " << kmax
    << "\">\n";

  f << "<PointData Scalars=\"values\">\n";
  f << "  <DataArray type=\"Float64\" Name=\"values\" format=\"ascii\">\n";
  
  for (k=kmin; k<=kmax; k++) {
    for (j=jmin; j<=jmax; j++) {
      for (i=imin; i<=imax; i++)
        f << " " << std::setw(7) << operator()(i,j,k);
      f << "\n";
    }
    f << "\n";
  }
  f << " </DataArray>\n";
   
  f << "</PointData>\n";

  f << " <Coordinates>\n";
  
  for (k=0; k<3; k++) {
    f << "   <DataArray type=\"Float64\" Name=\"" << char('X' + k) << "\"" 
      << " format=\"ascii\">";
    
    int imin = m_p.imin(k)-1;
    int imax = m_p.imax(k)+1;
    for (i=imin; i<=imax; i++)
      f << " " << i * m_p.dx(k);
    f << "   </DataArray>\n";
  }
  f << " </Coordinates>\n";
  
  f << "</Piece>\n"
    << "</RectilinearGrid>\n"
    << "</VTKFile>\n" <<std::endl;
}

void Values::operator= (const Values &other)
{
  int i;
  
  for (i=0; i<3; i++) {
    m_imin[i] = other.m_imin[i];
    m_imax[i] = other.m_imax[i];
    m_n[i] = other.m_n[i];
    m_xmin[i] = other.m_xmin[i];
    m_xmax[i] = other.m_xmax[i];
    m_dx[i] = other.m_dx[i];
  }
  h_synchronized = other.h_synchronized;

  if (other.h_synchronized)
     memcpy(h_u, other.h_u, nn*sizeof(double));

  copyDeviceToDevice(d_u, other.d_u, nn);
}
