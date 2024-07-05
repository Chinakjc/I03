
#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS
#include <direct.h>
#elif defined(__unix)
#include <unistd.h>
#endif

#include "os.hxx"
#include "arguments.hxx"
#include "parameters.hxx"
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <limits>

#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <mpi.h>

#include "timer.hxx"
#include "version.hxx"
void stime(char * buffer, int size)
{
  time_t curtime;
  struct tm *loctime;

  /* Get the current time. */
  curtime = time (NULL);

  /* Convert it to local time representation. */

  loctime = localtime (&curtime);
  strftime(buffer, size, "%F_%Hh%Mm%Ss", loctime);

}

Parameters::Parameters(int argc, char ** argv, int size, int rank)
  : Arguments(argc, argv), m_neighbour{-1, -1, -1, -1, -1, -1}
{
  m_command = argv[0];
  m_help = Get("h", 0) || Get("help", 0);
  m_path = "results";
  m_size = size;
  m_rank = rank;

  if (m_help) return;

  Timer& T_comm = GetTimer(3);
  T_comm.start();

  int dim[3] = {size, 1, 1};
  int period[3] = {0, 0, 0};
  int reorder = 0;
  int coord[3], coord2[3];

  MPI_Cart_create(MPI_COMM_WORLD, 3, dim, period, reorder, &m_comm);
  MPI_Cart_coords(m_comm, rank, 3, &(coord[0]));

  for (int i=0; i<3; i++) {
    coord2[0] = coord[0];
    coord2[1] = coord[1];
    coord2[2] = coord[2];
    if (coord[i] > 0) {
      coord2[i] = coord[i] - 1;
      MPI_Cart_rank(m_comm, &(coord2[0]), &m_neighbour[2*i]);
    }
    if (coord[i] < dim[i]-1) {
      coord2[i] = coord[i] + 1;
      MPI_Cart_rank(m_comm, &(coord2[0]), &m_neighbour[2*i+1]);
    }
  }
   
  T_comm.stop();

  m_n_global[0] = Get("n", -1);
  if (m_n_global[0] == -1) {
    m_n_global[0] = Get("n0", 401);
    m_n_global[1] = Get("n1", 401);
    m_n_global[2] = Get("n2", 401);
  }
  else {
    m_n_global[1] = m_n_global[0];
    m_n_global[2] = m_n_global[1];
  }
  m_itmax = Get("it", 10);

  double d;
  double dt_max = 0.1/(m_n_global[0]*m_n_global[0]);
  d = 0.1/(m_n_global[1]*m_n_global[1]);
  if (dt_max > d) dt_max = d;
  d = 0.1/(m_n_global[2]*m_n_global[2]);
  if (dt_max > d) dt_max = d;

  m_dt = Get("dt", dt_max);
  m_freq = Get("out", -1);

  if (rank == 0) {
    if (m_dt > dt_max)
      std::cerr << "Warning : provided dt (" << m_dt
        << ") is greater then the recommended maximum (" << dt_max
        << ")" << std::endl;
  }

  for (int i=0; i<3; i++) {
    m_dx[i] = m_n_global[i]>1 ? 1.0/(m_n_global[i]-1) : 0.0;

    int n = (m_n_global[i]-2)/dim[i];
    int nGlobal_int_min = 1 + coord[i]*n;
    int nGlobal_int_max;
    if (coord[i] < dim[i]-1) {
      nGlobal_int_max = nGlobal_int_min + n - 1;
    }
    else {
      nGlobal_int_max = m_n_global[i] - 2;
    }
    m_n[i] = nGlobal_int_max - nGlobal_int_min + 1;
    int nGlobal_ext_min = nGlobal_int_min - 1;
    int nGlobal_ext_max = nGlobal_int_max + 1;
    m_imin[i] = 1;
    m_imax[i] = nGlobal_int_max - nGlobal_int_min + 1;
    
    m_xmin[i] = m_dx[i] * nGlobal_ext_min;
    m_xmax[i] = m_dx[i] * nGlobal_ext_max;

    m_imin_global[i] = nGlobal_int_min;
    m_imax_global[i] = nGlobal_int_max;
  }
}

bool Parameters::help()
{
  if (m_rank == 0 && m_help) {
    std::cerr << "Usage : ./PoissonMPI <list of options>\n\n";
    std::cerr << "Options:\n\n"
              << "-h|--help       : display this message\n"
              << "--n <int>       : number of points in each direction (default: 401)\n"
              << "--n0 <int>      : number of points in the X direction (default: 401)\n"
              << "--n1 <int>      : number of points in the Y direction (default: 401)\n"
              << "--n2 <int>      : number of points in the Z direction (default: 401)\n"
              << "--dt <real>     : time step size (default : value to assure stable computations)\n"
              << "--it <int>      : number of time steps (default : 10)\n"
              << "--out <int>     : number of time steps between saving the solution on files\n"
              << "                 (default : no output)\n\n";
  }
  return m_help;
}

void sendString(const std::string& str, int dest, int tag, MPI_Comm comm)
{
  unsigned len = str.size();
  MPI_Send(&len, 1, MPI_UNSIGNED, dest, tag, comm);
  if (len != 0)
    MPI_Send(str.data(), len, MPI_CHAR, dest, tag, comm);
}

void recvString(std::string& str, int src, int tag, MPI_Comm comm)
{
  unsigned len;
  MPI_Status s;
  MPI_Recv(&len, 1, MPI_UNSIGNED, src, tag, comm, &s);
  if (len != 0) {
    char * tmp = new char[len+1];
    MPI_Recv(tmp, len, MPI_CHAR, src, tag, comm, &s);
    tmp[len] = '\0';
    str = tmp;
    delete [] tmp;
  }
  else
    str.clear();
}

std::ostream & operator<<(std::ostream &f, const Parameters & p)
{
  std::ostringstream s;
  s << "Process " << p.rank() << "\n";
  s << "  Internal domain :   "
    << "[" << p.xmin(0) + p.dx(0) << ", " << p.xmax(0) - p.dx(0) << "] x "
    << "[" << p.xmin(1) + p.dx(1) << ", " << p.xmax(1) - p.dx(1) << "] x "
    << "[" << p.xmin(2) + p.dx(2) << ", " << p.xmax(2) - p.dx(2) << "]\n";

  s << "  Point indices   :   "
    << "[" << p.imin_global(0) << " ... " << p.imax_global(0) << "] x "
    << "[" << p.imin_global(1) << " ... " << p.imax_global(1) << "] x "
    << "[" << p.imin_global(2) << " ... " << p.imax_global(2) << "]\n";

  std::string message = s.str();

  if (p.rank() == 0) {
    f << "Processes :" << p.size() << "\n"
      << "It. max :  " << p.itmax() << "\n"
      << "Dt :       " << p.dt() << "\n" << std::endl;
  }
  MPI_Barrier(p.comm());

  if (p.rank() == 0) {
    f << message;
    for (int i = 1; i < p.size(); i++) {
      recvString(message, i, i, p.comm());
      f << message;
    }
  }
  else {
    sendString(message, 0, p.rank(), p.comm());
  }

  return f;
}

