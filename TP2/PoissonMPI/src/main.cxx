#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <mpi.h>

#include "parameters.hxx"
#include "values.hxx"
#include "scheme.hxx"
#include "timer.hxx"
#include "os.hxx"

int main(int argc, char *argv[])
{
  AddTimer("total");
  AddTimer("init");
  AddTimer("calcul");
  AddTimer("comm");
  AddTimer("other");

  Timer &T_total = GetTimer(0);
  Timer &T_init = GetTimer(1);
  Timer &T_calcul = GetTimer(2);
  Timer &T_comm = GetTimer(3);
  Timer &T_other = GetTimer(4);

  T_total.start();

  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Parameters Prm(argc, argv, size, rank);
  if (Prm.help())
    return 0;
  std::cerr << Prm;

  int itMax = Prm.itmax();
  int freq = Prm.freq();
  
  T_init.start();

  Scheme C(Prm);
 
  Values u_0(Prm);
  u_0.boundaries();
  u_0.init();

  C.setInput(u_0);
  T_init.stop();

  MPI_Barrier(Prm.comm());
  if (Prm.rank() == 0)
    std::cerr << "\n  temps init "  << std::setw(12) << std::setprecision(6)
              << T_init.elapsed() << " s\n" << std::endl;

  double T_previous, T_comm_previous;
  
  for (int it = 0; it < itMax; it++)
  {

    T_comm_previous = T_comm.elapsed();
    T_comm.start();
    C.synchronize();
    T_comm.stop();
    
    if (freq > 0 && it % freq == 0) {
      T_other.start();
      C.getOutput().plot(it);
      T_other.stop();
    }

    T_previous = T_calcul.elapsed();
    T_calcul.start();
    C.iteration();
    T_calcul.stop();

    if (Prm.rank() == 0)
      std::cerr << "iteration " << std::setw(5) << it
              << "  variation   " << std::setw(10) << C.variation()
              << "  temps calcul " << std::setw(10) << std::setprecision(6)
                << T_calcul.elapsed() - T_previous << " s"
                << "  comm. " << std::setw(10)
                << T_comm.elapsed() - T_comm_previous << " s"
                << std::endl;
  }

  T_comm.start();
  C.synchronize();
  T_comm.stop();
    
  if (freq > 0 && itMax % freq == 0)
  {
    T_other.start();
    C.getOutput().plot(itMax);
    T_other.stop();
  }

  MPI_Finalize();

  T_total.stop();

  if (Prm.rank() == 0)
    std::cerr << "\n"
              << std::setw(26) << "temps total  "
              << std::setw(10) << T_total.elapsed() << " s\n"
              << std::endl;

  if (Prm.rank() == 0) {
    std::string s = Prm.resultPath();
    mkdir_p(s.c_str());
    s += "/temps";
    s += "_n_";
    s += std::to_string(Prm.size());
    s += ".dat";
    std::ofstream f(s.c_str());
    f << Prm.size() 
      << " " << T_total.elapsed() 
      << " " << T_calcul.elapsed()
      << " " << std::setprecision(10) << C.variation() << std::endl;
  }

  return 0;
}
