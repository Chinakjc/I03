#ifndef __PARAMETERS__
#define __PARAMETERS__

#include <iostream>
#include <string>
#include <memory>
#include <functional>

#include "arguments.hxx"

class Parameters : public Arguments {
public:

  Parameters(int argc, char **argv);
  void info();

  double dx(int i) const { return m_dx[i]; }
  double xmin(int i) const { return m_xmin[i]; }
  double xmax(int i) const { return m_xmax[i]; }

  int imin(int i) const { return m_imin[i]; }
  int imax(int i) const { return m_imax[i]; }
  
  int n(int i) const { return m_n[i]; }
  int itmax() const { return m_itmax; }
  double dt() const { return m_dt; }

  int freq() const { return m_freq; }
  std::string resultPath() const { return m_path; }
  bool help();

#ifdef _OPENMP

  int nthreads() const { return m_nthreads; }
  void nthreads(int n) { m_nthreads = n; }

#endif
  
private:

  std::string m_command;

#ifdef _OPENMP
  int m_nthreads;
#endif

  int m_n[3];
  double m_xmin[3], m_xmax[3], m_dx[3];
  int m_imin[3], m_imax[3];
  
  int m_itmax;
  double m_dt;
  
  int m_freq;

  std::string m_path;
  bool m_help;

};

std::ostream & operator<<(std::ostream &f, const Parameters & p);


#endif
