/*
 * scheme.hxx
 *
 *  Created on: 5 janv. 2016
 *      Author: tajchman
 */

#ifndef SCHEME_HXX_
#define SCHEME_HXX_

#include <vector>
#include "values.hxx"
#include "parameters.hxx"

class Scheme {

public:
  Scheme(Parameters &P);
  ~Scheme();

  double present();

  void iteration();
  double variation() { return m_duv; }

  const Values & getOutput();
  void setInput(const Values & u);

  std::string codeName;

protected:
  int m_n[3];
  double m_t, m_dt;
  double m_dx[3];
  double m_xmin[3];

  double iteration_domaine(int imin, int imax, 
                           int jmin, int jmax,
                           int kmin, int kmax);

  Values m_u, m_v;
  double m_duv;
  double *partialDiff, *diff;
  Parameters &m_P;
};

#endif /* SCHEME_HXX_ */
