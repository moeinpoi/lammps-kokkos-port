/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef BOND_CLASS

BondStyle(wlc/pow/all/visc,BondWLCPowAllVisc)

#else

#ifndef BOND_WLC_POW_ALL_VISC_H
#define BOND_WLC_POW_ALL_VISC_H

#include "stdio.h"
#include "bond.h"

namespace LAMMPS_NS {

class BondWLCPowAllVisc : public Bond {
 public:
  BondWLCPowAllVisc(class LAMMPS *);
  virtual ~BondWLCPowAllVisc();
  virtual void compute(int, int);
  void coeff(int, char **);
  double equilibrium_distance(int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_data(FILE *);
  void init_style();
  double single(int, double, int, int, double &);


 protected:

  double *temp, *r0, *mu_targ, *qp;
  double *gamc, *gamt, *sigc, *sigt;
  double wrr[4]; 
  double delx,dely,delz;
  class RanMars *random;
  char *id_fix;
  class FixStore *fix;
  int  init_on;
  uint64_t seed_ = 0;

  virtual void allocate();
  void generate_wrr();

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Incorrect args for bond coefficients

Self-explanatory.  Check the input script or data file.

*/
