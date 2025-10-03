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

#ifdef ANGLE_CLASS

AngleStyle(area/volume3,AngleAreaVolume3)

#else

#ifndef ANGLE_AREA_VOLUME3_H
#define ANGLE_AREA_VOLUME3_H

#include "stdio.h"
#include "angle.h"

namespace LAMMPS_NS {

class AngleAreaVolume3 : public Angle {
 public:
  AngleAreaVolume3(class LAMMPS *);
  virtual ~AngleAreaVolume3();
  virtual void compute(int, int);
  void coeff(int, char **);
  double equilibrium_angle(int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_data(FILE *);
  double single(int, int, int, int);
  void init_style() {};

 protected:
  double *press, *ka, *a0, *kv, *v0, *kl, *aa0, *r0, *qp, *vratio;
  int *nstep1,*nstep2;
  int *ttyp,*ttyp1;
  int init_on;
  double *h_area,*h_volume,*dath,*datt;
  int nm;

  void ev_tally3(int, int, int, int, int, double, double *, double *, double *, double *);
  void allocate();
};

}

#endif
#endif

