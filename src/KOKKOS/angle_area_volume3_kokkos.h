/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef ANGLE_CLASS
// clang-format off
AngleStyle(area/volume3/kk,AngleAreaVolume3Kokkos<LMPDeviceType>);
AngleStyle(area/volume3/kk/device,AngleAreaVolume3Kokkos<LMPDeviceType>);
AngleStyle(area/volume3/kk/host,AngleAreaVolume3Kokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_ANGLE_AREA_VOLUME3_KOKKOS_H
#define LMP_ANGLE_AREA_VOLUME3_KOKKOS_H

#include "angle_area_volume3.h"
#include "kokkos_type.h"
#include "kokkos_few.h"

namespace LAMMPS_NS {

template<int NEWTON_BOND, int EVFLAG>
struct TagAngleAreaVolume3Compute{};

template<int NEWTON_BOND>
struct TagAngleAreaVolume3LocalLoop{};

template<class DeviceType>
class AngleAreaVolume3Kokkos : public AngleAreaVolume3 {

 public:
  typedef DeviceType device_type;
  typedef EV_FLOAT value_type;

  AngleAreaVolume3Kokkos(class LAMMPS *);
  ~AngleAreaVolume3Kokkos() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void read_restart(FILE *) override;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagAngleAreaVolume3Compute<NEWTON_BOND,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagAngleAreaVolume3Compute<NEWTON_BOND,EVFLAG>, const int&) const;

  template<int NEWTON_BOND>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagAngleAreaVolume3LocalLoop<NEWTON_BOND>, const int&) const;

  //template<int NEWTON_BOND>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(EV_FLOAT &ev, const int i, const int j, const int k,
                     F_FLOAT &eangle, F_FLOAT *f1, F_FLOAT *f3,
                     const F_FLOAT &delx1, const F_FLOAT &dely1, const F_FLOAT &delz1,
                     const F_FLOAT &delx2, const F_FLOAT &dely2, const F_FLOAT &delz2) const;

 protected:

  class NeighborKokkos *neighborKK;

  typename ArrayTypes<DeviceType>::t_x_array_randomread x0;
  typename ArrayTypes<DeviceType>::t_x_array_randomread x;
  typename ArrayTypes<DeviceType>::t_f_array f;
  typename ArrayTypes<DeviceType>::t_tagint_1d tag;
  typename ArrayTypes<DeviceType>::t_tagint_1d molecule;
  typename ArrayTypes<DeviceType>::t_imageint_1d image;
  typename ArrayTypes<DeviceType>::t_int_2d anglelist;

  typename ArrayTypes<DeviceType>::tdual_efloat_1d k_eatom;
  typename ArrayTypes<DeviceType>::tdual_virial_array k_vatom;
  typename ArrayTypes<DeviceType>::t_efloat_1d d_eatom;
  typename ArrayTypes<DeviceType>::t_virial_array d_vatom;

  int nlocal,newton_bond;
  int eflag,vflag;

  Few<double,3> prd;
  Few<double,6> h;
  int triclinic;
  
  typename ArrayTypes<DeviceType>::tdual_ffloat_1d k_press, k_ka, k_a0, k_kv, k_v0, k_kl, k_aa0, k_r0, k_qp, k_vratio;
  typename ArrayTypes<DeviceType>::t_ffloat_1d d_press, d_ka, d_a0, d_kv, d_v0, d_kl, d_aa0, d_r0, d_qp, d_vratio;

  typename ArrayTypes<DeviceType>::tdual_int_1d k_nstep1, k_nstep2, k_ttyp, k_ttyp1;
  typename ArrayTypes<DeviceType>::t_int_1d d_nstep1, d_nstep2, d_ttyp, d_ttyp1;

  typename ArrayTypes<DeviceType>::tdual_ffloat_1d k_h_area, k_h_volume, k_dath, k_datt;
  typename ArrayTypes<DeviceType>::t_ffloat_1d d_h_area, d_h_volume, d_dath, d_datt;

  void allocate() override;
};

}

#endif
#endif

