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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(sph/taitwater/morris/kk,PairSPHTaitwaterMorrisKokkos<LMPDeviceType>);
PairStyle(sph/taitwater/morris/kk/device,PairSPHTaitwaterMorrisKokkos<LMPDeviceType>);
PairStyle(sph/taitwater/morris/kk/host,PairSPHTaitwaterMorrisKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_SPH_TAITWATER_MORRIS_KOKKOS_H
#define LMP_PAIR_SPH_TAITWATER_MORRIS_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_sph_taitwater_morris.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairSPHTaitwaterMorrisKernelA{};

template<class DeviceType>
class PairSPHTaitwaterMorrisKokkos : public PairSPHTaitwaterMorris {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF}; 
  enum {COUL_FLAG=0}; //-Changed to 0 by Moein(?)
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;
  PairSPHTaitwaterMorrisKokkos(class LAMMPS *);
  ~PairSPHTaitwaterMorrisKokkos() override;

  void compute(int, int) override;
  void init_style() override;
  void allocate() override;

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairSPHTaitwaterMorrisKernelA<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairSPHTaitwaterMorrisKernelA<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int&) const;

  template<int NEIGHFLAG, int NEWTON_PAIR>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &epair, const F_FLOAT &fpair, const F_FLOAT &delx,
                  const F_FLOAT &dely, const F_FLOAT &delz) const;

  KOKKOS_INLINE_FUNCTION
  int sbmask(const int& j) const;

 protected:

  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_v_array v;
  typename AT::t_float_1d_randomread rho;
  typename AT::t_float_1d drho;
  typename AT::t_float_1d desph;
  typename AT::t_float_1d_randomread mass;
  typename AT::t_int_1d_randomread type;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename ArrayTypes<DeviceType>::t_efloat_1d d_eatom;
  typename ArrayTypes<DeviceType>::t_virial_array d_vatom;
  
  Kokkos::DualView<F_FLOAT**, Kokkos::LayoutRight, DeviceType> k_cutsq;
  typename decltype(k_cutsq)::t_dev d_cutsq;
  Kokkos::DualView<F_FLOAT*, Kokkos::LayoutRight, DeviceType> k_rho0;
  typename decltype(k_rho0)::t_dev d_rho0;
  Kokkos::DualView<F_FLOAT*, Kokkos::LayoutRight, DeviceType> k_soundspeed;
  typename decltype(k_soundspeed)::t_dev d_soundspeed;
  Kokkos::DualView<F_FLOAT*, Kokkos::LayoutRight, DeviceType> k_B;
  typename decltype(k_B)::t_dev d_B;
  Kokkos::DualView<F_FLOAT**, Kokkos::LayoutRight, DeviceType> k_cut;
  typename decltype(k_cut)::t_dev d_cut;
  Kokkos::DualView<F_FLOAT**, Kokkos::LayoutRight, DeviceType> k_viscosity;
  typename decltype(k_viscosity)::t_dev d_viscosity;


  int neighflag,newton_pair;
  int nlocal,nall,eflag,vflag;
  int dimension;


  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;
  //NeighListKokkos<DeviceType> k_list;

  friend void pair_virial_fdotr_compute<PairSPHTaitwaterMorrisKokkos>(PairSPHTaitwaterMorrisKokkos*);

};

}

#endif
#endif

