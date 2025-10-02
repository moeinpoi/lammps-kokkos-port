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

#ifdef BOND_CLASS
// clang-format off
BondStyle(wlc/pow/all/visc/kk,BondWLCPowAllViscKokkos<LMPDeviceType>);
BondStyle(wlc/pow/all/visc/kk/device,BondWLCPowAllViscKokkos<LMPDeviceType>);
BondStyle(wlc/pow/all/visc/kk/host,BondWLCPowAllViscKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_BOND_WLC_POW_ALL_VISC_KOKKOS_H
#define LMP_BOND_WLC_POW_ALL_VISC_KOKKOS_H

#include "bond_wlc_pow_all_visc.h"
#include "Kokkos_Random.hpp"
#include "kokkos_type.h"

namespace LAMMPS_NS {

using VEC4 = Kokkos::Array<F_FLOAT,4>;

template<int NEWTON_BOND, int EVFLAG>
struct TagBondWLCPowAllViscCompute{};

template<class DeviceType>
class BondWLCPowAllViscKokkos : public BondWLCPowAllVisc {

 public:
  typedef DeviceType device_type;
  typedef EV_FLOAT value_type;

  BondWLCPowAllViscKokkos(class LAMMPS *);
  ~BondWLCPowAllViscKokkos() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void read_restart(FILE *) override;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagBondWLCPowAllViscCompute<NEWTON_BOND,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagBondWLCPowAllViscCompute<NEWTON_BOND,EVFLAG>, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &ebond, const F_FLOAT &fbond, const F_FLOAT &delx,
                  const F_FLOAT &dely, const F_FLOAT &delz) const;

  KOKKOS_INLINE_FUNCTION
  void generate_wrr(X_FLOAT delx, X_FLOAT dely, X_FLOAT delz, VEC4 &wrr) const;


 protected:

  class NeighborKokkos *neighborKK;
  
  Kokkos::Random_XorShift64_Pool<DeviceType> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<DeviceType>::generator_type rand_type;

  typedef ArrayTypes<DeviceType> AT;
  typename AT::t_x_array_randomread x0;
  typename AT::t_x_array_randomread x;
  typename AT::t_v_array_randomread v;
  typename Kokkos::View<double*[3],typename AT::t_f_array::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic> > f;
  typename AT::t_int_2d bondlist;

  typedef typename KKDevice<DeviceType>::value KKDeviceType;
  Kokkos::DualView<E_FLOAT*,Kokkos::LayoutRight,KKDeviceType> k_eatom;
  Kokkos::DualView<F_FLOAT*[6],Kokkos::LayoutRight,KKDeviceType> k_vatom;
  Kokkos::View<E_FLOAT*,Kokkos::LayoutRight,KKDeviceType,Kokkos::MemoryTraits<Kokkos::Atomic> > d_eatom;
  Kokkos::View<F_FLOAT*[6],Kokkos::LayoutRight,KKDeviceType,Kokkos::MemoryTraits<Kokkos::Atomic> > d_vatom;

  int nlocal,newton_bond;
  int eflag,vflag;

  DAT::tdual_ffloat_1d k_temp;
  DAT::tdual_ffloat_1d k_r0;
  DAT::tdual_ffloat_1d k_mu_targ;
  DAT::tdual_ffloat_1d k_qp;
  DAT::tdual_ffloat_1d k_gamc;
  DAT::tdual_ffloat_1d k_gamt;
  DAT::tdual_ffloat_1d k_sigc;
  DAT::tdual_ffloat_1d k_sigt;

  typename AT::t_ffloat_1d d_temp;
  typename AT::t_ffloat_1d d_r0;
  typename AT::t_ffloat_1d d_mu_targ;
  typename AT::t_ffloat_1d d_qp;
  typename AT::t_ffloat_1d d_gamc;
  typename AT::t_ffloat_1d d_gamt;
  typename AT::t_ffloat_1d d_sigc;
  typename AT::t_ffloat_1d d_sigt;

  void allocate() override;
};

}

#endif
#endif

