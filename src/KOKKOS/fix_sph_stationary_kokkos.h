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

#ifdef FIX_CLASS
// clang-format off
FixStyle(sph/stationary/kk,FixSPHStationaryKokkos<LMPDeviceType>);
FixStyle(sph/stationary/kk/device,FixSPHStationaryKokkos<LMPDeviceType>);
FixStyle(sph/stationary/kk/host,FixSPHStationaryKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_SPH_STATIONARY_KOKKOS_H
#define LMP_FIX_SPH_STATIONARY_KOKKOS_H

#include "fix_sph_stationary.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template <class DeviceType>
struct FixSPHStationaryKokkosInitialIntegrateFunctor;

template <class DeviceType>
struct FixSPHStationaryKokkosFinalIntegrateFunctor;

template<class DeviceType>
class FixSPHStationaryKokkos : public FixSPHStationary {
 public:
  FixSPHStationaryKokkos(class LAMMPS *, int, char **);
  void initial_integrate(int) override;
  void final_integrate() override;
  void cleanup_copy();

  KOKKOS_INLINE_FUNCTION
  void initial_integrate_item(int) const;
  KOKKOS_INLINE_FUNCTION
  void final_integrate_item(int) const;

 private:

  typename ArrayTypes<DeviceType>::t_float_1d rho;
  typename ArrayTypes<DeviceType>::t_float_1d drho;
  typename ArrayTypes<DeviceType>::t_float_1d esph;
  typename ArrayTypes<DeviceType>::t_float_1d desph;
  typename ArrayTypes<DeviceType>::t_int_1d mask;
};


template <class DeviceType>
struct FixSPHStationaryKokkosInitialIntegrateFunctor  {
  typedef DeviceType  device_type ;
  FixSPHStationaryKokkos<DeviceType> c;

  FixSPHStationaryKokkosInitialIntegrateFunctor(FixSPHStationaryKokkos<DeviceType>* c_ptr):
  c(*c_ptr) {c.cleanup_copy();};
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    c.initial_integrate_item(i);
  }
};

template <class DeviceType>
struct FixSPHStationaryKokkosFinalIntegrateFunctor  {
  typedef DeviceType  device_type ;
  FixSPHStationaryKokkos<DeviceType> c;

  FixSPHStationaryKokkosFinalIntegrateFunctor(FixSPHStationaryKokkos<DeviceType>* c_ptr):
  c(*c_ptr) {c.cleanup_copy();};
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    c.final_integrate_item(i);
  }
};


}

#endif
#endif

