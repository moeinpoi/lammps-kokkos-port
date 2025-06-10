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
FixStyle(sph/kk,FixSPHKokkos<LMPDeviceType>);
FixStyle(sph/kk/device,FixSPHKokkos<LMPDeviceType>);
FixStyle(sph/kk/host,FixSPHKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_SPH_KOKKOS_H
#define LMP_FIX_SPH_KOKKOS_H

#include "fix_sph.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<class DeviceType>
class FixSPHKokkos;

template <class DeviceType>
struct FixSPHKokkosPreForceFunctor;

template <class DeviceType>
struct FixSPHKokkosSetupPreForceFunctor;

template <class DeviceType, int RMass>
struct FixSPHKokkosInitialIntegrateFunctor;

template <class DeviceType, int RMass>
struct FixSPHKokkosFinalIntegrateFunctor;

template<class DeviceType>
class FixSPHKokkos : public FixSPH {
 public:
  FixSPHKokkos(class LAMMPS *, int, char **);

  void cleanup_copy();
  void init() override;
  void pre_force(int) override;
  void setup_pre_force(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;

  KOKKOS_INLINE_FUNCTION
  void pre_force_item(int) const;
  KOKKOS_INLINE_FUNCTION
  void setup_pre_force_item(int) const;
  KOKKOS_INLINE_FUNCTION
  void initial_integrate_item(int) const;
  KOKKOS_INLINE_FUNCTION
  void initial_integrate_rmass_item(int) const;
  KOKKOS_INLINE_FUNCTION
  void final_integrate_item(int) const;
  KOKKOS_INLINE_FUNCTION
  void final_integrate_rmass_item(int) const;

 private:


  typename ArrayTypes<DeviceType>::t_x_array x;
  typename ArrayTypes<DeviceType>::t_v_array v;
  typename ArrayTypes<DeviceType>::t_v_array vest;
  typename ArrayTypes<DeviceType>::t_float_1d rho;
  typename ArrayTypes<DeviceType>::t_float_1d drho;
  typename ArrayTypes<DeviceType>::t_float_1d esph;
  typename ArrayTypes<DeviceType>::t_float_1d desph;
  typename ArrayTypes<DeviceType>::t_f_array_const f;
  typename ArrayTypes<DeviceType>::t_float_1d rmass;
  typename ArrayTypes<DeviceType>::t_float_1d mass;
  typename ArrayTypes<DeviceType>::t_int_1d type;
  typename ArrayTypes<DeviceType>::t_int_1d mask;
};

template <class DeviceType>
struct FixSPHKokkosPreForceFunctor  {
  typedef DeviceType  device_type ;
  FixSPHKokkos<DeviceType> c;

  FixSPHKokkosPreForceFunctor(FixSPHKokkos<DeviceType>* c_ptr):
  c(*c_ptr) {c.cleanup_copy();}; 

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
      c.pre_force_item(i); 
  }
};

template <class DeviceType>
struct FixSPHKokkosSetupPreForceFunctor {
  typedef DeviceType device_type;
  FixSPHKokkos<DeviceType> c;

  FixSPHKokkosSetupPreForceFunctor(FixSPHKokkos<DeviceType>* c_ptr):
  c(*c_ptr) {c.cleanup_copy();};

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    c.setup_pre_force_item(i);
  }
};


template <class DeviceType, int RMass>
struct FixSPHKokkosInitialIntegrateFunctor  {
  typedef DeviceType  device_type ;
  FixSPHKokkos<DeviceType> c;

  FixSPHKokkosInitialIntegrateFunctor(FixSPHKokkos<DeviceType>* c_ptr):
  c(*c_ptr) {c.cleanup_copy();};
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    if (RMass) c.initial_integrate_rmass_item(i);
    else c.initial_integrate_item(i);
  }
};

template <class DeviceType, int RMass>
struct FixSPHKokkosFinalIntegrateFunctor  {
  typedef DeviceType  device_type ;
  FixSPHKokkos<DeviceType> c;

  FixSPHKokkosFinalIntegrateFunctor(FixSPHKokkos<DeviceType>* c_ptr):
  c(*c_ptr) {c.cleanup_copy();};
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    if (RMass) c.final_integrate_rmass_item(i);
    else c.final_integrate_item(i);
  }
};


}

#endif
#endif

