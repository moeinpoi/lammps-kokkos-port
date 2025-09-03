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
PairStyle(sph/rhosum/kk,PairSPHRhoSumKokkos<LMPDeviceType>);
PairStyle(sph/rhosum/kk/device,PairSPHRhoSumKokkos<LMPDeviceType>);
PairStyle(sph/rhosum/kk/host,PairSPHRhoSumKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_SPH_RHOSUM_KOKKOS_H
#define LMP_PAIR_SPH_RHOSUM_KOKKOS_H

#include "kokkos_base.h"
#include "pair_kokkos.h"
#include "pair_sph_rhosum.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {


struct TagPairSPHRhoSumCompute{};
struct TagPairSPHRhoSumPackForwardComm{};
struct TagPairSPHRhoSumUnpackForwardComm{};

template<class DeviceType>
class PairSPHRhoSumKokkos : public PairSPHRhoSum, public KokkosBase {
 public:
  enum {EnabledNeighFlags=FULL}; //Only full for rhosum
  enum {COUL_FLAG=0}; //-Changed to 0 by Moein(?)
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;
  PairSPHRhoSumKokkos(class LAMMPS *);
  ~PairSPHRhoSumKokkos() override;

  void compute(int, int) override;
  void init_style() override;
  void allocate() override;

  int pack_forward_comm_kokkos(int n, DAT::tdual_int_2d k_sendlist, int iswap_in, 
                              DAT::tdual_xfloat_1d &buf, int /*pbc_flag*/, int * /*pbc*/);

  void unpack_forward_comm_kokkos(int n, int first_in, DAT::tdual_xfloat_1d &buf);

  int pack_forward_comm(int n, int *list, double *buf,
                                                      int /*pbc_flag*/, int * /*pbc*/);

  void unpack_forward_comm(int n, int first, double *buf);                              

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairSPHRhoSumCompute, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairSPHRhoSumPackForwardComm, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairSPHRhoSumUnpackForwardComm, const int&) const;

  KOKKOS_INLINE_FUNCTION
  int sbmask(const int& j) const;

 protected:

  typename AT::t_x_array_randomread x;
  typename AT::t_float_1d rho;
  typename AT::t_float_1d_randomread mass;
  typename AT::t_int_1d_randomread type;

  
  Kokkos::DualView<F_FLOAT**, Kokkos::LayoutRight, DeviceType> k_cutsq;
  typename decltype(k_cutsq)::t_dev d_cutsq;
  Kokkos::DualView<F_FLOAT**, Kokkos::LayoutRight, DeviceType> k_cut;
  typename decltype(k_cut)::t_dev d_cut;


  int neighflag,newton_pair;
  int nlocal,nall,eflag,vflag;
  int dimension;

  int iswap, firstrecv;
  typename AT::t_int_2d d_sendlist;
  typename AT::t_xfloat_1d_um v_buf;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

};

}

#endif
#endif

