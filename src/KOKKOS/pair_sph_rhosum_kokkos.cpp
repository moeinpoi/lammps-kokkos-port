// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Moein Naderi (UIC), Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "pair_sph_rhosum_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"


#include <cmath>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairSPHRhoSumKokkos<DeviceType>::PairSPHRhoSumKokkos(LAMMPS *lmp) : PairSPHRhoSum(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | TYPE_MASK | RHO_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairSPHRhoSumKokkos<DeviceType>::~PairSPHRhoSumKokkos()
{
  if (copymode) return;
  
  if (allocated) {
    memoryKK->destroy_kokkos(k_cutsq, cutsq);
    memoryKK->destroy_kokkos(k_cut, cut);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairSPHRhoSumKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
	fprintf(screen, "compute() called on PairSPHRhoSumKokkos\n");
  //ev_init(eflag,vflag,0);  //safe, but is it required? -Moein

  atomKK->sync(execution_space,datamask_read);

  x = atomKK->k_x.view<DeviceType>();
  rho = atomKK->k_rho.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();

  // nlocal = atom->nlocal;
  // nall = atom->nlocal + atom->nghost;
  // newton_pair = force->newton_pair;
  dimension = domain->dimension;

  k_cutsq.template sync<DeviceType>();
  k_cut.template sync<DeviceType>();
  d_cutsq = k_cutsq.template view<DeviceType>();
  d_cut   = k_cut.template view<DeviceType>();

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  int inum = list->inum;

  // recompute density
  // we use a full neighborlist here
  if (nstep != 0) {
    if ((update->ntimestep % nstep) == 0) {
      copymode = 1;
      // loop over neighbors of my atoms
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairSPHRhoSumCompute>(0,inum), *this);
      copymode = 0;
      atomKK->modified(execution_space,RHO_MASK);
    }
  }

  // communicate densities
  comm->forward_comm(this);
  
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairSPHRhoSumKokkos<DeviceType>::init_style()
{
  PairSPHRhoSum::init_style();

  // adjust neighbor list request for KOKKOS

  neighflag = FULL; // need a full neighbor list for pair rhosum
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
                           !std::is_same<DeviceType,LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);
  request->enable_full(); 
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairSPHRhoSumKokkos<DeviceType>::operator()(TagPairSPHRhoSumCompute, const int &ii) const {

  const int i = d_ilist(ii);
  const X_FLOAT xtmp = x(i,0);
  const X_FLOAT ytmp = x(i,1);
  const X_FLOAT ztmp = x(i,2);
  const int itype = type(i);
  const F_FLOAT imass = mass(itype);
  F_FLOAT h = d_cut(itype, itype);
  F_FLOAT wf;

  // initialize density with self-contribution,
  if (dimension == 3) {
    /*
    // Lucy kernel, 3d
    wf = 2.0889086280811262819e0 / (h * h * h);
    */

    // quadric kernel, 3d
    wf = 2.1541870227086614782 / (h * h * h);
  } else {
      /*
      // Lucy kernel, 2d
      wf = 1.5915494309189533576e0 / (h * h);
      */

      // quadric kernel, 2d
      wf = 1.5915494309189533576e0 / (h * h);
  }
  rho(i) = imass * wf;


  // add density at each atom via kernel function overlap
  const int jnum = d_numneigh(i);

  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors(i,jj);
    j &= NEIGHMASK;
    const int jtype = type(j);
    const X_FLOAT delx = xtmp - x(j,0);
    const X_FLOAT dely = ytmp - x(j,1);
    const X_FLOAT delz = ztmp - x(j,2);
    const F_FLOAT rsq = delx * delx + dely * dely + delz * delz;

    if (rsq < d_cutsq(i, j)) {
      h = d_cut(itype, jtype);
      const F_FLOAT ih = 1.0 / h;
      const F_FLOAT ihsq = ih * ih;

      if (dimension == 3) {
        /*
        // Lucy kernel, 3d
        r = sqrt(rsq);
        wf = (h - r) * ihsq;
        wf =  2.0889086280811262819e0 * (h + 3. * r) * wf * wf * wf * ih;
        */

        // quadric kernel, 3d
        wf = 1.0 - rsq * ihsq;
        wf = wf * wf;
        wf = wf * wf;
        wf = 2.1541870227086614782e0 * wf * ihsq * ih;
      } else {
          // Lucy kernel, 2d
          //r = sqrt(rsq);
          //wf = (h - r) * ihsq;
          //wf = 1.5915494309189533576e0 * (h + 3. * r) * wf * wf * wf;

          // quadric kernel, 2d
          wf = 1.0 - rsq * ihsq;
          wf = wf * wf;
          wf = wf * wf;
          wf = 1.5915494309189533576e0 * wf * ihsq;
      }

      rho(i) += mass(jtype) * wf;
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairSPHRhoSumKokkos<DeviceType>::pack_forward_comm_kokkos(int n, DAT::tdual_int_2d k_sendlist,
                                                              int iswap_in, DAT::tdual_xfloat_1d &buf,                                                         
                                                              int /*pbc_flag*/, int * /*pbc*/) {
  
  d_sendlist = k_sendlist.view<DeviceType>();
  iswap = iswap_in;
  v_buf = buf.view<DeviceType>();
  rho = atomKK->k_rho.view<DeviceType>();
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairSPHRhoSumPackForwardComm>(0,n), *this);
  return n;

}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairSPHRhoSumKokkos<DeviceType>::operator()(TagPairSPHRhoSumPackForwardComm, const int &i) const {
  const int j = d_sendlist(iswap, i);
  v_buf(i) = rho(j);

} 

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairSPHRhoSumKokkos<DeviceType>::unpack_forward_comm_kokkos(int n, int first_in, 
                                                                DAT::tdual_xfloat_1d &buf) {
  
  firstrecv = first_in;
  v_buf = buf.view<DeviceType>();
  rho = atomKK->k_rho.view<DeviceType>();
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairSPHRhoSumUnpackForwardComm>(0, n), *this);
  atomKK->modified(execution_space, RHO_MASK);

}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairSPHRhoSumKokkos<DeviceType>::operator()(TagPairSPHRhoSumUnpackForwardComm, const int &i) const {
  rho(i + firstrecv) = v_buf(i);

} 

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairSPHRhoSumKokkos<DeviceType>::pack_forward_comm(int n, int *list, double *buf,
                                                      int /*pbc_flag*/, int * /*pbc*/)
{
  atomKK->k_rho.sync_host();

  int i,j;

  const auto h_rho = atomKK->k_rho.h_view;

  for (i = 0; i < n; i++) {
    j = list[i];
    buf[i] = h_rho[j];
  }
  return n;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairSPHRhoSumKokkos<DeviceType>::unpack_forward_comm(int n, int first, double *buf)
{
  atomKK->k_rho.sync_host();

  auto h_rho = atomKK->k_rho.h_view;
  for (int i = 0; i < n; i++) {
    h_rho[i + first] = buf[i];
  }

  atomKK->k_rho.modify_host();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairSPHRhoSumKokkos<DeviceType>::allocate() {

  PairSPHRhoSum::allocate();

  int n = atom->ntypes;
  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();
  memory->destroy(cut);
  memoryKK->create_kokkos(k_cut,cut,n+1,n+1,"pair:cut");
  d_cut = k_cut.template view<DeviceType>();

}; 

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int PairSPHRhoSumKokkos<DeviceType>::sbmask(const int& j) const {
  return j >> SBBITS & 3;
}

namespace LAMMPS_NS {
template class PairSPHRhoSumKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairSPHRhoSumKokkos<LMPHostType>;
#endif
}

