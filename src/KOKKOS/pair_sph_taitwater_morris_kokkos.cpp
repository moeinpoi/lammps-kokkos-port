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

#include "pair_sph_taitwater_morris_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairSPHTaitwaterMorrisKokkos<DeviceType>::PairSPHTaitwaterMorrisKokkos(LAMMPS *lmp) : PairSPHTaitwaterMorris(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  //read and modify masks needs to change for SPH fields -Moein
  datamask_read = X_MASK | F_MASK | VEST_MASK | TYPE_MASK | MASS_MASK | DESPH_MASK | RHO_MASK | DRHO_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | DESPH_MASK | DRHO_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairSPHTaitwaterMorrisKokkos<DeviceType>::~PairSPHTaitwaterMorrisKokkos()
{
  if (copymode) return;
  
  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->destroy_kokkos(k_cutsq, cutsq);
    memoryKK->destroy_kokkos(k_rho0, rho0);
    memoryKK->destroy_kokkos(k_soundspeed, soundspeed);
    memoryKK->destroy_kokkos(k_B, B);
    memoryKK->destroy_kokkos(k_cut, cut);
    memoryKK->destroy_kokkos(k_viscosity, viscosity);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairSPHTaitwaterMorrisKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
	fprintf(screen, "compute() called on PairSPHTaitwaterMorrisKokkos\n");
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space,datamask_read);
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK | DESPH_MASK | DRHO_MASK);

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  v = atomKK->k_vest.view<DeviceType>();
  rho = atomKK->k_rho.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  desph = atomKK->k_desph.view<DeviceType>();
  drho = atomKK->k_drho.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();

  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;
  dimension = domain->dimension;

  k_cutsq.template sync<DeviceType>();
  k_cut.template sync<DeviceType>();
  k_viscosity.template sync<DeviceType>();
  k_rho0.template sync<DeviceType>();
  k_B.template sync<DeviceType>();
  k_soundspeed.template sync<DeviceType>(); 

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;

  int inum = list->inum;

  copymode = 1;

  // loop over neighbors of my atoms

  EV_FLOAT ev;

  // compute kernel A

  if (evflag) {
    if (neighflag == HALF) {
      if (newton_pair) {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairSPHTaitwaterMorrisKernelA<HALF,1,1> >(0,inum),*this,ev);
      } else {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairSPHTaitwaterMorrisKernelA<HALF,0,1> >(0,inum),*this,ev);
      }
    } else if (neighflag == HALFTHREAD) {
      if (newton_pair) {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairSPHTaitwaterMorrisKernelA<HALFTHREAD,1,1> >(0,inum),*this,ev);
      } else {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairSPHTaitwaterMorrisKernelA<HALFTHREAD,0,1> >(0,inum),*this,ev);
      }
    } else if (neighflag == FULL) {
      if (newton_pair) {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairSPHTaitwaterMorrisKernelA<FULL,1,1> >(0,inum),*this,ev);
      } else {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairSPHTaitwaterMorrisKernelA<FULL,0,1> >(0,inum),*this,ev);
      }
    }
  } else {
    if (neighflag == HALF) {
      if (newton_pair) {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairSPHTaitwaterMorrisKernelA<HALF,1,0> >(0,inum),*this);
      } else {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairSPHTaitwaterMorrisKernelA<HALF,0,0> >(0,inum),*this);
      }
    } else if (neighflag == HALFTHREAD) {
      if (newton_pair) {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairSPHTaitwaterMorrisKernelA<HALFTHREAD,1,0> >(0,inum),*this);
      } else {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairSPHTaitwaterMorrisKernelA<HALFTHREAD,0,0> >(0,inum),*this);
      }
    } else if (neighflag == FULL) {
      if (newton_pair) {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairSPHTaitwaterMorrisKernelA<FULL,1,0> >(0,inum),*this);
      } else {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairSPHTaitwaterMorrisKernelA<FULL,0,0> >(0,inum),*this);
      }
    }
  }

  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  copymode = 0;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairSPHTaitwaterMorrisKokkos<DeviceType>::init_style()
{
  PairSPHTaitwaterMorris::init_style();

  // adjust neighbor list request for KOKKOS

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
                           !std::is_same<DeviceType,LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);
  if (neighflag == FULL) request->enable_full();
}

/* ---------------------------------------------------------------------- */

////Specialisation for Neighborlist types Half, HalfThread, Full
template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairSPHTaitwaterMorrisKokkos<DeviceType>::operator()(TagPairSPHTaitwaterMorrisKernelA<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int &ii, EV_FLOAT& ev) const {

  // The f array is atomic for Half/Thread neighbor style
  Kokkos::View<F_FLOAT*[3], typename DAT::t_f_array::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > a_f = f;
  Kokkos::View<F_FLOAT*, typename AT::t_float_1d::array_layout, typename KKDevice<DeviceType>::value, Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value>> a_drho = drho;
  Kokkos::View<F_FLOAT*, typename AT::t_float_1d::array_layout, typename KKDevice<DeviceType>::value, Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value>> a_desph = desph;

  const int i = d_ilist(ii);
  const X_FLOAT xtmp = x(i,0);
  const X_FLOAT ytmp = x(i,1);
  const X_FLOAT ztmp = x(i,2);
  const V_FLOAT vxtmp = v(i,0);
  const V_FLOAT vytmp = v(i,1);
  const V_FLOAT vztmp = v(i,2);
  const int itype = type(i);
  const F_FLOAT imass = mass(itype);

  // compute pressure of atom i with Tait EOS
  F_FLOAT tmp = rho(i) / d_rho0(itype);
  F_FLOAT fi = tmp * tmp * tmp;
  fi = d_B(itype) * (fi * fi * tmp - 1.0) / (d_rho(i) * d_rho(i));


  //const AtomNeighborsConst d_neighbors_i = k_list.get_neighbors_const(i);
  const int jnum = d_numneigh(i);

  F_FLOAT fxtmp = 0.0;
  F_FLOAT fytmp = 0.0;
  F_FLOAT fztmp = 0.0;
  F_FLOAT drhotmp = 0.0;
  F_FLOAT desphtmp = 0.0;

  for (int jj = 0; jj < jnum; jj++) {
    //int j = d_neighbors_i(jj);
    int j = d_neighbors(i,jj);
    j &= NEIGHMASK;

    const X_FLOAT delx = xtmp - x(j,0);
    const X_FLOAT dely = ytmp - x(j,1);
    const X_FLOAT delz = ztmp - x(j,2);
    const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;
    const int jtype = type(j);
    const F_FLOAT jmass = mass(jtype);


    if (rsq < d_cutsq(itype,jtype)) {
      const F_FLOAT h = d_cut(itype,jtype);
      const F_FLOAT ih = 1.0 / h;
      const F_FLOAT ihsq = ih * ih;

      F_FLOAT wfd = h - sqrt(rsq);
      if (dimension == 3) {
        // Lucy Kernel, 3d
        // Note that wfd, the derivative of the weight function with respect to r,
        // The missing factor of r is recovered by
        // (1) using delV . delX instead of delV . (delX/r) and
        // (2) using f[i][0] += delx * fpair instead of f[i][0] += (delx/r) * fpair
        wfd = -25.066903536973515383e0 * wfd * wfd * ihsq * ihsq * ihsq * ih;
        // is lacking a factor of r.
      } else {
        wfd = -19.098593171027440292e0 * wfd * wfd * ihsq * ihsq * ihsq;
        // Lucy Kernel, 2d
      }

      // compute pressure of atom j with Tait EOS
      F_FLOAT tmp = rho(j) / d_rho0(jtype);
      F_FLOAT fj = tmp * tmp * tmp;
      fj = d_B(jtype) * (fj * fj * tmp - 1.0) / (d_rho(j) * d_rho(j));
      
      V_FLOAT velx = vxtmp - v(j,0);
      V_FLOAT vely = vytmp - v(j,1);
      V_FLOAT velz = vztmp - v(j,2);

      // dot product of velocity delta and distance vector
      F_FLOAT delVdotDelR = delx * velx + dely * vely + delz * velz;

      // Morris Viscosity (Morris, 1996)

      F_FLOAT fvisc = 2 * d_viscosity(itype,jtype) / (d_rho(i) * d_rho(j));


      fvisc *= imass * jmass * wfd;

      // total pair force & thermal energy increment
      F_FLOAT fpair = -imass * jmass * (fi + fj) * wfd;
      F_FLOAT deltaE = -0.5 *(fpair * delVdotDelR + fvisc * (velx*velx + vely*vely + velz*velz));

      fxtmp += delx * fpair + velx * fvisc;
      fytmp += dely * fpair + vely * fvisc;
      fztmp += delz * fpair + velz * fvisc;

      // and change in density
      drhotmp += jmass * delVdotDelR * wfd;

      // change in thermal energy
      desphtmp += deltaE;

      if ((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || j < nlocal)) {
        a_f(j,0) -= delx * fpair + velx * fvisc;
        a_f(j,1) -= dely * fpair + vely * fvisc;
        a_f(j,2) -= delz * fpair + velz * fvisc;
        a_drho(j) += imass * delVdotDelR * wfd;
        a_desph(j) += deltaE;
      }


    if (EVFLAG) {
    // viscous forces do not contribute to virial
    if (vflag_either || eflag_atom)
      this->template ev_tally<NEIGHFLAG,NEWTON_PAIR>(ev,i,j,F_FLOAT(0.0),fpair,delx,dely,delz);
    }


    }
  }

  a_f(i,0) += fxtmp;
  a_f(i,1) += fytmp;
  a_f(i,2) += fztmp;
  a_drho(i) += drhotmp;
  a_desph(i) += desphtmp;
}

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairSPHTaitwaterMorrisKokkos<DeviceType>::operator()(TagPairSPHTaitwaterMorrisKernelA<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int &ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,NEWTON_PAIR,EVFLAG>(TagPairSPHTaitwaterMorrisKernelA<NEIGHFLAG,NEWTON_PAIR,EVFLAG>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR>
KOKKOS_INLINE_FUNCTION
void PairSPHTaitwaterMorrisKokkos<DeviceType>::ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &epair, const F_FLOAT &fpair, const F_FLOAT &delx,
                const F_FLOAT &dely, const F_FLOAT &delz) const
{
  const int EFLAG = eflag;
  const int VFLAG = vflag_either;

  // The eatom and vatom arrays are atomic for Half/Thread neighbor style
  Kokkos::View<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > v_eatom = k_eatom.view<DeviceType>();
  Kokkos::View<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > v_vatom = k_vatom.view<DeviceType>();

  if (EFLAG) {
    if (eflag_atom) {
      const E_FLOAT epairhalf = 0.5 * epair;
      if (NEIGHFLAG!=FULL) {
        if (NEWTON_PAIR || i < nlocal) v_eatom[i] += epairhalf;
        if (NEWTON_PAIR || j < nlocal) v_eatom[j] += epairhalf;
      } else {
        v_eatom[i] += epairhalf;
      }
    }
  }

  if (VFLAG) {
    const E_FLOAT v0 = delx*delx*fpair;
    const E_FLOAT v1 = dely*dely*fpair;
    const E_FLOAT v2 = delz*delz*fpair;
    const E_FLOAT v3 = delx*dely*fpair;
    const E_FLOAT v4 = delx*delz*fpair;
    const E_FLOAT v5 = dely*delz*fpair;

    if (vflag_global) {
      if (NEIGHFLAG!=FULL) {
        if (NEWTON_PAIR || i < nlocal) {
          ev.v[0] += 0.5*v0;
          ev.v[1] += 0.5*v1;
          ev.v[2] += 0.5*v2;
          ev.v[3] += 0.5*v3;
          ev.v[4] += 0.5*v4;
          ev.v[5] += 0.5*v5;
        }
        if (NEWTON_PAIR || j < nlocal) {
        ev.v[0] += 0.5*v0;
        ev.v[1] += 0.5*v1;
        ev.v[2] += 0.5*v2;
        ev.v[3] += 0.5*v3;
        ev.v[4] += 0.5*v4;
        ev.v[5] += 0.5*v5;
        }
      } else {
        ev.v[0] += 0.5*v0;
        ev.v[1] += 0.5*v1;
        ev.v[2] += 0.5*v2;
        ev.v[3] += 0.5*v3;
        ev.v[4] += 0.5*v4;
        ev.v[5] += 0.5*v5;
      }
    }

    if (vflag_atom) {
      if (NEIGHFLAG!=FULL) {
        if (NEWTON_PAIR || i < nlocal) {
          v_vatom(i,0) += 0.5*v0;
          v_vatom(i,1) += 0.5*v1;
          v_vatom(i,2) += 0.5*v2;
          v_vatom(i,3) += 0.5*v3;
          v_vatom(i,4) += 0.5*v4;
          v_vatom(i,5) += 0.5*v5;
        }
        if (NEWTON_PAIR || j < nlocal) {
        v_vatom(j,0) += 0.5*v0;
        v_vatom(j,1) += 0.5*v1;
        v_vatom(j,2) += 0.5*v2;
        v_vatom(j,3) += 0.5*v3;
        v_vatom(j,4) += 0.5*v4;
        v_vatom(j,5) += 0.5*v5;
        }
      } else {
        v_vatom(i,0) += 0.5*v0;
        v_vatom(i,1) += 0.5*v1;
        v_vatom(i,2) += 0.5*v2;
        v_vatom(i,3) += 0.5*v3;
        v_vatom(i,4) += 0.5*v4;
        v_vatom(i,5) += 0.5*v5;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairSPHTaitwaterMorrisKokkos<DeviceType>::allocate() {

  PairSPHTaitwaterMorris::allocate();

  int n = atom->ntypes;
  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();
  memory->destroy(rho0);
  memoryKK->create_kokkos(k_rho0,rho0,n+1,"pair:rho0");
  d_rho0 = k_rho0.template view<DeviceType>();
  memory->destroy(soundspeed);
  memoryKK->create_kokkos(k_soundspeed,soundspeed,n+1,"pair:soundspeed");
  d_soundspeed = k_soundspeed.template view<DeviceType>();
  memory->destroy(B);
  memoryKK->create_kokkos(k_B,B,n+1,"pair:B");
  d_B = k_B.template view<DeviceType>();
  memory->destroy(cut);
  memoryKK->create_kokkos(k_cut,cut,n+1,n+1,"pair:cut");
  d_cut = k_cut.template view<DeviceType>();
  memory->destroy(viscosity);
  memoryKK->create_kokkos(k_viscosity,viscosity,n+1,n+1,"pair:viscosity");
  d_viscosity = k_viscosity.template view<DeviceType>();

}; 

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int PairSPHTaitwaterMorrisKokkos<DeviceType>::sbmask(const int& j) const {
  return j >> SBBITS & 3;
}

namespace LAMMPS_NS {
template class PairSPHTaitwaterMorrisKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairSPHTaitwaterMorrisKokkos<LMPHostType>;
#endif
}

