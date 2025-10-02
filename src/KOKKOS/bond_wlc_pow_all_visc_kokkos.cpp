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
   Contributing author: Stan Moore (SNL), Moein Naderi (UIC), Zhangli Peng (UIC)
------------------------------------------------------------------------- */

#include "bond_wlc_pow_all_visc_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "force.h"
#include "memory_kokkos.h"
#include "neighbor_kokkos.h"

#include <cmath>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
BondWLCPowAllViscKokkos<DeviceType>::BondWLCPowAllViscKokkos(LAMMPS *lmp) : BondWLCPowAllVisc(lmp), rand_pool(seed_ + comm->me)
{
  atomKK = (AtomKokkos *) atom;
  neighborKK = (NeighborKokkos *) neighbor;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | X0_MASK | V_MASK | F_MASK | ENERGY_MASK | VIRIAL_MASK; 
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
BondWLCPowAllViscKokkos<DeviceType>::~BondWLCPowAllViscKokkos()
{
  if (!copymode) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void BondWLCPowAllViscKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{

	fprintf(screen, "compute() called on BondWLCPowAllVisc kokkos\n");
  eflag = eflag_in;
  vflag = vflag_in;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    //if(k_eatom.extent(0)<maxeatom) { // won't work without adding zero functor
      memoryKK->destroy_kokkos(k_eatom,eatom);
      memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"bond:eatom");
      d_eatom = k_eatom.template view<KKDeviceType>();
    //}
  }
  if (vflag_atom) {
    //if(k_vatom.extent(0)<maxvatom) { // won't work without adding zero functor
      memoryKK->destroy_kokkos(k_vatom,vatom);
      memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"bond:vatom");
      d_vatom = k_vatom.template view<KKDeviceType>();
    //}
  }

  k_temp.template sync<DeviceType>();
  k_r0.template sync<DeviceType>();
  k_mu_targ.template sync<DeviceType>();
  k_qp.template sync<DeviceType>();
  k_gamc.template sync<DeviceType>();
  k_gamt.template sync<DeviceType>();
  k_sigc.template sync<DeviceType>();
  k_sigt.template sync<DeviceType>();
  
  x0 = atomKK->k_x0.template view<DeviceType>(); 
  x = atomKK->k_x.template view<DeviceType>();
  v = atomKK->k_v.template view<DeviceType>();
  f = atomKK->k_f.template view<DeviceType>();
  neighborKK->k_bondlist.template sync<DeviceType>();
  bondlist = neighborKK->k_bondlist.template view<DeviceType>();
  int nbondlist = neighborKK->nbondlist;
  nlocal = atom->nlocal;
  newton_bond = force->newton_bond;

  copymode = 1;

  // loop over neighbors of my atoms

  EV_FLOAT ev;

  if (evflag) {
    if (newton_bond) {
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagBondWLCPowAllViscCompute<1,1> >(0,nbondlist),*this,ev);
    } else {
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagBondWLCPowAllViscCompute<0,1> >(0,nbondlist),*this,ev);
    }
  } else {
    if (newton_bond) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagBondWLCPowAllViscCompute<1,0> >(0,nbondlist),*this);
    } else {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagBondWLCPowAllViscCompute<0,0> >(0,nbondlist),*this);
    }
  }

  if (eflag_global) energy += ev.evdwl;
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

  copymode = 0;
}


/* ---------------------------------------------------------------------- */


template<class DeviceType>
template<int NEWTON_BOND, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void BondWLCPowAllViscKokkos<DeviceType>::operator()(TagBondWLCPowAllViscCompute<NEWTON_BOND,EVFLAG>, const int &n, EV_FLOAT& ev) const {

  const int i1 = bondlist(n,0);
  const int i2 = bondlist(n,1);
  const int type = bondlist(n,2);

  X_FLOAT delx = x0(i1,0) - x0(i2,0);
  X_FLOAT dely = x0(i1,1) - x0(i2,1);
  X_FLOAT delz = x0(i1,2) - x0(i2,2);
  const F_FLOAT l0 = sqrt(delx*delx + dely*dely + delz*delz);

  delx = x(i1,0) - x(i2,0);
  dely = x(i1,1) - x(i2,1);
  delz = x(i1,2) - x(i2,2);
  const V_FLOAT dvx = v(i1,0) - v(i2,0);
  const V_FLOAT dvy = v(i1,1) - v(i2,1);
  const V_FLOAT dvz = v(i1,2) - v(i2,2);

  const F_FLOAT temp = d_temp[type];
  const F_FLOAT r0 = d_r0[type];
  const F_FLOAT mu_targ = d_mu_targ[type];
  const F_FLOAT qp = d_qp[type];
  const F_FLOAT gamc = d_gamc[type];
  const F_FLOAT gamt = d_gamt[type];
  const F_FLOAT sigc = d_sigc[type];
  const F_FLOAT sigt = d_sigt[type];

  const F_FLOAT ra = sqrt(delx*delx + dely*dely + delz*delz);
  const F_FLOAT lmax = l0*r0;
  F_FLOAT rr = 1.0/r0;  

  F_FLOAT kph = pow(l0,qp)*temp*(0.25/(1.0-rr)/(1.0-rr)-0.25+rr);
  const F_FLOAT mu = 0.25*sqrt(3.0)*(temp*(-0.25/pow(1.0-rr,2) + 0.25 + 0.5*rr/pow(1.0-rr,3))/lmax/rr + kph*(qp+1.0)/pow(l0,qp+1.0));

  const F_FLOAT lambda = mu/mu_targ;
  kph = kph*mu_targ/mu;
  rr = ra/lmax; 
  const F_FLOAT rlogarg = pow(ra,qp+1.0);
  const F_FLOAT vv = (delx*dvx + dely*dvy +  delz*dvz)/ra;

  VEC4 wrr;
  generate_wrr(delx, dely, delz, wrr);

  F_FLOAT fforce = -temp*(0.25/(1.0-rr)/(1.0-rr)-0.25+rr)/lambda/ra + kph/rlogarg + (sigc*wrr[3] - gamc*vv)/ra;

  E_FLOAT ebond = 0.0;
    if (eflag) {
      ebond += 0.25*temp*lmax*(3.0*rr*rr-2.0*rr*rr*rr)/(1.0-rr)/lambda;
      if (qp == 1.0)
        ebond -= kph*log(ra);
      else
        ebond += kph/(qp-1.0)/pow(ra,qp-1.0);
    }


  // apply force to each of 2 atoms

  if (NEWTON_BOND || i1 < nlocal) {
    f(i1,0) += delx*fforce - gamt*dvx + sigt*wrr[0]/ra;
    f(i1,1) += dely*fforce - gamt*dvy + sigt*wrr[1]/ra;
    f(i1,2) += delz*fforce - gamt*dvz + sigt*wrr[2]/ra;
  }

  if (NEWTON_BOND || i2 < nlocal) {
    f(i2,0) -= delx*fforce - gamt*dvx + sigt*wrr[0]/ra;
    f(i2,1) -= dely*fforce - gamt*dvy + sigt*wrr[1]/ra;
    f(i2,2) -= delz*fforce - gamt*dvz + sigt*wrr[2]/ra;
  }

  E_FLOAT fbond;
  fbond = fforce - gamt*sqrt(dvx*dvx+dvy*dvy+dvz*dvz)/ra + sigt*sqrt(wrr[0]*wrr[0]+wrr[1]*wrr[1]+wrr[2]*wrr[2])/(ra*ra);

  if (EVFLAG) ev_tally(ev,i1,i2,ebond,fbond,delx,dely,delz);
}


template<class DeviceType>
template<int NEWTON_BOND, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void BondWLCPowAllViscKokkos<DeviceType>::operator()(TagBondWLCPowAllViscCompute<NEWTON_BOND,EVFLAG>, const int &n) const {
  EV_FLOAT ev;
  this->template operator()<NEWTON_BOND,EVFLAG>(TagBondWLCPowAllViscCompute<NEWTON_BOND,EVFLAG>(), n, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void BondWLCPowAllViscKokkos<DeviceType>::allocate()
{
  BondWLCPowAllViscKokkos::allocate();

  int n = atom->nbondtypes;
  k_temp = DAT::tdual_ffloat_1d("BondWLCPowAllViscKokkos::temp",n+1);
  k_r0 = DAT::tdual_ffloat_1d("BondWLCPowAllViscKokkos::r0",n+1);
  k_mu_targ = DAT::tdual_ffloat_1d("BondWLCPowAllViscKokkos::mu_targ",n+1);
  k_qp = DAT::tdual_ffloat_1d("BondWLCPowAllViscKokkos::qp",n+1);
  k_gamc = DAT::tdual_ffloat_1d("BondWLCPowAllViscKokkos::gamc",n+1);
  k_gamt = DAT::tdual_ffloat_1d("BondWLCPowAllViscKokkos::gamt",n+1);
  k_sigc = DAT::tdual_ffloat_1d("BondWLCPowAllViscKokkos::sigc",n+1);
  k_sigt = DAT::tdual_ffloat_1d("BondWLCPowAllViscKokkos::sigt",n+1);

  d_temp = k_temp.template view<DeviceType>();
  d_r0 = k_r0.template view<DeviceType>();
  d_mu_targ = k_mu_targ.template view<DeviceType>();
  d_qp = k_qp.template view<DeviceType>();
  d_gamc = k_gamc.template view<DeviceType>();
  d_gamt = k_gamt.template view<DeviceType>();
  d_sigc = k_sigc.template view<DeviceType>();
  d_sigt = k_sigt.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   set coeffs for one type
------------------------------------------------------------------------- */

template<class DeviceType>
void BondWLCPowAllViscKokkos<DeviceType>::coeff(int narg, char **arg)
{
  BondWLCPowAllVisc::coeff(narg, arg);

  int n = atom->nbondtypes;
  for (int i = 1; i <= n; i++) {
    k_temp.h_view[i] = temp[i];
    k_r0.h_view[i] = r0[i];
    k_mu_targ.h_view[i] = mu_targ[i];
    k_qp.h_view[i] = qp[i];
    k_gamc.h_view[i] = gamc[i];
    k_gamt.h_view[i] = gamt[i];
    k_sigc.h_view[i] = sigc[i];
    k_sigt.h_view[i] = sigt[i];
  }

  k_temp.template modify<LMPHostType>();
  k_r0.template modify<LMPHostType>();
  k_mu_targ.template modify<LMPHostType>();
  k_qp.template modify<LMPHostType>();
  k_gamc.template modify<LMPHostType>();
  k_gamt.template modify<LMPHostType>();
  k_sigc.template modify<LMPHostType>();
  k_sigt.template modify<LMPHostType>();
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

template<class DeviceType>
void BondWLCPowAllViscKokkos<DeviceType>::read_restart(FILE *fp)
{
  BondWLCPowAllVisc::read_restart(fp);

  int n = atom->nbondtypes;
  for (int i = 1; i <= n; i++) {
    k_temp.h_view[i] = temp[i];
    k_r0.h_view[i] = r0[i];
    k_mu_targ.h_view[i] = mu_targ[i];
    k_qp.h_view[i] = qp[i];
    k_gamc.h_view[i] = gamc[i];
    k_gamt.h_view[i] = gamt[i];
    k_sigc.h_view[i] = sigc[i];
    k_sigt.h_view[i] = sigt[i];
  }

  k_temp.template modify<LMPHostType>();
  k_r0.template modify<LMPHostType>();
  k_mu_targ.template modify<LMPHostType>();
  k_qp.template modify<LMPHostType>();
  k_gamc.template modify<LMPHostType>();
  k_gamt.template modify<LMPHostType>();
  k_sigc.template modify<LMPHostType>();
  k_sigt.template modify<LMPHostType>();
}

/* ----------------------------------------------------------------------
   tally energy and virial into global and per-atom accumulators
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void BondWLCPowAllViscKokkos<DeviceType>::ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &ebond, const F_FLOAT &fbond, const F_FLOAT &delx,
                const F_FLOAT &dely, const F_FLOAT &delz) const
{
  E_FLOAT ebondhalf;
  F_FLOAT v[6];

  if (eflag_either) {
    if (eflag_global) {
      if (newton_bond) ev.evdwl += ebond;
      else {
        ebondhalf = 0.5*ebond;
        if (i < nlocal) ev.evdwl += ebondhalf;
        if (j < nlocal) ev.evdwl += ebondhalf;
      }
    }
    if (eflag_atom) {
      ebondhalf = 0.5*ebond;
      if (newton_bond || i < nlocal) d_eatom[i] += ebondhalf;
      if (newton_bond || j < nlocal) d_eatom[j] += ebondhalf;
    }
  }

  if (vflag_either) {
    v[0] = delx*delx*fbond;
    v[1] = dely*dely*fbond;
    v[2] = delz*delz*fbond;
    v[3] = delx*dely*fbond;
    v[4] = delx*delz*fbond;
    v[5] = dely*delz*fbond;

    if (vflag_global) {
      if (newton_bond) {
        ev.v[0] += v[0];
        ev.v[1] += v[1];
        ev.v[2] += v[2];
        ev.v[3] += v[3];
        ev.v[4] += v[4];
        ev.v[5] += v[5];
      } else {
        if (i < nlocal) {
          ev.v[0] += 0.5*v[0];
          ev.v[1] += 0.5*v[1];
          ev.v[2] += 0.5*v[2];
          ev.v[3] += 0.5*v[3];
          ev.v[4] += 0.5*v[4];
          ev.v[5] += 0.5*v[5];
        }
        if (j < nlocal) {
          ev.v[0] += 0.5*v[0];
          ev.v[1] += 0.5*v[1];
          ev.v[2] += 0.5*v[2];
          ev.v[3] += 0.5*v[3];
          ev.v[4] += 0.5*v[4];
          ev.v[5] += 0.5*v[5];
        }
      }
    }

    if (vflag_atom) {
      if (newton_bond || i < nlocal) {
        d_vatom(i,0) += 0.5*v[0];
        d_vatom(i,1) += 0.5*v[1];
        d_vatom(i,2) += 0.5*v[2];
        d_vatom(i,3) += 0.5*v[3];
        d_vatom(i,4) += 0.5*v[4];
        d_vatom(i,5) += 0.5*v[5];
      }
      if (newton_bond || j < nlocal) {
        d_vatom(j,0) += 0.5*v[0];
        d_vatom(j,1) += 0.5*v[1];
        d_vatom(j,2) += 0.5*v[2];
        d_vatom(j,3) += 0.5*v[3];
        d_vatom(j,4) += 0.5*v[4];
        d_vatom(j,5) += 0.5*v[5];
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void BondWLCPowAllViscKokkos<DeviceType>::generate_wrr(X_FLOAT delx, X_FLOAT dely, X_FLOAT delz, VEC4 &wrr) const {

  int i;
  F_FLOAT ww[3][3];
  F_FLOAT v1, v2, factor, ss;
  rand_type rand_gen = rand_pool.get_state();

  for (i=0; i<5; i++){
    ss = 100.0;
    while ( ss > 1.0 ){
      v1 = 2.0*rand_gen.drand() - 1.0; 
      v2 = 2.0*rand_gen.drand() - 1.0; 
      ss = v1*v1 + v2*v2;
    }

    factor = sqrt(-2.0 * log(ss)/ss);

    if (i < 3){
      ww[i][0] = factor*v1;
      ww[i][1] = factor*v2; 
    }
    else if (i == 3){
      ww[0][2] = factor*v1;
      ww[1][2] = factor*v2;
    }
    else
      ww[2][2] = factor*v1; 
  }

  wrr[3] = (ww[0][0]+ww[1][1]+ww[2][2])/3.0;
  wrr[0] = (ww[0][0]-wrr[3])*delx + 0.5*(ww[0][1]+ww[1][0])*dely + 0.5*(ww[0][2]+ww[2][0])*delz;
  wrr[1] = 0.5*(ww[1][0]+ww[0][1])*delx + (ww[1][1]-wrr[3])*dely + 0.5*(ww[1][2]+ww[2][1])*delz;
  wrr[2] = 0.5*(ww[2][0]+ww[0][2])*delx + 0.5*(ww[2][1]+ww[1][2])*dely + (ww[2][2]-wrr[3])*delz;

  rand_pool.free_state(rand_gen);
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class BondWLCPowAllViscKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class BondWLCPowAllViscKokkos<LMPHostType>;
#endif
}

