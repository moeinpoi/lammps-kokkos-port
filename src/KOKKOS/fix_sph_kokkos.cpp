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

#include "fix_sph_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "domain.h"

#include <cstring>
using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixSPHKokkos<DeviceType>::FixSPHKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixSPH(lmp, narg, arg)
{
  kokkosable = 1;
  fuse_integrate_flag = 0;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = X_MASK | V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK | VEST_MASK | RHO_MASK | ESPH_MASK | DRHO_MASK | DESPH_MASK;
  datamask_modify = X_MASK | V_MASK | VEST_MASK | ESPH_MASK | RHO_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSPHKokkos<DeviceType>::init()
{
  FixSPH::init();

  atomKK->k_mass.modify<LMPHostType>();
  atomKK->k_mass.sync<DeviceType>();
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

template<class DeviceType>
void FixSPHKokkos<DeviceType>::pre_force(int /*vflag*/)
{
   if (!remap_v_flag) return;

   atomKK->sync(execution_space, V_MASK | VEST_MASK | MASK_MASK);

   v = atomKK->k_v.view<DeviceType>();
   vest = atomKK->k_vest.view<DeviceType>();
   mask = atomKK->k_mask.view<DeviceType>();
   int nlocal = atomKK->nlocal;
   if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

   FixSPHKokkosPreForceFunctor<DeviceType> functor(this);
   Kokkos::parallel_for(nlocal,functor);
   
   atomKK->modified(execution_space,VEST_MASK); 
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSPHKokkos<DeviceType>::pre_force_item(int i) const
{
  if (mask[i] & groupbit) {
    vest(i,0) += v(i,0);
    vest(i,1) += v(i,1);
    vest(i,2) += v(i,2); 
  }
}

template<class DeviceType>
void FixSPHKokkos<DeviceType>::setup_pre_force(int)
{
	fprintf(screen, "setup_pre_force called on FixSPHKokkos\n");
  remap_v_flag = domain->deform_vremap;
  if (remap_v_flag && (!comm->ghost_velocity))
  error->all(FLERR, "Fix sph requires ghost atoms store velocity when deforming with remap v");

  atomKK->sync(execution_space, V_MASK | VEST_MASK | MASK_MASK);

  v = atomKK->k_v.view<DeviceType>();
  vest = atomKK->k_vest.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  FixSPHKokkosSetupPreForceFunctor<DeviceType> functor(this);
  Kokkos::parallel_for(nlocal, functor);

  atomKK->modified(execution_space, VEST_MASK);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSPHKokkos<DeviceType>::setup_pre_force_item(int i) const
{
  if (mask[i] & groupbit) {
    vest(i,0) = v(i,0);
    vest(i,1) = v(i,1);
    vest(i,2) = v(i,2);
  }
}



template<class DeviceType>
void FixSPHKokkos<DeviceType>::initial_integrate(int /*vflag*/)
{
	fprintf(screen, "initial_integrate called on FixSPHKokkos\n");
  atomKK->sync(execution_space,datamask_read);

  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();

  vest = atomKK->k_vest.view<DeviceType>();
  rho = atomKK->k_rho.view<DeviceType>();
  drho = atomKK->k_drho.view<DeviceType>();
  esph = atomKK->k_esph.view<DeviceType>();
  desph = atomKK->k_desph.view<DeviceType>();

  rmass = atomKK->k_rmass.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  if (rmass.data()) {
    FixSPHKokkosInitialIntegrateFunctor<DeviceType,1> functor(this);
    Kokkos::parallel_for(nlocal,functor);
  } else {
    FixSPHKokkosInitialIntegrateFunctor<DeviceType,0> functor(this);
    Kokkos::parallel_for(nlocal,functor);
  }

  atomKK->modified(execution_space,datamask_modify); 
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSPHKokkos<DeviceType>::initial_integrate_item(int i) const
{
  if (mask[i] & groupbit) {
	
    esph(i) += dtf * desph(i); // half-step update of particle internal energy
    rho(i) += dtf * drho(i); // ... and density

    const double dtfm = dtf / mass[type[i]];
    // extrapolate velocity for use with velocity-dependent potentials, e.g. SPH

    vest(i,0) = v(i,0) + 2.0 * dtfm * f(i, 0);
    vest(i,1) = v(i,1) + 2.0 * dtfm * f(i, 1);
    vest(i,2) = v(i,2) + 2.0 * dtfm * f(i, 2);
    v(i,0) += dtfm * f(i,0);
    v(i,1) += dtfm * f(i,1);
    v(i,2) += dtfm * f(i,2);
    x(i,0) += dtv * v(i,0);
    x(i,1) += dtv * v(i,1);
    x(i,2) += dtv * v(i,2); 
 
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSPHKokkos<DeviceType>::initial_integrate_rmass_item(int i) const
{
  if (mask[i] & groupbit) {

    esph(i) += dtf * desph(i); // half-step update of particle internal energy
    rho(i) += dtf * drho(i); // ... and density

    const double dtfm = dtf / rmass[i];
    vest(i,0) = v(i,0) + 2.0 * dtfm * f(i, 0);
    vest(i,1) = v(i,1) + 2.0 * dtfm * f(i, 1);
    vest(i,2) = v(i,2) + 2.0 * dtfm * f(i, 2);
    v(i,0) += dtfm * f(i,0);
    v(i,1) += dtfm * f(i,1);
    v(i,2) += dtfm * f(i,2);
    x(i,0) += dtv * v(i,0);
    x(i,1) += dtv * v(i,1);
    x(i,2) += dtv * v(i,2);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSPHKokkos<DeviceType>::final_integrate()
{
	fprintf(screen, "final_integrate called on FixSPHKokkos\n");
  atomKK->sync(execution_space,datamask_read);

  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();

  rho = atomKK->k_rho.view<DeviceType>();
  drho = atomKK->k_drho.view<DeviceType>();
  esph = atomKK->k_esph.view<DeviceType>();
  desph = atomKK->k_desph.view<DeviceType>();
 
  rmass = atomKK->k_rmass.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  if (rmass.data()) {
    FixSPHKokkosFinalIntegrateFunctor<DeviceType,1> functor(this);
    Kokkos::parallel_for(nlocal,functor);
  } else {
    FixSPHKokkosFinalIntegrateFunctor<DeviceType,0> functor(this);
    Kokkos::parallel_for(nlocal,functor);
  }

  atomKK->modified(execution_space,V_MASK | ESPH_MASK | RHO_MASK);

  // debug
  //atomKK->sync(Host,datamask_read);
}


template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSPHKokkos<DeviceType>::final_integrate_item(int i) const
{
  if (mask[i] & groupbit) {
    const double dtfm = dtf / mass[type[i]];
    
    v(i,0) += dtfm * f(i,0);
    v(i,1) += dtfm * f(i,1);
    v(i,2) += dtfm * f(i,2);
    esph(i) += dtf * desph(i); 
    rho(i) += dtf * drho(i); 
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSPHKokkos<DeviceType>::final_integrate_rmass_item(int i) const
{
  if (mask[i] & groupbit) {
    const double dtfm = dtf / rmass[i];
    v(i,0) += dtfm * f(i,0);
    v(i,1) += dtfm * f(i,1);
    v(i,2) += dtfm * f(i,2);
    esph(i) += dtf * desph(i); 
    rho(i) += dtf * drho(i);  
  }
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSPHKokkos<DeviceType>::cleanup_copy()
{
	fprintf(screen, "cleanup_copy called on FixSPHKokkos\n");
  id = style = nullptr;
  vatom = nullptr;
}

/* ---------------------------------------------------------------------- */


namespace LAMMPS_NS {
template class FixSPHKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixSPHKokkos<LMPHostType>;
#endif
}

