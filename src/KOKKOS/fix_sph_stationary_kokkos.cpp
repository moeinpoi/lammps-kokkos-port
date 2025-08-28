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

#include "fix_sph_stationary_kokkos.h"

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
FixSPHStationaryKokkos<DeviceType>::FixSPHStationaryKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixSPHStationary(lmp, narg, arg)
{
  kokkosable = 1;
  fuse_integrate_flag = 0;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = MASK_MASK | RHO_MASK | ESPH_MASK | DRHO_MASK | DESPH_MASK;
  datamask_modify = ESPH_MASK | RHO_MASK;
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

template<class DeviceType>
void FixSPHStationaryKokkos<DeviceType>::initial_integrate(int /*vflag*/)
{
	fprintf(screen, "initial_integrate called on FixSPHStationaryKokkos\n");

  atomKK->sync(execution_space,datamask_read);

  rho = atomKK->k_rho.view<DeviceType>();
  drho = atomKK->k_drho.view<DeviceType>();
  esph = atomKK->k_esph.view<DeviceType>();
  desph = atomKK->k_desph.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;

  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  FixSPHStationaryKokkosInitialIntegrateFunctor<DeviceType> functor(this);
  Kokkos::parallel_for(nlocal,functor);

  atomKK->modified(execution_space,datamask_modify); 
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSPHStationaryKokkos<DeviceType>::initial_integrate_item(int i) const
{
  if (mask(i) & groupbit) {
	
    esph(i) += dtf * desph(i); // half-step update of particle internal energy
    rho(i) += dtf * drho(i); // ... and density
 
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSPHStationaryKokkos<DeviceType>::final_integrate()
{
	fprintf(screen, "final_integrate called on FixSPHStationaryKokkos\n");

  atomKK->sync(execution_space,datamask_read);

  rho = atomKK->k_rho.view<DeviceType>();
  drho = atomKK->k_drho.view<DeviceType>();
  esph = atomKK->k_esph.view<DeviceType>();
  desph = atomKK->k_desph.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;

  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  FixSPHStationaryKokkosFinalIntegrateFunctor<DeviceType> functor(this);
  Kokkos::parallel_for(nlocal,functor);

  atomKK->modified(execution_space,datamask_modify);

  // debug
  //atomKK->sync(Host,datamask_read);
}


template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixSPHStationaryKokkos<DeviceType>::final_integrate_item(int i) const
{
  if (mask(i) & groupbit) {

    esph(i) += dtf * desph(i); 
    rho(i) += dtf * drho(i); 

  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSPHStationaryKokkos<DeviceType>::cleanup_copy()
{
	fprintf(screen, "cleanup_copy called on FixSPHStationaryKokkos\n");
  id = style = nullptr;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixSPHStationaryKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixSPHStationaryKokkos<LMPHostType>;
#endif
}

