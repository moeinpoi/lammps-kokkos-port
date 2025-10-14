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

#include "dihedral_bend2_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory_kokkos.h"
#include "neighbor_kokkos.h"

#include <cmath>

using namespace LAMMPS_NS;

#define TOLERANCE 0.05
#define SMALL     0.001

/* ---------------------------------------------------------------------- */

template<class DeviceType>
DihedralBend2Kokkos<DeviceType>::DihedralBend2Kokkos(LAMMPS *lmp) : DihedralBend2(lmp)
{
  atomKK = (AtomKokkos *) atom;
  neighborKK = (NeighborKokkos *) neighbor;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  // datamask_read = X_MASK | F_MASK | Q_MASK | ENERGY_MASK | VIRIAL_MASK;
  // datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;

  k_warning_flag = DAT::tdual_int_scalar("Dihedral:warning_flag");
  d_warning_flag = k_warning_flag.view<DeviceType>();
  h_warning_flag = k_warning_flag.h_view;

  centroidstressflag = CENTROID_NOTAVAIL;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
DihedralBend2Kokkos<DeviceType>::~DihedralBend2Kokkos()
{
  if (!copymode) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void DihedralBend2Kokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{

  eflag = eflag_in;
  vflag = vflag_in;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"dihedral:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"dihedral:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  k_k.template sync<DeviceType>();
  k_theta0.template sync<DeviceType>();

  x0 = atomKK->k_x0.view<DeviceType>();
  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  neighborKK->k_dihedrallist.template sync<DeviceType>();
  dihedrallist = neighborKK->k_dihedrallist.view<DeviceType>();
  int ndihedrallist = neighborKK->ndihedrallist;
  nlocal = atom->nlocal;
  newton_bond = force->newton_bond;

  h_warning_flag() = 0;
  k_warning_flag.template modify<LMPHostType>();
  k_warning_flag.template sync<DeviceType>();

  copymode = 1;

  // loop over neighbors of my atoms

  EV_FLOAT ev;

  if (evflag) {
    if (newton_bond) {
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagDihedralBend2Compute<1,1> >(0,ndihedrallist),*this,ev);
    } else {
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagDihedralBend2Compute<0,1> >(0,ndihedrallist),*this,ev);
    }
  } else {
    if (newton_bond) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagDihedralBend2Compute<1,0> >(0,ndihedrallist),*this);
    } else {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagDihedralBend2Compute<0,0> >(0,ndihedrallist),*this);
    }
  }

  // error check

  k_warning_flag.template modify<DeviceType>();
  k_warning_flag.template sync<LMPHostType>();
  if (h_warning_flag())
    error->warning(FLERR,"Dihedral problem");

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

template<class DeviceType>
template<int NEWTON_BOND, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void DihedralBend2Kokkos<DeviceType>::operator()(TagDihedralBend2Compute<NEWTON_BOND,EVFLAG>, const int &n, EV_FLOAT& ev) const {

  // The f array is atomic
  Kokkos::View<F_FLOAT*[3], typename DAT::t_f_array::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > a_f = f;

  const int i1 = dihedrallist(n,0);
  const int i2 = dihedrallist(n,1);
  const int i3 = dihedrallist(n,2);
  const int i4 = dihedrallist(n,3);
  const int type = dihedrallist(n,4);

  //Initial position
  // 2-1 distance
  X_FLOAT d21x = x0(i2,0) - x0(i1,0);
  X_FLOAT d21y = x0(i2,1) - x0(i1,1);
  X_FLOAT d21z = x0(i2,2) - x0(i1,2);
  //domain->minimum_image(d21x,d21y,d21z);

  // 3-1 distance
  X_FLOAT d31x = x0(i3,0) - x0(i1,0);
  X_FLOAT d31y = x0(i3,1) - x0(i1,1);
  X_FLOAT d31z = x0(i3,2) - x0(i1,2);

  // 3-2 distance
  X_FLOAT d32x = x0(i3,0) - x0(i2,0);
  X_FLOAT d32y = x0(i3,1) - x0(i2,1);
  X_FLOAT d32z = x0(i3,2) - x0(i2,2);

  // 3-4 distance
  X_FLOAT d34x = x0(i3,0) - x0(i4,0);
  X_FLOAT d34y = x0(i3,1) - x0(i4,0);
  X_FLOAT d34z = x0(i3,2) - x0(i4,0);

  // 2-4 distance
  X_FLOAT d24x = x0(i2,0) - x0(i4,0);
  X_FLOAT d24y = x0(i2,1) - x0(i4,1);
  X_FLOAT d24z = x0(i2,2) - x0(i4,2);

  // 1-4 distance
  X_FLOAT d14x = x0(i1,0) - x0(i4,0);
  X_FLOAT d14y = x0(i1,1) - x0(i4,1);
  X_FLOAT d14z = x0(i1,2) - x0(i4,2);

  // calculate normals
  X_FLOAT n1x = d21y*d31z - d31y*d21z;
  X_FLOAT n1y = d31x*d21z - d21x*d31z;
  X_FLOAT n1z = d21x*d31y - d31x*d21y;
  X_FLOAT n2x = d34y*d24z - d24y*d34z;
  X_FLOAT n2y = d24x*d34z - d34x*d24z;
  X_FLOAT n2z = d34x*d24y - d24x*d34y;
  X_FLOAT n1 = n1x*n1x + n1y*n1y + n1z*n1z;
  X_FLOAT n2 = n2x*n2x + n2y*n2y + n2z*n2z;
  X_FLOAT nn = sqrt(n1*n2);

  // cos(theta) and sin(theta) calculation 
  F_FLOAT costheta0 = (n1x*n2x + n1y*n2y + n1z*n2z)/nn; 
  if (costheta0 > 1.0) costheta0 = 1.0;
  if (costheta0 < -1.0) costheta0 = -1.0;
  F_FLOAT sintheta0 = sqrt(1.0-costheta0*costheta0); 
  if (sintheta0 < SMALL) sintheta0 = SMALL;
  F_FLOAT mx = (n1x-n2x)*d14x + (n1y-n2y)*d14y + (n1z-n2z)*d14z;
  if (mx < 0) sintheta0 = -sintheta0;

  // 2-1 distance
  d21x = x(i2,0) - x(i1,0);
  d21y = x(i2,1) - x(i1,1);
  d21z = x(i2,2) - x(i1,2);
  //domain->minimum_image(d21x,d21y,d21z);

  // 3-1 distance
  d31x = x(i3,0) - x(i1,0);
  d31y = x(i3,1) - x(i1,1);
  d31z = x(i3,2) - x(i1,2);

  // 3-2 distance
  d32x = x(i3,0) - x(i2,0);
  d32y = x(i3,1) - x(i2,1);
  d32z = x(i3,2) - x(i2,2);

  // 3-4 distance
  d34x = x(i3,0) - x(i4,0);
  d34y = x(i3,1) - x(i4,0);
  d34z = x(i3,2) - x(i4,0);

  // 2-4 distance
  d24x = x(i2,0) - x(i4,0);
  d24y = x(i2,1) - x(i4,1);
  d24z = x(i2,2) - x(i4,2);

  // 1-4 distance
  d14x = x(i1,0) - x(i4,0);
  d14y = x(i1,1) - x(i4,1);
  d14z = x(i1,2) - x(i4,2);

  n1x = d21y*d31z - d31y*d21z;
  n1y = d31x*d21z - d21x*d31z;
  n1z = d21x*d31y - d31x*d21y;
  n2x = d34y*d24z - d24y*d34z;
  n2y = d24x*d34z - d34x*d24z;
  n2z = d34x*d24y - d24x*d34y;
  n1 = n1x*n1x + n1y*n1y + n1z*n1z;
  n2 = n2x*n2x + n2y*n2y + n2z*n2z;
  nn = sqrt(n1*n2);  

  // cos(theta) and sin(theta) calculation 
  F_FLOAT costheta = (n1x*n2x + n1y*n2y + n1z*n2z)/nn; 
  if (costheta > 1.0) costheta = 1.0;
  if (costheta < -1.0) costheta = -1.0;
  F_FLOAT sintheta = sqrt(1.0-costheta*costheta); 
  if (sintheta < SMALL) sintheta = SMALL;
  mx = (n1x-n2x)*d14x + (n1y-n2y)*d14y + (n1z-n2z)*d14z;
  if (mx < 0) sintheta = -sintheta;

  const F_FLOAT k = d_k(type);

  // coeffs calculation
  const F_FLOAT alfa = k*(costheta0-costheta*sintheta0/sintheta);   
  const F_FLOAT a11 = -alfa*costheta/n1;
  const F_FLOAT a12 = alfa/nn;
  const F_FLOAT a22 = -alfa*costheta/n2;

  // forces calculation
  const F_FLOAT s1x = a11*(n1y*d32z - n1z*d32y) + a12*(n2y*d32z - n2z*d32y);
  const F_FLOAT s1y = a11*(n1z*d32x - n1x*d32z) + a12*(n2z*d32x - n2x*d32z);
  const F_FLOAT s1z = a11*(n1x*d32y - n1y*d32x) + a12*(n2x*d32y - n2y*d32x);
  const F_FLOAT s2x = a11*(n1z*d31y - n1y*d31z) + a22*(n2y*d34z - n2z*d34y) +  
        a12*(n2z*d31y - n2y*d31z + n1y*d34z - n1z*d34y);
  const F_FLOAT s2y = a11*(n1x*d31z - n1z*d31x) + a22*(n2z*d34x - n2x*d34z) +  
        a12*(n2x*d31z - n2z*d31x + n1z*d34x - n1x*d34z);
  const F_FLOAT s2z = a11*(n1y*d31x - n1x*d31y) + a22*(n2x*d34y - n2y*d34x) +  
        a12*(n2y*d31x - n2x*d31y + n1x*d34y - n1y*d34x);
  const F_FLOAT s3x = a11*(n1y*d21z - n1z*d21y) + a22*(n2z*d24y - n2y*d24z) +  
        a12*(n2y*d21z - n2z*d21y + n1z*d24y - n1y*d24z);     
  const F_FLOAT s3y = a11*(n1z*d21x - n1x*d21z) + a22*(n2x*d24z - n2z*d24x) +  
        a12*(n2z*d21x - n2x*d21z + n1x*d24z - n1z*d24x);
  const F_FLOAT s3z = a11*(n1x*d21y - n1y*d21x) + a22*(n2y*d24x - n2x*d24y) +  
        a12*(n2x*d21y - n2y*d21x + n1y*d24x - n1x*d24y);
  const F_FLOAT s4x = a22*(n2z*d32y - n2y*d32z) + a12*(n1z*d32y - n1y*d32z);
  const F_FLOAT s4y = a22*(n2x*d32z - n2z*d32x) + a12*(n1x*d32z - n1z*d32x);
  const F_FLOAT s4z = a22*(n2y*d32x - n2x*d32y) + a12*(n1y*d32x - n1x*d32y);

  F_FLOAT edihedral = 0.0;
  F_FLOAT energy = 0.0;

  if (eflag){
      mx = costheta*costheta0 + sintheta*sintheta0;
      edihedral = k*(1.0-mx);
      energy += k*(1.0-mx);
    }
  
  // apply force to each of 4 atoms
  F_FLOAT f1[3],f2[3],f3[3],f4[3];

  f1[0] = s1x;
  f1[1] = s1y;
  f1[2] = s1z;

  f2[0] = s2x;
  f2[1] = s2y;
  f2[2] = s2z;

  f3[0] = s3x;
  f3[1] = s3y;
  f3[2] = s3z;

  f4[0] = s4x;
  f4[1] = s4y;
  f4[2] = s4z;

  if (NEWTON_BOND || i1 < nlocal) {
    a_f(i1,0) += f1[0];
    a_f(i1,1) += f1[1];
    a_f(i1,2) += f1[2];
  }  
  if (NEWTON_BOND || i2 < nlocal) {
    a_f(i2,0) += f2[0];
    a_f(i2,1) += f2[1];
    a_f(i2,2) += f2[2];
  }  
  if (NEWTON_BOND || i3 < nlocal) {
    a_f(i3,0) += f3[0];
    a_f(i3,1) += f3[1];
    a_f(i3,2) += f3[2];
  }  
  if (NEWTON_BOND || i4 < nlocal) {
    a_f(i4,0) += f4[0];
    a_f(i4,1) += f4[1];
    a_f(i4,2) += f4[2];
  }  

  if (EVFLAG)
    ev_tally(ev,i1,i2,i3,i4,edihedral,f1,f3,f4,
             -d21x,-d21y,-d21z,d32x,d32y,d32z,-d34x,-d34y,-d34z);

}

template<class DeviceType>
template<int NEWTON_BOND, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void DihedralBend2Kokkos<DeviceType>::operator()(TagDihedralBend2Compute<NEWTON_BOND,EVFLAG>, const int &n) const {
  EV_FLOAT ev;
  this->template operator()<NEWTON_BOND,EVFLAG>(TagDihedralBend2Compute<NEWTON_BOND,EVFLAG>(), n, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void DihedralBend2Kokkos<DeviceType>::allocate()
{
  DihedralBend2::allocate();

  int n = atom->ndihedraltypes;
  k_k = DAT::tdual_ffloat_1d("DihedralBend2::k",n+1);
  k_theta0 = DAT::tdual_ffloat_1d("DihedralBend2::theta0",n+1);

  d_k = k_k.template view<DeviceType>();
  d_theta0 = k_theta0.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   set coeffs for one type
------------------------------------------------------------------------- */

template<class DeviceType>
void DihedralBend2Kokkos<DeviceType>::coeff(int narg, char **arg)
{
  DihedralBend2::coeff(narg, arg);

  int n = atom->ndihedraltypes;
  for (int i = 1; i <= n; i++) {
    k_k.h_view[i] = k[i];
    k_theta0.h_view[i] = theta0[i];
  }

  k_k.template modify<LMPHostType>();
  k_theta0.template modify<LMPHostType>();
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

template<class DeviceType>
void DihedralBend2Kokkos<DeviceType>::read_restart(FILE *fp)
{
  DihedralBend2::read_restart(fp);

  int n = atom->ndihedraltypes;
  for (int i = 1; i <= n; i++) {
    k_k.h_view[i] = k[i];
    k_theta0.h_view[i] = theta0[i];
  }

  k_k.template modify<LMPHostType>();
  k_theta0.template modify<LMPHostType>();
}

/* ----------------------------------------------------------------------
   tally energy and virial into global and per-atom accumulators
   virial = r1F1 + r2F2 + r3F3 + r4F4 = (r1-r2) F1 + (r3-r2) F3 + (r4-r2) F4
          = (r1-r2) F1 + (r3-r2) F3 + (r4-r3 + r3-r2) F4
          = vb1*f1 + vb2*f3 + (vb3+vb2)*f4
------------------------------------------------------------------------- */

template<class DeviceType>
//template<int NEWTON_BOND>
KOKKOS_INLINE_FUNCTION
void DihedralBend2Kokkos<DeviceType>::ev_tally(EV_FLOAT &ev, const int i1, const int i2, const int i3, const int i4,
                        F_FLOAT &edihedral, F_FLOAT *f1, F_FLOAT *f3, F_FLOAT *f4,
                        const F_FLOAT &vb1x, const F_FLOAT &vb1y, const F_FLOAT &vb1z,
                        const F_FLOAT &vb2x, const F_FLOAT &vb2y, const F_FLOAT &vb2z,
                        const F_FLOAT &vb3x, const F_FLOAT &vb3y, const F_FLOAT &vb3z) const
{
  E_FLOAT edihedralquarter;
  F_FLOAT v[6];

  // The eatom and vatom arrays are atomic
  Kokkos::View<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > v_eatom = k_eatom.view<DeviceType>();
  Kokkos::View<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > v_vatom = k_vatom.view<DeviceType>();

  if (eflag_either) {
    if (eflag_global) {
      if (newton_bond) ev.evdwl += edihedral;
      else {
        edihedralquarter = 0.25*edihedral;
        if (i1 < nlocal) ev.evdwl += edihedralquarter;
        if (i2 < nlocal) ev.evdwl += edihedralquarter;
        if (i3 < nlocal) ev.evdwl += edihedralquarter;
        if (i4 < nlocal) ev.evdwl += edihedralquarter;
      }
    }
    if (eflag_atom) {
      edihedralquarter = 0.25*edihedral;
      if (newton_bond || i1 < nlocal) v_eatom[i1] += edihedralquarter;
      if (newton_bond || i2 < nlocal) v_eatom[i2] += edihedralquarter;
      if (newton_bond || i3 < nlocal) v_eatom[i3] += edihedralquarter;
      if (newton_bond || i4 < nlocal) v_eatom[i4] += edihedralquarter;
    }
  }

  if (vflag_either) {
    v[0] = vb1x*f1[0] + vb2x*f3[0] + (vb3x+vb2x)*f4[0];
    v[1] = vb1y*f1[1] + vb2y*f3[1] + (vb3y+vb2y)*f4[1];
    v[2] = vb1z*f1[2] + vb2z*f3[2] + (vb3z+vb2z)*f4[2];
    v[3] = vb1x*f1[1] + vb2x*f3[1] + (vb3x+vb2x)*f4[1];
    v[4] = vb1x*f1[2] + vb2x*f3[2] + (vb3x+vb2x)*f4[2];
    v[5] = vb1y*f1[2] + vb2y*f3[2] + (vb3y+vb2y)*f4[2];

    if (vflag_global) {
      if (newton_bond) {
        ev.v[0] += v[0];
        ev.v[1] += v[1];
        ev.v[2] += v[2];
        ev.v[3] += v[3];
        ev.v[4] += v[4];
        ev.v[5] += v[5];
      } else {
        if (i1 < nlocal) {
          ev.v[0] += 0.25*v[0];
          ev.v[1] += 0.25*v[1];
          ev.v[2] += 0.25*v[2];
          ev.v[3] += 0.25*v[3];
          ev.v[4] += 0.25*v[4];
          ev.v[5] += 0.25*v[5];
        }
        if (i2 < nlocal) {
          ev.v[0] += 0.25*v[0];
          ev.v[1] += 0.25*v[1];
          ev.v[2] += 0.25*v[2];
          ev.v[3] += 0.25*v[3];
          ev.v[4] += 0.25*v[4];
          ev.v[5] += 0.25*v[5];
        }
        if (i3 < nlocal) {
          ev.v[0] += 0.25*v[0];
          ev.v[1] += 0.25*v[1];
          ev.v[2] += 0.25*v[2];
          ev.v[3] += 0.25*v[3];
          ev.v[4] += 0.25*v[4];
          ev.v[5] += 0.25*v[5];
        }
        if (i4 < nlocal) {
          ev.v[0] += 0.25*v[0];
          ev.v[1] += 0.25*v[1];
          ev.v[2] += 0.25*v[2];
          ev.v[3] += 0.25*v[3];
          ev.v[4] += 0.25*v[4];
          ev.v[5] += 0.25*v[5];
        }
      }
    }

    if (vflag_atom) {
      if (newton_bond || i1 < nlocal) {
        v_vatom(i1,0) += 0.25*v[0];
        v_vatom(i1,1) += 0.25*v[1];
        v_vatom(i1,2) += 0.25*v[2];
        v_vatom(i1,3) += 0.25*v[3];
        v_vatom(i1,4) += 0.25*v[4];
        v_vatom(i1,5) += 0.25*v[5];
      }
      if (newton_bond || i2 < nlocal) {
        v_vatom(i2,0) += 0.25*v[0];
        v_vatom(i2,1) += 0.25*v[1];
        v_vatom(i2,2) += 0.25*v[2];
        v_vatom(i2,3) += 0.25*v[3];
        v_vatom(i2,4) += 0.25*v[4];
        v_vatom(i2,5) += 0.25*v[5];
      }
      if (newton_bond || i3 < nlocal) {
        v_vatom(i3,0) += 0.25*v[0];
        v_vatom(i3,1) += 0.25*v[1];
        v_vatom(i3,2) += 0.25*v[2];
        v_vatom(i3,3) += 0.25*v[3];
        v_vatom(i3,4) += 0.25*v[4];
        v_vatom(i3,5) += 0.25*v[5];
      }
      if (newton_bond || i4 < nlocal) {
        v_vatom(i4,0) += 0.25*v[0];
        v_vatom(i4,1) += 0.25*v[1];
        v_vatom(i4,2) += 0.25*v[2];
        v_vatom(i4,3) += 0.25*v[3];
        v_vatom(i4,4) += 0.25*v[4];
        v_vatom(i4,5) += 0.25*v[5];
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class DihedralBend2Kokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class DihedralBend2Kokkos<LMPHostType>;
#endif
}

