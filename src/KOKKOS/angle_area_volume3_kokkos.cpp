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

#include "angle_area_volume3_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "domain_kokkos.h"
#include "force.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neighbor_kokkos.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.001

/* ---------------------------------------------------------------------- */

template<class DeviceType>
AngleAreaVolume3Kokkos<DeviceType>::AngleAreaVolume3Kokkos(LAMMPS *lmp) : AngleAreaVolume3(lmp)
{
  atomKK = (AtomKokkos *) atom;
  neighborKK = (NeighborKokkos *) neighbor;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | X0_MASK | F_MASK | ENERGY_MASK | MOLECULE_MASK | IMAGE_MASK | VIRIAL_MASK; 
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;

  prd = Few<double,3>(domain->prd);
  h   = Few<double,6>(domain->h);
  triclinic = domain->triclinic;

  centroidstressflag = CENTROID_NOTAVAIL;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
AngleAreaVolume3Kokkos<DeviceType>::~AngleAreaVolume3Kokkos()
{
  if (!copymode) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);

    if (init_on) {
      memoryKK->destroy_kokkos(k_h_area,h_area);
      memoryKK->destroy_kokkos(k_h_volume,h_volume);
      memoryKK->destroy_kokkos(k_ttyp,ttyp);
      memoryKK->destroy_kokkos(k_ttyp1,ttyp1);
      memoryKK->destroy_kokkos(k_dath,dath);
      memoryKK->destroy_kokkos(k_datt,datt);
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void AngleAreaVolume3Kokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"angle:eatom");
    d_eatom = k_eatom.template view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"angle:vatom");
    d_vatom = k_vatom.template view<DeviceType>();
  }

  k_press.template sync<DeviceType>();
  k_ka.template sync<DeviceType>();
  k_a0.template sync<DeviceType>();
  k_kv.template sync<DeviceType>();
  k_v0.template sync<DeviceType>();
  k_kl.template sync<DeviceType>();
  k_aa0.template sync<DeviceType>();
  k_r0.template sync<DeviceType>();
  k_qp.template sync<DeviceType>();
  k_vratio.template sync<DeviceType>();
  k_nstep1.template sync<DeviceType>();
  k_nstep2.template sync<DeviceType>();

  x0 = atomKK->k_x0.template view<DeviceType>();
  x = atomKK->k_x.template view<DeviceType>();
  f = atomKK->k_f.template view<DeviceType>();
  tag = atomKK->k_tag.template view<DeviceType>();
  molecule = atomKK->k_molecule.template view<DeviceType>();
  image = atomKK->k_image.template view<DeviceType>();

  neighborKK->k_anglelist.template sync<DeviceType>();
  anglelist = neighborKK->k_anglelist.template view<DeviceType>();
  int nanglelist = neighborKK->nanglelist;
  nlocal = atom->nlocal;
  newton_bond = force->newton_bond;

  // tagint nm, nmol; //already declared in base header
  nm = 0;
  nmol = 0;
  //find max mol ID in this rank
  atomKK->sync(execution_space, ALL_MASK);
  auto d_molecule = molecule; 
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0,nlocal),
                          KOKKOS_LAMBDA(const int i, tagint& nmol_loc){if (nmol_loc < d_molecule(i)) nmol_loc = d_molecule(i);},
                          Kokkos::Max<tagint>(nmol));

  //highest mol ID in the entire sim (across all ranks) will be stored in nm
  MPI_Allreduce(&nmol,&nm,1,MPI_LMP_TAGINT,MPI_MAX,world);

  if (init_on == 0) {
    memoryKK->destroy_kokkos(k_h_area,h_area);
    memoryKK->create_kokkos(k_h_area,h_area,nm,"angle:h_area");
    d_h_area = k_h_area.template view<DeviceType>();
    memoryKK->destroy_kokkos(k_h_volume,h_volume);
    memoryKK->create_kokkos(k_h_volume,h_volume,nm,"angle:h_volume");
    d_h_volume = k_h_volume.template view<DeviceType>();
    memoryKK->destroy_kokkos(k_ttyp,ttyp);
    memoryKK->create_kokkos(k_ttyp,ttyp,nm,"angle:ttyp");
    d_ttyp = k_ttyp.template view<DeviceType>();
    memoryKK->destroy_kokkos(k_ttyp1,ttyp1);
    memoryKK->create_kokkos(k_ttyp1,ttyp1,nm,"angle:ttyp1");
    d_ttyp1 = k_ttyp1.template view<DeviceType>();
    memoryKK->destroy_kokkos(k_dath,dath);
    memoryKK->create_kokkos(k_dath,dath,2*nm,"angle:dath");
    d_dath = k_dath.template view<DeviceType>();
    memoryKK->destroy_kokkos(k_datt,datt);
    memoryKK->create_kokkos(k_datt,datt,2*nm,"angle:datt");
    d_datt = k_datt.template view<DeviceType>();

    Kokkos::deep_copy(d_ttyp1, 0);
    Kokkos::deep_copy(d_ttyp, 0);

    atomKK->sync(execution_space, MOLECULE_MASK | ANGLE_MASK);
    const int nlocal_ = nlocal;
    d_molecule = molecule; 
    auto d_anglelist = anglelist; 
    auto d_d_ttyp1 = d_ttyp1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,nanglelist),
                         KOKKOS_LAMBDA(int n){const int i1 = d_anglelist(n,0);
                                              const int i2 = d_anglelist(n,1);
                                              const int i3 = d_anglelist(n,2);
                                              int m = -1;
                                              if(i1<nlocal_) m = d_molecule(i1)-1;
                                              if(i2<nlocal_) m = d_molecule(i2)-1;
                                              if(i3<nlocal_) m = d_molecule(i3)-1;
                                              if (m >= 0) Kokkos::atomic_max(&d_d_ttyp1(m), d_anglelist(n,3));
                                              });

    k_ttyp1.template modify<DeviceType>();                                          
    k_ttyp1.template sync<LMPHostType>();

    init_on = 1;
    MPI_Allreduce(ttyp1,ttyp,nm,MPI_INT,MPI_MAX,world);
  }

  Kokkos::deep_copy(d_h_area, 0);
  Kokkos::deep_copy(d_h_volume, 0);
  Kokkos::deep_copy(d_datt, 0);
  Kokkos::deep_copy(d_dath, 0);

  // auto prd = Few<double,3>(domain->prd);
  // auto h = Few<double,6>(domain->h);
  // auto triclinic = domain->triclinic;

  copymode = 1;

  if (newton_bond)
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagAngleAreaVolume3LocalLoop<1> >(0,nanglelist),*this);
  else
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagAngleAreaVolume3LocalLoop<0> >(0,nanglelist),*this);

  copymode = 0;

  k_datt.template modify<DeviceType>();
  k_datt.template sync<LMPHostType>();
  k_dath.template modify<DeviceType>();
  k_dath.template sync<LMPHostType>();
  k_h_area.template modify<DeviceType>();
  k_h_area.template sync<LMPHostType>();
  k_h_volume.template modify<DeviceType>();
  k_h_volume.template sync<LMPHostType>();

  MPI_Allreduce(dath,datt,2*nm,MPI_DOUBLE,MPI_SUM,world);

  F_FLOAT voltemp, tempxxx, energy_a, energy_v;
  ntimestep = update->ntimestep;
  energy_a = 0;
  energy_v = 0;
  if (eflag) {
    for (int m = 0; m < nm; m++) {
      const int type = ttyp[m];

      if(ntimestep < nstep1[type]) {
        voltemp = v0[type];
      }
      if(ntimestep>=nstep1[type] && ntimestep<=nstep2[type]) {
        tempxxx=(ntimestep-nstep1[type])*1.0/((nstep2[type]-nstep1[type])*1.0);
        voltemp = v0[type]*(1.0+tempxxx*(vratio[type]-1.0));
      }
      if(ntimestep > nstep2[type]) {
        voltemp=v0[type]*vratio[type];
      }

      energy_a += 0.5*ka[type]*(a0[type]-datt[m])*(a0[type]-datt[m])/a0[type];
      energy_v += 0.5*kv[type]*(voltemp-datt[m+nm])*(voltemp-datt[m+nm])/voltemp;

      if(abs(a0[type]) > 0) tempxxx=ka[type]*(a0[type]-datt[m])/a0[type];

      if(comm->me==0) {
        printf("mol: %d/%d, area is %f, volume is %f, desired is %f %f, tension is %f, in step %d\n", 
                m+1,nm, datt[m], datt[m+nm], a0[type], voltemp, tempxxx, ntimestep);
      }
    }
  }

  copymode = 1;

  EV_FLOAT ev;
  if (evflag) {
    if (newton_bond) {
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagAngleAreaVolume3Compute<1,1> >(0,nanglelist),*this,ev);
    } else {
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagAngleAreaVolume3Compute<0,1> >(0,nanglelist),*this,ev);
    }
  } else {
    if (newton_bond) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagAngleAreaVolume3Compute<1,0> >(0,nanglelist),*this);
    } else {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagAngleAreaVolume3Compute<0,0> >(0,nanglelist),*this);
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

template<class DeviceType>
template<int NEWTON_BOND>
KOKKOS_INLINE_FUNCTION
void AngleAreaVolume3Kokkos<DeviceType>::operator()(TagAngleAreaVolume3LocalLoop<NEWTON_BOND>, const int &n) const {

  //make views atomic
  Kokkos::View<F_FLOAT*, typename DAT::t_ffloat_1d::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > a_datt = d_datt;
  Kokkos::View<F_FLOAT*, typename DAT::t_ffloat_1d::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > a_dath = d_dath;
  Kokkos::View<F_FLOAT*, typename DAT::t_ffloat_1d::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > a_h_area = d_h_area;
  Kokkos::View<F_FLOAT*, typename DAT::t_ffloat_1d::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > a_h_volume = d_h_volume;

  const int i1 = anglelist(n,0);
  const int i2 = anglelist(n,1);
  const int i3 = anglelist(n,2);
  const int type = anglelist(n,3);

  int m;
  if(i1<nlocal) m = molecule(i1)-1;
  if(i2<nlocal) m = molecule(i2)-1;
  if(i3<nlocal) m = molecule(i3)-1;

  const X_FLOAT d21x = x(i2,0) - x(i1,0);
  const X_FLOAT d21y = x(i2,1) - x(i1,1);
  const X_FLOAT d21z = x(i2,2) - x(i1,2);

  // 3-1 distance
  const X_FLOAT d31x = x(i3,0) - x(i1,0);
  const X_FLOAT d31y = x(i3,1) - x(i1,1);
  const X_FLOAT d31z = x(i3,2) - x(i1,2);

  // calculate normal
  const X_FLOAT nx = d21y*d31z - d31y*d21z;
  const X_FLOAT ny = d31x*d21z - d21x*d31z;
  const X_FLOAT nz = d21x*d31y - d31x*d21y;
  const X_FLOAT nn = sqrt(nx*nx + ny*ny + nz*nz);

  // X_FLOAT xx1[3], xx2[3], xx3[3];
  Few<X_FLOAT,3> xx1, xx2, xx3;
  // calculate center
  for (int j = 0; j < 3; j++) {
    xx1[j] = x(i1,j);
    xx2[j] = x(i2,j);
    xx3[j] = x(i3,j); 
  }

  // auto prd = Few<double,3>(domain->prd);
  // auto h = Few<double,6>(domain->h);
  // auto triclinic = domain->triclinic;
  if(i1<nlocal) {
    xx1 = DomainKokkos::unmap(prd,h,triclinic,xx1,image(i1));
    xx2 = DomainKokkos::unmap(prd,h,triclinic,xx2,image(i1));
    xx3 = DomainKokkos::unmap(prd,h,triclinic,xx3,image(i1));
  }
  else {
    if(i2<nlocal) {
      xx1 = DomainKokkos::unmap(prd,h,triclinic,xx1,image(i2));
      xx2 = DomainKokkos::unmap(prd,h,triclinic,xx2,image(i2));
      xx3 = DomainKokkos::unmap(prd,h,triclinic,xx3,image(i2));
    }
    else {
      if(i3<nlocal) {
        xx1 = DomainKokkos::unmap(prd,h,triclinic,xx1,image(i3));
        xx2 = DomainKokkos::unmap(prd,h,triclinic,xx2,image(i3));
        xx3 = DomainKokkos::unmap(prd,h,triclinic,xx3,image(i3));
      }
    }
  } 

  const X_FLOAT mx =  xx1[0] + xx2[0] + xx3[0];
  const X_FLOAT my =  xx1[1] + xx2[1] + xx3[1];
  const X_FLOAT mz =  xx1[2] + xx2[2] + xx3[2];
    
  // calculate area and volume
  const F_FLOAT aa = 0.5*nn;
  const F_FLOAT vv = (nx*mx + ny*my + nz*mz)/18.0;

  if (NEWTON_BOND) {
    a_dath(m) += aa;
    a_dath(m+nm) += vv;
    a_h_area(m) += aa;
    a_h_volume(m) += vv;
  }
  else {
    if(i1<nlocal && i2<nlocal && i3<nlocal) {
      a_dath(m) += aa;
      a_dath(m+nm) += vv;
      a_h_area(m) += aa;
      a_h_volume(m) += vv;
    }
    else {
      if(i1<nlocal && tag[i1]<= tag[i2] && tag[i1]<=tag[i3]) {
        a_dath(m) += aa;
        a_dath(m+nm) += vv;
        a_h_area(m) += aa;
        a_h_volume(m) += vv;
      }
      if(i2<nlocal && tag[i2]<= tag[i1] && tag[i2]<=tag[i3]) {
          a_dath(m) += aa;
          a_dath(m+nm) += vv;
          a_h_area(m) += aa;
          a_h_volume(m) += vv;
      }
      if(i3<nlocal && tag[i3]<= tag[i1] && tag[i3]<=tag[i2]) {
        a_dath(m) += aa;
        a_dath(m+nm) += vv;
        a_h_area(m) += aa;
        a_h_volume(m) += vv;
      }
    }
  } 
}

template<class DeviceType>
template<int NEWTON_BOND, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void AngleAreaVolume3Kokkos<DeviceType>::operator()(TagAngleAreaVolume3Compute<NEWTON_BOND,EVFLAG>, const int &n, EV_FLOAT& ev) const {

  // The f array is atomic
  Kokkos::View<F_FLOAT*[3], typename DAT::t_f_array::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > a_f = f;

  const int i1 = anglelist(n,0);
  const int i2 = anglelist(n,1);
  const int i3 = anglelist(n,2);
  const int type = anglelist(n,3);

  int m;
  if(i1<nlocal) m = molecule(i1)-1;
  if(i2<nlocal) m = molecule(i2)-1;
  if(i3<nlocal) m = molecule(i3)-1;

  //Initial local area
  // 2-1 distance
  X_FLOAT d21x = x0(i2,0) - x0(i1,0);
  X_FLOAT d21y = x0(i2,1) - x0(i1,1);
  X_FLOAT d21z = x0(i2,2) - x0(i1,2);

  // 3-1 distance
  X_FLOAT d31x = x0(i3,0) - x0(i1,0);
  X_FLOAT d31y = x0(i3,1) - x0(i1,1);
  X_FLOAT d31z = x0(i3,2) - x0(i1,2);

  // calculate normal
  X_FLOAT nx = d21y*d31z - d31y*d21z;
  X_FLOAT ny = d31x*d21z - d21x*d31z;
  X_FLOAT nz = d21x*d31y - d31x*d21y;
  X_FLOAT nn = sqrt(nx*nx + ny*ny + nz*nz);
  const X_FLOAT ar0 = 0.5*nn;

  // 2-1 distance
  d21x = x(i2,0) - x(i1,0);
  d21y = x(i2,1) - x(i1,1);
  d21z = x(i2,2) - x(i1,2);

  // 3-1 distance
  d31x = x(i3,0) - x(i1,0);
  d31y = x(i3,1) - x(i1,1);
  d31z = x(i3,2) - x(i1,2);

  // 3-2 distance
  X_FLOAT d32x = x(i3,0) - x(i2,0);
  X_FLOAT d32y = x(i3,1) - x(i2,1);
  X_FLOAT d32z = x(i3,2) - x(i2,2);
    
  // calculate normal
  nx = d21y*d31z - d31y*d21z;
  ny = d31x*d21z - d21x*d31z;
  nz = d21x*d31y - d31x*d21y;
  nn = sqrt(nx*nx + ny*ny + nz*nz);

  // calculate center
  // X_FLOAT xx1[3], xx2[3], xx3[3];
  Few<X_FLOAT,3> xx1, xx2, xx3;
  for (int j = 0; j < 3; j++) {
    xx1[j] = x(i1,j);
    xx2[j] = x(i2,j);
    xx3[j] = x(i3,j); 
  }

  if(i1<nlocal) {
    xx1 = DomainKokkos::unmap(prd,h,triclinic,xx1,image(i1));
    xx2 = DomainKokkos::unmap(prd,h,triclinic,xx2,image(i1));
    xx3 = DomainKokkos::unmap(prd,h,triclinic,xx3,image(i1));
  }
  else {
    if(i2<nlocal) {
      xx1 = DomainKokkos::unmap(prd,h,triclinic,xx1,image(i2));
      xx2 = DomainKokkos::unmap(prd,h,triclinic,xx2,image(i2));
      xx3 = DomainKokkos::unmap(prd,h,triclinic,xx3,image(i2));
    }
    else {
      if(i3<nlocal) {
        xx1 = DomainKokkos::unmap(prd,h,triclinic,xx1,image(i3));
        xx2 = DomainKokkos::unmap(prd,h,triclinic,xx2,image(i3));
        xx3 = DomainKokkos::unmap(prd,h,triclinic,xx3,image(i3));
      }
    }
  } 

  const X_FLOAT mx =  xx1[0] + xx2[0] + xx3[0];
  const X_FLOAT my =  xx1[1] + xx2[1] + xx3[1];
  const X_FLOAT mz =  xx1[2] + xx2[2] + xx3[2];

  const F_FLOAT press = d_press[type];
  const F_FLOAT ka = d_ka[type];
  const F_FLOAT a0 = d_a0[type];
  const F_FLOAT kv = d_kv[type];
  const F_FLOAT v0 = d_v0[type];
  const F_FLOAT kl = d_kl[type];
  const F_FLOAT aa0 = d_aa0[type];
  const F_FLOAT r0 = d_r0[type];
  const F_FLOAT qp = d_qp[type];
  const F_FLOAT vratio = d_vratio[type];
  const F_FLOAT nstep1 = d_nstep1[type];
  const F_FLOAT nstep2 = d_nstep2[type];
  // const F_FLOAT ttyp = d_ttyp[type];
  // const F_FLOAT ttyp1 = d_ttyp1[type];

  F_FLOAT voltemp, tempxxx;

  if(ntimestep < nstep1) {
    voltemp = v0;
  }

  if(ntimestep >= nstep1 && ntimestep <= nstep2) {
    tempxxx = (ntimestep-nstep1)*1.0/((nstep2-nstep1)*1.0);
    voltemp = v0*(1.0+tempxxx*(vratio-1.0));
  }

  if(ntimestep > nstep2) {
    voltemp = v0*vratio;
  }

  const F_FLOAT coefc = 0.25*press*qp / pow(0.5*nn,qp+2);
  const F_FLOAT coefl = 0.5*kl*(ar0-0.5*nn)/ar0/nn; 
  const F_FLOAT coefa = 0.5*ka*(a0-datt[m])/a0/nn;
  const F_FLOAT coefca = coefc + coefl + coefa;        
  const F_FLOAT coefv = kv*(voltemp-datt[m+nm])/voltemp/18.0;

  // force & energy

  E_FLOAT eangle = 0.0;
  E_FLOAT energy_l = 0.0;
  if (eflag) {
    energy_l += 0.5*kl*(ar0-0.5*nn)*(ar0-0.5*nn)/ar0;
    eangle = 0.5*kl*(ar0-0.5*nn)*(ar0-0.5*nn)/ar0;
  }

  const F_FLOAT s1x = coefca*(ny*d32z - nz*d32y);
  const F_FLOAT s1y = coefca*(nz*d32x - nx*d32z);    
  const F_FLOAT s1z = coefca*(nx*d32y - ny*d32x);
    
  const F_FLOAT s2x = coefca*(nz*d31y - ny*d31z);
  const F_FLOAT s2y = coefca*(nx*d31z - nz*d31x);
  const F_FLOAT s2z = coefca*(ny*d31x - nx*d31y);
   
  const F_FLOAT s3x = coefca*(ny*d21z - nz*d21y);
  const F_FLOAT s3y = coefca*(nz*d21x - nx*d21z);
  const F_FLOAT s3z = coefca*(nx*d21y - ny*d21x);

  const F_FLOAT s1xv = coefv*(nx);
  const F_FLOAT s1yv = coefv*(ny);
  const F_FLOAT s1zv = coefv*(nz);

  const F_FLOAT s2xv = coefv*(nx);
  const F_FLOAT s2yv = coefv*(ny);
  const F_FLOAT s2zv = coefv*(nz);

  const F_FLOAT s3xv = coefv*(nx);
  const F_FLOAT s3yv = coefv*(ny);
  const F_FLOAT s3zv = coefv*(nz);

  
  F_FLOAT f1[3], f2[3], f3[3];
  f1[0] = s1x + s1xv;
  f1[1] = s1y + s1yv;
  f1[2] = s1z + s1zv;

  f2[0] = s2x + s2xv;
  f2[1] = s2y + s2yv;
  f2[2] = s2z + s2zv;

  f3[0] = s3x+s3xv;
  f3[1] = s3y+s3yv;
  f3[2] = s3z+s3zv;

  F_FLOAT f1x[3], f2x[3], f3x[3];
  f1x[0] = s1x;
  f1x[1] = s1y;
  f1x[2] = s1z;
 
  f2x[0] = s2x;
  f2x[1] = s2y;
  f2x[2] = s2z;

  f3x[0] = s3x;
  f3x[1] = s3y;
  f3x[2] = s3z; 

  // apply force to each of 3 atoms

  if (NEWTON_BOND || i1 < nlocal) {
    a_f(i1,0) += f1[0];
    a_f(i1,1) += f1[1];
    a_f(i1,2) += f1[2];
  }

  if (NEWTON_BOND || i2 < nlocal) {
    a_f(i2,0) += f2[0];  //s2x+coefv*nx;
    a_f(i2,1) += f2[1];  //s2y+coefv*ny;
    a_f(i2,2) += f2[2];  //s2z+coefv*nz;
  }
  if (NEWTON_BOND || i3 < nlocal) {
    a_f(i3,0) += f3[0];
    a_f(i3,1) += f3[1];
    a_f(i3,2) += f3[2];
  }

  F_FLOAT ff[6];
  F_FLOAT vv = 0.0; //2.0*datt[m+nm]*coefv/n_atoms[m];
  ff[0] = d21x*s2x + d31x*s3x + (d21x*(s2xv-s1xv)+d31x*(s3xv-s1xv)+d32x*(s3xv-s2xv))/3.0 + vv;
  ff[1] = d21y*s2y + d31y*s3y + (d21y*(s2yv-s1yv)+d31y*(s3yv-s1yv)+d32y*(s3yv-s2yv))/3.0 + vv;
  ff[2] = d21z*s2z + d31z*s3z + (d21z*(s2zv-s1zv)+d31z*(s3zv-s1zv)+d32z*(s3zv-s2zv))/3.0 + vv;
  ff[3] = d21x*s2y + d31x*s3y + (d21x*(s2yv-s1yv)+d31x*(s3yv-s1yv)+d32x*(s3yv-s2yv))/3.0;
  ff[4] = d21x*s2z + d31x*s3z + (d21x*(s2zv-s1zv)+d31x*(s3zv-s1zv)+d32x*(s3zv-s2zv))/3.0;
  ff[5] = d21y*s2z + d31y*s3z + (d21y*(s2zv-s1zv)+d31y*(s3zv-s1zv)+d32y*(s3zv-s2zv))/3.0;

  const X_FLOAT d12x = -d21x;
  const X_FLOAT d12y = -d21y;
  const X_FLOAT d12z = -d21z;

  if (EVFLAG) ev_tally(ev,i1,i2,i3,eangle,f1,f3,
                       d12x,d12y,d12z,d32x,d32y,d32z);

}

template<class DeviceType>
template<int NEWTON_BOND, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void AngleAreaVolume3Kokkos<DeviceType>::operator()(TagAngleAreaVolume3Compute<NEWTON_BOND,EVFLAG>, const int &n) const {
  EV_FLOAT ev;
  this->template operator()<NEWTON_BOND,EVFLAG>(TagAngleAreaVolume3Compute<NEWTON_BOND,EVFLAG>(), n, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void AngleAreaVolume3Kokkos<DeviceType>::allocate()
{ 
  AngleAreaVolume3::allocate();

  int n = atom->nangletypes;

  k_ka = typename ArrayTypes<DeviceType>::tdual_ffloat_1d("AngleAreaVolume3::ka",n+1);
  d_ka = k_ka.template view<DeviceType>();
  k_a0 = typename ArrayTypes<DeviceType>::tdual_ffloat_1d("AngleAreaVolume3::a0",n+1);
  d_a0 = k_a0.template view<DeviceType>();
  k_kv = typename ArrayTypes<DeviceType>::tdual_ffloat_1d("AngleAreaVolume3::kv",n+1);
  d_kv = k_kv.template view<DeviceType>();
  k_v0 = typename ArrayTypes<DeviceType>::tdual_ffloat_1d("AngleAreaVolume3::v0",n+1);
  d_v0 = k_v0.template view<DeviceType>();
  k_kl = typename ArrayTypes<DeviceType>::tdual_ffloat_1d("AngleAreaVolume3::kl",n+1);
  d_kl = k_kl.template view<DeviceType>();
  k_aa0 = typename ArrayTypes<DeviceType>::tdual_ffloat_1d("AngleAreaVolume3::aa0",n+1);
  d_aa0 = k_aa0.template view<DeviceType>();
  k_press = typename ArrayTypes<DeviceType>::tdual_ffloat_1d("AngleAreaVolume3::press",n+1);
  d_press = k_press.template view<DeviceType>();
  k_qp = typename ArrayTypes<DeviceType>::tdual_ffloat_1d("AngleAreaVolume3::qp",n+1);
  d_qp = k_qp.template view<DeviceType>();
  k_r0 = typename ArrayTypes<DeviceType>::tdual_ffloat_1d("AngleAreaVolume3::r0",n+1);
  d_r0 = k_r0.template view<DeviceType>();
  k_nstep1 = typename ArrayTypes<DeviceType>::tdual_int_1d("AngleAreaVolume3::nstep1",n+1);
  d_nstep1 = k_nstep1.template view<DeviceType>();
  k_nstep2 = typename ArrayTypes<DeviceType>::tdual_int_1d("AngleAreaVolume3::nstep2",n+1);
  d_nstep2 = k_nstep2.template view<DeviceType>();
  k_vratio = typename ArrayTypes<DeviceType>::tdual_ffloat_1d("AngleAreaVolume3::vratio",n+1);
  d_vratio = k_vratio.template view<DeviceType>();
  k_setflag = typename ArrayTypes<DeviceType>::tdual_int_1d("AngleAreaVolume3::setflag",n+1);
  d_setflag = k_setflag.template view<DeviceType>();

}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

template<class DeviceType>
void AngleAreaVolume3Kokkos<DeviceType>::coeff(int narg, char **arg)
{
  AngleAreaVolume3::coeff(narg, arg);

  int n = atom->nangletypes;
  for (int i = 1; i <= n; i++) {
    k_ka.h_view[i] = ka[i];
    k_a0.h_view[i] = a0[i];
    k_kv.h_view[i] = kv[i];
    k_v0.h_view[i] = v0[i];
    k_kl.h_view[i] = kl[i];
    k_aa0.h_view[i] = aa0[i];
    k_press.h_view[i] = press[i];
    k_qp.h_view[i] = qp[i];
    k_r0.h_view[i] = r0[i];
    k_nstep1.h_view[i] = nstep1[i];
    k_nstep2.h_view[i] = nstep2[i];
    k_vratio.h_view[i] = vratio[i];
    k_setflag.h_view[i] = setflag[i];
  }
  
  k_ka.template modify<LMPHostType>();
  k_a0.template modify<LMPHostType>();
  k_kv.template modify<LMPHostType>();
  k_v0.template modify<LMPHostType>();
  k_kl.template modify<LMPHostType>();
  k_aa0.template modify<LMPHostType>();
  k_press.template modify<LMPHostType>();
  k_qp.template modify<LMPHostType>();
  k_r0.template modify<LMPHostType>();
  k_nstep1.template modify<LMPHostType>();
  k_nstep2.template modify<LMPHostType>();
  k_vratio.template modify<LMPHostType>();
  k_setflag.template modify<LMPHostType>();

}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

template<class DeviceType>
void AngleAreaVolume3Kokkos<DeviceType>::read_restart(FILE *fp)
{
  AngleAreaVolume3::read_restart(fp);

  int n = atom->nangletypes;
  for (int i = 1; i <= n; i++) {
    k_ka.h_view[i] = ka[i];
    k_a0.h_view[i] = a0[i];
    k_kv.h_view[i] = kv[i];
    k_v0.h_view[i] = v0[i];
    k_kl.h_view[i] = kl[i];
    k_aa0.h_view[i] = aa0[i];
    k_press.h_view[i] = press[i];
    k_qp.h_view[i] = qp[i];
    k_r0.h_view[i] = r0[i];
    k_nstep1.h_view[i] = nstep1[i];
    k_nstep2.h_view[i] = nstep2[i];
    k_vratio.h_view[i] = vratio[i];
    k_setflag.h_view[i] = setflag[i];
  }
  
  k_ka.template modify<LMPHostType>();
  k_a0.template modify<LMPHostType>();
  k_kv.template modify<LMPHostType>();
  k_v0.template modify<LMPHostType>();
  k_kl.template modify<LMPHostType>();
  k_aa0.template modify<LMPHostType>();
  k_press.template modify<LMPHostType>();
  k_qp.template modify<LMPHostType>();
  k_r0.template modify<LMPHostType>();
  k_nstep1.template modify<LMPHostType>();
  k_nstep2.template modify<LMPHostType>();
  k_vratio.template modify<LMPHostType>();
  k_setflag.template modify<LMPHostType>();
  
}

/* ----------------------------------------------------------------------
   tally energy and virial into global and per-atom accumulators
   virial = r1F1 + r2F2 + r3F3 = (r1-r2) F1 + (r3-r2) F3 = del1*f1 + del2*f3
------------------------------------------------------------------------- */

template<class DeviceType>
//template<int NEWTON_BOND>
KOKKOS_INLINE_FUNCTION
void AngleAreaVolume3Kokkos<DeviceType>::ev_tally(EV_FLOAT &ev, const int i, const int j, const int k,
                     F_FLOAT &eangle, F_FLOAT *f1, F_FLOAT *f3,
                     const F_FLOAT &delx1, const F_FLOAT &dely1, const F_FLOAT &delz1,
                     const F_FLOAT &delx2, const F_FLOAT &dely2, const F_FLOAT &delz2) const
{
  E_FLOAT eanglethird;
  F_FLOAT v[6];

  // The eatom and vatom arrays are atomic
  Kokkos::View<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > v_eatom = k_eatom.template view<DeviceType>();
  Kokkos::View<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > v_vatom = k_vatom.template view<DeviceType>();

  if (eflag_either) {
    if (eflag_global) {
      if (newton_bond) ev.evdwl += eangle;
      else {
        eanglethird = THIRD*eangle;

        if (i < nlocal) ev.evdwl += eanglethird;
        if (j < nlocal) ev.evdwl += eanglethird;
        if (k < nlocal) ev.evdwl += eanglethird;
      }
    }
    if (eflag_atom) {
      eanglethird = THIRD*eangle;

      if (newton_bond || i < nlocal) v_eatom[i] += eanglethird;
      if (newton_bond || j < nlocal) v_eatom[j] += eanglethird;
      if (newton_bond || k < nlocal) v_eatom[k] += eanglethird;
    }
  }

  if (vflag_either) {
    v[0] = delx1*f1[0] + delx2*f3[0];
    v[1] = dely1*f1[1] + dely2*f3[1];
    v[2] = delz1*f1[2] + delz2*f3[2];
    v[3] = delx1*f1[1] + delx2*f3[1];
    v[4] = delx1*f1[2] + delx2*f3[2];
    v[5] = dely1*f1[2] + dely2*f3[2];

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
          ev.v[0] += THIRD*v[0];
          ev.v[1] += THIRD*v[1];
          ev.v[2] += THIRD*v[2];
          ev.v[3] += THIRD*v[3];
          ev.v[4] += THIRD*v[4];
          ev.v[5] += THIRD*v[5];
        }
        if (j < nlocal) {
          ev.v[0] += THIRD*v[0];
          ev.v[1] += THIRD*v[1];
          ev.v[2] += THIRD*v[2];
          ev.v[3] += THIRD*v[3];
          ev.v[4] += THIRD*v[4];
          ev.v[5] += THIRD*v[5];
        }
        if (k < nlocal) {
          ev.v[0] += THIRD*v[0];

          ev.v[1] += THIRD*v[1];
          ev.v[2] += THIRD*v[2];
          ev.v[3] += THIRD*v[3];
          ev.v[4] += THIRD*v[4];
          ev.v[5] += THIRD*v[5];
        }
      }
    }

    if (vflag_atom) {
      if (newton_bond || i < nlocal) {
        v_vatom(i,0) += THIRD*v[0];
        v_vatom(i,1) += THIRD*v[1];
        v_vatom(i,2) += THIRD*v[2];
        v_vatom(i,3) += THIRD*v[3];
        v_vatom(i,4) += THIRD*v[4];
        v_vatom(i,5) += THIRD*v[5];
      }
      if (newton_bond || j < nlocal) {
        v_vatom(j,0) += THIRD*v[0];
        v_vatom(j,1) += THIRD*v[1];
        v_vatom(j,2) += THIRD*v[2];
        v_vatom(j,3) += THIRD*v[3];
        v_vatom(j,4) += THIRD*v[4];
        v_vatom(j,5) += THIRD*v[5];
      }
      if (newton_bond || k < nlocal) {
        v_vatom(k,0) += THIRD*v[0];
        v_vatom(k,1) += THIRD*v[1];
        v_vatom(k,2) += THIRD*v[2];
        v_vatom(k,3) += THIRD*v[3];
        v_vatom(k,4) += THIRD*v[4];
        v_vatom(k,5) += THIRD*v[5];

      }
    }
  }
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class AngleAreaVolume3Kokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class AngleAreaVolume3Kokkos<LMPHostType>;
#endif
}

