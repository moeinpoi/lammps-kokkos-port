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

#include "atom_vec_sph_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm_kokkos.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "math_const.h"
#include "memory.h"
#include "memory_kokkos.h"
#include "modify.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

AtomVecSPHKokkos::AtomVecSPHKokkos(LAMMPS *lmp) : AtomVec(lmp),
AtomVecKokkos(lmp), AtomVecSPH(lmp)
{
  no_border_vel_flag = 0;
  unpack_exchange_indices_flag = 1;
  comm_f_only = 0; 
  no_comm_vel_flag = 0;
}

/* ----------------------------------------------------------------------
   grow atom arrays
   n = 0 grows arrays by a chunk
   n > 0 allocates arrays to size n
------------------------------------------------------------------------- */

void AtomVecSPHKokkos::grow(int n)
{
	fprintf(screen, "grow called on AtomVecSPHKokkos\n");
  auto DELTA = LMP_KOKKOS_AV_DELTA;
  int step = MAX(DELTA,nmax*0.01);
  if (n == 0) nmax += step;
  else nmax = n;
  atom->nmax = nmax;
  if (nmax < 0 || nmax > MAXSMALLINT)
    error->one(FLERR,"Per-processor system is too big");

  atomKK->sync(Device,ALL_MASK);
  atomKK->modified(Device,ALL_MASK);

  memoryKK->grow_kokkos(atomKK->k_tag,atomKK->tag,nmax,"atom:tag");
  memoryKK->grow_kokkos(atomKK->k_type,atomKK->type,nmax,"atom:type");
  memoryKK->grow_kokkos(atomKK->k_mask,atomKK->mask,nmax,"atom:mask");
  memoryKK->grow_kokkos(atomKK->k_image,atomKK->image,nmax,"atom:image");
  memoryKK->grow_kokkos(atomKK->k_x,atomKK->x,nmax,"atom:x");
  memoryKK->grow_kokkos(atomKK->k_v,atomKK->v,nmax,"atom:v");
  memoryKK->grow_kokkos(atomKK->k_f,atomKK->f,nmax,"atom:f");

  memoryKK->grow_kokkos(atomKK->k_vest,atomKK->vest,nmax,"atom:vest");
  memoryKK->grow_kokkos(atomKK->k_rho,atomKK->rho,nmax,"atom:rho");
  memoryKK->grow_kokkos(atomKK->k_drho,atomKK->drho,nmax,"atom:drho");
  memoryKK->grow_kokkos(atomKK->k_esph,atomKK->esph,nmax,"atom:esph");
  memoryKK->grow_kokkos(atomKK->k_desph,atomKK->desph,nmax,"atom:desph");
  memoryKK->grow_kokkos(atomKK->k_cv,atomKK->cv,nmax,"atom:cv");

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->grow_arrays(nmax);

  grow_pointers();
  atomKK->sync(Host,ALL_MASK);
}

/* ----------------------------------------------------------------------
   reset local array ptrs
------------------------------------------------------------------------- */

void AtomVecSPHKokkos::grow_pointers()
{
	fprintf(screen, "grow_pointers called on AtomVecSPHKokkos\n");
  tag = atomKK->tag;
  d_tag = atomKK->k_tag.d_view;
  h_tag = atomKK->k_tag.h_view;

  type = atomKK->type;
  d_type = atomKK->k_type.d_view;
  h_type = atomKK->k_type.h_view;
  mask = atomKK->mask;
  d_mask = atomKK->k_mask.d_view;
  h_mask = atomKK->k_mask.h_view;
  image = atomKK->image;
  d_image = atomKK->k_image.d_view;
  h_image = atomKK->k_image.h_view;

  x = atomKK->x;
  d_x = atomKK->k_x.d_view;
  h_x = atomKK->k_x.h_view;
  v = atomKK->v;
  d_v = atomKK->k_v.d_view;
  h_v = atomKK->k_v.h_view;
  f = atomKK->f;
  d_f = atomKK->k_f.d_view;
  h_f = atomKK->k_f.h_view;
  
  vest = atomKK->vest;
  d_vest = atomKK->k_vest.d_view;
  h_vest = atomKK->k_vest.h_view;
  rho = atomKK->rho;
  d_rho = atomKK->k_rho.d_view;
  h_rho = atomKK->k_rho.h_view;
  drho = atomKK->drho;
  d_drho = atomKK->k_drho.d_view;
  h_drho = atomKK->k_drho.h_view;
  esph = atomKK->esph;
  d_esph = atomKK->k_esph.d_view;
  h_esph = atomKK->k_esph.h_view;
  desph = atomKK->desph;
  d_desph = atomKK->k_desph.d_view;
  h_desph = atomKK->k_desph.h_view;
  cv = atomKK->cv;
  d_cv = atomKK->k_cv.d_view;
  h_cv = atomKK->k_cv.h_view;  
}

/* ----------------------------------------------------------------------
   clear SPH force properties drho and desph
------------------------------------------------------------------------- */

void AtomVecSPHKokkos::force_clear_kokkos(int n, size_t nbytes)
{
	fprintf(screen, "force_clear_kokkos called on AtomVecSPHKokkos\n");
  int nzero = static_cast<int>(nbytes / sizeof(double));
  if (nzero == 0) return;

  atomKK->k_drho.clear_sync_state();
  atomKK->k_desph.clear_sync_state();

  auto d_drho = atomKK->k_drho.d_view;
  auto d_desph = atomKK->k_desph.d_view;

  Kokkos::parallel_for(
    Kokkos::RangePolicy<LMPDeviceType>(n, n + nzero),
    LAMMPS_LAMBDA(int i) {
      d_drho(i) = 0.0;
      d_desph(i) = 0.0;
    });

  atomKK->modified(Device, DRHO_MASK | DESPH_MASK);
}

/* ----------------------------------------------------------------------
   sort atom arrays on device
------------------------------------------------------------------------- */

void AtomVecSPHKokkos::sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter)
{
  atomKK->sync(Device, ALL_MASK & ~F_MASK);

  Sorter.sort(LMPDeviceType(), d_tag);
  Sorter.sort(LMPDeviceType(), d_type);
  Sorter.sort(LMPDeviceType(), d_mask);
  Sorter.sort(LMPDeviceType(), d_image);
  Sorter.sort(LMPDeviceType(), d_x);
  Sorter.sort(LMPDeviceType(), d_v);

  Sorter.sort(LMPDeviceType(), d_vest);
  Sorter.sort(LMPDeviceType(), d_rho);
  Sorter.sort(LMPDeviceType(), d_drho);
  Sorter.sort(LMPDeviceType(), d_esph);
  Sorter.sort(LMPDeviceType(), d_desph);
  Sorter.sort(LMPDeviceType(), d_cv);

  atomKK->modified(Device, ALL_MASK & ~F_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC>
struct AtomVecSPHKokkos_PackComm {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_x_array_randomread _x;

  typename ArrayTypes<DeviceType>::t_v_array _vest;
  typename ArrayTypes<DeviceType>::t_float_1d _rho,_esph;

  typename ArrayTypes<DeviceType>::t_xfloat_2d_um _buf;
  typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;
  X_FLOAT _xprd,_yprd,_zprd,_xy,_xz,_yz;
  X_FLOAT _pbc[6];

  AtomVecSPHKokkos_PackComm(
    const typename DAT::tdual_x_array &x,

    const typename DAT::tdual_v_array &vest,
    const typename DAT::tdual_float_1d &rho,
    const typename DAT::tdual_float_1d &esph,

    const typename DAT::tdual_xfloat_2d &buf,
    const typename DAT::tdual_int_2d &list,
    const int & iswap,
    const X_FLOAT &xprd, const X_FLOAT &yprd, const X_FLOAT &zprd,
    const X_FLOAT &xy, const X_FLOAT &xz, const X_FLOAT &yz, const int* const pbc):
    _x(x.view<DeviceType>()),

    _vest(vest.view<DeviceType>()),
    _rho(rho.view<DeviceType>()),
    _esph(esph.view<DeviceType>()),

    _list(list.view<DeviceType>()),_iswap(iswap),
    _xprd(xprd),_yprd(yprd),_zprd(zprd),
    _xy(xy),_xz(xz),_yz(yz) {
    const size_t elements = 8;
    const size_t maxsend = (buf.view<DeviceType>().extent(0)*buf.view<DeviceType>().extent(1))/elements;
    _buf = typename ArrayTypes<DeviceType>::t_xfloat_2d_um(buf.view<DeviceType>().data(),maxsend,elements);
    _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
    _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(_iswap,i);
    if (PBC_FLAG == 0) {
      _buf(i,0) = _x(j,0);
      _buf(i,1) = _x(j,1);
      _buf(i,2) = _x(j,2);
    } else {
      if (TRICLINIC == 0) {
        _buf(i,0) = _x(j,0) + _pbc[0]*_xprd;
        _buf(i,1) = _x(j,1) + _pbc[1]*_yprd;
        _buf(i,2) = _x(j,2) + _pbc[2]*_zprd;
      } else {
        _buf(i,0) = _x(j,0) + _pbc[0]*_xprd + _pbc[5]*_xy + _pbc[4]*_xz;
        _buf(i,1) = _x(j,1) + _pbc[1]*_yprd + _pbc[3]*_yz;
        _buf(i,2) = _x(j,2) + _pbc[2]*_zprd;
      }
    }
    _buf(i,3) = _rho(j);
    _buf(i,4) = _esph(j);
    _buf(i,5) = _vest(j,0);
    _buf(i,6) = _vest(j,1);
    _buf(i,7) = _vest(j,2);
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecSPHKokkos::pack_comm_kokkos(
  const int &n,
  const DAT::tdual_int_2d &list,
  const int & iswap,
  const DAT::tdual_xfloat_2d &buf,
  const int &pbc_flag,
  const int* const pbc)
{
  // Fallback to AtomVecKokkos if radvary == 0
//  if (radvary == 0)
//    return AtomVecKokkos::pack_comm_kokkos(n,list,iswap,buf,pbc_flag,pbc);  //RADVARY check is commented out because SPH doesn't need it. -Moein
  // Check whether to always run forward communication on the host
  // Choose correct forward PackComm kernel
  fprintf(screen, "pack_comm_kokkos called on AtomVecSPHKokkos\n");
  if (commKK->forward_comm_on_host) {
    atomKK->sync(Host,X_MASK|RHO_MASK|ESPH_MASK|VEST_MASK);
    if (pbc_flag) {
      if (domain->triclinic) {
        struct AtomVecSPHKokkos_PackComm<LMPHostType,1,1> f(
          atomKK->k_x,
          atomKK->k_vest,
          atomKK->k_rho,atomKK->k_esph,
          buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecSPHKokkos_PackComm<LMPHostType,1,0> f(
          atomKK->k_x,
          atomKK->k_vest,
          atomKK->k_rho,atomKK->k_esph,
          buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecSPHKokkos_PackComm<LMPHostType,0,1> f(
          atomKK->k_x,
          atomKK->k_vest,
          atomKK->k_rho,atomKK->k_esph,
          buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecSPHKokkos_PackComm<LMPHostType,0,0> f(
          atomKK->k_x,
          atomKK->k_vest,
          atomKK->k_rho,atomKK->k_esph,
          buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    }
  } else {
    atomKK->sync(Device,X_MASK|RHO_MASK|ESPH_MASK|VEST_MASK);
    if (pbc_flag) {
      if (domain->triclinic) {
        struct AtomVecSPHKokkos_PackComm<LMPDeviceType,1,1> f(
          atomKK->k_x,
          atomKK->k_vest,
          atomKK->k_rho,atomKK->k_esph,
          buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecSPHKokkos_PackComm<LMPDeviceType,1,0> f(
          atomKK->k_x,
          atomKK->k_vest,
          atomKK->k_rho,atomKK->k_esph,
          buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecSPHKokkos_PackComm<LMPDeviceType,0,1> f(
          atomKK->k_x,
          atomKK->k_vest,
          atomKK->k_rho,atomKK->k_esph,
          buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecSPHKokkos_PackComm<LMPDeviceType,0,0> f(
          atomKK->k_x,
          atomKK->k_vest,
          atomKK->k_rho,atomKK->k_esph,
          buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    }
  }
  return n*size_forward;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC,int DEFORM_VREMAP>
struct AtomVecSPHKokkos_PackCommVel {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_x_array_randomread _x;
  typename ArrayTypes<DeviceType>::t_int_1d _mask;
 
  typename ArrayTypes<DeviceType>::t_v_array _vest;
  typename ArrayTypes<DeviceType>::t_float_1d _rho,_esph;

  typename ArrayTypes<DeviceType>::t_v_array _v;
  typename ArrayTypes<DeviceType>::t_xfloat_2d_um _buf;
  typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;
  X_FLOAT _xprd,_yprd,_zprd,_xy,_xz,_yz;
  X_FLOAT _pbc[6];
  X_FLOAT _h_rate[6];
  const int _deform_vremap;

  AtomVecSPHKokkos_PackCommVel(
    const typename DAT::tdual_x_array &x,
    const typename DAT::tdual_int_1d &mask,

    const typename DAT::tdual_v_array &vest,
    const typename DAT::tdual_float_1d &rho,
    const typename DAT::tdual_float_1d &esph,

    const typename DAT::tdual_v_array &v,
    const typename DAT::tdual_xfloat_2d &buf,
    const typename DAT::tdual_int_2d &list,
    const int &iswap,
    const X_FLOAT &xprd, const X_FLOAT &yprd, const X_FLOAT &zprd,
    const X_FLOAT &xy, const X_FLOAT &xz, const X_FLOAT &yz, const int* const pbc,
    const double * const h_rate,
    const int &deform_vremap):
    _x(x.view<DeviceType>()),
    _mask(mask.view<DeviceType>()),

    _vest(vest.view<DeviceType>()),
    _rho(rho.view<DeviceType>()),
    _esph(esph.view<DeviceType>()),

    _v(v.view<DeviceType>()),
    _list(list.view<DeviceType>()),_iswap(iswap),
    _xprd(xprd),_yprd(yprd),_zprd(zprd),
    _xy(xy),_xz(xz),_yz(yz),
    _deform_vremap(deform_vremap)
  {
    const size_t elements = 11;
    const int maxsend = (buf.template view<DeviceType>().extent(0)*buf.template view<DeviceType>().extent(1))/elements;
    _buf = typename ArrayTypes<DeviceType>::t_xfloat_2d_um(buf.view<DeviceType>().data(),maxsend,elements);
    _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
    _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
    _h_rate[0] = h_rate[0]; _h_rate[1] = h_rate[1]; _h_rate[2] = h_rate[2];
    _h_rate[3] = h_rate[3]; _h_rate[4] = h_rate[4]; _h_rate[5] = h_rate[5];
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(_iswap,i);
    if (PBC_FLAG == 0) {
      _buf(i,0) = _x(j,0);
      _buf(i,1) = _x(j,1);
      _buf(i,2) = _x(j,2);
    } else {
      if (TRICLINIC == 0) {
        _buf(i,0) = _x(j,0) + _pbc[0]*_xprd;
        _buf(i,1) = _x(j,1) + _pbc[1]*_yprd;
        _buf(i,2) = _x(j,2) + _pbc[2]*_zprd;
      } else {
        _buf(i,0) = _x(j,0) + _pbc[0]*_xprd + _pbc[5]*_xy + _pbc[4]*_xz;
        _buf(i,1) = _x(j,1) + _pbc[1]*_yprd + _pbc[3]*_yz;
        _buf(i,2) = _x(j,2) + _pbc[2]*_zprd;
      }
    }
    if (DEFORM_VREMAP == 0) {
      _buf(i,3) = _v(j,0);
      _buf(i,4) = _v(j,1);
      _buf(i,5) = _v(j,2);
      _buf(i,6) = _vest(j,0);
      _buf(i,7) = _vest(j,1);
      _buf(i,8) = _vest(j,2);
    } else {
      if (_mask(i) & _deform_vremap) {
        _buf(i,3) = _v(j,0) + _pbc[0]*_h_rate[0] + _pbc[5]*_h_rate[5] + _pbc[4]*_h_rate[4];
        _buf(i,4) = _v(j,1) + _pbc[1]*_h_rate[1] + _pbc[3]*_h_rate[3];
        _buf(i,5) = _v(j,2) + _pbc[2]*_h_rate[2];
        _buf(i,6) = _vest(j,0) + _pbc[0]*_h_rate[0] + _pbc[5]*_h_rate[5] + _pbc[4]*_h_rate[4];
        _buf(i,7) = _vest(j,1) + _pbc[1]*_h_rate[1] + _pbc[3]*_h_rate[3];
        _buf(i,8) = _vest(j,2) + _pbc[2]*_h_rate[2];
      } else {
        _buf(i,3) = _v(j,0);
        _buf(i,4) = _v(j,1);
        _buf(i,5) = _v(j,2);
        _buf(i,6) = _vest(j,0);
        _buf(i,7) = _vest(j,1);
        _buf(i,8) = _vest(j,2);
      }
    }
    _buf(i,9) = _rho(j);
    _buf(i,10) = _esph(j);
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecSPHKokkos::pack_comm_vel_kokkos(
  const int &n,
  const DAT::tdual_int_2d &list,
  const int & iswap,
  const DAT::tdual_xfloat_2d &buf,
  const int &pbc_flag,
  const int* const pbc)
{
	fprintf(screen, "pack_comm_vel_kokkos called on AtomVecSPHKokkos\n");
  if (commKK->forward_comm_on_host) {
    atomKK->sync(Host,X_MASK|V_MASK|VEST_MASK|RHO_MASK|ESPH_MASK);
    if (pbc_flag) {
      if (deform_vremap) {
        if (domain->triclinic) {
            struct AtomVecSPHKokkos_PackCommVel<LMPHostType,1,1,1> f(
              atomKK->k_x,atomKK->k_mask,
              atomKK->k_vest,atomKK->k_rho,atomKK->k_esph,
              atomKK->k_v,
              buf,list,iswap,
              domain->xprd,domain->yprd,domain->zprd,
              domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
            Kokkos::parallel_for(n,f);
          }
          else {
            struct AtomVecSPHKokkos_PackCommVel<LMPHostType,1,0,1> f(
              atomKK->k_x,atomKK->k_mask,
              atomKK->k_vest,atomKK->k_rho,atomKK->k_esph,
              atomKK->k_v,
              buf,list,iswap,
              domain->xprd,domain->yprd,domain->zprd,
              domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
            Kokkos::parallel_for(n,f);       
        }
      } else {
        if (domain->triclinic) {    
            struct AtomVecSPHKokkos_PackCommVel<LMPHostType,1,1,0> f(
              atomKK->k_x,atomKK->k_mask,
              atomKK->k_vest,atomKK->k_rho,atomKK->k_esph,
              atomKK->k_v,
              buf,list,iswap,
              domain->xprd,domain->yprd,domain->zprd,
              domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
            Kokkos::parallel_for(n,f);
        } else {
            struct AtomVecSPHKokkos_PackCommVel<LMPHostType,1,0,0> f(
              atomKK->k_x,atomKK->k_mask,
              atomKK->k_vest,atomKK->k_rho,atomKK->k_esph,
              atomKK->k_v,
              buf,list,iswap,
              domain->xprd,domain->yprd,domain->zprd,
              domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
            Kokkos::parallel_for(n,f);  
        }
      } 
    } else {
      if (domain->triclinic) {
          struct AtomVecSPHKokkos_PackCommVel<LMPHostType,0,1,0> f(
            atomKK->k_x,atomKK->k_mask,
            atomKK->k_vest,atomKK->k_rho,atomKK->k_esph,
            atomKK->k_v,
            buf,list,iswap,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
          Kokkos::parallel_for(n,f);
      } 
      else {
          struct AtomVecSPHKokkos_PackCommVel<LMPHostType,0,0,0> f(
            atomKK->k_x,atomKK->k_mask,
            atomKK->k_vest,atomKK->k_rho,atomKK->k_esph,
            atomKK->k_v,
            buf,list,iswap,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
          Kokkos::parallel_for(n,f);
      }  
    }
  } else {
    atomKK->sync(Device,X_MASK|V_MASK|VEST_MASK|RHO_MASK|ESPH_MASK);
    if (pbc_flag) {
      if (deform_vremap) {
        if (domain->triclinic) {
            struct AtomVecSPHKokkos_PackCommVel<LMPHostType,1,1,1> f(
              atomKK->k_x,atomKK->k_mask,
              atomKK->k_vest,atomKK->k_rho,atomKK->k_esph,
              atomKK->k_v,
              buf,list,iswap,
              domain->xprd,domain->yprd,domain->zprd,
              domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
            Kokkos::parallel_for(n,f);
          }
          else {
            struct AtomVecSPHKokkos_PackCommVel<LMPHostType,1,0,1> f(
              atomKK->k_x,atomKK->k_mask,
              atomKK->k_vest,atomKK->k_rho,atomKK->k_esph,
              atomKK->k_v,
              buf,list,iswap,
              domain->xprd,domain->yprd,domain->zprd,
              domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
            Kokkos::parallel_for(n,f);       
        }
      } else {
        if (domain->triclinic) {    
            struct AtomVecSPHKokkos_PackCommVel<LMPHostType,1,1,0> f(
              atomKK->k_x,atomKK->k_mask,
              atomKK->k_vest,atomKK->k_rho,atomKK->k_esph,
              atomKK->k_v,
              buf,list,iswap,
              domain->xprd,domain->yprd,domain->zprd,
              domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
            Kokkos::parallel_for(n,f);
        } else {
            struct AtomVecSPHKokkos_PackCommVel<LMPHostType,1,0,0> f(
              atomKK->k_x,atomKK->k_mask,
              atomKK->k_vest,atomKK->k_rho,atomKK->k_esph,
              atomKK->k_v,
              buf,list,iswap,
              domain->xprd,domain->yprd,domain->zprd,
              domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
            Kokkos::parallel_for(n,f);  
        }
      } 
    } else {
      if (domain->triclinic) {
          struct AtomVecSPHKokkos_PackCommVel<LMPHostType,0,1,0> f(
            atomKK->k_x,atomKK->k_mask,
            atomKK->k_vest,atomKK->k_rho,atomKK->k_esph,
            atomKK->k_v,
            buf,list,iswap,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
          Kokkos::parallel_for(n,f);
      } 
      else {
          struct AtomVecSPHKokkos_PackCommVel<LMPHostType,0,0,0> f(
            atomKK->k_x,atomKK->k_mask,
            atomKK->k_vest,atomKK->k_rho,atomKK->k_esph,
            atomKK->k_v,
            buf,list,iswap,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
          Kokkos::parallel_for(n,f);
      }  
    }
  } 
  return n*(size_forward+size_velocity);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC>
struct AtomVecSPHKokkos_PackCommSelf {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_x_array_randomread _x;
  typename ArrayTypes<DeviceType>::t_x_array _xw;

  typename ArrayTypes<DeviceType>::t_v_array _vest;
  typename ArrayTypes<DeviceType>::t_float_1d _rho,_esph;

  int _nfirst;
  typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;
  X_FLOAT _xprd,_yprd,_zprd,_xy,_xz,_yz;
  X_FLOAT _pbc[6];

  AtomVecSPHKokkos_PackCommSelf(
    const typename DAT::tdual_x_array &x,

    const typename DAT::tdual_v_array &vest,
    const typename DAT::tdual_float_1d &rho,
    const typename DAT::tdual_float_1d &esph,

    const int &nfirst,
    const typename DAT::tdual_int_2d &list,
    const int & iswap,
    const X_FLOAT &xprd, const X_FLOAT &yprd, const X_FLOAT &zprd,
    const X_FLOAT &xy, const X_FLOAT &xz, const X_FLOAT &yz, const int* const pbc):
    _x(x.view<DeviceType>()),_xw(x.view<DeviceType>()),
    _vest(vest.view<DeviceType>()),
    _rho(rho.view<DeviceType>()),
    _esph(esph.view<DeviceType>()),
    _nfirst(nfirst),_list(list.view<DeviceType>()),_iswap(iswap),
    _xprd(xprd),_yprd(yprd),_zprd(zprd),
    _xy(xy),_xz(xz),_yz(yz) {
    _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
    _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(_iswap,i);
    if (PBC_FLAG == 0) {
      _xw(i+_nfirst,0) = _x(j,0);
      _xw(i+_nfirst,1) = _x(j,1);
      _xw(i+_nfirst,2) = _x(j,2);
    } else {
      if (TRICLINIC == 0) {
        _xw(i+_nfirst,0) = _x(j,0) + _pbc[0]*_xprd;
        _xw(i+_nfirst,1) = _x(j,1) + _pbc[1]*_yprd;
        _xw(i+_nfirst,2) = _x(j,2) + _pbc[2]*_zprd;
      } else {
        _xw(i+_nfirst,0) = _x(j,0) + _pbc[0]*_xprd + _pbc[5]*_xy + _pbc[4]*_xz;
        _xw(i+_nfirst,1) = _x(j,1) + _pbc[1]*_yprd + _pbc[3]*_yz;
        _xw(i+_nfirst,2) = _x(j,2) + _pbc[2]*_zprd;
      }
    }
    _vest(i+_nfirst,0) = _vest(j,0);
    _vest(i+_nfirst,1) = _vest(j,1);
    _vest(i+_nfirst,2) = _vest(j,2);
    _rho(i+_nfirst) = _rho(j);
    _esph(i+_nfirst) = _esph(j);
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecSPHKokkos::pack_comm_self(
  const int &n, const DAT::tdual_int_2d &list, const int &iswap,
  const int nfirst, const int &pbc_flag, const int* const pbc) { 
  fprintf(screen, "pack_comm_self called on AtomVecSPHKokkos\n");
  if (commKK->forward_comm_on_host) {
    atomKK->sync(Host,X_MASK|VEST_MASK|RHO_MASK|ESPH_MASK);
    if (pbc_flag) {
      if (domain->triclinic) {
        struct AtomVecSPHKokkos_PackCommSelf<LMPHostType,1,1> f(
          atomKK->k_x,
          atomKK->k_vest,
          atomKK->k_rho,atomKK->k_esph,
          nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecSPHKokkos_PackCommSelf<LMPHostType,1,0> f(
          atomKK->k_x,
          atomKK->k_vest,
          atomKK->k_rho,atomKK->k_esph,
          nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecSPHKokkos_PackCommSelf<LMPHostType,0,1> f(
          atomKK->k_x,
          atomKK->k_vest,
          atomKK->k_rho,atomKK->k_esph,
          nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecSPHKokkos_PackCommSelf<LMPHostType,0,0> f(
          atomKK->k_x,
          atomKK->k_vest,
          atomKK->k_rho,atomKK->k_esph,
          nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    }
    atomKK->modified(Host,X_MASK|VEST_MASK|RHO_MASK|ESPH_MASK);
  } else {
    atomKK->sync(Device,X_MASK|VEST_MASK|RHO_MASK|ESPH_MASK);
    atomKK->modified(Device,X_MASK|VEST_MASK|RHO_MASK|ESPH_MASK);
    if (pbc_flag) {
      if (domain->triclinic) {
        struct AtomVecSPHKokkos_PackCommSelf<LMPDeviceType,1,1> f(
          atomKK->k_x,
          atomKK->k_vest,
          atomKK->k_rho,atomKK->k_esph,
          nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecSPHKokkos_PackCommSelf<LMPDeviceType,1,0> f(
          atomKK->k_x,
          atomKK->k_vest,
          atomKK->k_rho,atomKK->k_esph,
          nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecSPHKokkos_PackCommSelf<LMPDeviceType,0,1> f(
          atomKK->k_x,
          atomKK->k_vest,
          atomKK->k_rho,atomKK->k_esph,
          nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecSPHKokkos_PackCommSelf<LMPDeviceType,0,0> f(
          atomKK->k_x,
          atomKK->k_vest,
          atomKK->k_rho,atomKK->k_esph,
          nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    }
  }
  return n*size_forward;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecSPHKokkos_UnpackComm {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_x_array _x;
  typename ArrayTypes<DeviceType>::t_v_array _vest;
  typename ArrayTypes<DeviceType>::t_float_1d _rho,_esph;
  typename ArrayTypes<DeviceType>::t_xfloat_2d_const_um _buf;
  int _first;

  AtomVecSPHKokkos_UnpackComm(
    const typename DAT::tdual_x_array &x,
    const typename DAT::tdual_v_array &vest,
    const typename DAT::tdual_float_1d &rho,
    const typename DAT::tdual_float_1d &esph,
    const typename DAT::tdual_xfloat_2d &buf,
    const int& first):
    _x(x.view<DeviceType>()),
    _vest(vest.view<DeviceType>()),
    _rho(rho.view<DeviceType>()),
    _esph(esph.view<DeviceType>()),
    _first(first)
  {
    const size_t elements = 8;
    const size_t maxsend = (buf.view<DeviceType>().extent(0)*buf.view<DeviceType>().extent(1))/elements;
    _buf = typename ArrayTypes<DeviceType>::t_xfloat_2d_const_um(buf.view<DeviceType>().data(),maxsend,elements);
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    _x(i+_first,0) = _buf(i,0);
    _x(i+_first,1) = _buf(i,1);
    _x(i+_first,2) = _buf(i,2);
    _rho(i+_first) = _buf(i,3);
    _esph(i+_first) = _buf(i,4);
    _vest(i+_first,0) = _buf(i,5);
    _vest(i+_first,1) = _buf(i,6);
    _vest(i+_first,2) = _buf(i,7);
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecSPHKokkos::unpack_comm_kokkos(
  const int &n, const int &first,
  const DAT::tdual_xfloat_2d &buf) {
	fprintf(screen, "unpack_comm_kokkos called on AtomVecSPHKokkos\n");
  if (commKK->forward_comm_on_host) {
    struct AtomVecSPHKokkos_UnpackComm<LMPHostType> f(
      atomKK->k_x,
      atomKK->k_vest,
      atomKK->k_rho,atomKK->k_esph,
      buf,first);
    Kokkos::parallel_for(n,f);
    atomKK->modified(Host,X_MASK|VEST_MASK|RHO_MASK|ESPH_MASK);
  } else {
    struct AtomVecSPHKokkos_UnpackComm<LMPDeviceType> f(
      atomKK->k_x,
      atomKK->k_vest,
      atomKK->k_rho,atomKK->k_esph,
      buf,first);
    Kokkos::parallel_for(n,f);
    atomKK->modified(Device,X_MASK|VEST_MASK|RHO_MASK|ESPH_MASK);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecSPHKokkos_UnpackCommVel {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_x_array _x;
  typename ArrayTypes<DeviceType>::t_v_array _vest;
  typename ArrayTypes<DeviceType>::t_float_1d _rho,_esph;
  typename ArrayTypes<DeviceType>::t_v_array _v;
  typename ArrayTypes<DeviceType>::t_xfloat_2d_const _buf;
  int _first;

  AtomVecSPHKokkos_UnpackCommVel(
    const typename DAT::tdual_x_array &x,
    const typename DAT::tdual_v_array &vest,
    const typename DAT::tdual_float_1d &rho,
    const typename DAT::tdual_float_1d &esph,
    const typename DAT::tdual_v_array &v,
    const typename DAT::tdual_xfloat_2d &buf,
    const int& first):
    _x(x.view<DeviceType>()),
    _vest(vest.view<DeviceType>()),
    _rho(rho.view<DeviceType>()),
    _esph(esph.view<DeviceType>()),
    _v(v.view<DeviceType>()),
    _first(first)
  {
    const size_t elements = 11;
    const int maxsend = (buf.template view<DeviceType>().extent(0)*buf.template view<DeviceType>().extent(1))/elements;
    buffer_view<DeviceType>(_buf,buf,maxsend,elements);
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    _x(i+_first,0) = _buf(i,0);
    _x(i+_first,1) = _buf(i,1);
    _x(i+_first,2) = _buf(i,2);
    _v(i+_first,0) = _buf(i,3);
    _v(i+_first,1) = _buf(i,4);
    _v(i+_first,2) = _buf(i,5);
    _vest(i+_first,0) = _buf(i,6);
    _vest(i+_first,1) = _buf(i,7);
    _vest(i+_first,2) = _buf(i,8); 
    _rho(i+_first) = _buf(i,9);
    _esph(i+_first) = _buf(i,10); 
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecSPHKokkos::unpack_comm_vel_kokkos(
  const int &n, const int &first,
  const DAT::tdual_xfloat_2d &buf) {
  if (commKK->forward_comm_on_host) {
      struct AtomVecSPHKokkos_UnpackCommVel<LMPHostType> f(
        atomKK->k_x,
        atomKK->k_vest,
        atomKK->k_rho,atomKK->k_esph,
        atomKK->k_v,
        buf,first);
      Kokkos::parallel_for(n,f);
      atomKK->modified(Host,X_MASK|V_MASK|VEST_MASK|RHO_MASK|ESPH_MASK);
   }
   else {
      struct AtomVecSPHKokkos_UnpackCommVel<LMPDeviceType> f(
        atomKK->k_x,
        atomKK->k_vest,
        atomKK->k_rho,atomKK->k_esph,
        atomKK->k_v,
        buf,first);
      Kokkos::parallel_for(n,f);
      atomKK->modified(Device,X_MASK|V_MASK|VEST_MASK|RHO_MASK|ESPH_MASK);
   }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecSPHKokkos_PackReverse {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_f_array_randomread _f;
  typename ArrayTypes<DeviceType>::t_float_1d _drho, _desph;
  typename ArrayTypes<DeviceType>::t_ffloat_2d _buf;
  int _first;

  AtomVecSPHKokkos_PackReverse(
      const typename DAT::tdual_f_array &f,
      const typename DAT::tdual_float_1d &drho,
      const typename DAT::tdual_float_1d &desph,
      const typename DAT::tdual_ffloat_2d &buf,
      const int& first):_f(f.view<DeviceType>()),_drho(drho.view<DeviceType>()),
                        _desph(desph.view<DeviceType>()),_buf(buf.view<DeviceType>()),
                        _first(first) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    _buf(i,0) = _f(i+_first,0);
    _buf(i,1) = _f(i+_first,1);
    _buf(i,2) = _f(i+_first,2);
    _buf(i,3) = _drho(i+_first);
    _buf(i,4) = _desph(i+_first);
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecSPHKokkos::pack_reverse_kokkos(const int &n, const int &first,
    const DAT::tdual_ffloat_2d &buf) {
    fprintf(screen, "pack_reverse_kokkos called on AtomVecSPHKokkos\n");
  if (commKK->reverse_comm_on_host) {
    atomKK->sync(Host,F_MASK | DRHO_MASK | DESPH_MASK);
    struct AtomVecSPHKokkos_PackReverse<LMPHostType> f(atomKK->k_f, atomKK->k_drho, atomKK->k_desph, buf,first);
    Kokkos::parallel_for(n,f);
  } else {
    atomKK->sync(Device,F_MASK | DRHO_MASK | DESPH_MASK);
    struct AtomVecSPHKokkos_PackReverse<LMPDeviceType> f(atomKK->k_f, atomKK->k_drho, atomKK->k_desph,buf,first);
    Kokkos::parallel_for(n,f);
  }
  fprintf(screen, "size_reverse is= %d\n", size_reverse);
  return n*size_reverse;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecSPHKokkos_UnPackReverseSelf {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_f_array_randomread _f;
  typename ArrayTypes<DeviceType>::t_float_1d _drho, _desph;
  typename ArrayTypes<DeviceType>::t_f_array _fw;
  typename ArrayTypes<DeviceType>::t_float_1d _drhow, _desphw;
  int _nfirst;
  typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;

  AtomVecSPHKokkos_UnPackReverseSelf(
      const typename DAT::tdual_f_array &f,
      const typename DAT::tdual_float_1d &drho,
      const typename DAT::tdual_float_1d &desph,
      const int &nfirst,
      const typename DAT::tdual_int_2d &list,
      const int & iswap):
      _f(f.view<DeviceType>()),_drho(drho.view<DeviceType>()), _desph(desph.view<DeviceType>()),
      _fw(f.view<DeviceType>()),_drhow(drho.view<DeviceType>()), _desphw(desph.view<DeviceType>()),
      _nfirst(nfirst),_list(list.view<DeviceType>()),_iswap(iswap) {
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(_iswap,i);
    _fw(j,0) += _f(i+_nfirst,0);
    _fw(j,1) += _f(i+_nfirst,1);
    _fw(j,2) += _f(i+_nfirst,2);
    _drhow(j) += _drho(i+_nfirst);
    _desphw(j) += _desph(i+_nfirst);
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecSPHKokkos::unpack_reverse_self(const int &n, const DAT::tdual_int_2d &list, const int & iswap,
                                        const int nfirst) {
	fprintf(screen, "unpack_reverse_self called on AtomVecSPHKokkos\n");
  if (commKK->reverse_comm_on_host) {
    atomKK->sync(Host,F_MASK | DRHO_MASK | DESPH_MASK);
    struct AtomVecSPHKokkos_UnPackReverseSelf<LMPHostType> f(atomKK->k_f,atomKK->k_drho,atomKK->k_desph,nfirst,list,iswap);
    Kokkos::parallel_for(n,f);
    atomKK->modified(Host,F_MASK | DRHO_MASK | DESPH_MASK);
  } else {
    atomKK->sync(Device,F_MASK | DRHO_MASK | DESPH_MASK);
    struct AtomVecSPHKokkos_UnPackReverseSelf<LMPDeviceType> f(atomKK->k_f,atomKK->k_drho, atomKK->k_desph,nfirst,list,iswap);
    Kokkos::parallel_for(n,f);
    atomKK->modified(Device,F_MASK | DRHO_MASK | DESPH_MASK);
  }
  return n*5;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecSPHKokkos_UnPackReverse {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_f_array _f;
  typename ArrayTypes<DeviceType>::t_float_1d _drho, _desph;
  typename ArrayTypes<DeviceType>::t_ffloat_2d_const _buf;
  typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;

  AtomVecSPHKokkos_UnPackReverse(
      const typename DAT::tdual_f_array &f,
      const typename DAT::tdual_float_1d &drho,
      const typename DAT::tdual_float_1d &desph,
      const typename DAT::tdual_ffloat_2d &buf,
      const typename DAT::tdual_int_2d &list,
      const int & iswap):
      _f(f.view<DeviceType>()),_drho(drho.view<DeviceType>()),_desph(desph.view<DeviceType>()),
      _list(list.view<DeviceType>()),_iswap(iswap) {
        const size_t maxsend = (buf.view<DeviceType>().extent(0)*buf.view<DeviceType>().extent(1))/5;
        const size_t elements = 5;
        buffer_view<DeviceType>(_buf,buf,maxsend,elements);
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(_iswap,i);
    _f(j,0) += _buf(i,0);
    _f(j,1) += _buf(i,1);
    _f(j,2) += _buf(i,2);
    _drho(j) += _buf(i,3);
    _desph(j) += _buf(i,4);
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecSPHKokkos::unpack_reverse_kokkos(const int &n,
                                          const DAT::tdual_int_2d &list,
                                          const int & iswap,
                                          const DAT::tdual_ffloat_2d &buf)
{
  // Check whether to always run reverse communication on the host
  // Choose correct reverse UnPackReverse kernel
   fprintf(screen, "unpack_reverse_kokkos called on AtomVecSPHKokkos\n");
  if (commKK->reverse_comm_on_host) {
    struct AtomVecSPHKokkos_UnPackReverse<LMPHostType> f(atomKK->k_f,atomKK->k_drho,atomKK->k_desph,buf,list,iswap);
    Kokkos::parallel_for(n,f);
    atomKK->modified(Host,F_MASK | DRHO_MASK | DESPH_MASK);
  } else {
    struct AtomVecSPHKokkos_UnPackReverse<LMPDeviceType> f(atomKK->k_f,atomKK->k_drho,atomKK->k_desph,buf,list,iswap);
    Kokkos::parallel_for(n,f);
    atomKK->modified(Device,F_MASK | DRHO_MASK | DESPH_MASK);
  }
}

/* ---------------------------------------------------------------------- */


template<class DeviceType,int PBC_FLAG>
struct AtomVecSPHKokkos_PackBorder {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_xfloat_2d_um _buf;
  const typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;
  const typename ArrayTypes<DeviceType>::t_x_array_randomread _x;
  const typename ArrayTypes<DeviceType>::t_v_array _vest;
  const typename ArrayTypes<DeviceType>::t_tagint_1d _tag;
  const typename ArrayTypes<DeviceType>::t_int_1d _type;
  const typename ArrayTypes<DeviceType>::t_int_1d _mask;
  typename ArrayTypes<DeviceType>::t_float_1d _rho,_esph,_cv;
  X_FLOAT _dx,_dy,_dz;

  AtomVecSPHKokkos_PackBorder(
    const typename ArrayTypes<DeviceType>::t_xfloat_2d &buf,
    const typename ArrayTypes<DeviceType>::t_int_2d_const &list,
    const int &iswap,
    const typename ArrayTypes<DeviceType>::t_x_array &x,
    const typename ArrayTypes<DeviceType>::t_v_array &vest,
    const typename ArrayTypes<DeviceType>::t_tagint_1d &tag,
    const typename ArrayTypes<DeviceType>::t_int_1d &type,
    const typename ArrayTypes<DeviceType>::t_int_1d &mask,
    const typename ArrayTypes<DeviceType>::t_float_1d &rho,
    const typename ArrayTypes<DeviceType>::t_float_1d &esph,
    const typename ArrayTypes<DeviceType>::t_float_1d &cv,
    const X_FLOAT &dx, const X_FLOAT &dy, const X_FLOAT &dz):
    _list(list),_iswap(iswap),
    _x(x),_vest(vest),_tag(tag),_type(type),_mask(mask),
    _rho(rho),
    _esph(esph),
    _cv(cv),
    _dx(dx),_dy(dy),_dz(dz)
  {
    const size_t elements = 12;
    const int maxsend = (buf.extent(0)*buf.extent(1))/elements;
    _buf = typename ArrayTypes<DeviceType>::t_xfloat_2d_um(buf.data(),maxsend,elements);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(_iswap,i);
    if (PBC_FLAG == 0) {
      _buf(i,0) = _x(j,0);
      _buf(i,1) = _x(j,1);
      _buf(i,2) = _x(j,2);
    } else {
      _buf(i,0) = _x(j,0) + _dx;
      _buf(i,1) = _x(j,1) + _dy;
      _buf(i,2) = _x(j,2) + _dz;
    }
    _buf(i,3) = d_ubuf(_tag(j)).d;
    _buf(i,4) = d_ubuf(_type(j)).d;
    _buf(i,5) = d_ubuf(_mask(j)).d;
    _buf(i,6) = _vest(j,0);
    _buf(i,7) = _vest(j,1);
    _buf(i,8) = _vest(j,2);
    _buf(i,9) = _rho(j);
    _buf(i,10) = _esph(j);
    _buf(i,11) = _cv(j);
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecSPHKokkos::pack_border_kokkos(
  int n, DAT::tdual_int_2d k_sendlist, DAT::tdual_xfloat_2d buf,int iswap,
  int pbc_flag, int *pbc, ExecutionSpace space)
{
	fprintf(screen, "pack_border_kokos called on AtomVecSPHKokkos\n");
  X_FLOAT dx,dy,dz;

  // This was in atom_vec_dpd_kokkos but doesn't appear in any other atom_vec
  atomKK->sync(space,ALL_MASK);

  if (pbc_flag != 0) {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    if (space==Host) {
      AtomVecSPHKokkos_PackBorder<LMPHostType,1> f(
        buf.view<LMPHostType>(), k_sendlist.view<LMPHostType>(),
        iswap,h_x,h_vest,h_tag,h_type,h_mask,
        h_rho,h_esph,h_cv,
        dx,dy,dz);
      Kokkos::parallel_for(n,f);
    } else {
      AtomVecSPHKokkos_PackBorder<LMPDeviceType,1> f(
        buf.view<LMPDeviceType>(), k_sendlist.view<LMPDeviceType>(),
        iswap,d_x,d_vest,d_tag,d_type,d_mask,
        d_rho,d_esph,d_cv,
        dx,dy,dz);
      Kokkos::parallel_for(n,f);
    }
  } else {
    dx = dy = dz = 0;
    if (space==Host) {
      AtomVecSPHKokkos_PackBorder<LMPHostType,0> f(
        buf.view<LMPHostType>(), k_sendlist.view<LMPHostType>(),
        iswap,h_x,h_vest,h_tag,h_type,h_mask,
        h_rho,h_esph,h_cv,
        dx,dy,dz);
      Kokkos::parallel_for(n,f);
    } else {
      AtomVecSPHKokkos_PackBorder<LMPDeviceType,0> f(
        buf.view<LMPDeviceType>(), k_sendlist.view<LMPDeviceType>(),
        iswap,d_x,d_vest,d_tag,d_type,d_mask,
        d_rho,d_esph,d_cv,
        dx,dy,dz);
      Kokkos::parallel_for(n,f);
    }
  }
  return n*size_border;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int DEFORM_VREMAP>
struct AtomVecSPHKokkos_PackBorderVel {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_xfloat_2d_um _buf;
  const typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;
  const typename ArrayTypes<DeviceType>::t_x_array_randomread _x;
  const typename ArrayTypes<DeviceType>::t_v_array _vest;
  const typename ArrayTypes<DeviceType>::t_tagint_1d _tag;
  const typename ArrayTypes<DeviceType>::t_int_1d _type;
  const typename ArrayTypes<DeviceType>::t_int_1d _mask;
  typename ArrayTypes<DeviceType>::t_float_1d _rho,_esph,_cv;
  typename ArrayTypes<DeviceType>::t_v_array _v;
  X_FLOAT _dx,_dy,_dz, _dvx, _dvy, _dvz;
  const int _deform_groupbit;

  AtomVecSPHKokkos_PackBorderVel(
    const typename ArrayTypes<DeviceType>::t_xfloat_2d &buf,
    const typename ArrayTypes<DeviceType>::t_int_2d_const &list,
    const int &iswap,
    const typename ArrayTypes<DeviceType>::t_x_array &x,
    const typename ArrayTypes<DeviceType>::t_v_array &vest,
    const typename ArrayTypes<DeviceType>::t_tagint_1d &tag,
    const typename ArrayTypes<DeviceType>::t_int_1d &type,
    const typename ArrayTypes<DeviceType>::t_int_1d &mask,
    const typename ArrayTypes<DeviceType>::t_float_1d &rho,
    const typename ArrayTypes<DeviceType>::t_float_1d &esph,
    const typename ArrayTypes<DeviceType>::t_float_1d &cv,
    const typename ArrayTypes<DeviceType>::t_v_array &v,
    const X_FLOAT &dx, const X_FLOAT &dy, const X_FLOAT &dz,
    const X_FLOAT &dvx, const X_FLOAT &dvy, const X_FLOAT &dvz,
    const int &deform_groupbit):
    _buf(buf),_list(list),_iswap(iswap),
    _x(x),_vest(vest),_tag(tag),_type(type),_mask(mask),
    _rho(rho),
    _esph(esph),
    _cv(cv),
    _v(v),
    _dx(dx),_dy(dy),_dz(dz),
    _dvx(dvx),_dvy(dvy),_dvz(dvz),
    _deform_groupbit(deform_groupbit)
  {
    const size_t elements = 15;
    const int maxsend = (buf.extent(0)*buf.extent(1))/elements;
    _buf = typename ArrayTypes<DeviceType>::t_xfloat_2d_um(buf.data(),maxsend,elements);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(_iswap,i);
    if (PBC_FLAG == 0) {
      _buf(i,0) = _x(j,0);
      _buf(i,1) = _x(j,1);
      _buf(i,2) = _x(j,2);
    } else {
      _buf(i,0) = _x(j,0) + _dx;
      _buf(i,1) = _x(j,1) + _dy;
      _buf(i,2) = _x(j,2) + _dz;
    }
    _buf(i,3) = d_ubuf(_tag(j)).d;
    _buf(i,4) = d_ubuf(_type(j)).d;
    _buf(i,5) = d_ubuf(_mask(j)).d;
    _buf(i,6) = _rho(j);
    _buf(i,7) = _esph(j);
    _buf(i,8) = _cv(j);
    if (DEFORM_VREMAP) {
      if (_mask(i) & _deform_groupbit) {
        _buf(i,9) = _v(j,0) + _dvx;
        _buf(i,10) = _v(j,1) + _dvy;
        _buf(i,11) = _v(j,2) + _dvz;
        _buf(i,12) = _vest(j,0) + _dvx;
        _buf(i,13) = _vest(j,1) + _dvy;
        _buf(i,14) = _vest(j,2) + _dvz;
      }
    }
    else {
      _buf(i,9) = _v(j,0);
      _buf(i,10) = _v(j,1);
      _buf(i,11) = _v(j,2);
      _buf(i,12) = _vest(j,0);
      _buf(i,13) = _vest(j,1);
      _buf(i,14) = _vest(j,2);
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecSPHKokkos::pack_border_vel_kokkos(
  int n, DAT::tdual_int_2d k_sendlist, DAT::tdual_xfloat_2d buf,int iswap,
  int pbc_flag, int *pbc, ExecutionSpace space)
{
	fprintf(screen, "pack_border_vel_kokkos called on AtomVecSPHKokkos\n");
  X_FLOAT dx=0,dy=0,dz=0;
  X_FLOAT dvx=0,dvy=0,dvz=0;

  // This was in atom_vec_dpd_kokkos but doesn't appear in any other atom_vec
  atomKK->sync(space,ALL_MASK);

  if (pbc_flag != 0) {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    if (!deform_vremap) {
      if (space==Host) {
        AtomVecSPHKokkos_PackBorderVel<LMPHostType,1,0> f(
          buf.view<LMPHostType>(), k_sendlist.view<LMPHostType>(),
          iswap,h_x,h_vest,h_tag,h_type,h_mask,
          h_rho,h_esph,h_cv,
          h_v, 
          dx,dy,dz,dvx,dvy,dvz,
          deform_groupbit);
        Kokkos::parallel_for(n,f);
      } else {
        AtomVecSPHKokkos_PackBorderVel<LMPDeviceType,1,0> f(
          buf.view<LMPDeviceType>(), k_sendlist.view<LMPDeviceType>(),
          iswap,d_x,d_vest,d_tag,d_type,d_mask,
          d_rho,d_esph,d_cv,
          d_v,
          dx,dy,dz,dvx,dvy,dvz,
          deform_groupbit);
        Kokkos::parallel_for(n,f);
      }
    }
    else {
      dvx = pbc[0]*h_rate[0] + pbc[5]*h_rate[5] + pbc[4]*h_rate[4];
      dvy = pbc[1]*h_rate[1] + pbc[3]*h_rate[3];
      dvz = pbc[2]*h_rate[2];
      if (space==Host) {
        AtomVecSPHKokkos_PackBorderVel<LMPHostType,1,1> f(
          buf.view<LMPHostType>(), k_sendlist.view<LMPHostType>(),
          iswap,h_x,h_vest,h_tag,h_type,h_mask,
          h_rho,h_esph,h_cv,
          h_v, 
          dx,dy,dz,dvx,dvy,dvz,
          deform_groupbit);
        Kokkos::parallel_for(n,f);
      } else {
        AtomVecSPHKokkos_PackBorderVel<LMPDeviceType,1,1> f(
          buf.view<LMPDeviceType>(), k_sendlist.view<LMPDeviceType>(),
          iswap,d_x,d_vest,d_tag,d_type,d_mask,
          d_rho,d_esph,d_cv,
          d_v,
          dx,dy,dz,dvx,dvy,dvz,
          deform_groupbit);
        Kokkos::parallel_for(n,f);
      }
    }
  } else {
    if (space==Host) {
      AtomVecSPHKokkos_PackBorderVel<LMPHostType,0,0> f(
        buf.view<LMPHostType>(), k_sendlist.view<LMPHostType>(),
        iswap,h_x,h_vest,h_tag,h_type,h_mask,
        h_rho,h_esph,h_cv,
        h_v,
        dx,dy,dz,dvx,dvy,dvz,
        deform_groupbit);
      Kokkos::parallel_for(n,f);
    } else {
      AtomVecSPHKokkos_PackBorderVel<LMPDeviceType,0,0> f(
        buf.view<LMPDeviceType>(), k_sendlist.view<LMPDeviceType>(),
        iswap,d_x,d_vest,d_tag,d_type,d_mask,
        d_rho,d_esph,d_cv,
        d_v,
        dx,dy,dz,dvx,dvy,dvz,
        deform_groupbit);
      Kokkos::parallel_for(n,f);
    }
  }

  return n*(size_border + size_velocity);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecSPHKokkos_UnpackBorder {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_xfloat_2d_const_um _buf;
  typename ArrayTypes<DeviceType>::t_x_array _x;
  typename ArrayTypes<DeviceType>::t_v_array _v;
  typename ArrayTypes<DeviceType>::t_v_array _vest;
  typename ArrayTypes<DeviceType>::t_tagint_1d _tag;
  typename ArrayTypes<DeviceType>::t_int_1d _type;
  typename ArrayTypes<DeviceType>::t_int_1d _mask;
  typename ArrayTypes<DeviceType>::t_float_1d _rho,_esph,_cv;
  int _first;

  AtomVecSPHKokkos_UnpackBorder(
    const typename ArrayTypes<DeviceType>::t_xfloat_2d &buf,
    const typename ArrayTypes<DeviceType>::t_x_array &x,
    const typename ArrayTypes<DeviceType>::t_v_array &v,
    const typename ArrayTypes<DeviceType>::t_v_array &vest,
    const typename ArrayTypes<DeviceType>::t_tagint_1d &tag,
    const typename ArrayTypes<DeviceType>::t_int_1d &type,
    const typename ArrayTypes<DeviceType>::t_int_1d &mask,
    const typename ArrayTypes<DeviceType>::t_float_1d &rho,
    const typename ArrayTypes<DeviceType>::t_float_1d &esph,
    const typename ArrayTypes<DeviceType>::t_float_1d &cv,
    const int& first):
    _buf(buf),_x(x),_v(v),_vest(vest),_tag(tag),_type(type),_mask(mask),
    _rho(rho),
    _esph(esph),
    _cv(cv),
    _first(first)
  {
    const size_t elements = 12;
    const int maxsend = (buf.extent(0)*buf.extent(1))/elements;
    _buf = typename ArrayTypes<DeviceType>::t_xfloat_2d_const_um(buf.data(),maxsend,elements);
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    _x(i+_first,0) = _buf(i,0);
    _x(i+_first,1) = _buf(i,1);
    _x(i+_first,2) = _buf(i,2);
    _tag(i+_first) = static_cast<tagint> (d_ubuf(_buf(i,3)).i);
    _type(i+_first) = static_cast<int>  (d_ubuf(_buf(i,4)).i);
    _mask(i+_first) = static_cast<int>  (d_ubuf(_buf(i,5)).i);
    _vest(i+_first,0) = _buf(i,6);
    _vest(i+_first,1) = _buf(i,7);
    _vest(i+_first,2) = _buf(i,8);
    _rho(i+_first) = _buf(i,9);
    _esph(i+_first) = _buf(i,10);
    _cv(i+_first) = _buf(i,11);
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecSPHKokkos::unpack_border_kokkos(const int &n, const int &first,
                                               const DAT::tdual_xfloat_2d &buf,ExecutionSpace space) {
	fprintf(screen, "unpack_border_kokkos called on AtomVecSPHKokkos\n");
  while (first+n >= nmax) grow(0);

  if (space==Host) {
    struct AtomVecSPHKokkos_UnpackBorder<LMPHostType> f(buf.view<LMPHostType>(),
      h_x,h_v,h_vest,h_tag,h_type,h_mask,
      h_rho,h_esph,h_cv,
      first);
    Kokkos::parallel_for(n,f);
  } else {
    struct AtomVecSPHKokkos_UnpackBorder<LMPDeviceType> f(buf.view<LMPDeviceType>(),
      d_x,d_v,d_vest,d_tag,d_type,d_mask,
      d_rho,d_esph,d_cv,
      first);
    Kokkos::parallel_for(n,f);
  }

  atomKK->modified(space,X_MASK|VEST_MASK|TAG_MASK|TYPE_MASK|MASK_MASK|
                 RHO_MASK|ESPH_MASK|CV_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecSPHKokkos_UnpackBorderVel {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_xfloat_2d_const_um _buf;
  typename ArrayTypes<DeviceType>::t_x_array _x;
  typename ArrayTypes<DeviceType>::t_v_array _v;
  typename ArrayTypes<DeviceType>::t_v_array _vest;
  typename ArrayTypes<DeviceType>::t_tagint_1d _tag;
  typename ArrayTypes<DeviceType>::t_int_1d _type;
  typename ArrayTypes<DeviceType>::t_int_1d _mask;
  typename ArrayTypes<DeviceType>::t_float_1d _rho,_esph,_cv;
  int _first;

  AtomVecSPHKokkos_UnpackBorderVel(
    const typename ArrayTypes<DeviceType>::t_xfloat_2d_const &buf,
    const typename ArrayTypes<DeviceType>::t_x_array &x,
    const typename ArrayTypes<DeviceType>::t_v_array &v,
    const typename ArrayTypes<DeviceType>::t_v_array &vest,
    const typename ArrayTypes<DeviceType>::t_tagint_1d &tag,
    const typename ArrayTypes<DeviceType>::t_int_1d &type,
    const typename ArrayTypes<DeviceType>::t_int_1d &mask,
    const typename ArrayTypes<DeviceType>::t_float_1d &rho,
    const typename ArrayTypes<DeviceType>::t_float_1d &esph,
    const typename ArrayTypes<DeviceType>::t_float_1d &cv,
    const int& first):
    _buf(buf),_x(x),_v(v),_vest(vest),_tag(tag),_type(type),_mask(mask),
    _rho(rho),
    _esph(esph),
    _cv(cv),
    _first(first)
  {
    const size_t elements = 15;
    const int maxsend = (buf.extent(0)*buf.extent(1))/elements;
    _buf = typename ArrayTypes<DeviceType>::t_xfloat_2d_const_um(buf.data(),maxsend,elements);
  };
  
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    _x(i+_first,0) = _buf(i,0);
    _x(i+_first,1) = _buf(i,1);
    _x(i+_first,2) = _buf(i,2);
    _tag(i+_first) = static_cast<tagint> (d_ubuf(_buf(i,3)).i);
    _type(i+_first) = static_cast<int>  (d_ubuf(_buf(i,4)).i);
    _mask(i+_first) = static_cast<int>  (d_ubuf(_buf(i,5)).i);
    _rho(i+_first) = _buf(i,6);
    _esph(i+_first) = _buf(i,7);
    _cv(i+_first) = _buf(i,8);
    _v(i+_first,0) = _buf(i,9);
    _v(i+_first,1) = _buf(i,10);
    _v(i+_first,2) = _buf(i,11);
    _vest(i+_first,0) = _buf(i,12);
    _vest(i+_first,1) = _buf(i,13);
    _vest(i+_first,2) = _buf(i,14);
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecSPHKokkos::unpack_border_vel_kokkos(
  const int &n, const int &first,
  const DAT::tdual_xfloat_2d &buf,ExecutionSpace space) {
	fprintf(screen, "pack_border_vel_kokkos called on AtomVecSPHKokkos\n");
  while (first+n >= nmax) grow(0);
  //buf.sync<space>(); //added by Moein

  if (space==Host) {
    struct AtomVecSPHKokkos_UnpackBorderVel<LMPHostType> f(buf.view<LMPHostType>(),
      h_x,h_v,h_vest,h_tag,h_type,h_mask,
      h_rho,h_esph,h_cv, 
      first);
    Kokkos::parallel_for(n,f);
  } else {
    struct AtomVecSPHKokkos_UnpackBorderVel<LMPDeviceType> f(buf.view<LMPDeviceType>(),
      d_x,d_v,d_vest,d_tag,d_type,d_mask,
      d_rho,d_esph,d_cv,
      first);
    Kokkos::parallel_for(n,f);
  }

  atomKK->modified(space,X_MASK|VEST_MASK|TAG_MASK|TYPE_MASK|MASK_MASK|
                 RHO_MASK|ESPH_MASK|CV_MASK|V_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecSPHKokkos_PackExchangeFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typename AT::t_x_array_randomread _x;
  typename AT::t_v_array_randomread _v;
  typename AT::t_v_array_randomread _vest;
  typename AT::t_tagint_1d_randomread _tag;
  typename AT::t_int_1d_randomread _type;
  typename AT::t_int_1d_randomread _mask;
  typename AT::t_imageint_1d_randomread _image;
  typename AT::t_float_1d_randomread _rho,_esph,_cv;
  typename AT::t_x_array _xw;
  typename AT::t_v_array _vw;
  typename AT::t_v_array _vestw;
  typename AT::t_tagint_1d _tagw;
  typename AT::t_int_1d _typew;
  typename AT::t_int_1d _maskw;
  typename AT::t_imageint_1d _imagew;
  typename AT::t_float_1d _rhow,_esphw,_cvw;
  typename AT::t_xfloat_2d_um _buf;
  typename AT::t_int_1d_const _sendlist;
  typename AT::t_int_1d_const _copylist;
  int _size_exchange;

  AtomVecSPHKokkos_PackExchangeFunctor(
    const AtomKokkos* atom,
    const typename AT::tdual_xfloat_2d buf,
    typename AT::tdual_int_1d sendlist,
    typename AT::tdual_int_1d copylist):
    _size_exchange(atom->avecKK->size_exchange),
    _x(atom->k_x.view<DeviceType>()),
    _v(atom->k_v.view<DeviceType>()),
    _vest(atom->k_vest.view<DeviceType>()),
    _tag(atom->k_tag.view<DeviceType>()),
    _type(atom->k_type.view<DeviceType>()),
    _mask(atom->k_mask.view<DeviceType>()),
    _image(atom->k_image.view<DeviceType>()),
    _rho(atom->k_rho.view<DeviceType>()),
    _esph(atom->k_esph.view<DeviceType>()),
    _cv(atom->k_cv.view<DeviceType>()),
    _xw(atom->k_x.view<DeviceType>()),
    _vw(atom->k_v.view<DeviceType>()),
    _vestw(atom->k_vest.view<DeviceType>()),
    _tagw(atom->k_tag.view<DeviceType>()),
    _typew(atom->k_type.view<DeviceType>()),
    _maskw(atom->k_mask.view<DeviceType>()),
    _imagew(atom->k_image.view<DeviceType>()),
    _rhow(atom->k_rho.view<DeviceType>()),
    _esphw(atom->k_esph.view<DeviceType>()),
    _cvw(atom->k_cv.view<DeviceType>()),
    _sendlist(sendlist.template view<DeviceType>()),
    _copylist(copylist.template view<DeviceType>()) {
    const int maxsend = (buf.template view<DeviceType>().extent(0)*buf.template view<DeviceType>().extent(1))/_size_exchange;

    _buf = typename AT::t_xfloat_2d_um(buf.template view<DeviceType>().data(),maxsend,_size_exchange);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int &mysend) const {
    const int i = _sendlist(mysend);
    _buf(mysend,0) = _size_exchange;
    _buf(mysend,1) = _x(i,0);
    _buf(mysend,2) = _x(i,1);
    _buf(mysend,3) = _x(i,2);
    _buf(mysend,4) = _v(i,0);
    _buf(mysend,5) = _v(i,1);
    _buf(mysend,6) = _v(i,2);
    _buf(mysend,7) = d_ubuf(_tag[i]).d;
    _buf(mysend,8) = d_ubuf(_type[i]).d;
    _buf(mysend,9) = d_ubuf(_mask[i]).d;
    _buf(mysend,10) = d_ubuf(_image[i]).d;
    _buf(mysend,11) = _rho[i];
    _buf(mysend,12) = _esph[i];
    _buf(mysend,13) = _vest(i,0);
    _buf(mysend,14) = _vest(i,1);
    _buf(mysend,15) = _vest(i,2);
    _buf(mysend,16) = _cv[i];
    const int j = _copylist(mysend);

    if (j>-1) {
      _xw(i,0) = _x(j,0);
      _xw(i,1) = _x(j,1);
      _xw(i,2) = _x(j,2);
      _vw(i,0) = _v(j,0);
      _vw(i,1) = _v(j,1);
      _vw(i,2) = _v(j,2);
      _tagw[i] = _tag(j);
      _typew[i] = _type(j);
      _maskw[i] = _mask(j);
      _imagew[i] = _image(j);
      _rhow[i] = _rho(j);
      _esphw[i] = _esph(j);
      _vestw(i,0) = _vest(j,0);
      _vestw(i,1) = _vest(j,1);
      _vestw(i,2) = _vest(j,2);
      _cvw[i] = _cv(j);
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecSPHKokkos::pack_exchange_kokkos(
  const int &nsend,
  DAT::tdual_xfloat_2d &k_buf,
  DAT::tdual_int_1d k_sendlist,
  DAT::tdual_int_1d k_copylist,
  ExecutionSpace space)
{
	fprintf(screen, "pack_exchange_kokkos called on AtomVecSPHKokkos\n");
  size_exchange = 17;

  if (nsend > (int) (k_buf.view<LMPHostType>().extent(0)*k_buf.view<LMPHostType>().extent(1))/size_exchange) {
    int newsize = nsend*18/k_buf.view<LMPHostType>().extent(1)+1;
    k_buf.resize(newsize,k_buf.view<LMPHostType>().extent(1));
  }
  atomKK->sync(space,X_MASK | V_MASK | TAG_MASK | TYPE_MASK |
             MASK_MASK | IMAGE_MASK| RHO_MASK | ESPH_MASK |
             VEST_MASK | CV_MASK);

  if (space == Host) {
    AtomVecSPHKokkos_PackExchangeFunctor<LMPHostType> f(atomKK,k_buf,k_sendlist,k_copylist);
    Kokkos::parallel_for(nsend,f);
  } else {
    AtomVecSPHKokkos_PackExchangeFunctor<LMPDeviceType> f(atomKK,k_buf,k_sendlist,k_copylist);
    Kokkos::parallel_for(nsend,f);
  }
  return nsend*size_exchange;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int OUTPUT_INDICES>
struct AtomVecSPHKokkos_UnpackExchangeFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typename AT::t_x_array _x;
  typename AT::t_v_array _v;
  typename AT::t_tagint_1d _tag;
  typename AT::t_int_1d _type;
  typename AT::t_int_1d _mask;
  typename AT::t_imageint_1d _image;
  typename AT::t_float_1d _rho;
  typename AT::t_float_1d _esph;
  typename AT::t_float_1d _cv;
  typename AT::t_v_array _vest;
  typename AT::t_xfloat_2d_um _buf;
  typename AT::t_int_1d _nlocal;
  typename AT::t_int_1d _indices;
  int _dim;
  X_FLOAT _lo,_hi;
  int _size_exchange;

  AtomVecSPHKokkos_UnpackExchangeFunctor(
    const AtomKokkos* atom,
    const typename AT::tdual_xfloat_2d buf,
    typename AT::tdual_int_1d nlocal,
    typename AT::tdual_int_1d indices,
    int dim, X_FLOAT lo, X_FLOAT hi):
      _size_exchange(atom->avecKK->size_exchange),
      _x(atom->k_x.view<DeviceType>()),
      _v(atom->k_v.view<DeviceType>()),
      _tag(atom->k_tag.view<DeviceType>()),
      _type(atom->k_type.view<DeviceType>()),
      _mask(atom->k_mask.view<DeviceType>()),
      _image(atom->k_image.view<DeviceType>()),
      _rho(atom->k_rho.view<DeviceType>()),
      _esph(atom->k_esph.view<DeviceType>()),
      _cv(atom->k_cv.view<DeviceType>()),
      _vest(atom->k_vest.view<DeviceType>()),
      _nlocal(nlocal.template view<DeviceType>()),
      _indices(indices.template view<DeviceType>()),
      _dim(dim),
      _lo(lo),_hi(hi)
  {
    const size_t size_exchange = 17;
    const int maxsendlist = (buf.template view<DeviceType>().extent(0)*buf.template view<DeviceType>().extent(1))/size_exchange;

    buffer_view<DeviceType>(_buf,buf,maxsendlist,size_exchange);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int &myrecv) const {
    X_FLOAT x = _buf(myrecv,_dim+1);
    int i = -1;
    if (x >= _lo && x < _hi) {
      i = Kokkos::atomic_fetch_add(&_nlocal(0),1);
      _x(i,0) = _buf(myrecv,1);
      _x(i,1) = _buf(myrecv,2);
      _x(i,2) = _buf(myrecv,3);
      _v(i,0) = _buf(myrecv,4);
      _v(i,1) = _buf(myrecv,5);
      _v(i,2) = _buf(myrecv,6);
      _tag[i] = (tagint) d_ubuf(_buf(myrecv,7)).i;
      _type[i] = (int) d_ubuf(_buf(myrecv,8)).i;
      _mask[i] = (int) d_ubuf(_buf(myrecv,9)).i;
      _image[i] = (imageint) d_ubuf(_buf(myrecv,10)).i;
      _rho[i] = _buf(myrecv,11);
      _esph[i] = _buf(myrecv,12);
      _vest(i,0) = _buf(myrecv,13);
      _vest(i,1) = _buf(myrecv,14);
      _vest(i,2) = _buf(myrecv,15);
      _cv[i] = _buf(myrecv,16);
    }
    if (OUTPUT_INDICES)
      _indices(myrecv) = i;
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecSPHKokkos::unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, int nrecv, int nlocal,
                                                int dim, X_FLOAT lo, X_FLOAT hi, ExecutionSpace space,
                                                DAT::tdual_int_1d &k_indices)
{
	fprintf(screen, "unpack_exchange_kokkos called on AtomVecSPHKokkos\n");
  while (nlocal + nrecv/size_exchange >= nmax) grow(0);

  if (space == Host) {
    k_count.h_view(0) = nlocal;
    if (k_indices.h_view.data()) {
      AtomVecSPHKokkos_UnpackExchangeFunctor<LMPHostType,1> f(atomKK,k_buf,k_count,k_indices,dim,lo,hi);
      Kokkos::parallel_for(nrecv/size_exchange,f);
    } else {
      AtomVecSPHKokkos_UnpackExchangeFunctor<LMPHostType,0> f(atomKK,k_buf,k_count,k_indices,dim,lo,hi);
      Kokkos::parallel_for(nrecv/size_exchange,f);
    }
  } else {
    k_count.h_view(0) = nlocal;
    k_count.modify<LMPHostType>();
    k_count.sync<LMPDeviceType>();
    if (k_indices.h_view.data()) {
      AtomVecSPHKokkos_UnpackExchangeFunctor<LMPDeviceType,1> f(atomKK,k_buf,k_count,k_indices,dim,lo,hi);
      Kokkos::parallel_for(nrecv/size_exchange,f);
    } else {
      AtomVecSPHKokkos_UnpackExchangeFunctor<LMPDeviceType,0> f(atomKK,k_buf,k_count,k_indices,dim,lo,hi);
      Kokkos::parallel_for(nrecv/size_exchange,f);
    }
    k_count.modify<LMPDeviceType>();
    k_count.sync<LMPHostType>();
  }

  atomKK->modified(space,X_MASK | V_MASK | TAG_MASK | TYPE_MASK |
                 MASK_MASK | IMAGE_MASK| RHO_MASK | ESPH_MASK |
                 VEST_MASK | CV_MASK);

  return k_count.h_view(0);
}

/* ---------------------------------------------------------------------- */

void AtomVecSPHKokkos::sync(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if (mask & X_MASK) atomKK->k_x.sync<LMPDeviceType>();
    if (mask & V_MASK) atomKK->k_v.sync<LMPDeviceType>();
    if (mask & F_MASK) atomKK->k_f.sync<LMPDeviceType>();
    if (mask & TAG_MASK) atomKK->k_tag.sync<LMPDeviceType>();
    if (mask & TYPE_MASK) atomKK->k_type.sync<LMPDeviceType>();
    if (mask & MASK_MASK) atomKK->k_mask.sync<LMPDeviceType>();
    if (mask & IMAGE_MASK) atomKK->k_image.sync<LMPDeviceType>();
    if (mask & RHO_MASK) atomKK->k_rho.sync<LMPDeviceType>();
    if (mask & ESPH_MASK) atomKK->k_esph.sync<LMPDeviceType>();
    if (mask & VEST_MASK) atomKK->k_vest.sync<LMPDeviceType>();
    if (mask & CV_MASK) atomKK->k_cv.sync<LMPDeviceType>();
  } else {
    if (mask & X_MASK) atomKK->k_x.sync<LMPHostType>();
    if (mask & V_MASK) atomKK->k_v.sync<LMPHostType>();
    if (mask & F_MASK) atomKK->k_f.sync<LMPHostType>();
    if (mask & TAG_MASK) atomKK->k_tag.sync<LMPHostType>();
    if (mask & TYPE_MASK) atomKK->k_type.sync<LMPHostType>();
    if (mask & MASK_MASK) atomKK->k_mask.sync<LMPHostType>();
    if (mask & IMAGE_MASK) atomKK->k_image.sync<LMPHostType>();
    if (mask & RHO_MASK) atomKK->k_rho.sync<LMPHostType>();
    if (mask & ESPH_MASK) atomKK->k_esph.sync<LMPHostType>();
    if (mask & VEST_MASK) atomKK->k_vest.sync<LMPHostType>();
    if (mask & CV_MASK) atomKK->k_cv.sync<LMPHostType>();
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecSPHKokkos::sync_overlapping_device(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if ((mask & X_MASK) && atomKK->k_x.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_x_array>(atomKK->k_x,space);
    if ((mask & V_MASK) && atomKK->k_v.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_v_array>(atomKK->k_v,space);
    if ((mask & F_MASK) && atomKK->k_f.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_f_array>(atomKK->k_f,space);
    if ((mask & TAG_MASK) && atomKK->k_tag.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_tagint_1d>(atomKK->k_tag,space);
    if ((mask & TYPE_MASK) && atomKK->k_type.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_type,space);
    if ((mask & MASK_MASK) && atomKK->k_mask.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_mask,space);
    if ((mask & IMAGE_MASK) && atomKK->k_image.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_imageint_1d>(atomKK->k_image,space);
    if ((mask & RHO_MASK) && atomKK->k_rho.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_float_1d>(atomKK->k_rho,space);
    if ((mask & ESPH_MASK) && atomKK->k_esph.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_float_1d>(atomKK->k_esph,space);
    if ((mask & VEST_MASK) && atomKK->k_vest.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_v_array>(atomKK->k_vest,space);
    if ((mask & CV_MASK) && atomKK->k_cv.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_float_1d>(atomKK->k_cv,space);
  } else {
    if ((mask & X_MASK) && atomKK->k_x.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_x_array>(atomKK->k_x,space);
    if ((mask & V_MASK) && atomKK->k_v.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_v_array>(atomKK->k_v,space);
    if ((mask & F_MASK) && atomKK->k_f.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_f_array>(atomKK->k_f,space);
    if ((mask & TAG_MASK) && atomKK->k_tag.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_tagint_1d>(atomKK->k_tag,space);
    if ((mask & TYPE_MASK) && atomKK->k_type.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_type,space);
    if ((mask & MASK_MASK) && atomKK->k_mask.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_mask,space);
    if ((mask & IMAGE_MASK) && atomKK->k_image.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_imageint_1d>(atomKK->k_image,space);
    if ((mask & RHO_MASK) && atomKK->k_rho.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_float_1d>(atomKK->k_rho,space);
    if ((mask & ESPH_MASK) && atomKK->k_esph.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_float_1d>(atomKK->k_esph,space);
    if ((mask & VEST_MASK) && atomKK->k_vest.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_v_array>(atomKK->k_vest,space);
    if ((mask & CV_MASK) && atomKK->k_cv.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_float_1d>(atomKK->k_cv,space);
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecSPHKokkos::modified(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if (mask & X_MASK) atomKK->k_x.modify<LMPDeviceType>();
    if (mask & V_MASK) atomKK->k_v.modify<LMPDeviceType>();
    if (mask & F_MASK) atomKK->k_f.modify<LMPDeviceType>();
    if (mask & TAG_MASK) atomKK->k_tag.modify<LMPDeviceType>();
    if (mask & TYPE_MASK) atomKK->k_type.modify<LMPDeviceType>();
    if (mask & MASK_MASK) atomKK->k_mask.modify<LMPDeviceType>();
    if (mask & IMAGE_MASK) atomKK->k_image.modify<LMPDeviceType>();
    if (mask & RHO_MASK) atomKK->k_rho.modify<LMPDeviceType>();
    if (mask & ESPH_MASK) atomKK->k_esph.modify<LMPDeviceType>();
    if (mask & VEST_MASK) atomKK->k_vest.modify<LMPDeviceType>();
    if (mask & CV_MASK) atomKK->k_cv.modify<LMPDeviceType>();
  } else {
    if (mask & X_MASK) atomKK->k_x.modify<LMPHostType>();
    if (mask & V_MASK) atomKK->k_v.modify<LMPHostType>();
    if (mask & F_MASK) atomKK->k_f.modify<LMPHostType>();
    if (mask & TAG_MASK) atomKK->k_tag.modify<LMPHostType>();
    if (mask & TYPE_MASK) atomKK->k_type.modify<LMPHostType>();
    if (mask & MASK_MASK) atomKK->k_mask.modify<LMPHostType>();
    if (mask & IMAGE_MASK) atomKK->k_image.modify<LMPHostType>();
    if (mask & RHO_MASK) atomKK->k_rho.modify<LMPHostType>();
    if (mask & ESPH_MASK) atomKK->k_esph.modify<LMPHostType>();
    if (mask & VEST_MASK) atomKK->k_vest.modify<LMPHostType>();
    if (mask & CV_MASK) atomKK->k_cv.modify<LMPHostType>();
  }
}
