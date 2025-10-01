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

#ifdef ATOM_CLASS
// clang-format off
AtomStyle(full/sph/x0image/kk,AtomVecFullSPHX0ImageKokkos);
AtomStyle(full/sph/x0image/kk/device,AtomVecFullSPHX0ImageKokkos);
AtomStyle(full/sph/x0image/kk/host,AtomVecFullSPHX0ImageKokkos);
// clang-format on
#else

// clang-format off
#ifndef LMP_ATOM_VEC_FULL_SPH_X0IMAGE_KOKKOS_H
#define LMP_ATOM_VEC_FULL_SPH_X0IMAGE_KOKKOS_H

#include "atom_vec_kokkos.h"
#include "atom_vec_full_sph_x0image.h"

namespace LAMMPS_NS {

class AtomVecFullSPHX0ImageKokkos : public AtomVecKokkos, public AtomVecFullSPHX0Image {
 public:
  AtomVecFullSPHX0ImageKokkos(class LAMMPS *);

  void grow(int) override;
  void grow_pointers() override;
  void sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter) override;
  int pack_border_kokkos(int n, DAT::tdual_int_2d k_sendlist,
                         DAT::tdual_xfloat_2d buf,int iswap,
                         int pbc_flag, int *pbc, ExecutionSpace space) override;
  void unpack_border_kokkos(const int &n, const int &nfirst,
                            const DAT::tdual_xfloat_2d &buf,
                            ExecutionSpace space) override;

  void force_clear_kokkos(int, size_t) override;
  
  /*
  int pack_exchange_kokkos(const int &nsend,DAT::tdual_xfloat_2d &buf,
                           DAT::tdual_int_1d k_sendlist,
                           DAT::tdual_int_1d k_copylist,
                           ExecutionSpace space) override;
  int unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, int nrecv,
                             int nlocal, int dim, X_FLOAT lo, X_FLOAT hi,
                             ExecutionSpace space,
                             DAT::tdual_int_1d &k_indices) override;

  */

  void sync(ExecutionSpace space, unsigned int mask) override;
  void modified(ExecutionSpace space, unsigned int mask) override;
  void sync_overlapping_device(ExecutionSpace space, unsigned int mask) override;

  /*

  int pack_comm_kokkos(const int &n, const DAT::tdual_int_2d &k_sendlist,
                       const int & iswap,
                       const DAT::tdual_xfloat_2d &buf,
                       const int &pbc_flag, const int pbc[]) override;
  void unpack_comm_kokkos(const int &n, const int &nfirst,
                          const DAT::tdual_xfloat_2d &buf) override;
  int pack_comm_vel_kokkos(const int &n, const DAT::tdual_int_2d &k_sendlist,
                           const int & iswap,
                           const DAT::tdual_xfloat_2d &buf,
                           const int &pbc_flag, const int pbc[]) override;
  void unpack_comm_vel_kokkos(const int &n, const int &nfirst,
                              const DAT::tdual_xfloat_2d &buf) override;
  int pack_comm_self(const int &n, const DAT::tdual_int_2d &list,
                     const int & iswap, const int nfirst,
                     const int &pbc_flag, const int pbc[]) override;
  int pack_reverse_kokkos(const int &n, const int &nfirst,
                        const DAT::tdual_ffloat_2d &buf) override;
  int unpack_reverse_self(const int &n, const DAT::tdual_int_2d &list,
                      const int & iswap, const int nfirst) override;
  void unpack_reverse_kokkos(const int &n, const DAT::tdual_int_2d &list,
                          const int & iswap, const DAT::tdual_ffloat_2d &buf) override;
  int pack_border_vel_kokkos(int n, DAT::tdual_int_2d k_sendlist,
                             DAT::tdual_xfloat_2d buf,int iswap,
                             int pbc_flag, int *pbc, ExecutionSpace space) override;
  void unpack_border_vel_kokkos(const int &n, const int &nfirst,
                                const DAT::tdual_xfloat_2d &buf,
                                ExecutionSpace space) override;

  */

 protected:
  double *q;
  tagint *molecule;
  tagint **special;
  tagint **bond_atom;
  tagint **angle_atom1,**angle_atom2,**angle_atom3;
  tagint **dihedral_atom1,**dihedral_atom2,**dihedral_atom3,**dihedral_atom4;
  tagint **improper_atom1,**improper_atom2,**improper_atom3,**improper_atom4;

  DAT::t_tagint_1d d_tag;
  DAT::t_int_1d d_type, d_mask;
  HAT::t_tagint_1d h_tag;
  HAT::t_int_1d h_type, h_mask;

  DAT::t_imageint_1d d_image;
  HAT::t_imageint_1d h_image;

  DAT::t_x_array d_x;
  DAT::t_v_array d_v;
  DAT::t_f_array d_f;

  DAT::t_float_1d d_q;
  HAT::t_float_1d h_q;

  DAT::t_tagint_1d d_molecule;
  DAT::t_int_2d d_nspecial;
  DAT::t_tagint_2d d_special;
  DAT::t_int_1d d_num_bond;
  DAT::t_int_2d d_bond_type;
  DAT::t_tagint_2d d_bond_atom;

  HAT::t_tagint_1d h_molecule;
  HAT::t_int_2d h_nspecial;
  HAT::t_tagint_2d h_special;
  HAT::t_int_1d h_num_bond;
  HAT::t_int_2d h_bond_type;
  HAT::t_tagint_2d h_bond_atom;

  DAT::t_int_1d d_num_angle;
  DAT::t_int_2d d_angle_type;
  DAT::t_tagint_2d d_angle_atom1,d_angle_atom2,d_angle_atom3;

  HAT::t_int_1d h_num_angle;
  HAT::t_int_2d h_angle_type;
  HAT::t_tagint_2d h_angle_atom1,h_angle_atom2,h_angle_atom3;

  DAT::t_int_1d d_num_dihedral;
  DAT::t_int_2d d_dihedral_type;
  DAT::t_tagint_2d d_dihedral_atom1,d_dihedral_atom2,
    d_dihedral_atom3,d_dihedral_atom4;
  DAT::t_int_1d d_num_improper;
  DAT::t_int_2d d_improper_type;
  DAT::t_tagint_2d d_improper_atom1,d_improper_atom2,
    d_improper_atom3,d_improper_atom4;

  HAT::t_int_1d h_num_dihedral;
  HAT::t_int_2d h_dihedral_type;
  HAT::t_tagint_2d h_dihedral_atom1,h_dihedral_atom2,
    h_dihedral_atom3,h_dihedral_atom4;
  HAT::t_int_1d h_num_improper;
  HAT::t_int_2d h_improper_type;
  HAT::t_tagint_2d h_improper_atom1,h_improper_atom2,
    h_improper_atom3,h_improper_atom4;


  DAT::t_v_array d_vest;
  DAT::t_float_1d d_rho;
  DAT::t_float_1d d_drho;
  DAT::t_float_1d d_esph;
  DAT::t_float_1d d_desph;
  DAT::t_float_1d d_cv;
  HAT::t_v_array h_vest;
  HAT::t_float_1d h_rho;
  HAT::t_float_1d h_drho;
  HAT::t_float_1d h_esph;
  HAT::t_float_1d h_desph;
  HAT::t_float_1d h_cv;
  DAT::t_float_1d d_rmass;
  HAT::t_float_1d h_rmass;

  DAT::t_x_array d_x0;
  HAT::t_x_array h_x0;
};

}

#endif
#endif
