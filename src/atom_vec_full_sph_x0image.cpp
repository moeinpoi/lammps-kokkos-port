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

#include "atom_vec_full_sph_x0image.h"
#include "atom.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecFullSPHX0Image::AtomVecFullSPHX0Image(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = Atom::MOLECULAR;
  bonds_allow = angles_allow = dihedrals_allow = impropers_allow = 1;
  mass_type = PER_TYPE;
  atom->molecule_flag = atom->q_flag = 1;

  forceclearflag = 1;
  atom->esph_flag = 1;
  atom->rho_flag = 1;
  atom->cv_flag = 1;
  atom->vest_flag = 1;
  

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file

  // clang-format off
  fields_grow = {"q", "molecule", "num_bond", "bond_type", "bond_atom", "num_angle", "angle_type",
    "angle_atom1", "angle_atom2", "angle_atom3", "num_dihedral", "dihedral_type", "dihedral_atom1",
    "dihedral_atom2", "dihedral_atom3", "dihedral_atom4", "num_improper", "improper_type",
    "improper_atom1", "improper_atom2", "improper_atom3", "improper_atom4", "nspecial", "special",
    "rho", "drho", "esph", "desph", "cv", "vest",
    "x0"};
  fields_copy = {"q", "molecule", "num_bond", "bond_type", "bond_atom", "num_angle", "angle_type",
    "angle_atom1", "angle_atom2", "angle_atom3", "num_dihedral", "dihedral_type", "dihedral_atom1",
    "dihedral_atom2", "dihedral_atom3", "dihedral_atom4", "num_improper", "improper_type",
    "improper_atom1", "improper_atom2", "improper_atom3", "improper_atom4", "nspecial", "special",
    "rho", "drho", "esph", "desph", "cv", "vest",
    "x0"};
  fields_comm = {"rho", "esph", "vest"};
  fields_comm_vel = {"rho", "esph", "vest"};
  fields_reverse = {"drho", "desph"};
  fields_border = {"q", "molecule", "rho", "esph", "cv", "vest", "x0"};
  fields_border_vel = {"q", "molecule", "rho", "esph", "cv", "vest", "x0"};
  fields_exchange = {"q", "molecule", "num_bond", "bond_type", "bond_atom",
    "num_angle", "angle_type", "angle_atom1", "angle_atom2", "angle_atom3",
    "num_dihedral", "dihedral_type", "dihedral_atom1", "dihedral_atom2",
    "dihedral_atom3", "dihedral_atom4", "num_improper", "improper_type", "improper_atom1",
    "improper_atom2", "improper_atom3", "improper_atom4", "nspecial", "special",
    "rho", "esph", "cv", "vest", "x0"};
 fields_restart = {"q", "molecule", "num_bond", "bond_type", "bond_atom", "num_angle",
   "angle_type", "angle_atom1", "angle_atom2", "angle_atom3", "num_dihedral", "dihedral_type",
   "dihedral_atom1", "dihedral_atom2", "dihedral_atom3", "dihedral_atom4", "num_improper",
   "improper_type", "improper_atom1", "improper_atom2", "improper_atom3", "improper_atom4",
    "rho", "esph", "cv", "vest", "x0"};
  fields_create = {"q", "molecule", "num_bond", "num_angle",
    "num_dihedral", "num_improper", "nspecial",
    "rho", "esph", "cv", "vest", "x0"};
  fields_data_atom = {"id", "molecule", "type", "q", "rho", "esph", "cv", "x0", "x"};
  fields_data_vel = {"id", "v"};
  // clang-format on
  setup_fields();

  bond_per_atom = angle_per_atom = dihedral_per_atom = improper_per_atom = 0;
  bond_negative = angle_negative = dihedral_negative = improper_negative = nullptr;
}

/* ---------------------------------------------------------------------- */

AtomVecFullSPHX0Image::~AtomVecFullSPHX0Image()
{
  delete[] bond_negative;
  delete[] angle_negative;
  delete[] dihedral_negative;
  delete[] improper_negative;
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()
------------------------------------------------------------------------- */

void AtomVecFullSPHX0Image::grow_pointers()
{
  num_bond = atom->num_bond;
  bond_type = atom->bond_type;
  num_angle = atom->num_angle;
  angle_type = atom->angle_type;
  num_dihedral = atom->num_dihedral;
  dihedral_type = atom->dihedral_type;
  num_improper = atom->num_improper;
  improper_type = atom->improper_type;
  nspecial = atom->nspecial;

  rho = atom->rho;
  drho = atom->drho;
  esph = atom->esph;
  desph = atom->desph;
  cv = atom->cv;
  vest = atom->vest;

  x0 = atom->x0;
}

/* ----------------------------------------------------------------------
   clear extra forces starting at atom N
   nbytes = # of bytes to clear for a per-atom vector
------------------------------------------------------------------------- */

void AtomVecFullSPHX0Image::force_clear(int n, size_t nbytes)
{
  memset(&desph[n], 0, nbytes);
  memset(&drho[n], 0, nbytes);
}

/* ----------------------------------------------------------------------
   initialize non-zero atom quantities
------------------------------------------------------------------------- */

void AtomVecFullSPHX0Image::create_atom_post(int ilocal)
{
  cv[ilocal] = 1.0;
  const auto xinit = atom->x;
  x0[ilocal][0] = xinit[ilocal][0];
  x0[ilocal][1] = xinit[ilocal][1];
  x0[ilocal][2] = xinit[ilocal][2];
}

/* ----------------------------------------------------------------------
   assign an index to named atom property and return index
   return -1 if name is unknown to this atom style
------------------------------------------------------------------------- */

int AtomVecFullSPHX0Image::property_atom(const std::string &name)
{
  if (name == "rho") return 0;
  if (name == "drho") return 1;
  if (name == "esph") return 2;
  if (name == "desph") return 3;
  if (name == "cv") return 4;

  if (name == "x0") return 5;
  if (name == "y0") return 6;
  if (name == "z0") return 7;

  return -1;
}

/* ----------------------------------------------------------------------
   pack per-atom data into buf for ComputePropertyAtom
   index maps to data specific to this atom style
------------------------------------------------------------------------- */

void AtomVecFullSPHX0Image::pack_property_atom(int index, double *buf, int nvalues, int groupbit)
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int n = 0;

  if (index == 0) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit)
        buf[n] = rho[i];
      else
        buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 1) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit)
        buf[n] = drho[i];
      else
        buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 2) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit)
        buf[n] = esph[i];
      else
        buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 3) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit)
        buf[n] = desph[i];
      else
        buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 4) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit)
        buf[n] = cv[i];
      else
        buf[n] = 0.0;
      n += nvalues;
    }
  }
  else if (index == 5) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit)
        buf[n] = x0[i][0];
      else
        buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 6) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit)
        buf[n] = x0[i][1];
      else
        buf[n] = 0.0;
      n += nvalues;
    } 
  } else if (index == 7) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit)
        buf[n] = x0[i][2];
      else
        buf[n] = 0.0;
      n += nvalues;
    }
  }
}

/* ----------------------------------------------------------------------
   modify values for AtomVec::pack_restart() to pack
------------------------------------------------------------------------- */

void AtomVecFullSPHX0Image::pack_restart_pre(int ilocal)
{
  // ensure negative vectors are needed length

  if (bond_per_atom < atom->bond_per_atom) {
    delete[] bond_negative;
    bond_per_atom = atom->bond_per_atom;
    bond_negative = new int[bond_per_atom];
  }
  if (angle_per_atom < atom->angle_per_atom) {
    delete[] angle_negative;
    angle_per_atom = atom->angle_per_atom;
    angle_negative = new int[angle_per_atom];
  }
  if (dihedral_per_atom < atom->dihedral_per_atom) {
    delete[] dihedral_negative;
    dihedral_per_atom = atom->dihedral_per_atom;
    dihedral_negative = new int[dihedral_per_atom];
  }
  if (improper_per_atom < atom->improper_per_atom) {
    delete[] improper_negative;
    improper_per_atom = atom->improper_per_atom;
    improper_negative = new int[improper_per_atom];
  }

  // flip any negative types to positive and flag which ones

  any_bond_negative = 0;
  for (int m = 0; m < num_bond[ilocal]; m++) {
    if (bond_type[ilocal][m] < 0) {
      bond_negative[m] = 1;
      bond_type[ilocal][m] = -bond_type[ilocal][m];
      any_bond_negative = 1;
    } else
      bond_negative[m] = 0;
  }

  any_angle_negative = 0;
  for (int m = 0; m < num_angle[ilocal]; m++) {
    if (angle_type[ilocal][m] < 0) {
      angle_negative[m] = 1;
      angle_type[ilocal][m] = -angle_type[ilocal][m];
      any_angle_negative = 1;
    } else
      angle_negative[m] = 0;
  }

  any_dihedral_negative = 0;
  for (int m = 0; m < num_dihedral[ilocal]; m++) {
    if (dihedral_type[ilocal][m] < 0) {
      dihedral_negative[m] = 1;
      dihedral_type[ilocal][m] = -dihedral_type[ilocal][m];
      any_dihedral_negative = 1;
    } else
      dihedral_negative[m] = 0;
  }

  any_improper_negative = 0;
  for (int m = 0; m < num_improper[ilocal]; m++) {
    if (improper_type[ilocal][m] < 0) {
      improper_negative[m] = 1;
      improper_type[ilocal][m] = -improper_type[ilocal][m];
      any_improper_negative = 1;
    } else
      improper_negative[m] = 0;
  }
}

/* ----------------------------------------------------------------------
   unmodify values packed by AtomVec::pack_restart()
------------------------------------------------------------------------- */

void AtomVecFullSPHX0Image::pack_restart_post(int ilocal)
{
  // restore the flagged types to their negative values

  if (any_bond_negative) {
    for (int m = 0; m < num_bond[ilocal]; m++)
      if (bond_negative[m]) bond_type[ilocal][m] = -bond_type[ilocal][m];
  }

  if (any_angle_negative) {
    for (int m = 0; m < num_angle[ilocal]; m++)
      if (angle_negative[m]) angle_type[ilocal][m] = -angle_type[ilocal][m];
  }

  if (any_dihedral_negative) {
    for (int m = 0; m < num_dihedral[ilocal]; m++)
      if (dihedral_negative[m]) dihedral_type[ilocal][m] = -dihedral_type[ilocal][m];
  }

  if (any_improper_negative) {
    for (int m = 0; m < num_improper[ilocal]; m++)
      if (improper_negative[m]) improper_type[ilocal][m] = -improper_type[ilocal][m];
  }
}

/* ----------------------------------------------------------------------
   initialize other atom quantities after AtomVec::unpack_restart()
------------------------------------------------------------------------- */

void AtomVecFullSPHX0Image::unpack_restart_init(int ilocal)
{
  nspecial[ilocal][0] = 0;
  nspecial[ilocal][1] = 0;
  nspecial[ilocal][2] = 0;
}

/* ----------------------------------------------------------------------
   modify what AtomVec::data_atom() just unpacked
   or initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecFullSPHX0Image::data_atom_post(int ilocal)
{
  num_bond[ilocal] = 0;
  num_angle[ilocal] = 0;
  num_dihedral[ilocal] = 0;
  num_improper[ilocal] = 0;
  nspecial[ilocal][0] = 0;
  nspecial[ilocal][1] = 0;
  nspecial[ilocal][2] = 0;

  vest[ilocal][0] = 0.0;
  vest[ilocal][1] = 0.0;
  vest[ilocal][2] = 0.0;
  desph[ilocal] = 0.0;
  drho[ilocal] = 0.0;

  const auto xinit = atom->x;
  x0[ilocal][0] = xinit[ilocal][0];
  x0[ilocal][1] = xinit[ilocal][1];
  x0[ilocal][2] = xinit[ilocal][2];
}
