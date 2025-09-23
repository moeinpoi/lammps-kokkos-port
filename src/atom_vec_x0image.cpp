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
   Contributing author: Mike Parks (SNL)
------------------------------------------------------------------------- */

#include "atom_vec_x0image.h"

#include "atom.h"
#include "citeme.h"
#include "error.h"

#include <cfloat>

using namespace LAMMPS_NS;

static const char cite_peri_package[] =
    "PERI package for X0Imagedynamics: doi:10.1016/j.cpc.2008.06.011\n\n"
    "@Article{Parks08,\n"
    " author = {M. L. Parks and R. B. Lehoucq and S. J. Plimpton and S. A. Silling},\n"
    " title = {Implementing X0Imagedynamics Within a Molecular Dynamics Code},\n"
    " journal = {Comput.\\ Phys.\\ Commun.},\n"
    " year =    2008,\n"
    " volume =  179,\n"
    " number =  11,\n"
    " pages =   {777--783}\n"
    "}\n\n";

/* ---------------------------------------------------------------------- */

AtomVecX0Image::AtomVecX0Image(LAMMPS *lmp) : AtomVec(lmp)
{
  if (lmp->citeme) lmp->citeme->add(cite_peri_package);

  molecular = Atom::ATOMIC;

//  atom->rmass_flag = 1;
//  atom->peri_flag = 1;
//  atom->vfrac_flag = 1;

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file


  fields_grow = {"x0"};
  fields_copy = {"x0"};
  fields_border = {"x0"};
  fields_border_vel = {"x0"};
  fields_exchange = {"x0"};
  fields_restart = {"x0"};
  fields_create = {"x0"};
  fields_data_atom = {"id", "type", "x0", "x"};
  fields_data_vel = {"id", "v"};

  setup_fields();
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()
------------------------------------------------------------------------- */

void AtomVecX0Image::grow_pointers()
{
  x0 = atom->x0;
}

/* ----------------------------------------------------------------------
   initialize non-zero atom quantities
------------------------------------------------------------------------- */

void AtomVecX0Image::create_atom_post(int ilocal)
{
  const auto xinit = atom->x;
  x0[ilocal][0] = xinit[ilocal][0];
  x0[ilocal][1] = xinit[ilocal][1];
  x0[ilocal][2] = xinit[ilocal][2];
}

/* ----------------------------------------------------------------------
   modify what AtomVec::data_atom() just unpacked
   or initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecX0Image::data_atom_post(int ilocal)
{
  const auto xinit = atom->x;
  x0[ilocal][0] = xinit[ilocal][0];
  x0[ilocal][1] = xinit[ilocal][1];
  x0[ilocal][2] = xinit[ilocal][2];

//  if (rmass[ilocal] <= 0.0) error->one(FLERR, "Invalid mass in Atoms section of data file");
}

/* ----------------------------------------------------------------------
   assign an index to named atom property and return index
   return -1 if name is unknown to this atom style
------------------------------------------------------------------------- */

int AtomVecX0Image::property_atom(const std::string &name)
{
//  if (name == "vfrac") return 0;
//  if (name == "s0") return 1;
  return -1;
}

/* ----------------------------------------------------------------------
   pack per-atom data into buf for ComputePropertyAtom
   index maps to data specific to this atom style
------------------------------------------------------------------------- */

void AtomVecX0Image::pack_property_atom(int index, double *buf, int nvalues, int groupbit)
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int n = 0;

}
