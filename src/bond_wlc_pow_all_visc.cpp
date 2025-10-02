/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "math.h"
#include "stdlib.h"
#include "bond_wlc_pow_all_visc.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "random_mars.h"
#include "modify.h"
#include "string.h"
#include "update.h"
#include "fix.h"
#include "fix_store_atom.h"
#include "fix_store_global.h"


using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondWLCPowAllVisc::BondWLCPowAllVisc(LAMMPS *lmp) : Bond(lmp) {

    random = new RanMars(lmp, 1000 + comm->me);
    seed_ = (uint64_t)((comm->me + 3) * 2846u);

}

/* ---------------------------------------------------------------------- */

BondWLCPowAllVisc::~BondWLCPowAllVisc()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(temp);
    memory->destroy(r0);
    memory->destroy(mu_targ);
    memory->destroy(qp);
    memory->destroy(gamc);
    memory->destroy(gamt);
    memory->destroy(sigc);
    memory->destroy(sigt);
  }

}
/* ---------------------------------------------------------------------- */

void BondWLCPowAllVisc::init_style()
{

    init_on=0;

  double sdtt = sqrt(update->dt);
  for (int i = 1; i <= atom->nbondtypes; i++){
  //  if (setflag[i] == 0) error->all("All bond coeffs are not set");
  //  if (gamt[i]>3.0*gamc[i]) error->all("Gamma_t > 3*Gamma_c");
    sigc[i] = sqrt(2.0*temp[i]*(3.0*gamc[i]-gamt[i]))/sdtt;
    sigt[i] = 2.0*sqrt(gamt[i]*temp[i])/sdtt;
  }


}

/* ---------------------------------------------------------------------- */

void BondWLCPowAllVisc::compute(int eflag, int vflag)
{

  double ebond,fbond;

  int i1,i2,n,type,factor,i;
  double rr,fforce,rfactor,ra,rlogarg,kph,l0,lmax,mu,lambda;
  double dvx, dvy, dvz, vv;
  char warning[128];
  double ff[6];
  
  double **x0 = atom->x0;    
  double **x = atom->x;
  int nlocal = atom->nlocal;
  int nghost= atom->nghost; //do we need it?

  ebond = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **f = atom->f;
  double **v = atom->v;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];

    delx = x0[i1][0] - x0[i2][0];
    dely = x0[i1][1] - x0[i2][1];
    delz = x0[i1][2] - x0[i2][2];
    l0 = sqrt(delx*delx + dely*dely + delz*delz);

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];
    dvx = v[i1][0] - v[i2][0];
    dvy = v[i1][1] - v[i2][1];
    dvz = v[i1][2] - v[i2][2];

    // force from log term
    ra = sqrt(delx*delx + dely*dely + delz*delz);
    lmax = l0*r0[type];
    rr = 1.0/r0[type];  

    kph = pow(l0,qp[type])*temp[type]*(0.25/(1.0-rr)/(1.0-rr)-0.25+rr);
    mu = 0.25*sqrt(3.0)*(temp[type]*(-0.25/pow(1.0-rr,2) + 0.25 + 0.5*rr/pow(1.0-rr,3))/lmax/rr + kph*(qp[type]+1.0)/pow(l0,qp[type]+1.0));

    lambda = mu/mu_targ[type];
    kph = kph*mu_targ[type]/mu;
    rr = ra/lmax; 
    rlogarg = pow(ra,qp[type]+1.0);
    vv = (delx*dvx + dely*dvy +  delz*dvz)/ra;

    generate_wrr();
       
    fforce = -temp[type]*(0.25/(1.0-rr)/(1.0-rr)-0.25+rr)/lambda/ra + kph/rlogarg + (sigc[type]*wrr[3] - gamc[type]*vv)/ra;

    ebond = 0.0;
    if (eflag) {
      ebond += 0.25*temp[type]*lmax*(3.0*rr*rr-2.0*rr*rr*rr)/(1.0-rr)/lambda;
      if (qp[type] == 1.0)
        ebond -= kph*log(ra);
      else
        ebond += kph/(qp[type]-1.0)/pow(ra,qp[type]-1.0);
    }

    // apply force to each of 2 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += delx*fforce - gamt[type]*dvx + sigt[type]*wrr[0]/ra;
      f[i1][1] += dely*fforce - gamt[type]*dvy + sigt[type]*wrr[1]/ra;
      f[i1][2] += delz*fforce - gamt[type]*dvz + sigt[type]*wrr[2]/ra;
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= delx*fforce - gamt[type]*dvx + sigt[type]*wrr[0]/ra;
      f[i2][1] -= dely*fforce - gamt[type]*dvy + sigt[type]*wrr[1]/ra;
      f[i2][2] -= delz*fforce - gamt[type]*dvz + sigt[type]*wrr[2]/ra;
    }
    
    fbond=fforce-gamt[type]*sqrt(dvx*dvx+dvy*dvy+dvz*dvz)/ra + sigt[type]*sqrt(wrr[0]*wrr[0]+wrr[1]*wrr[1]+wrr[2]*wrr[2])/(ra*ra);
    if (evflag) ev_tally(i1,i2,nlocal,newton_bond,ebond,fbond,delx,dely,delz);
  }
}

/* ---------------------------------------------------------------------- */

void BondWLCPowAllVisc::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  seed_ = (comm->me + 3) * 2846u;
  srand48((long)seed_);

  memory->create(temp,n+1,"bond:temp");
  memory->create(r0,n+1,"bond:r0");
  memory->create(mu_targ,n+1,"bond:mu_targ");
  memory->create(qp,n+1,"bond:qp");
  memory->create(gamc,n+1,"bond:gamc");
  memory->create(gamt,n+1,"bond:gamt");
  memory->create(sigc,n+1,"bond:sigc");
  memory->create(sigt,n+1,"bond:sigt"); 
 
  memory->create(setflag,n+1,"bond:setflag");

  for (int i = 1; i <= n; i++) setflag[i] = 0;
 
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondWLCPowAllVisc::coeff(int narg, char **arg)
{
  if (narg != 7) error->all(FLERR,"Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  utils::bounds(FLERR,arg[0],1,atom->nbondtypes,ilo,ihi,error);

  double temp_one = utils::numeric(FLERR, arg[1], true, lmp);
  double r0_one = utils::numeric(FLERR, arg[2], true, lmp);
  double mu_one = utils::numeric(FLERR, arg[3], true, lmp);
  double qp_one = utils::numeric(FLERR, arg[4],true, lmp);
  double gamc_one = utils::numeric(FLERR, arg[5], true, lmp);
  double gamt_one = utils::numeric(FLERR, arg[6], true, lmp);


  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    temp[i] = temp_one;
    r0[i] = r0_one;
    mu_targ[i] = mu_one;
    qp[i] = qp_one;
    gamc[i] = gamc_one;
    gamt[i] = gamt_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double BondWLCPowAllVisc::equilibrium_distance(int i)
{
  return r0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondWLCPowAllVisc::write_restart(FILE *fp)
{
  fwrite(&temp[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&r0[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&mu_targ[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&qp[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&gamc[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&gamt[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondWLCPowAllVisc::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&temp[1],sizeof(double),atom->nbondtypes,fp);
    fread(&r0[1],sizeof(double),atom->nbondtypes,fp);
    fread(&mu_targ[1],sizeof(double),atom->nbondtypes,fp);
    fread(&qp[1],sizeof(double),atom->nbondtypes,fp);
    fread(&gamc[1],sizeof(double),atom->nbondtypes,fp);
    fread(&gamt[1],sizeof(double),atom->nbondtypes,fp);
  }
  MPI_Bcast(&temp[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r0[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&mu_targ[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&qp[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&gamc[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&gamt[1],atom->nbondtypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;

}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondWLCPowAllVisc::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g %g %g %g %g\n",i,temp[i],r0[i],mu_targ[i],qp[i],gamc[i],gamt[i]);

}

/* ---------------------------------------------------------------------- */
//check later now is a fake function
double BondWLCPowAllVisc::single(int type, double rsq, int i, int j,
                        double &fforce)
{
  return 1.0;
}

/* ---------------------------------------------------------------------- */


void BondWLCPowAllVisc::generate_wrr()
{
  int i;
  double ww[3][3] = {{0}};
  double v1, v2, factor, ss;

  for (i=0; i<5; i++){
    ss = 100.0;
    while ( ss > 1.0 ){
      //v1 = 2.0 * drand48() - 1.0;
      //v2 = 2.0 * drand48() - 1.0;
      v1 = 2.0 * random->uniform() - 1.0;
      v2 = 2.0 * random->uniform() - 1.0;
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
}




