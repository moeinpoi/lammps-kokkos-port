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
#include "angle_area_volume3.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "fix.h"
#include "fix_store_atom.h"
#include "fix_store_global.h"
#include "update.h"
#include "modify.h"
#include "string.h"


using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.001

/* ---------------------------------------------------------------------- */

AngleAreaVolume3::AngleAreaVolume3(LAMMPS *lmp) : Angle(lmp) 
{
}

/* ---------------------------------------------------------------------- */

AngleAreaVolume3::~AngleAreaVolume3()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(ka);
    memory->destroy(a0);
    memory->destroy(kv);
    memory->destroy(v0);
    memory->destroy(kl);
    memory->destroy(aa0);   
    memory->destroy(press);
    memory->destroy(qp);
    memory->destroy(r0);
    memory->destroy(nstep1);
    memory->destroy(nstep2);
    memory->destroy(vratio);
  }

  if (init_on && !copymode) {
    memory->destroy(h_area);
    memory->destroy(h_volume);
    memory->destroy(ttyp);
    memory->destroy(ttyp1);
    memory->destroy(dath);
    memory->destroy(datt);
  }
}

/* ---------------------------------------------------------------------- */

void AngleAreaVolume3::compute(int eflag, int vflag)
{

  double eangle,f1[3],f2[3],f3[3], f1x[3],f2x[3],f3x[3];
  int i1,i2,i3,n,m,j,type;
  double d21x,d21y,d21z,d31x,d31y,d31z,d32x,d32y,d32z;
  double d12x,d12y,d12z;
  double nx,ny,nz,nn,mx,my,mz,aa,vv,ar0;
  double coefc, coefl, coefa, coefv, coefca;
  double s1x,s1y,s1z,s2x,s2y,s2z,s3x,s3y,s3z;
  double s1xv,s1yv,s1zv,s2xv,s2yv,s2zv,s3xv,s3yv,s3zv;
  double s1xvx,s1yvx,s1zvx,s2xvx,s2yvx,s2zvx,s3xvx,s3yvx,s3zvx;

  double xx1[3],xx2[3],xx3[3]; 
  double ff[6];
  double voltemp,tempxxx;
  double energy_l,energy_a,energy_v;

  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int newton_bond = force->newton_bond;

  double **x0 = atom->x0;
  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;

  int i, nnmol;
  nm = 0; 
  nnmol = 0;

  //find max mol ID in this rank
  for (i=0; i<nlocal; i++) {
    if (nnmol < atom->molecule[i]) {
      nnmol = atom->molecule[i];
    }
  }

  //highest mol ID in the entire sim (across all ranks) will be stored in nm
  MPI_Allreduce(&nnmol,&nm,1,MPI_INT,MPI_MAX,world);

  if (init_on == 0) {

    memory->create(h_area,nm,"angle_area_volume3:h_area");
    memory->create(h_volume,nm,"angle_area_volume3:h_volume");
    memory->create(ttyp,nm,"angle_area_volume3:ttyp");
    memory->create(ttyp1,nm,"angle_area_volume3:ttyp1");
    memory->create(dath,2*nm,"angle_area_volume3:dath");
    memory->create(datt,2*nm,"angle_area_volume3:datt");

    for (n = 0; n < nm; n++) {
      ttyp[n] = 0;
      ttyp1[n] = 0;
    }

    for (n = 0; n < nanglelist; n++) {
    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];
    if(i1<nlocal) m = atom->molecule[i1]-1;
    if(i2<nlocal) m = atom->molecule[i2]-1;
    if(i3<nlocal) m = atom->molecule[i3]-1;
    ttyp1[m] = anglelist[n][3]; 
    }

    init_on =1;
    MPI_Allreduce(ttyp1,ttyp,nm,MPI_INT,MPI_MAX,world);
  }

  eangle = 0.0
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  for (n = 0; n < nm; n++) {
    h_area[n] = 0.0;
    h_volume[n] = 0.0;
  }

  energy = 0.0;
  energy_a = 0.0;
  energy_v = 0.0;
  energy_l = 0.0;
   
  for (n = 0; n < 2*nm; n++) {
    dath[n] = 0.0;
    datt[n] = 0.0;   
  }

  for (n = 0; n < nanglelist; n++) {
    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];
    type = anglelist[n][3];
    if(i1<nlocal) m = atom->molecule[i1]-1;
    if(i2<nlocal) m = atom->molecule[i2]-1;
    if(i3<nlocal) m = atom->molecule[i3]-1;
    d21x = x[i2][0] - x[i1][0];
    d21y = x[i2][1] - x[i1][1];
    d21z = x[i2][2] - x[i1][2];

    // 3-1 distance
    d31x = x[i3][0] - x[i1][0];
    d31y = x[i3][1] - x[i1][1];
    d31z = x[i3][2] - x[i1][2];

    // calculate normal
    nx = d21y*d31z - d31y*d21z;
    ny = d31x*d21z - d21x*d31z;
    nz = d21x*d31y - d31x*d21y;
    nn = sqrt(nx*nx + ny*ny + nz*nz);
    
    // calculate center
    for (j = 0; j < 3; j++) {
      xx1[j] = x[i1][j];
      xx2[j] = x[i2][j];
      xx3[j] = x[i3][j]; 
    }

    if(i1<nlocal) {
      domain->unmap(xx1,atom->image[i1]);
      domain->unmap(xx2,atom->image[i1]);
      domain->unmap(xx3,atom->image[i1]);
    }
    else {
      if(i2<nlocal) {
        domain->unmap(xx1,atom->image[i2]);
        domain->unmap(xx2,atom->image[i2]);
        domain->unmap(xx3,atom->image[i2]);
      }
      else {
        if(i3<nlocal) {
          domain->unmap(xx1,atom->image[i3]);
          domain->unmap(xx2,atom->image[i3]);
          domain->unmap(xx3,atom->image[i3]);
        }
      }
    }    

    mx =  xx1[0] + xx2[0] + xx3[0];
    my =  xx1[1] + xx2[1] + xx3[1];
    mz =  xx1[2] + xx2[2] + xx3[2];
    
    // calculate area and volume
    aa = 0.5*nn;
    vv = (nx*mx + ny*my + nz*mz)/18.0;

    if(newton_bond) {
      dath[m] += aa;
      dath[m+nm] += vv;
      h_area[m] += aa;
      h_volume[m] += vv;
    }
    else {
      if(i1<nlocal&&i2<nlocal&&i3<nlocal) {
        dath[m] += aa;
        dath[m+nm] += vv;
        h_area[m] += aa;
        h_volume[m] += vv;
      }
      else {
        if(i1<nlocal && tag[i1]<= tag[i2] && tag[i1]<=tag[i3]) {
          dath[m] += aa;
          dath[m+nm] += vv;
          h_area[m] += aa;
          h_volume[m] += vv;
        }
        if(i2<nlocal && tag[i2]<= tag[i1] && tag[i2]<=tag[i3]) {
          dath[m] += aa;
          dath[m+nm] += vv;
          h_area[m] += aa;
          h_volume[m] += vv;
        }
        if(i3<nlocal && tag[i3]<= tag[i1] && tag[i3]<=tag[i2]) {
          dath[m] += aa;
          dath[m+nm] += vv;
          h_area[m] += aa;
          h_volume[m] += vv;
        }
      } 
    }
  }
  
  MPI_Allreduce(dath,datt,2*nm,MPI_DOUBLE,MPI_SUM,world);

  if (eflag) {
    for (m = 0; m < nm; m++) {
      type = ttyp[m];

      if(update->ntimestep<nstep1[type]) {
        voltemp=v0[type];
      }
      if(update->ntimestep>=nstep1[type]&&update->ntimestep<=nstep2[type]) {
        tempxxx=(update->ntimestep-nstep1[type])*1.0/((nstep2[type]-nstep1[type])*1.0);
        voltemp = v0[type]*(1.0+tempxxx*(vratio[type]-1.0));
      }
      if(update->ntimestep>nstep2[type]) {
        voltemp=v0[type]*vratio[type];
      }

      energy_a += 0.5*ka[type]*(a0[type]-datt[m])*(a0[type]-datt[m])/a0[type];
      energy_v += 0.5*kv[type]*(voltemp-datt[m+nm])*(voltemp-datt[m+nm])/voltemp;

      if(abs(a0[type])>0) tempxxx=ka[type]*(a0[type]-datt[m])/a0[type];

      if(comm->me==0) {
        printf("mol: %d/%d, area is %f, volume is %f, desired is %f %f, tension is %f, in step %d\n", 
                m+1,nm, datt[m], datt[m+nm], a0[type], voltemp, tempxxx, update->ntimestep);
      }
    }
  } 

  for (n = 0; n < nanglelist; n++) {

    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];
    type = anglelist[n][3];

    if(i1<nlocal) m = atom->molecule[i1]-1;
    if(i2<nlocal) m = atom->molecule[i2]-1;
    if(i3<nlocal) m = atom->molecule[i3]-1; 

    //Initial local area
    // 2-1 distance
    d21x = x0[i2][0] - x0[i1][0];
    d21y = x0[i2][1] - x0[i1][1];
    d21z = x0[i2][2] - x0[i1][2];

    // 3-1 distance
    d31x = x0[i3][0] - x0[i1][0];
    d31y = x0[i3][1] - x0[i1][1];
    d31z = x0[i3][2] - x0[i1][2];

    // calculate normal
    nx = d21y*d31z - d31y*d21z;
    ny = d31x*d21z - d21x*d31z;
    nz = d21x*d31y - d31x*d21y;
    nn = sqrt(nx*nx + ny*ny + nz*nz);
    ar0 = 0.5*nn;

    // 2-1 distance
    d21x = x[i2][0] - x[i1][0];
    d21y = x[i2][1] - x[i1][1];
    d21z = x[i2][2] - x[i1][2];

    // 3-1 distance
    d31x = x[i3][0] - x[i1][0];
    d31y = x[i3][1] - x[i1][1];
    d31z = x[i3][2] - x[i1][2];

    // 3-2 distance
    d32x = x[i3][0] - x[i2][0];
    d32y = x[i3][1] - x[i2][1];
    d32z = x[i3][2] - x[i2][2];
    
    // calculate normal
    nx = d21y*d31z - d31y*d21z;
    ny = d31x*d21z - d21x*d31z;
    nz = d21x*d31y - d31x*d21y;
    nn = sqrt(nx*nx + ny*ny + nz*nz);
    
    // calculate center
    for (j = 0; j < 3; j++) {
      xx1[j] = x[i1][j];
      xx2[j] = x[i2][j];
      xx3[j] = x[i3][j]; 
    }

    if(i1<nlocal) {
      domain->unmap(xx1,atom->image[i1]);
      domain->unmap(xx2,atom->image[i1]);
      domain->unmap(xx3,atom->image[i1]);
    }
    else {
      if(i2<nlocal) {
        domain->unmap(xx1,atom->image[i2]);
        domain->unmap(xx2,atom->image[i2]);
        domain->unmap(xx3,atom->image[i2]);
      }
      else {
        if(i3<nlocal) {
          domain->unmap(xx1,atom->image[i3]);
          domain->unmap(xx2,atom->image[i3]);
          domain->unmap(xx3,atom->image[i3]);
        }
      }
    }

    mx =  xx1[0] + xx2[0] + xx3[0];
    my =  xx1[1] + xx2[1] + xx3[1];
    mz =  xx1[2] + xx2[2] + xx3[2];
        
    if(update->ntimestep<nstep1[type]) {
      voltemp=v0[type];
    }

    if(update->ntimestep>=nstep1[type]&&update->ntimestep<=nstep2[type]) {
      tempxxx=(update->ntimestep-nstep1[type])*1.0/((nstep2[type]-nstep1[type])*1.0);
      voltemp = v0[type]*(1.0+tempxxx*(vratio[type]-1.0));
    }

    if(update->ntimestep>nstep2[type]) {
      voltemp=v0[type]*vratio[type];
    } 

    coefc = 0.25*press[type]*qp[type]/pow(0.5*nn,qp[type]+2);
    coefl = 0.5*kl[type]*(ar0-0.5*nn)/ar0/nn; 
    coefa = 0.5*ka[type]*(a0[type]-datt[m])/a0[type]/nn;
    coefca = coefc + coefl + coefa;        
    coefv = kv[type]*(voltemp-datt[m+nm])/voltemp/18.0;  

    if (eflag) {
      energy_l += 0.5*kl[type]*(ar0-0.5*nn)*(ar0-0.5*nn)/ar0;
      eangle = 0.5*kl[type]*(ar0-0.5*nn)*(ar0-0.5*nn)/ar0;
    }

    s1x = coefca*(ny*d32z - nz*d32y);
    s1y = coefca*(nz*d32x - nx*d32z);    
    s1z = coefca*(nx*d32y - ny*d32x);
    
    s2x = coefca*(nz*d31y - ny*d31z);
    s2y = coefca*(nx*d31z - nz*d31x);
    s2z = coefca*(ny*d31x - nx*d31y);
   
    s3x = coefca*(ny*d21z - nz*d21y);
    s3y = coefca*(nz*d21x - nx*d21z);
    s3z = coefca*(nx*d21y - ny*d21x);

    s1xv = coefv*(nx);
    s1yv = coefv*(ny);
    s1zv = coefv*(nz);

    s2xv = coefv*(nx);
    s2yv = coefv*(ny);
    s2zv = coefv*(nz);

    s3xv = coefv*(nx);
    s3yv = coefv*(ny);
    s3zv = coefv*(nz);

    // apply force to each of 3 atoms
    f1[0] = s1x + s1xv;
    f1[1] = s1y + s1yv;
    f1[2] = s1z + s1zv;
 
    f2[0] = s2x + s2xv;
    f2[1] = s2y + s2yv;
    f2[2] = s2z + s2zv;

    f3[0] = s3x+s3xv;
    f3[1] = s3y+s3yv;
    f3[2] = s3z+s3zv;

    f1x[0] = s1x;
    f1x[1] = s1y;
    f1x[2] = s1z;
 
    f2x[0] = s2x;
    f2x[1] = s2y;
    f2x[2] = s2z;

    f3x[0] = s3x;
    f3x[1] = s3y;
    f3x[2] = s3z; 
  
    if (newton_bond || i1 < nlocal) {
      f[i1][0] += f1[0];
      f[i1][1] += f1[1];
      f[i1][2] += f1[2];
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] += f2[0];  //s2x+coefv*nx;
      f[i2][1] += f2[1];  //s2y+coefv*ny;
      f[i2][2] += f2[2];  //s2z+coefv*nz;
    }
    if (newton_bond || i3 < nlocal) {
      f[i3][0] += f3[0];
      f[i3][1] += f3[1];
      f[i3][2] += f3[2];
    }

    vv = 0.0; //2.0*datt[m+nm]*coefv/n_atoms[m];
    ff[0] = d21x*s2x + d31x*s3x + (d21x*(s2xv-s1xv)+d31x*(s3xv-s1xv)+d32x*(s3xv-s2xv))/3.0 + vv;
    ff[1] = d21y*s2y + d31y*s3y + (d21y*(s2yv-s1yv)+d31y*(s3yv-s1yv)+d32y*(s3yv-s2yv))/3.0 + vv;
    ff[2] = d21z*s2z + d31z*s3z + (d21z*(s2zv-s1zv)+d31z*(s3zv-s1zv)+d32z*(s3zv-s2zv))/3.0 + vv;
    ff[3] = d21x*s2y + d31x*s3y + (d21x*(s2yv-s1yv)+d31x*(s3yv-s1yv)+d32x*(s3yv-s2yv))/3.0;
    ff[4] = d21x*s2z + d31x*s3z + (d21x*(s2zv-s1zv)+d31x*(s3zv-s1zv)+d32x*(s3zv-s2zv))/3.0;
    ff[5] = d21y*s2z + d31y*s3z + (d21y*(s2zv-s1zv)+d31y*(s3zv-s1zv)+d32y*(s3zv-s2zv))/3.0;

    d12x=-d21x;
    d12y=-d21y;
    d12z=-d21z;

    if (evflag) ev_tally(i1,i2,i3,nlocal,newton_bond,eangle,f1x,f3x,d12x,d12y,d12z,d32x,d32y,d32z);
  }
}

/* ---------------------------------------------------------------------- */

void AngleAreaVolume3::allocate()
{
  allocated = 1;
  int n = atom->nangletypes;
  init_on = 0;

  memory->create(ka,n+1,"angle:ka");
  memory->create(a0,n+1,"angle:a0");
  memory->create(kv,n+1,"angle:kv");
  memory->create(v0,n+1,"angle:v0");
  memory->create(kl,n+1,"angle:kl");
  memory->create(aa0,n+1,"angle:aa0");
  memory->create(press,n+1,"angle:press");
  memory->create(qp,n+1,"angle:qp");
  memory->create(r0,n+1,"angle:r0");
  memory->create(nstep1,n+1,"angle:nstep1");
  memory->create(nstep2,n+1,"angle:nstep2");
  memory->create(vratio,n+1,"angle:vratio");
  memory->create(setflag,n+1,"angle:setflag");

  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void AngleAreaVolume3::coeff(int narg, char **arg)
{
  if (narg != 13) error->all(FLERR,"Incorrect args for angle coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  utils::bounds(FLERR,arg[0],1,atom->nangletypes,ilo,ihi,error);

  double ka_one =  utils::numeric(FLERR, arg[1],true, lmp);
  double a0_one = utils::numeric(FLERR, arg[2], true,lmp);
  double kv_one = utils::numeric(FLERR, arg[3], true,lmp);
  double v0_one = utils::numeric(FLERR,  arg[4],true, lmp);
  double kl_one = utils::numeric(FLERR,  arg[5],true, lmp);
  double aa0_one = utils::numeric(FLERR, arg[6],true, lmp);
  double press_one = utils::numeric(FLERR, arg[7],true, lmp);
  double qp_one = utils::numeric(FLERR, arg[8], true,lmp);
  double r0_one = utils::numeric(FLERR,  arg[9],true, lmp);
  int nstep1_one = utils::numeric(FLERR, arg[10],true, lmp);
  int nstep2_one = utils::numeric(FLERR,  arg[11],true, lmp);
  double vratio_one = utils::numeric(FLERR, arg[12],true, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    ka[i] = ka_one;
    a0[i] = a0_one;
    kv[i] = kv_one;
    v0[i] = v0_one;
    kl[i] = kl_one;
    aa0[i] = aa0_one;
    press[i] = press_one;
    qp[i] = qp_one;
    r0[i] = r0_one;
    nstep1[i]=nstep1_one;
    nstep2[i]=nstep2_one;
    vratio[i]=vratio_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for angle coefficients");
}

/* ---------------------------------------------------------------------- */

double AngleAreaVolume3::equilibrium_angle(int i)
{
  return -1;
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void AngleAreaVolume3::write_restart(FILE *fp)
{
  fwrite(&ka[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&a0[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&kv[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&v0[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&kl[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&aa0[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&press[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&qp[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&r0[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&nstep1[1],sizeof(int),atom->nangletypes,fp);
  fwrite(&nstep2[1],sizeof(int),atom->nangletypes,fp);
  fwrite(&vratio[1],sizeof(double),atom->nangletypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void AngleAreaVolume3::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0){ 
    fread(&ka[1],sizeof(double),atom->nangletypes,fp);
    fread(&a0[1],sizeof(double),atom->nangletypes,fp);
    fread(&kv[1],sizeof(double),atom->nangletypes,fp);
    fread(&v0[1],sizeof(double),atom->nangletypes,fp);
    fread(&kl[1],sizeof(double),atom->nangletypes,fp);
    fread(&aa0[1],sizeof(double),atom->nangletypes,fp);  
    fread(&press[1],sizeof(double),atom->nangletypes,fp);
    fread(&qp[1],sizeof(double),atom->nangletypes,fp);
    fread(&r0[1],sizeof(double),atom->nangletypes,fp);
    fread(&nstep1[1],sizeof(int),atom->nangletypes,fp);
    fread(&nstep2[1],sizeof(int),atom->nangletypes,fp);
    fread(&vratio[1],sizeof(double),atom->nangletypes,fp);
  }

  MPI_Bcast(&ka[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&a0[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&kv[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&v0[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&kl[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&aa0[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&press[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&qp[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r0[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&nstep1[1],atom->nangletypes,MPI_INT,0,world);
  MPI_Bcast(&nstep2[1],atom->nangletypes,MPI_INT,0,world);
  MPI_Bcast(&vratio[1],atom->nangletypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nangletypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void AngleAreaVolume3::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nangletypes; i++)
    fprintf(fp,"%d %g %g %g %g %g %g %g %g %g %d %d %g\n",i,ka[i],a0[i],kv[i],v0[i],kl[i],aa0[i],press[i],
            qp[i],r0[i],nstep1[i],nstep2[i],vratio[i]);
}

/* ---------------------------------------------------------------------- */

double AngleAreaVolume3::single(int type, int i1, int i2, int i3)
{
    return -1;
}

