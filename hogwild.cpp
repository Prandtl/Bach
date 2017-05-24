#include <petsctao.h>
#include <petscsys.h>

static char help[] = "using tao to solve x^2 + (x-y)^2 ";


typedef struct {
        PetscReal alpha;
        PetscInt maxIter;
} AppCtx;

/* -------------- User-defined routines ---------- */
PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);

int main(int argc,char **argv)
{
        PetscErrorCode ierr;                /* used to check for functions returning nonzeros */
        PetscReal zero=0.0, lambda=0.1, *xlocal, delta_norm=0.0;
        PetscReal xinitial[] = {11.0, -7.0, 5};
        PetscInt rstart, rend, i, N=2;
        Vec x, x_old, delta;                              /* solution vector */
        Tao tao;                            /* Tao solver context */
        PetscBool flg;
        PetscMPIInt size,rank;                   /* number of processes running */
        AppCtx user;                        /* user-defined application context */

        PetscInt iter = 0;
        PetscReal f;
        Vec G;

        PetscViewer viewer;

        /* Initialize TAO and PETSc */
        ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;
        ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);
        ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);

        /*  create viewer */
        ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "hogwild.out", &viewer); CHKERRQ(ierr);
        PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_PYTHON);

        /* Initialize problem parameters */
        user.alpha = 0.01;
        user.maxIter = 5000;
        /* Check for command line arguments to override defaults */
        ierr = PetscOptionsGetReal(NULL,NULL,"-alpha",&user.alpha,&flg); CHKERRQ(ierr);

        /* Allocate vectors for the solution and gradient */
        // ierr = VecCreateSeq(PETSC_COMM_SELF, 2, &x); CHKERRQ(ierr);
        ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &x); CHKERRQ(ierr);
        // ierr = VecCreateSeq(PETSC_COMM_SELF, 2, &x_old); CHKERRQ(ierr);
        ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &x_old); CHKERRQ(ierr);
        // ierr = VecCreateSeq(PETSC_COMM_SELF, 2, &delta); CHKERRQ(ierr);
        ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &delta); CHKERRQ(ierr);
        // ierr = VecCreateSeq(PETSC_COMM_SELF, 2, &G); CHKERRQ(ierr);
        ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &G); CHKERRQ(ierr);

        /* The TAO code begins here */
        /* Create TAO solver with desired solution method */
        ierr = TaoCreate(PETSC_COMM_SELF, &tao); CHKERRQ(ierr);
        ierr = TaoSetType(tao,TAOLMVM); CHKERRQ(ierr);

        /* Set solution vec and an initial guess */
        ierr = VecSet(x, zero); CHKERRQ(ierr);
        VecView(x, viewer);

        VecGetOwnershipRange(x, &rstart, &rend);
        PetscInfo1(NULL, "rstart: %i\n", rstart);
        PetscInfo1(NULL, "rend: %i\n", rend);

        VecGetArray(x, &xlocal);
        for (i = rstart; i < rend; i++) {
                // xlocal[i] = xinitial[i];
                // v    = (PetscReal)(rank*i);
                VecSetValues(x,1,&i,&xinitial[i],INSERT_VALUES );
        }
        VecAssemblyBegin(x);
        VecAssemblyEnd(x);

        // VecRestoreArray(x,&xlocal);

        VecView(x, PETSC_VIEWER_STDOUT_WORLD);

        ierr = TaoSetInitialVector(tao,x); CHKERRQ(ierr);
        ierr = VecSet(x_old, zero); CHKERRQ(ierr);
        ierr = VecSet(delta, zero); CHKERRQ(ierr);
        ierr = VecSet(G, zero); CHKERRQ(ierr);

        /* Set routines for function, gradient, hessian evaluation */
        ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,&user); CHKERRQ(ierr);

        /* Check for TAO command line options */
        ierr = TaoSetFromOptions(tao); CHKERRQ(ierr);



        /*solve*/
        iter = 0;

        do {
                VecView(x, viewer);
                ierr = VecCopy(x, x_old); CHKERRQ(ierr);
                ierr = FormFunctionGradient(tao, x, &f, G, &user); CHKERRQ(ierr);
                ierr = VecAXPY(x, -lambda, G); CHKERRQ(ierr);
                ierr = VecWAXPY(delta, -1, x_old, x); CHKERRQ(ierr); // delta = x - x_old
                ierr = VecNorm(delta, NORM_2, &delta_norm); CHKERRQ(ierr);

                PetscInfo1(NULL, "iteration: %i\n", iter);
                PetscInfo1(NULL, "delta: %g\n", delta_norm);

                iter+=1;
                if(iter>user.maxIter)
                {
                        PetscInfo1(NULL, "did not converge in maxIter (%i) iterations\n", user.maxIter);
                        break;
                }
        } while(delta_norm > user.alpha);
        ierr = TaoDestroy(&tao); CHKERRQ(ierr);
        ierr = VecDestroy(&x); CHKERRQ(ierr);
        ierr = VecDestroy(&G); CHKERRQ(ierr);
        PetscViewerPopFormat(viewer);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

        ierr = PetscFinalize();
        return ierr;
}

PetscErrorCode FormFunctionGradient(Tao tao,Vec X,PetscReal *f, Vec G,void *ptr)
{
        PetscErrorCode ierr;
        PetscReal ff=0;
        PetscReal           *x,*g;

        /* Get pointers to vector data */
        ierr = VecGetArray(X,&x); CHKERRQ(ierr);
        ierr = VecGetArray(G,&g); CHKERRQ(ierr);
        /* Compute f(x) */
        ff = PetscSqr(x[0]) + PetscSqr(x[0] - x[1]);

        /* Compute G(X) */
        g[0] = 4*x[0] - 2*x[1];
        g[1] = -2*x[0] + 2*x[1];

        /* Restore vectors */
        ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
        ierr = VecRestoreArray(G,&g); CHKERRQ(ierr);
        *f=ff;

        return 0;
}
