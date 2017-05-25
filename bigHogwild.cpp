#include <petsctao.h>
#include <petscsys.h>

static char help[] = "using tao to solve x^2 + (x-y)^2 ";


typedef struct {
        PetscReal alpha;
        PetscInt maxIter;
        PetscReal lambda;
} AppCtx;

/* -------------- User-defined routines ---------- */
PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);

int main(int argc,char **argv)
{
        PetscErrorCode ierr;                /* used to check for functions returning nonzeros */
        PetscReal zero=0.0, *xlocal, *antigLocal, delta_norm=0.0;
        PetscReal xinitial[] = {11.0, 7.0, 5.0, 0.0, 7.0, -3.5, -3.0, 2.0};
        PetscInt rstart, rend, i, N=8;
        Vec x, xOld, delta;                              /* solution vector */
        Tao tao;                            /* Tao solver context */
        PetscBool flg;
        PetscMPIInt size,rank;                   /* number of processes running */
        AppCtx user;                        /* user-defined application context */

        PetscInt iter = 0;
        PetscReal f;
        Vec G, antiG, minusLambda;

        PetscViewer viewer;

        /* Initialize TAO and PETSc */
        ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;
        ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(ierr);
        ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

        /*  create viewer */
        ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "bigHogwild.out", &viewer); CHKERRQ(ierr);
        PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_COMMON);
        /* Initialize problem parameters */
        user.alpha = 0.01;
        user.lambda = 0.05;
        user.maxIter = 5000;

        /* Check for command line arguments to override defaults */
        ierr = PetscOptionsGetReal(NULL, NULL, "-alpha", &user.alpha, &flg); CHKERRQ(ierr);
        ierr = PetscOptionsGetReal(NULL, NULL, "-lambda", &user.lambda, &flg); CHKERRQ(ierr);
        ierr = PetscOptionsGetInt(NULL, NULL, "-maxIter", &user.maxIter, &flg); CHKERRQ(ierr);

        /* Allocate vectors for the solution and gradient */
        ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &x); CHKERRQ(ierr);
        ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &xOld); CHKERRQ(ierr);
        ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &delta); CHKERRQ(ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF, N, &G); CHKERRQ(ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF, N, &antiG); CHKERRQ(ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF, N, &minusLambda); CHKERRQ(ierr);

        /* The TAO code begins here */
        /* Create TAO solver with desired solution method */
        ierr = TaoCreate(PETSC_COMM_SELF, &tao); CHKERRQ(ierr);
        ierr = TaoSetType(tao,TAOLMVM); CHKERRQ(ierr);

        /* Set solution vec and an initial guess */
        ierr = VecSet(x, zero); CHKERRQ(ierr);

        VecGetOwnershipRange(x, &rstart, &rend);
        PetscInfo1(NULL, "rstart: %i\n", rstart);
        PetscInfo1(NULL, "rend: %i\n", rend);

        VecGetArray(x, &xlocal);
        for (i = rstart; i < rend; i++) {
                VecSetValues(x, 1, &i, &xinitial[i], INSERT_VALUES );
        }
        VecAssemblyBegin(x);
        VecAssemblyEnd(x);

        ierr = TaoSetInitialVector(tao,x); CHKERRQ(ierr);
        ierr = VecSet(xOld, zero); CHKERRQ(ierr);
        ierr = VecSet(delta, zero); CHKERRQ(ierr);
        ierr = VecSet(G, zero); CHKERRQ(ierr);
        ierr = VecSet(antiG, zero); CHKERRQ(ierr);
        ierr = VecSet(minusLambda, -user.lambda); CHKERRQ(ierr);

        /* Set routines for function, gradient, hessian evaluation */
        ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,&user); CHKERRQ(ierr);

        /* Check for TAO command line options */
        ierr = TaoSetFromOptions(tao); CHKERRQ(ierr);

        /*solve*/
        iter = 0;

        do {
                PetscInfo1(NULL, "----iteration: %i----------------------------------------------------\n", iter);
                VecView(x, viewer);
                ierr = VecCopy(x, xOld); CHKERRQ(ierr);
                ierr = FormFunctionGradient(tao, x, &f, G, &user); CHKERRQ(ierr);
                ierr = VecPointwiseMult(antiG, G, minusLambda);
                VecGetArray(x, &xlocal);
                VecGetArray(antiG, &antigLocal);

                for (i = 0; i < N; i++) {
                        VecSetValues(x, 1, &i, &antigLocal[i], ADD_VALUES );
                }

                VecView(antiG, PETSC_VIEWER_STDOUT_SELF);

                VecRestoreArray(antiG, &antigLocal);
                VecAssemblyBegin(x);
                VecAssemblyEnd(x);

                ierr = VecWAXPY(delta, -1, xOld, x); CHKERRQ(ierr); // delta = x - xOld
                ierr = VecNorm(delta, NORM_2, &delta_norm); CHKERRQ(ierr);

                VecView(delta, PETSC_VIEWER_STDOUT_WORLD);

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
        ierr = PetscViewerPopFormat(viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

        ierr = PetscFinalize();
        return ierr;
}

PetscErrorCode FormFunctionGradient(Tao tao,Vec X,PetscReal *f, Vec G,void *ptr)
{
        PetscErrorCode ierr;
        PetscReal ff=0;
        PetscReal *x,*g;
        PetscInt xsize, rstart, rend;

        VecGetSize(X, &xsize);
        PetscReal xreal[xsize];

        /* Get pointers to vector data */
        ierr = VecGetArray(X,&x); CHKERRQ(ierr);
        VecGetOwnershipRange(X, &rstart, &rend);
        int k = 0;
        for(int i=0; i<xsize; i++)
        {
                if(rstart<=i && i<rend)
                {
                        xreal[i] = x[k];
                        k++;
                }
                else
                {
                        xreal[i] = 0;
                }
        }
        ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);

        ierr = VecGetArray(G,&g); CHKERRQ(ierr);
        /* Compute f(x) */
        ff = PetscSqr(xreal[0]) + PetscSqr(xreal[0] - xreal[1]);

        /* Compute G(X) */
        g[0] = 4*xreal[0] - 2*xreal[1];
        g[1] = -2*xreal[0] + 2*xreal[1];

        g[2] = 4*xreal[2] - 2*xreal[3];
        g[3] = -2*xreal[2] + 2*xreal[3];

        g[4] = 0;
        g[5] = 0;

        g[6] = 0;
        g[7] = 0;

        /* Restore vectors */
        ierr = VecRestoreArray(G,&g); CHKERRQ(ierr);

        *f=ff;
        return 0;
}
