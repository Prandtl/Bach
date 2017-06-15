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

        #if defined(PETSC_USE_LOG)
          PetscLogStage stage;
        #endif

        /* Initialize TAO and PETSc */
        ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;
        ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(ierr);
        ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

        /*  create viewer */
        ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "bin/bigHogwild.out", &viewer); CHKERRQ(ierr);
        PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_COMMON);
        /* Initialize problem parameters */
        user.alpha = 0.01;
        user.lambda = 0.015;
        user.maxIter = 1000;

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
        //PetscInfo1(NULL, "rstart: %i\n", rstart);
        //PetscInfo1(NULL, "rend: %i\n", rend);

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
        PetscLogStageRegister("Work", &stage);
        PetscLogStagePush(stage);
        do {
                ierr = VecCopy(x, xOld); CHKERRQ(ierr);
                ierr = FormFunctionGradient(tao, x, &f, G, &user); CHKERRQ(ierr);
                ierr = VecPointwiseMult(antiG, G, minusLambda);
                VecGetArray(x, &xlocal);
                VecGetArray(antiG, &antigLocal);

                for (i = 0; i < N; i++) {
                        VecSetValues(x, 1, &i, &antigLocal[i], ADD_VALUES );
                }

                VecRestoreArray(antiG, &antigLocal);
                VecAssemblyBegin(x);
                VecAssemblyEnd(x);

                ierr = VecWAXPY(delta, -1, xOld, x); CHKERRQ(ierr);
                ierr = VecNorm(delta, NORM_2, &delta_norm); CHKERRQ(ierr);

                iter+=1;
                if(iter>user.maxIter)
                {
                        break;
                }
        } while(delta_norm > user.alpha);
        PetscLogStagePop();
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
        ff += PetscSqr(xreal[2]) + PetscSqr(xreal[2] - xreal[3]);
        ff += PetscSqr(xreal[4]) + PetscSqr(xreal[4] - xreal[5]);
        ff += PetscSqr(xreal[6]) + PetscSqr(xreal[6] - xreal[7]);

        /* Compute G(X) */
        g[0] = 4*xreal[0] - 2*xreal[1];
        g[1] = -2*xreal[0] + 2*xreal[1];

        g[2] = 4*xreal[2] - 2*xreal[3];
        g[3] = -2*xreal[2] + 2*xreal[3];

        g[4] = 4*xreal[4] - 2*xreal[5];
        g[5] = -2*xreal[4] + 2*xreal[5];

        g[6] = 4*xreal[6] - 2*xreal[7];
        g[7] = -2*xreal[6] + 2*xreal[7];

        // g[4] = -3*PetscPowReal(xreal[4],2)*PetscPowReal(xreal[5],3) + 6*xreal[4] + 20*xreal[5] - 120;
        // g[5] = -3*PetscPowReal(xreal[4],3)*PetscPowReal(xreal[5],2) + 20*xreal[4];
        // g[4] = 0;
        // g[5] = 0;
        // g[6] = 2*xreal[6] + 20 * PETSC_PI * PetscSinReal(2 * PETSC_PI * xreal[6]);
        // g[7] = 2*xreal[7] + 20 * PETSC_PI * PetscSinReal(2 * PETSC_PI * xreal[7]);

        /* Restore vectors */
        ierr = VecRestoreArray(G,&g); CHKERRQ(ierr);

        *f=ff;
        return 0;
}
