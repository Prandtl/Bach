#include <petsctao.h>
#include <petscsys.h>

static char help[] = "using tao to solve x^2 + (x-y)^2";


typedef struct {
        PetscReal alpha;
} AppCtx;

/* -------------- User-defined routines ---------- */
PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);

int main(int argc,char **argv)
{
        PetscErrorCode ierr;                /* used to check for functions returning nonzeros */
        PetscReal zero=0.0, lambda=0.01, *xinitial;
        Vec x;                              /* solution vector */
        Tao tao;                            /* Tao solver context */
        PetscBool flg;
        PetscMPIInt size,rank;                   /* number of processes running */
        AppCtx user;                        /* user-defined application context */

        PetscReal f;
        Vec G;

        PetscViewer    viewer;

        /* Initialize TAO and PETSc */
        ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;
        ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);
        ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);
        if (size >1) SETERRQ(PETSC_COMM_SELF,1,"Incorrect number of processors");

        /* Initialize problem parameters */
        user.alpha = 0.01;
        /* Check for command line arguments to override defaults */
        ierr = PetscOptionsGetReal(NULL,NULL,"-alpha",&user.alpha,&flg); CHKERRQ(ierr);

        /* Allocate vectors for the solution and gradient */
        ierr = VecCreateSeq(PETSC_COMM_SELF, 2, &x); CHKERRQ(ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF, 2, &G); CHKERRQ(ierr);

        /* The TAO code begins here */
        /* Create TAO solver with desired solution method */
        ierr = TaoCreate(PETSC_COMM_SELF,&tao); CHKERRQ(ierr);
        ierr = TaoSetType(tao,TAOLMVM); CHKERRQ(ierr);

        /* Set solution vec and an initial guess */
        ierr = VecSet(x, zero); CHKERRQ(ierr);
        ierr = VecSet(G, zero); CHKERRQ(ierr);
        ierr = VecGetArray(x, &xinitial); CHKERRQ(ierr);
        xinitial[0]=5.0;
        xinitial[1]=5.0;
        ierr = VecRestoreArray(x, &xinitial); CHKERRQ(ierr);
        ierr = TaoSetInitialVector(tao,x); CHKERRQ(ierr);

        /* Set routines for function, gradient, hessian evaluation */
        ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,&user); CHKERRQ(ierr);

        /* Check for TAO command line options */
        ierr = TaoSetFromOptions(tao); CHKERRQ(ierr);

        /*  create viewer */
        ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "example.out", &viewer); CHKERRQ(ierr);
         PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_COMMON);
        /*solver was here*/
        // PetscInfo2(NULL, "position = %g, %g\n",x[0],x[1]);
        for(int i=0; i<300; i++)
        {
                VecView(x, viewer);
                ierr = FormFunctionGradient(tao, x, &f, G, &user); CHKERRQ(ierr);
                ierr = VecAXPY(x, -lambda, G); CHKERRQ(ierr);
        }
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
        AppCtx         *user = (AppCtx *) ptr;
        PetscErrorCode ierr;
        PetscReal alpha=user->alpha, ff=0;
        PetscReal      *x,*g;

        /* Get pointers to vector data */
        ierr = VecGetArray(X,&x); CHKERRQ(ierr);
        ierr = VecGetArray(G,&g); CHKERRQ(ierr);
        /* Compute f(x) */
        ff = PetscSqr(x[0]) + PetscSqr(x[0] - x[1]);

        /* Compute G(X) */
        g[0]=4*x[0] - 2*x[1];
        g[1]=-2*x[0] + 2*x[1];

        /* Restore vectors */
        ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
        ierr = VecRestoreArray(G,&g); CHKERRQ(ierr);
        *f=ff;

        return 0;
}
