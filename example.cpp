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
        PetscReal zero=0.0;
        Vec x;                              /* solution vector */
        Tao tao;                            /* Tao solver context */
        PetscBool flg;
        PetscMPIInt size,rank;                   /* number of processes running */
        AppCtx user;                        /* user-defined application context */

        /* Initialize TAO and PETSc */
        ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;
        ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);
        ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);
        if (size >1) SETERRQ(PETSC_COMM_SELF,1,"Incorrect number of processors");

        /* Initialize problem parameters */
        user.alpha = 99.0;
        /* Check for command line arguments to override defaults */
        ierr = PetscOptionsGetReal(NULL,NULL,"-alpha",&user.alpha,&flg); CHKERRQ(ierr);

        /* Allocate vectors for the solution and gradient */
        ierr = VecCreateSeq(PETSC_COMM_SELF, 2, &x); CHKERRQ(ierr);

        /* The TAO code begins here */
        /* Create TAO solver with desired solution method */
        ierr = TaoCreate(PETSC_COMM_SELF,&tao); CHKERRQ(ierr);
        ierr = TaoSetType(tao,TAOLMVM); CHKERRQ(ierr);

        /* Set solution vec and an initial guess */
        ierr = VecSet(x, zero); CHKERRQ(ierr);
        ierr = TaoSetInitialVector(tao,x); CHKERRQ(ierr);

        /* Set routines for function, gradient, hessian evaluation */
        ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,&user); CHKERRQ(ierr);

        /* Check for TAO command line options */
        ierr = TaoSetFromOptions(tao); CHKERRQ(ierr);

        /*solver was here*/
        PetscInfo(NULL, "Matrix uses parameter alpha\n");

        ierr = TaoDestroy(&tao); CHKERRQ(ierr);
        ierr = VecDestroy(&x); CHKERRQ(ierr);

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
