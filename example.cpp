#include <petsctao.h>
#include <petscsys.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines that evaluate the function,
   gradient, and hessian.
 */

static char help[] = "This example demonstrates use of the TAO package to \n\
 solve an unconstrained minimization problem on a single processor.  We \n\
 minimize the extended Rosenbrock function: \n\
    sum_{i=0}^{n/2-1} ( alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2 ) \n\
 or the chained Rosenbrock function:\n\
    sum_{i=0}^{n-1} alpha*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2\n";


typedef struct {
        PetscInt n;     /* dimension */
        PetscReal alpha; /* condition parameter */
        PetscBool chained;
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
        user.n = 2; user.alpha = 99.0; user.chained = PETSC_FALSE;
        /* Check for command line arguments to override defaults */
        ierr = PetscOptionsGetInt(NULL,NULL,"-n",&user.n,&flg); CHKERRQ(ierr);
        ierr = PetscOptionsGetReal(NULL,NULL,"-alpha",&user.alpha,&flg); CHKERRQ(ierr);
        ierr = PetscOptionsGetBool(NULL,NULL,"-chained",&user.chained,&flg); CHKERRQ(ierr);

        /* Allocate vectors for the solution and gradient */
        ierr = VecCreateSeq(PETSC_COMM_SELF,user.n,&x); CHKERRQ(ierr);

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

        PetscInfo(NULL, "Matrix uses parameter alpha\n");

        ierr = TaoDestroy(&tao); CHKERRQ(ierr);
        ierr = VecDestroy(&x); CHKERRQ(ierr);

        ierr = PetscFinalize();
        return ierr;
}

PetscErrorCode FormFunctionGradient(Tao tao,Vec X,PetscReal *f, Vec G,void *ptr)
{
        AppCtx         *user = (AppCtx *) ptr;
        PetscInt i,nn=user->n/2;
        PetscErrorCode ierr;
        PetscReal ff=0,t1,t2,alpha=user->alpha;
        PetscReal      *x,*g;

        /* Get pointers to vector data */
        ierr = VecGetArray(X,&x); CHKERRQ(ierr);
        ierr = VecGetArray(G,&g); CHKERRQ(ierr);

        /* Compute G(X) */
        if (user->chained) {
                g[0] = 0;
                for (i=0; i<user->n-1; i++) {
                        t1 = x[i+1] - x[i]*x[i];
                        ff += PetscSqr(1 - x[i]) + alpha*t1*t1;
                        g[i] += -2*(1 - x[i]) + 2*alpha*t1*(-2*x[i]);
                        g[i+1] = 2*alpha*t1;
                }
        } else {
                for (i=0; i<nn; i++) {
                        t1 = x[2*i+1]-x[2*i]*x[2*i]; t2= 1-x[2*i];
                        ff += alpha*t1*t1 + t2*t2;
                        g[2*i] = -4*alpha*t1*x[2*i]-2.0*t2;
                        g[2*i+1] = 2*alpha*t1;
                }
        }

        /* Restore vectors */
        ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
        ierr = VecRestoreArray(G,&g); CHKERRQ(ierr);
        *f=ff;

        ierr = PetscLogFlops(nn*15); CHKERRQ(ierr);
        return 0;
}
