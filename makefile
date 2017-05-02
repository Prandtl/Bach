ALL: toast

CFLAGS =
FFLAGS =
CPPFLAGS =
FPPFLAGS =
CLEANFILES = toast

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

toast: toast.o chkopts
	-${CLINKER} -o  toast toast.o ${PETSC_LIB}
	${RM} toast.o


runToast:
	-@${MPIEXEC}  -n 2 ./toast
