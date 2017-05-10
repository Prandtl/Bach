ALL: toast

CFLAGS =
FFLAGS =
CPPFLAGS =
FPPFLAGS =
CLEANFILES = toast, example

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

toast: toast.o chkopts
	-${CLINKER} -o  toast toast.o ${PETSC_LIB}
	${RM} toast.o

runToast:
	-@${MPIEXEC}  -n 2 ./toast

example: example.o chkopts
	-${CLINKER} -o  example example.o ${PETSC_LIB}
	${RM} example.o

runExample:
	-@${MPIEXEC}  -n 3 ./example
