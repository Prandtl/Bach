ALL: toast, example, hogwild

CFLAGS =
FFLAGS =
CPPFLAGS =
FPPFLAGS =
CLEANFILES = toast, example, example.out, hogwild, hogwild.out

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

toast: toast.o chkopts
	-${CLINKER} -o  bin/toast toast.o ${PETSC_LIB}
	${RM} toast.o

runToast:
	-@${MPIEXEC}  -n 2 ./toast

example: example.o chkopts
	-${CLINKER} -o  bin/example example.o ${PETSC_LIB}
	${RM} example.o

runExample: example
	-@${MPIEXEC}  -n 1 bin/example -info

runShowExample: example
	-@${MPIEXEC}  -n 1 bin/example -info
	python3 ariadna.py bin/example.out

hogwild: hogwild.o chkopts
	-${CLINKER} -o  bin/hogwild hogwild.o ${PETSC_LIB}
	${RM} hogwild.o

runHogwild: hogwild
	-@${MPIEXEC}  -n 2 bin/hogwild -info

runSerialHogwild: hogwild
	-@${MPIEXEC}  -n 1 bin/hogwild -info

runShowHogwild: hogwild
	-@${MPIEXEC}  -n 2 bin/hogwild -info
	python3 ariadna.py bin/hogwild.out

bigHog: bigHogwild.o chkopts
		-${CLINKER} -o  bin/bigHogwild bigHogwild.o ${PETSC_LIB}
		${RM} bigHogwild.o

runBigHog: bigHog
	-@${MPIEXEC} -n 2 bin/bigHogwild -info

runShowBigHog: bigHog
	-@${MPIEXEC} -n 8 bin/bigHogwild -info
	python3 ariadna-big.py bin/bigHogwild.out
