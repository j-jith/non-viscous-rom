CFLAGS = 
CPPFLAGS =
LIBFILES =
SOAR = soar
SOAR_OBJ = soar_main.o multi_soar.o
FULL = full
FULL_OBJ = full_main.o
OBJFILES = file_io.o misc.o block_matrices.o pod.o reduce.o

#include ${PETSC_DIR}/lib/petsc/conf/variables
#include ${PETSC_DIR}/lib/petsc/conf/rules
include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

all: ${SOAR}

${SOAR}: ${SOAR_OBJ} ${OBJFILES} globals.h chkopts
	-${CLINKER} -o ${SOAR} ${SOAR_OBJ} ${OBJFILES} ${PETSC_LIB} ${SLEPC_LIB}
	${RM} ${SOAR_OBJ} ${OBJFILES}

${FULL}: ${FULL_OBJ} ${OBJFILES} globals.h chkopts
	-${CLINKER} -o ${FULL} ${FULL_OBJ} ${OBJFILES} ${PETSC_LIB} ${SLEPC_LIB}
	${RM} ${FULL_OBJ} ${OBJFILES}
