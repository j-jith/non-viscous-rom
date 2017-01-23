CFLAGS = 
CPPFLAGS =
LIBFILES =
SOAR = soar
SOAR_OBJ = soar_main.o multi_soar.o
REDUCE = reduce
REDUCE_OBJ = reduce_main.o reduce.o
OBJFILES = file_io.o misc.o block_matrices.o

include ${PETSC_DIR}/petsc/conf/variables
include ${PETSC_DIR}/petsc/conf/rules

all: ${SOAR} ${REDUCE}

${SOAR}: ${SOAR_OBJ} ${OBJFILES} globals.h chkopts
	-${CLINKER} -o ${SOAR} ${SOAR_OBJ} ${OBJFILES} ${PETSC_LIB}
	${RM} ${SOAR_OBJ} ${OBJFILES}

${REDUCE}: ${REDUCE_OBJ} ${OBJFILES} globals.h chkopts
	-${CLINKER} -o ${REDUCE} ${REDUCE_OBJ} ${OBJFILES} ${PETSC_LIB}
	${RM} ${REDUCE_OBJ} ${OBJFILES}
