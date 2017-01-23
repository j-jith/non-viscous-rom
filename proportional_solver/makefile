CFLAGS = 
CPPFLAGS =
LIBFILES =
TARGET = solver
OBJFILES = solver.o

include ${PETSC_DIR}/petsc/conf/variables
include ${PETSC_DIR}/petsc/conf/rules

all: ${TARGET}

${TARGET}: ${OBJFILES} chkopts
	-${CLINKER} -o ${TARGET} ${OBJFILES} ${PETSC_LIB}
	${RM} ${OBJFILES}
