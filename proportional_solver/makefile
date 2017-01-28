CFLAGS = 
CPPFLAGS =
LIBFILES =
TARGET = solver
OBJFILES = solver.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

all: ${TARGET}

${TARGET}: ${OBJFILES} chkopts
	-${CLINKER} -o ${TARGET} ${OBJFILES} ${PETSC_LIB}
	${RM} ${OBJFILES}
