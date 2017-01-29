CFLAGS = 
CPPFLAGS =
LIBFILES =
TARGET = solver
OBJFILES = solver.o
TARGET1 = undamped
OBJFILES1 = undamped.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

all: ${TARGET} ${TARGET1}

${TARGET}: ${OBJFILES} chkopts
	-${CLINKER} -o ${TARGET} ${OBJFILES} ${PETSC_LIB}
	${RM} ${OBJFILES}

${TARGET1}: ${OBJFILES1} chkopts
	-${CLINKER} -o ${TARGET1} ${OBJFILES1} ${PETSC_LIB}
	${RM} ${OBJFILES1}
