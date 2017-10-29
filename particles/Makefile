#
# Edison - NERSC 
#
# Intel Compilers are loaded by default; for other compilers please check the module list
#
CC = g++
MPCC = mpic++
OPENMP = -fopenmp
CFLAGS = -O3
LIBS = -lm


TARGETS = serial openmp mpi autograder

all:	$(TARGETS)

serial: serial.o common.o
	$(CC) -o $@  serial.o common.o $(LIBS)
autograder: autograder.o common.o
	$(CC) -o $@ autograder.o common.o $(LIBS)
openmp: openmp.o common.o
	$(CC) -o $@ $(OPENMP) openmp.o common.o $(LIBS)
mpi: mpi.o common.o
	$(MPCC) -o $@ $(MPILIBS) mpi.o common.o $(LIBS)

autograder.o: autograder.cpp common.h
	$(CC) -c $(CFLAGS) autograder.cpp
openmp.o: openmp.cpp common.h
	$(CC) -c $(OPENMP) $(CFLAGS) openmp.cpp
serial.o: serial.cpp common.h
	$(CC) -c $(CFLAGS) serial.cpp
mpi.o: mpi.cpp common.h
	$(MPCC) -c $(CFLAGS) mpi.cpp
common.o: common.cpp common.h
	$(CC) -c $(CFLAGS) common.cpp

clean:
	rm -f *.o $(TARGETS) *.stdout *.out
