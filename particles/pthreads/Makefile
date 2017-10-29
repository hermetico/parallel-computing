#
# Edison
#
CC = CC
MPCC = CC
OPENMP = -mp
CFLAGS = -O3
LIBS =


TARGETS = pthreads

all:	$(TARGETS)

pthreads: pthreads.o common.o
	$(CC) -o $@ $(LIBS) -lpthread pthreads.o common.o

pthreads.o: pthreads.cpp common.h
	$(CC) -c $(CFLAGS) pthreads.cpp

clean:
	rm -f *.o $(TARGETS)
