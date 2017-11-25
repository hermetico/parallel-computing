CC = g++
MPICC = mpic++
TARGETS = virtual_topo

all: ${TARGETS}

virtual_topo: mpi_virtual_topo.c
	$(MPICC) -o $@ mpi_virtual_topo.c


clean:
	rm -f ${TARGETS} *.o
