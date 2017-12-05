CC = g++
MPICC = mpicc
TARGETS = virtual_topo

all: ${TARGETS}

virtual_topo: mpi_virtual_topo.c
	$(MPICC) -o $@ mpi_virtual_topo.c

cube_topo: mpi_cube_topo.c
	$(MPICC) -o $@ mpi_cube_topo.c

clean:
	rm -f ${TARGETS} *.o
