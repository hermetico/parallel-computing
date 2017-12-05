#include "mpi.h"
#include <stdio.h>
#define SIZE 8


int main(int argc, char **argv){
	int p, rank, cube_rank;
    int ndims[3]={2,2,2};
	int dims = 3;
	int coords[3];
	int periods[3]={0,0, 0}; // wraparaound?
	int reorder = 0;
   

	MPI_Init(&argc,&argv);

	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Comm whole_cube_comm;   // required variable

	MPI_Cart_create(MPI_COMM_WORLD, dims, ndims, periods, reorder, &whole_cube_comm);
	MPI_Comm_rank(whole_cube_comm, &cube_rank);
	MPI_Cart_coords(whole_cube_comm, cube_rank, 3, coords);

	printf("rank=%i of %i with cube_rank of %d and coords x=%i, y=%i, z=%i\n", rank,p, cube_rank, coords[0], coords[1], coords[2]);

	MPI_Finalize();
}
