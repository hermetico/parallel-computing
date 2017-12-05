

#include <mpi.h>
#include "comm.h"

/* Divide and send/recieve the matrix before calculation */
void spread_matrix(/* PARAMETERS */)
{
    
}

/* Gather the finished result to P(0,0,0) */
void gather_matrix(/* PARAMETERS */)
{
    
}

/* Splits the world comm into cartesian comms */
int comms_split(MPI_Comm *comm_i, MPI_Comm *comm_j, MPI_Comm *comm_k, MPI_Comm *comm_k0, int *coords, int q)
{
	MPI_Comm_split(MPI_COMM_WORLD, coords[2], coords[0] * q + coords[1], comm_k0); // communicates along k0

	MPI_Comm_split(MPI_COMM_WORLD, coords[2] * q + coords[1], coords[0], comm_i);  // communicates along i
	MPI_Comm_split(MPI_COMM_WORLD, coords[2] * q + coords[0], coords[1], comm_j);  // communicates along j
	MPI_Comm_split(MPI_COMM_WORLD, coords[0] * q + coords[1], coords[2], comm_k);  // communicates along k
}
