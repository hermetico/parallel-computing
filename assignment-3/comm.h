
#ifndef __COMM_H
#define __COMM_H
#include "comm.c"
/* Function prototypes */
//void spread_matrix(/* PARAMETERS */);
//void gather_matrix(/* PARAMETERS */);
int comms_split(MPI_Comm *comm_i, MPI_Comm *comm_j, MPI_Comm *comm_k, MPI_Comm *comm_k0, int *coords, int q);

#endif /* __COMM_H */
