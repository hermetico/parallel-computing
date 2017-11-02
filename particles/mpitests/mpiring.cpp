#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);


    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int token;
    if(world_rank != 0){
	    MPI_Recv(&token, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    printf("Process %i received number %d from process %i\n", world_rank, 
            token, world_rank -1);
    }else{
        // rank 0 decide sthe value of the token
        token = 77;
    }

    // Sending token to next node
    MPI_Send(&token, 1, MPI_INT, (world_rank + 1) % world_size, 0, MPI_COMM_WORLD);
    
    if(world_rank == 0){
        // rank 0 receives from last node
	    MPI_Recv(&token, 1, MPI_INT, world_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    printf("Process %i received number %d from process %i\n", world_rank, 
            token, world_size -1);
    }
     
    // Finalize the MPI environment.
    MPI_Finalize();
}
