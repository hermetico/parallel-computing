#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);


    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const int MAX_NUMBERS = 100;
    int numbers[MAX_NUMBERS];
    int number_amount;

    if(world_rank == 0){
	    srand(time(NULL));
        number_amount = (rand() /(float)RAND_MAX) * MAX_NUMBERS;
    	MPI_Send(&numbers, number_amount, MPI_INT, 1, 0, MPI_COMM_WORLD);
	    printf("Process 0 sends %d numbers\n", number_amount);
    }else if(world_rank == 1){
        MPI_Status status;
	    MPI_Recv(&numbers, MAX_NUMBERS, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        MPI_Get_count(&status, MPI_INT, &number_amount);
	    // Print off a hello world message
	    printf("Process 1 received number %i instad of %i\n", number_amount, MAX_NUMBERS);
    }

    // Finalize the MPI environment.
    MPI_Finalize();
}
