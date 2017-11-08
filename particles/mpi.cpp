#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "common.h"
#include <iostream>
#include <math.h>
#include <vector>
#include <cstring>
#include "mmpiutils.cpp"


// the cutoff
#define cutoff  0.01
#define BINS_LEVEL 6
#define PARTICLES_LEVEL 5
#define MOVING_PARTICLES_LEVEL 4
#define VERBOSE_LEVEL 6

// the size of the grid
extern double size;



//
//  benchmarking program
//
int main( int argc, char **argv )
{	
	int navg, nabsavg=0;
	double dmin, absmin=1.0,davg,absavg=0.0;
	double rdavg,rdmin;
	int rnavg; 
 
	//
	//  process command line parameters
	//
	if( find_option( argc, argv, "-h" ) >= 0 )
	{
		printf( "Options:\n" );
		printf( "-h to see this help\n" );
		printf( "-n <int> to set the number of particles\n" );
		printf( "-o <filename> to specify the output file name\n" );
		printf( "-s <filename> to specify a summary file name\n" );
		printf( "-no turns off all correctness checks and particle output\n");
		return 0;
	}
	
	int n = read_int( argc, argv, "-n", 1000 );
	char *savename = read_string( argc, argv, "-o", NULL );
	char *sumname = read_string( argc, argv, "-s", NULL );
	
	//
	//  set up MPI
	//
	int n_proc, rank;
	MPI_Init( &argc, &argv );
	MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );

	
	//
	//  allocate generic resources
	//
	FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
	FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


	particle_t *particles;
	if(rank == 0){
		particles = (particle_t*) malloc( n * sizeof(particle_t) );
	}

	
    /* Creates a datatype for the particle, it contains 7 double
     * which are the values and the pointer to the next one
     */
    MPI_Datatype PARTICLE;
	MPI_Type_contiguous( 10, MPI_DOUBLE, &PARTICLE ); // as many doubles as the structure has
	MPI_Type_commit( &PARTICLE );

	MPI_Datatype BIN;
	MPI_Type_contiguous( 9, MPI_INT, &BIN );
	MPI_Type_commit( &BIN );


	//
	//  initialize and distribute the particles
	//
	set_size( n );
	if( rank == 0 )
		init_particles( n, particles );


	//
	// setup the bin partitioning across processors
	// so far, only slicing rows, would be nice to slice in a grid manner
	// everyone should know the general layout of the problem
	int total_bins, bins_per_row, bins_per_proc;
	double bin_size = cutoff; // do not touch
	bins_per_row = ceil(size / bin_size);
	total_bins = bins_per_row * bins_per_row;


	bins_per_proc = total_bins / n_proc;
	// bins_per_proc should be divisible by bins_per_row ( easier to handle I hope)
	if(bins_per_proc % bins_per_row != 0){
		bins_per_proc += bins_per_row - (bins_per_proc % bins_per_row);
	}

	//MPI_Barrier(MPI_COMM_WORLD);
	//std::cout << "receive bins\n";
	//
	//
	//Populate and distribute bins, (work for process 0)
	//
	int *proc_bins_from, *proc_bins_until;
	proc_bins_from = (int *) malloc(n_proc * sizeof(int));
	proc_bins_until = (int *) malloc(n_proc * sizeof(int));

	bin_t* local_bins;
	int local_nbins = 0;
	local_bins = organize_and_send_bins( &local_nbins, n_proc, bins_per_proc, total_bins, rank, bins_per_row, BIN, proc_bins_from, proc_bins_until);
	// knowing the bins we can populate the placeholders, which are the first entry for linked lists
	// of particles
	// the first particle to be pointing from the bin

	particle_ph* local_bins_particles_ph = (particle_ph*) malloc(local_nbins * sizeof(particle_ph));
	reset_particles_placeholders(local_bins_particles_ph, local_nbins);

	std::cout << "Process " << rank << std::endl;
	for(int i = 0; i < n_proc; i++){
		std::cout << "Process " << i << " begins at " << proc_bins_from[i] << std::endl;
		std::cout << "Process " << i << " ends at " << proc_bins_until[i] << std::endl;
	}
	//MPI_Barrier(MPI_COMM_WORLD);
	//std::cout << "bins received\n";

	// Also, knowing the number of bins we can also define the
	// outter bins ( the ones in the grey area ) and populate them
	// define local grey bins for process, this number will be used to define the bins to store outter data
	// for the bottom and top row we only share one row, two for the rest, unless its only one row of bins
	int local_ngrey_bins = bins_per_row;
	// for those with double grey bins: bottom outter bins will be on the left half, top outter bins on the right half
	if(rank != 0 && rank == n_proc -1 && local_nbins > local_ngrey_bins)
		local_ngrey_bins *= 2;

	//MPI_Barrier(MPI_COMM_WORLD);
	//std::cout << "receive particles\n";
	//
	// Distribute particles across the bins ( mainly work for process 0)
	//
	particle_t *local_particles;
	int  nlocal = 0;
	local_particles = receive_particles(&nlocal, n_proc, particles, total_bins, rank, n, bin_size, bins_per_row,
	                                    bins_per_proc, PARTICLE);

	//MPI_Barrier(MPI_COMM_WORLD);
	//std::cout << "particles received\n";
	// once we have our particles locally, we can assign them to the bins
	assign_local_particles_to_ph(local_particles, local_bins_particles_ph, nlocal, bins_per_proc);


	//MPI_Barrier(MPI_COMM_WORLD);
	//std::cout << "particles assigned to placeholders\n";


	//MPI_Barrier(MPI_COMM_WORLD);
	//std::cout << "sending grey particles\n";
	//
	// Distribute grey area particles
	//
	particle_t* local_grey_particles;
	int grey_nlocal = 0;
	local_grey_particles = send_and_receive_grey_area_particles( &grey_nlocal, n_proc, nlocal, local_particles, bins_per_row,
	                                                             local_nbins, rank, PARTICLE, proc_bins_from, proc_bins_until);

	//MPI_Barrier(MPI_COMM_WORLD);
	//std::cout << "grey particles sent\n";
	// assign local grey particles to grey bins
	// we can define the place holders for
	particle_ph* local_grey_bins_particles_ph = (particle_ph*) malloc(local_ngrey_bins * sizeof(particle_ph));
	reset_particles_placeholders(local_grey_bins_particles_ph, local_ngrey_bins);
	assign_local_grey_particles_to_ph(local_grey_particles, local_grey_bins_particles_ph, grey_nlocal, bins_per_row,
	                                  rank, n_proc,proc_bins_from, proc_bins_until);

	//MPI_Barrier(MPI_COMM_WORLD);
	//std::cout << "grey particles assigned to grey bins\n";

	//
	//  set up particle partitioning across processors
	//
	//int particle_per_proc = (n + n_proc - 1) / n_proc;
	//int *partition_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
	//for( int i = 0; i < n_proc+1; i++ )
	//	partition_offsets[i] = min( i * particle_per_proc, n );

	//int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
	//for( int i = 0; i < n_proc; i++ )
	//	partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];

	//
	//  allocate storage for local partition
	//
	//int nlocal = partition_sizes[rank];
	//particle_t *local = (particle_t*) malloc( nlocal * sizeof(particle_t) );

	/*
	MPI_Scatterv( particles, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD );
	// DEBUG INFO
	MPI_Barrier(MPI_COMM_WORLD);
	std::cout << "proces " << rank << " particles assigned " << nlocal << std::endl;
	MPI_Barrier(MPI_COMM_WORLD);


	std::cout << "bins per proc" << bins_per_proc<< std::endl;
	MPI_Barrier(MPI_COMM_WORLD);
	int matches=0, non_matches =0;
	for(int i = 0; i < nlocal; i++){
		int global_bin_id = get_bin_id(bins_per_row, bin_size, particles[i].x, particles[i].y);
		int particle_proc_owner = get_proc_id(global_bin_id, bins_offsets, n_proc);
		int local_bin_position = global_bin_id - (bins_per_proc * (particle_proc_owner));
		std::cout << "proces " << rank << " particle owner  " << particle_proc_owner << std::endl;
		std::cout << "proces " << rank << " particle bin  " << global_bin_id << std::endl;
		std::cout << "proces " << rank << " local bin  " << local_bin_position << std::endl;
		if( global_bin_id == local_bins[local_bin_position].global_id)
			matches++;
		else
			non_matches++;

		//bin_t local_owner_bin = local_bins[]
		//particles[i].next = bins[particle_owner].first;
		//bins[particle_owner].first = &particles[i];
	}
	MPI_Barrier(MPI_COMM_WORLD);
	std::cout << "proces " << rank << " matches  " << matches << std::endl;
	std::cout << "proces " << rank << " non matches  " << non_matches << std::endl;
	//
	//  simulate a number of time steps
	//
	double simulation_time = read_timer( );
	for( int step = 0; step < NSTEPS; step++ )
	{
		navg = 0;
		dmin = 1.0;
		davg = 0.0;

		//
		// Assign particles to bins
		//
		int matches=0, non_matches =0;
		for(int i = 0; i < nlocal; i++){
			int global_bin_id = get_bin_id(bins_per_row, bin_size, particles[i].x, particles[i].y);
			int particle_proc_owner = get_proc_id(global_bin_id, bins_offsets, n_proc);
			int local_bin_position = global_bin_id - (bins_per_proc * particle_proc_owner);
			if( global_bin_id == local_bins[local_bin_position].global_id)
				matches++;
			else
				non_matches++;

			//bin_t local_owner_bin = local_bins[]
			//particles[i].next = bins[particle_owner].first;
			//bins[particle_owner].first = &particles[i];
		}
		std::cout << "proces " << rank << " matches  " << matches << std::endl;
		std::cout << "proces " << rank << " non matches  " << non_matches << std::endl;

		//
		//  collect all global data locally (not good idea to do)
		//
		///*UNCOMMENT
		//MPI_Allgatherv( local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD );
		
		//
		//  save current step if necessary (slightly different semantics than in other codes)
		//
		if( find_option( argc, argv, "-no" ) == -1 )
		  if( fsave && (step%SAVEFREQ) == 0 )
			save( fsave, n, particles );
		
		//
		//  compute all forces
		//
		for( int i = 0; i < nlocal; i++ )
		{
			local[i].ax = local[i].ay = 0;
			for (int j = 0; j < n; j++ )
				apply_force( local[i], particles[j], &dmin, &davg, &navg );
		}
	 
		if( find_option( argc, argv, "-no" ) == -1 )
		{
		  
		  MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		  MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
		  MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

 
		  if (rank == 0){
			//
			// Computing statistical data
			//
			if (rnavg) {
			  absavg +=  rdavg/rnavg;
			  nabsavg++;
			}
			if (rdmin < absmin) absmin = rdmin;
		  }
		}

		//
		//  move particles
		//
		for( int i = 0; i < nlocal; i++ )
			move( local[i] );
	}
	simulation_time = read_timer( ) - simulation_time;
  
	if (rank == 0) {  
	  printf( "n = %d, simulation time = %g seconds", n, simulation_time);

	  if( find_option( argc, argv, "-no" ) == -1 )
	  {
		if (nabsavg) absavg /= nabsavg;
	  // 
	  //  -The minimum distance absmin between 2 particles during the run of the simulation
	  //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
	  //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
	  //
	  //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
	  //
	  printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
	  if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
	  if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
	  }
	  printf("\n");	 
		
	  //  
	  // Printing summary data
	  //  
	  if( fsum)
		fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
	}
    */
	//
	//  release resources
	//
	if ( fsum )
		fclose( fsum );

	free(local_bins);

	free(proc_bins_from);
	free(proc_bins_until);

	free( local_particles );
	free( local_grey_particles);


	free( local_bins_particles_ph);
	free( local_grey_bins_particles_ph);

	if (rank == 0) {
		free(particles);
	}
	if( fsave )
		fclose( fsave );
	
	MPI_Finalize( );
	
	return 0;
}
