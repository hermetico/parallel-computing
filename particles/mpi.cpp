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
#define UPDATING_PARTICLES_LEVEL 3
#define UPDATING_GREY_PARTICLES_LEVEL 2
#define CHECKING_INTERACTION_LEVEL 1
#define VERBOSE_LEVEL 1

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
	MPI_Type_contiguous( 11, MPI_DOUBLE, &PARTICLE ); // as many doubles as the structure has
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
	double bin_size = cutoff * 1.5;
	bins_per_row = ceil(size / bin_size);
	total_bins = bins_per_row * bins_per_row;


	bins_per_proc = total_bins / n_proc;
	// bins_per_proc should be divisible by bins_per_row ( easier to handle I hope)
	if(bins_per_proc % bins_per_row != 0){
		bins_per_proc += (bins_per_row - (bins_per_proc % bins_per_row));
	}

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

	/*
	if(rank == 0){
		for(int i = 0; i < n_proc; i++){
			std::cout << "Process " << i << " with bins from " << proc_bins_from[i] << " to " << proc_bins_until[i] << std::endl;
		}
	}
	 */
	// knowing the bins we can populate the placeholders, which are the first entry for linked lists
	// of particles
	// the first particle to be pointing from the bin
	particle_ph* local_bins_particles_ph = (particle_ph*) malloc(local_nbins * sizeof(particle_ph));
	reset_particles_placeholders(local_bins_particles_ph, local_nbins);


	// Also, knowing the number of bins we can also define the
	// outter bins ( the ones in the grey area ) and populate them
	// define local grey bins for process, this number will be used to define the bins to store outter data
	// for the bottom and top row we only share one row, two for the rest, unless its only one row of bins
	int local_ngrey_bins = bins_per_row;
	// for those with double grey bins: bottom outter bins will be on the left half, top outter bins on the right half
	if(rank > 0 && rank < n_proc -1)// && local_nbins > local_ngrey_bins)
		local_ngrey_bins *= 2;

	//
	// Distribute particles across the bins
	//
	particle_t *local_particles;
	int  nlocal = 0;

	if (rank == 0) {
		local_particles = send_and_receive_local_particles(particles, &nlocal, n, n_proc, rank,
		                                                   PARTICLE, bins_per_row, bin_size, bins_per_proc);
	}else{
		particle_t *dummy_particles = (particle_t*) malloc(0 * sizeof(particle_t));
		local_particles = send_and_receive_local_particles(dummy_particles, &nlocal, 0, n_proc, rank,
		                                                   PARTICLE, bins_per_row, bin_size, bins_per_proc);
		free(dummy_particles);
	}


	// once we have our particles locally, we can assign them to the bins
	assign_local_particles_to_ph(local_particles, local_bins_particles_ph, nlocal, bins_per_proc);

	//
	// Distribute grey area particles
	//
	particle_t* local_grey_particles;
	int grey_nlocal = 0;
	local_grey_particles = send_and_receive_grey_area_particles( &grey_nlocal, n_proc, nlocal, local_particles, bins_per_row,
	                                                             local_nbins, rank, PARTICLE, proc_bins_from, proc_bins_until);

	// assign local grey particles to grey bins
	// we can define the place holders for
	particle_ph* local_grey_bins_particles_ph = (particle_ph*) malloc(local_ngrey_bins * sizeof(particle_ph));
	reset_particles_placeholders(local_grey_bins_particles_ph, local_ngrey_bins);
	assign_local_grey_particles_to_ph(local_grey_particles, local_grey_bins_particles_ph, grey_nlocal, bins_per_row,
	                                  rank, n_proc,proc_bins_from, proc_bins_until);


	//
	//  simulate a number of time steps
	//

	double simulation_time = read_timer( );
	for( int step = 0; step < NSTEPS; step++ )
	{
		navg = 0;
		dmin = 1.0;
		davg = 0.0;


		int* vis_counts = (int*) malloc(n_proc * sizeof(int));
		int* vis_displs = (int*) malloc(n_proc * sizeof(int));

		MPI_Gather(&nlocal, 1, MPI_INT, vis_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
		vis_displs[0] = 0;
		for(int i = 1; i< n_proc; i++)
		{
			vis_displs[i] = vis_displs[i-1] + vis_counts[i-1];
		}

		MPI_Gatherv(local_particles, nlocal, PARTICLE, particles, vis_counts, vis_displs, PARTICLE,0,  MPI_COMM_WORLD);
		//
		//  save current step if necessary (slightly different semantics than in other codes)
		//
		if( find_option( argc, argv, "-no" ) == -1 )
		  if( fsave && (step%SAVEFREQ) == 0 ) {
			  //MPI_gather( local_particles, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD );
			  save(fsave, n, particles);
		  }
		free(vis_counts);
		free(vis_displs);
		//
		//
		//  compute all forces
		//

		// compute forces
		for(int y = 0; y < local_nbins; y++ )
		{
			bin_t* c_bin = &local_bins[y];
			particle_ph* ph = &local_bins_particles_ph[y];

			particle_t* c_particle = ph->first;
			particle_t* other;
			while(c_particle)
			{

				other = NULL;
				c_particle->ax = 0;
				c_particle->ay = 0;

				// same bin
				//std::cout << "Particle at bin " << c_bin->global_id << " interacting with:\n ";
				apply_forces_linked_particles(c_particle, ph->first, &dmin, &davg, &navg);


				if(c_bin->top != -1){
					//std::cout << "Process " << rank << " local bin " << y << std::endl;
					int bin_id = c_bin->top;

					if( bin_id > proc_bins_until[rank]) // must be in grey bins
					{
						bin_id = get_local_grey_bin_id_from_global(bin_id, rank, n_proc, bins_per_row, proc_bins_from, proc_bins_until);
						other = local_grey_bins_particles_ph[bin_id].first;
						//std::cout << "Process " << rank << " grey neighbor in " << bin_id << std::endl;

					}else{ // local bins
						bin_id = get_local_bin_from_global_bin(bin_id, bins_per_proc);
						other = local_bins_particles_ph[bin_id].first;
						//std::cout << "Process " << rank << " local neighbor in " << bin_id << std::endl;

					}
					//std::cout << "Bin " << c_bin->top << " \n ";
					apply_forces_linked_particles(c_particle, other, &dmin, &davg, &navg);
				}


				if(c_bin->bottom != -1) {
					int bin_id = c_bin->bottom;
					if( bin_id < proc_bins_from[rank]) // must be in grey bins
					{
						bin_id = get_local_grey_bin_id_from_global(bin_id, rank, n_proc, bins_per_row, proc_bins_from, proc_bins_until);
						other = local_grey_bins_particles_ph[bin_id].first;
						//std::cout << "Process " << rank << " grey neighbor in " << bin_id << std::endl;

					}else{ // local bins
						bin_id = get_local_bin_from_global_bin(bin_id, bins_per_proc);
						other = local_bins_particles_ph[bin_id].first;
						//std::cout << "Process " << rank << " local neighbor in " << bin_id << std::endl;
					}
					//std::cout << "Bin " << c_bin->bottom << " \n ";
					apply_forces_linked_particles(c_particle, other, &dmin, &davg, &navg);

				}

				if(c_bin->left != -1) { // always in local bin
					int bin_id = c_bin->left;
					bin_id = get_local_bin_from_global_bin(bin_id, bins_per_proc);
					other = local_bins_particles_ph[bin_id].first;
					//std::cout << "Bin " << c_bin->left << " \n ";
					apply_forces_linked_particles(c_particle, other, &dmin, &davg, &navg);

				}

				if(c_bin->right != -1){ // always in local bin
					int bin_id = c_bin->right;
					bin_id = get_local_bin_from_global_bin(bin_id, bins_per_proc);
					other = local_bins_particles_ph[bin_id].first;
					//std::cout << "Bin " << c_bin->right << " \n ";
					apply_forces_linked_particles(c_particle, other, &dmin, &davg, &navg);

				}

				if(c_bin->top_left != -1) {
					int bin_id = c_bin->top_left;
					if( bin_id > proc_bins_until[rank]) // must be in grey bins
					{
						bin_id = get_local_grey_bin_id_from_global(bin_id, rank, n_proc, bins_per_row, proc_bins_from, proc_bins_until);
						other = local_grey_bins_particles_ph[bin_id].first;
						//std::cout << "Process " << rank << " grey neighbor in " << bin_id << std::endl;

					}else{ // local bins
						bin_id = get_local_bin_from_global_bin(bin_id, bins_per_proc);
						other = local_bins_particles_ph[bin_id].first;
						//std::cout << "Process " << rank << " local neighbor in " << bin_id << std::endl;

					}
					//std::cout << "Bin " << c_bin->top_left << " \n ";
					apply_forces_linked_particles(c_particle, other, &dmin, &davg, &navg);

				}
				if(c_bin->top_right != -1) {
					int bin_id = c_bin->top_right;
					if( bin_id > proc_bins_until[rank]) // must be in grey bins
					{
						bin_id = get_local_grey_bin_id_from_global(bin_id, rank, n_proc, bins_per_row, proc_bins_from, proc_bins_until);
						other = local_grey_bins_particles_ph[bin_id].first;
						//std::cout << "Process " << rank << " grey neighbor in " << bin_id << std::endl;

					}else{ // local bins
						bin_id = get_local_bin_from_global_bin(bin_id, bins_per_proc);
						other = local_bins_particles_ph[bin_id].first;
						//std::cout << "Process " << rank << " local neighbor in " << bin_id << std::endl;
					}
					//std::cout << "Bin " << c_bin->top_right << " \n ";
					apply_forces_linked_particles(c_particle, other, &dmin, &davg, &navg);

				}
				if(c_bin->bottom_left != -1) {
					int bin_id = c_bin->bottom_left;
					if( bin_id < proc_bins_from[rank]) // must be in grey bins
					{
						bin_id = get_local_grey_bin_id_from_global(bin_id, rank, n_proc, bins_per_row, proc_bins_from, proc_bins_until);
						other = local_grey_bins_particles_ph[bin_id].first;
						//std::cout << "Process " << rank << " grey neighbor in " << bin_id << std::endl;


					}else{ // local bins
						bin_id = get_local_bin_from_global_bin(bin_id, bins_per_proc);
						other = local_bins_particles_ph[bin_id].first;
						//std::cout << "Process " << rank << " local neighbor in " << bin_id << std::endl;

					}
					//std::cout << "Bin " << c_bin->bottom_left << " \n ";
					apply_forces_linked_particles(c_particle, other, &dmin, &davg, &navg);

				}
				if(c_bin->bottom_right != -1) {
					int bin_id = c_bin->bottom_right;
					if( bin_id < proc_bins_from[rank]) // must be in grey bins
					{
						bin_id = get_local_grey_bin_id_from_global(bin_id, rank, n_proc, bins_per_row, proc_bins_from, proc_bins_until);
						other = local_grey_bins_particles_ph[bin_id].first;
						//std::cout << "Process " << rank << " grey neighbor in " << bin_id << std::endl;


					}else{ // local bins
						bin_id = get_local_bin_from_global_bin(bin_id, bins_per_proc);
						other = local_bins_particles_ph[bin_id].first;
						//std::cout << "Process " << rank << " local neighbor in " << bin_id << std::endl;

					}
					//std::cout << "Bin " << c_bin->bottom_right << " \n ";
					apply_forces_linked_particles(c_particle, other, &dmin, &davg, &navg);


				}
				c_particle = c_particle->next;
			}

		}

		//
		//  move particles
		//
		for( int i = 0; i < nlocal; i++ )
			move( local_particles[i] );


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
		// Update particles bins
		//
		particle_t* old_local_particles = local_particles;
		int updated_nlocal = 0;
		local_particles = send_and_receive_local_particles(old_local_particles, &updated_nlocal, nlocal, n_proc, rank,
									PARTICLE, bins_per_row, bin_size, bins_per_proc);

		free(old_local_particles);

		nlocal = updated_nlocal;
		reset_particles_placeholders(local_bins_particles_ph, local_nbins);
		assign_local_particles_to_ph(local_particles, local_bins_particles_ph, nlocal, bins_per_proc);


		//
		// Distribute grey area particles
		//

		particle_t* old_local_grey_particles = local_grey_particles;
		local_grey_particles = send_and_receive_grey_area_particles( &grey_nlocal, n_proc, nlocal, local_particles, bins_per_row,
		                                                             local_nbins, rank, PARTICLE, proc_bins_from, proc_bins_until);
		free(old_local_grey_particles);

		// assign local grey particles to grey bins
		// we can define the place holders for

		reset_particles_placeholders(local_grey_bins_particles_ph, local_ngrey_bins);
		assign_local_grey_particles_to_ph(local_grey_particles, local_grey_bins_particles_ph, grey_nlocal, bins_per_row,
		                                  rank, n_proc,proc_bins_from, proc_bins_until);

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
