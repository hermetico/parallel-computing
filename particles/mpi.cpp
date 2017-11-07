#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "common.h"
#include <iostream>
#include <math.h>
#include <vector>


// the cutoff
#define cutoff  0.01

// the size of the grid
extern double size;
typedef struct bin_t bin_t;
struct bin_t
{
	// neighbor bins
	int top;
	int bottom;
	int left;
	int right;
	int top_left;
	int top_right;
	int bottom_left;
	int bottom_right;
	int global_id;

};


int get_bin_id(int bins_per_row, double bin_size,  double x, double y){
	int binx = floor(x / bin_size);
	int biny = floor(y / bin_size);

	return  bins_per_row * biny + binx;
}


int get_proc_id(int bin_id, int* bins_offsets, int num_procs){
	for( int i = 0; i < num_procs + 1; i++ ) {
		//TODO check >= >
		if (bin_id >= bins_offsets[i]) {
			return i;
		}
	}
}

void reset_bins_particles(particle_t** bins_particles, int local_nbins){
	for(int i = 0; i < local_nbins; i++)
		bins_particles[i] = 0;
}

void link_bins(int bins_per_row, bin_t* bins){
	for(int y = 0; y < bins_per_row; y++ )
	{
		for(int x = 0; x < bins_per_row; x++)
		{
			bin_t* c_bin = &bins[y * bins_per_row + x];
			if( y > 0)
			{
				c_bin->bottom = (y-1) * bins_per_row + x;

				if (x > 0)
					c_bin->bottom_left = (y - 1) * bins_per_row + (x - 1);

				if (x < bins_per_row - 1)
					c_bin->bottom_right = (y - 1) * bins_per_row + (x + 1);
			}

			if( y < bins_per_row - 1)
			{
				c_bin->top = (y+1) * bins_per_row + x;

				if (x > 0)
					c_bin->top_left = (y + 1) * bins_per_row + (x - 1);

				if (x < bins_per_row - 1)
					c_bin->top_right = (y + 1) * bins_per_row + (x + 1);
			}

			if (x > 0)
				c_bin->left = y * bins_per_row + (x - 1);

			if (x < bins_per_row - 1)
				c_bin->right = y * bins_per_row + (x + 1);
		}
	}
}

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
	//std::cout << "proces " << rank << " of "<< n_proc << " processes" << std::endl;
	
	//
	//  allocate generic resources
	//
	FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
	FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


	particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
	
    /* Creates a datatype for the particle, it contains 7 double
     * which are the values and the pointer to the next one
     */
    MPI_Datatype PARTICLE;
	MPI_Type_contiguous( 7, MPI_DOUBLE, &PARTICLE ); // as many doubles as the structure has
	MPI_Type_commit( &PARTICLE );

	MPI_Datatype BIN;
	MPI_Type_contiguous( 9, MPI_INT, &BIN );
	MPI_Type_commit( &BIN );


	//
	//  set up the data partitioning across processors
	//
	int particle_per_proc = (n + n_proc - 1) / n_proc;
	int *partition_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
	for( int i = 0; i < n_proc+1; i++ )
		partition_offsets[i] = min( i * particle_per_proc, n );

	int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
	for( int i = 0; i < n_proc; i++ )
		partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];

	//
	//  allocate storage for local partition
	//
	int nlocal = partition_sizes[rank];
	particle_t *local = (particle_t*) malloc( nlocal * sizeof(particle_t) );
	
	//
	//  initialize and distribute the particles (that's fine to leave it unoptimized)
	//
	set_size( n );
	if( rank == 0 )
		init_particles( n, particles );

	//
	// setup the bin partitioning across processors
	//
	int total_bins, bins_per_row, bins_per_proc;
	double bin_size = cutoff;
	bins_per_row = ceil(size / bin_size);
	total_bins = bins_per_row * bins_per_row;


	bins_per_proc = (total_bins +  n_proc - 1) / n_proc;
	int* bins_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
	for( int i = 0; i < n_proc + 1; i++ )
		bins_offsets[i] = min(i * bins_per_proc, total_bins);

	int* bins_per_proc_sizes = (int*) malloc( n_proc * sizeof(int) );
	for( int i = 0; i < n_proc; i++ )
		bins_per_proc_sizes[i] = bins_offsets[i+1] - bins_offsets[i];


	bin_t* local_bins;
	bin_t* global_bins;
	// the real local number of bins
	int local_nbins = bins_per_proc_sizes[rank];
	local_bins = (bin_t*) malloc( local_nbins * sizeof(bin_t));
	// the first particle to be pointing from the bin
	particle_t** bins_particles = static_cast<particle_t **>(malloc(local_nbins * sizeof(particle_t*)));
	reset_bins_particles(bins_particles, local_nbins);

	if(rank == 0) // global bins setup from process 0
	{
		global_bins = (bin_t*) malloc( total_bins * sizeof(bin_t));
		for (int i = 0; i < total_bins; i++) {
			bin_t new_bin;
			new_bin.top = -1;
			new_bin.bottom = -1;
			new_bin.left = -1;
			new_bin.right = -1;
			new_bin.top_left = -1;
			new_bin.top_right = -1;
			new_bin.bottom_left = -1;
			new_bin.bottom_right = -1;
			new_bin.global_id = i;
			global_bins[i] = new_bin;
		}
		link_bins(bins_per_row, global_bins);
	}

	MPI_Scatterv(global_bins, bins_per_proc_sizes, bins_offsets, BIN, local_bins, local_nbins, BIN, 0, MPI_COMM_WORLD);


	// DEBUG INFO
		MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0)
		std::cout << "total bins " << total_bins << std::endl;
	std::cout << "proces " << rank << " bins assigned " << local_nbins << std::endl;
	std::cout << "proces " << rank << ":"<< std::endl;
	for( int i = 0; i < local_nbins; i++){
		std::cout << "bin with id " << local_bins[i].global_id << std::endl;
	}


	MPI_Scatterv( particles, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD );
	/*// DEBUG INFO
	MPI_Barrier(MPI_COMM_WORLD);
	std::cout << "proces " << rank << " particles assigned " << nlocal << std::endl;
	MPI_Barrier(MPI_COMM_WORLD);
	*/

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
		/*int matches=0, non_matches =0;
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
		 */
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
  
	//
	//  release resources
	//
	if ( fsum )
		fclose( fsum );
	free(local_bins);
	free( bins_offsets );
	free( bins_per_proc_sizes );
	free( local );
	free( particles );
	if( fsave )
		fclose( fsave );
	
	MPI_Finalize( );
	
	return 0;
}
