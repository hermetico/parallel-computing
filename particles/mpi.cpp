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
#define VERBOSE 5

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
	if(VERBOSE > 5 && rank == 0) {
		std::cout << "A total of " << n_proc << " processes" << std::endl;
	}
	
	//
	//  allocate generic resources
	//
	FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
	FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


	particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
	particle_t *ordered_particles = (particle_t*) malloc( n * sizeof(particle_t) );
	
    /* Creates a datatype for the particle, it contains 7 double
     * which are the values and the pointer to the next one
     */
    MPI_Datatype PARTICLE;
	MPI_Type_contiguous( 9, MPI_DOUBLE, &PARTICLE ); // as many doubles as the structure has
	MPI_Type_commit( &PARTICLE );

	MPI_Datatype BIN;
	MPI_Type_contiguous( 9, MPI_INT, &BIN );
	MPI_Type_commit( &BIN );


	//
	//  initialize and distribute the particles (that's fine to leave it unoptimized)
	//
	set_size( n );
	if( rank == 0 )
		init_particles( n, particles );


	//
	// setup the bin partitioning across processors
	// so far, only slicing rows, would be nice to slice in a grid manner
	//
	int total_bins, bins_per_row, bins_per_proc;
	double bin_size = cutoff;
	bins_per_row = ceil(size / bin_size);
	total_bins = bins_per_row * bins_per_row;

	if(VERBOSE > 5 && rank == 0)
		std::cout << "Total global bins " << total_bins <<  std::endl;

	bins_per_proc = (total_bins +  n_proc - 1) / n_proc;
	// bins_per_proc should be divisible by bins_per_row ( easier to handle I hope)
	//TODO CHECK DOUBLE CHECK TRIPPPPLE CHECK THIS
	if(bins_per_proc % bins_per_row != 0){
		bins_per_proc += bins_per_row - (bins_per_proc % bins_per_row);
	}
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
	particle_ph* local_bins_particles_ph = (particle_ph*) malloc(local_nbins * sizeof(particle_ph));
	reset_particles_placeholders(local_bins_particles_ph, local_nbins);

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

	// Send binds to the people
	MPI_Scatterv(global_bins, bins_per_proc_sizes, bins_offsets, BIN, local_bins, local_nbins, BIN, 0, MPI_COMM_WORLD);

	// define local grey bins for process, this number will be used to define the bins to store outter data
	// for the bottom and top row we only share one row, two for the rest, unless its only one row of bins
	int local_ngrey_bins = bins_per_row;
	if(rank != 0 && rank == n_proc -1 && local_nbins > local_ngrey_bins)
		local_ngrey_bins *= 2;

	particle_ph* local_grey_bins_particles_ph = (particle_ph*) malloc(local_ngrey_bins * sizeof(particle_ph));
	for(int i = 0; i < local_ngrey_bins; i++)
		reset_particles_placeholders(local_bins_particles_ph, local_nbins);


	if( VERBOSE > 5) {
		// DEBUG INFO

		if (rank == 0)
			std::cout << "total bins " << total_bins << std::endl;

		MPI_Barrier(MPI_COMM_WORLD);
		std::cout << "proces " << rank << " bins assigned " << local_nbins << std::endl;
		std::cout << "proces " << rank << ":" << std::endl;

		for( int i = 0; i < local_nbins; i++){

			int global_id = local_bins[i].global_id;
			std::cout << "Global id " << local_bins[i].global_id << " this bin should be at processor ";
			std::cout << get_proc_from_bin(global_id, bins_per_proc);
			std::cout << " ; local id "  << i ;
			std::cout << " and  computed local id " << get_local_bin_from_global_bin(global_id, bins_per_proc) << std::endl;
		}

	}

	//
	// Distribute particles across the bins
	//
	particle_ph* total_bins_particles_ph;
	int particle_per_proc = (n + n_proc - 1) / n_proc; // lets hope this is always true

	if(rank == 0) // global bins setup from process 0
	{
		total_bins_particles_ph = (particle_ph*) malloc(total_bins * sizeof(particle_ph));
		reset_particles_placeholders(total_bins_particles_ph, total_bins);
		// assign particles to bins
		for (int i = 0; i < n; i++) {
			int global_bin_id = get_bin_id(bins_per_row, bin_size, particles[i].x, particles[i].y);
			particles[i].next = total_bins_particles_ph[global_bin_id].first;
			particles[i].global_bin_id = (double) global_bin_id;
			particles[i].proc_id = (double) get_proc_from_bin(global_bin_id, bins_per_proc);
			//std::cout << "A particle for proc " << particles[i].proc_id << std::endl;
			total_bins_particles_ph[global_bin_id].first = &particles[i];
			total_bins_particles_ph[global_bin_id].size++;

		}


	}
	if (VERBOSE > 5) {
		MPI_Barrier(MPI_COMM_WORLD);
		for (int i = 0; i < total_bins; i++) {
			std::cout << "Bin " << i << " with " << total_bins_particles_ph[i].size << std::endl;
		}
	}

	int* partition_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
	for(int i = 0; i< n_proc+1;i++)
		partition_offsets[i] = 0;

	int* partition_sizes = (int*) malloc( n_proc * sizeof(int) );
	for(int i = 0; i< n_proc;i++)
		partition_sizes[i] = 0;

	// prepare data to be sent ot other processes
	if(rank == 0){
		int particles_copied = 0;
		for (int i = 0; i < total_bins; i++) {
			particle_t* c_p = total_bins_particles_ph[i].first;
			while(c_p)
			{
				//std::cout << "Adding particle for proc " << (int)c_p->proc_id;
				partition_sizes[(int)c_p->proc_id]++;
				//memcpy(ordered_particles + particles_copied, &c_p, sizeof(c_p));
				pt_copy(ordered_particles + particles_copied, c_p);
				particles_copied++;
				//std::cout << ", it has " << partition_sizes[(int)c_p->proc_id] <<" so far" <<std::endl;
				c_p = c_p->next;
			}
		}
		for(int i = 1; i< n_proc+1;i++) {
			partition_offsets[i] = min(partition_offsets[i - 1] + partition_sizes[i - 1], n);

		}
	}
	if(VERBOSE == 5 && rank == 0){
		for(int i = 0; i< n_proc;i++) {
			std::cout << "Size at " << i << " = " << partition_sizes[i] << std::endl;
			std::cout << "Offset at " << i << " = " << partition_offsets[i] << std::endl;
		}
	}

	particle_t *local_particles;
	int nlocal;
	if(rank == 0){
		for(int i = 1; i<n_proc; i++)
		{
			//std::cout << "Sending " << partition_sizes[i] << " to " << i << std::endl;
			MPI_Send((ordered_particles + partition_offsets[i]), partition_sizes[i], PARTICLE, i, 0, MPI_COMM_WORLD);
		}

		// no need for comunication, no need for copy this here either though, but :/
		nlocal = partition_sizes[0];
		local_particles  = (particle_t*) malloc( nlocal * sizeof(particle_t) );
		for(int i = 0; i < nlocal; i++){
			//memcpy(local_particles + i, ordered_particles + i, sizeof(particle_t));
			pt_copy(local_particles + i, ordered_particles + i);
			(local_particles+i)->next = NULL;
		}
	}else{

		MPI_Status status;
		MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status, PARTICLE, &nlocal);
		local_particles  = (particle_t*) malloc( nlocal * sizeof(particle_t) );
		MPI_Recv(local_particles, nlocal, PARTICLE, 0, 0, MPI_COMM_WORLD, &status);
	}

	if(VERBOSE == 5){
		MPI_Barrier(MPI_COMM_WORLD);
		std::cout << "Process " << rank << " with " << nlocal << " particles " << std::endl;
		for(int i = 0; i < nlocal; i++){
			std::cout << "Received particle for process " << local_particles[i].proc_id << " for bin " << local_particles[i].global_bin_id << std::endl;
		}
	}

	// once we have our particles locally, we can assign them to the bins
	for(int i = 0; i < nlocal; i++)
	{
		int local_bin = get_local_bin_from_global_bin(local_particles[i].global_bin_id, bins_per_proc);
		local_bins_particles_ph[local_bin].first = &local_particles[i];
		local_bins_particles_ph[local_bin].size++;
	}
	if(VERBOSE == 5) {
		MPI_Barrier(MPI_COMM_WORLD);
		std::cout << "Process " << rank << " assigned particles locally " << std::endl;
	}

	// once we have all the particles, we can distribute the ones in our grey zone

	// prepare structure
	// first check to send to upper process
	//particle_t* grey_send_buff  = (particle_t*) malloc( 2 * n * sizeof(particle_t) );
	//particle_t* grey_recv_buff  = (particle_t*) malloc( 2 * n * sizeof(particle_t) );
	particle_ph* grey_send_ph  = (particle_ph*) malloc( n_proc * sizeof(particle_ph) );
	reset_particles_placeholders(grey_send_ph, n_proc);
	int* send_counts = (int*) malloc(n_proc * sizeof(int));
	for(int i = 0; i< n_proc; i++){
		send_counts[i] = 0;
	}
	int* recv_counts = (int*) malloc(n_proc * sizeof(int));
	int* send_displs = (int*) malloc(1 + n_proc * sizeof(int));
	int* recv_displs = (int*) malloc(1 + n_proc * sizeof(int));


	int total_send_counts = 0;
	// collect info about data to be sent
	for(int i = 0; i < nlocal; i++){
		int bin_id = local_particles[i].global_bin_id;
		// needs to go up?
		if(bin_id + bins_per_row >= local_nbins * rank && rank < n_proc - 1)
		{
			int recv_id = rank + 1;
			//pt_copy(&grey_send_buff[particle_per_proc * recv_id + send_counts[recv_id]], &local_particles[i]);
			local_particles[i].next = grey_send_ph[recv_id].first;
			grey_send_ph[recv_id].first = &local_particles[i];
			grey_send_ph[recv_id].size++;
			send_counts[recv_id]++;
			total_send_counts++;
			std::cout << "Particle at process " << rank << " sending it to " <<recv_id <<std::endl;
		}else{
			std::cout << "Particle at process " << rank << " not going anywhere\n";
		}
		// needs to go down?
		if(bin_id - bins_per_row < local_nbins * rank && rank != 0)
		{
			int recv_id = rank - 1;
			local_particles[i].next = grey_send_ph[recv_id].first;
			grey_send_ph[recv_id].first = &local_particles[i];
			grey_send_ph[recv_id].size++;
			send_counts[recv_id]++;
			total_send_counts++;
		}
	}
	// organize data to be sent
	particle_t* grey_send_buff  = (particle_t*) malloc( total_send_counts *  sizeof(particle_t) );
	int pos = 0;
	for(int i = 0; i< n_proc; i++)
	{
		particle_t* c_p = grey_send_ph[i].first;
		for(int j = 0; j < grey_send_ph[i].size; j++){

			pt_copy(&grey_send_buff[pos], c_p);
			c_p = c_p->next;
			pos++;
		}
	}

	if(VERBOSE == 5) {
		MPI_Barrier(MPI_COMM_WORLD);
		std::cout << "Process " << rank << " grey particles assigned " << std::endl;
	}
	// prepare the offsets
	send_displs[0] = 0;
	for(int i = 1; i< n_proc+1;i++) {
		send_displs[i] = min(send_displs[i - 1] + send_counts[i - 1], n);
	}
	if(VERBOSE == 5){
		MPI_Barrier(MPI_COMM_WORLD);
		for(int i = 0; i < n_proc; i++){
			std::cout << "Process " << rank << " will send: " ;
			std::cout << send_counts[i] << " particles to process " << i  << std::endl;
		}
	}

	// let them know the quantity of data to receive
	MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1,MPI_INT,  MPI_COMM_WORLD);
	if(VERBOSE == 5){
		MPI_Barrier(MPI_COMM_WORLD);
		for(int i = 0; i < n_proc; i++){
			std::cout << "Process " << rank << " will receive: " ;
			std::cout << recv_counts[i] << " particles from process " << i  << std::endl;
		}
	}

	int total_recv_counts = 0;
	recv_displs[0] = 0;
	for(int i = 1; i< n_proc+1;i++) {
		total_recv_counts +=recv_counts[i - 1];
		recv_displs[i] = min(recv_displs[i - 1] + recv_counts[i - 1], n);
	}
	particle_t* grey_recv_buff  = (particle_t*) malloc( total_recv_counts * sizeof(particle_t) );
	MPI_Alltoallv(grey_send_buff, send_counts, send_displs, PARTICLE, grey_recv_buff, recv_counts, recv_displs, PARTICLE, MPI_COMM_WORLD);
	if(VERBOSE == 5){
		MPI_Barrier(MPI_COMM_WORLD);
		std::cout << "All processes have received data\n";
	}



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
	free( bins_offsets );
	free( bins_per_proc_sizes );
	//free( local );
	free( particles );
	if( fsave )
		fclose( fsave );
	
	MPI_Finalize( );
	
	return 0;
}
