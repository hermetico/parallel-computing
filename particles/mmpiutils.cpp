#ifndef __MY_MPI_UTILS__
#define __MY_MPI_UTILS__
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "common.h"
#include <iostream>
#define BINS_LEVEL 6
#define PARTICLES_LEVEL 5
#define MOVING_PARTICLES_LEVEL 4
#define VERBOSE_LEVEL 4


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

// particles placeholder
typedef struct particle_ph particle_ph;
struct particle_ph
{
	particle_t* first;
	int size;
};

int get_bin_id(int bins_per_row, double bin_size,  double x, double y){
	int binx = floor(x / bin_size);
	int biny = floor(y / bin_size);

	return  bins_per_row * biny + binx;
}

void pt_copy(particle_t* to, particle_t* from){
	to->ax = from->ax;
	to->ay = from->ay;
	to->x = from->x;
	to->y = from->y;
	to->vx = from->vx;
	to->vy = from->vy;
	to->proc_id = from->proc_id;
	to->global_bin_id = from->global_bin_id;
	to->next = NULL;
	to->next_grey = NULL;
}

int get_proc_from_bin(int bin_id, int bins_per_proc){
	return (int)floor(bin_id / bins_per_proc);
}

int get_local_bin_from_global_bin(int bin_id, int bins_per_proc){
	return bin_id % bins_per_proc;
}

void reset_particles_placeholders(particle_ph* bins_particles, int local_nbins){
	for(int i = 0; i < local_nbins; i++) {
		particle_ph new_ph;
		new_ph.first = NULL;
		new_ph.size = 0;
		bins_particles[i] = new_ph;
	}
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

void reset_bins(bin_t* bins, int total_bins){
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
		bins[i] = new_bin;
	}
}

void show_placeholders(particle_ph* ph, int num_bins, int bins_per_row){
	std::cout << std::endl;
	std::cout << std::endl;
	for(int y = 0; y < num_bins; y++ )
	{
		std::cout <<  ph[y].size << ",   " ;
		if((y + 1) % bins_per_row == 0) std::cout << std::endl;

	}
	std::cout << std::endl;
}

void show_bins(bin_t* bins, int num_bins, int bins_per_row){
	std::cout << std::endl;
	std::cout << std::endl;
	for(int y = 0; y < num_bins; y++ )
	{
		std::cout <<  bins[y].global_id << ",   " ;
		if((y + 1) % bins_per_row == 0) std::cout << std::endl;

	}
	std::cout << std::endl;
}

void assign_particles_to_ph(particle_t* particles, particle_ph* particle_ph, int n, int bins_per_row, int bins_per_proc, double bin_size){
	for (int i = 0; i < n; i++) {
		int global_bin_id = get_bin_id(bins_per_row, bin_size, particles[i].x, particles[i].y);
		particles[i].next = particle_ph[global_bin_id].first;
		particles[i].global_bin_id = (double) global_bin_id;
		particles[i].proc_id = (double) get_proc_from_bin(global_bin_id, bins_per_proc);
		//std::cout << "A particle for proc " << particles[i].proc_id << std::endl;
		particle_ph[global_bin_id].first = &particles[i];
		particle_ph[global_bin_id].size++;

	}
}

void assign_local_particles_to_ph(particle_t* particles, particle_ph* local_placeholders,  int nlocal, int bins_per_proc){
	for(int i = 0; i < nlocal; i++)
	{
		int local_bin = get_local_bin_from_global_bin(particles[i].global_bin_id, bins_per_proc);
		local_placeholders[local_bin].first = &particles[i];
		local_placeholders[local_bin].size++;
	}
}

// populates buff_length
particle_t* send_and_receive_grey_area_particles( int* buff_length, int n_proc, int nlocal,
	particle_t* local_particles, int bins_per_row, int local_nbins, int rank, MPI_Datatype PARTICLE){


	particle_ph* grey_send_ph;
	int *send_counts, *recv_counts, *send_displs, *recv_displs;
	particle_t* grey_send_buff;

	grey_send_ph  = (particle_ph*) malloc( n_proc * sizeof(particle_ph) );
	reset_particles_placeholders(grey_send_ph, n_proc);

	send_counts = (int*) malloc(n_proc * sizeof(int));
	recv_counts = (int*) malloc(n_proc * sizeof(int));
	send_displs = (int*) malloc((1 + n_proc) * sizeof(int));
	recv_displs = (int*) malloc((1 + n_proc) * sizeof(int));

	for(int i = 0; i< n_proc; i++){
		send_counts[i] = 0;
	}

	int total_send_counts = 0;
	// collect info about data to be sent
	for(int i = 0; i < nlocal; i++){
		int bin_id = local_particles[i].global_bin_id;

		// needs to go up?
		if(bin_id + bins_per_row >= local_nbins * (rank + 1) && rank < n_proc - 1)
		{
			int recv_id = rank + 1;
			local_particles[i].next_grey = grey_send_ph[recv_id].first;
			grey_send_ph[recv_id].first = &local_particles[i];
			grey_send_ph[recv_id].size++;
			send_counts[recv_id]++;
			total_send_counts++;
			//std::cout << "Particle at process " << rank << " sending it to " <<recv_id <<std::endl;
		}
		// needs to go down?
		if(bin_id - bins_per_row < local_nbins * (rank + 1) && rank != 0)
		{
			int recv_id = rank - 1;
			local_particles[i].next_grey = grey_send_ph[recv_id].first;
			grey_send_ph[recv_id].first = &local_particles[i];
			grey_send_ph[recv_id].size++;
			send_counts[recv_id]++;
			total_send_counts++;
			//std::cout << "Particle at process " << rank << " sending it to " <<recv_id <<std::endl;
		}
	}

	// organize data to be sent
	grey_send_buff  = (particle_t*) malloc( total_send_counts *  sizeof(particle_t) );
	int pos = 0;
	for(int i = 0; i< n_proc; i++)
	{
		particle_t* c_p = grey_send_ph[i].first;

		for (int j = 0; j < grey_send_ph[i].size; j++) {

			pt_copy(&grey_send_buff[pos], c_p);
			c_p = c_p->next_grey;
			pos++;
		}

	}

	if(VERBOSE_LEVEL == PARTICLES_LEVEL) {
		MPI_Barrier(MPI_COMM_WORLD);
		std::cout << "Process " << rank << " grey particles assigned " << std::endl;
	}
	// prepare the offsets
	send_displs[0] = 0;
	for(int i = 1; i < n_proc + 1; i++) {
		send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
	}


	if(VERBOSE_LEVEL == PARTICLES_LEVEL){
		MPI_Barrier(MPI_COMM_WORLD);
		for(int i = 0; i < n_proc; i++){
			std::cout << "Process " << rank << " will send: " ;
			std::cout << send_counts[i] << " particles to process " << i  << std::endl;
		}
	}

	// let them know the quantity of data to receive
	MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1,MPI_INT,  MPI_COMM_WORLD);
	if(VERBOSE_LEVEL == PARTICLES_LEVEL){
		MPI_Barrier(MPI_COMM_WORLD);
		for(int i = 0; i < n_proc; i++){
			std::cout << "Process " << rank << " will receive: " ;
			std::cout << recv_counts[i] << " particles from process " << i  << std::endl;
		}
	}

	int total_recv_counts = 0;
	recv_displs[0] = 0;
	for(int i = 1; i< n_proc + 1;i++) {
		total_recv_counts += recv_counts[i - 1];
		recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
	}

	particle_t* recv_buff  = (particle_t*) malloc( total_recv_counts * sizeof(particle_t) );
	MPI_Alltoallv(grey_send_buff, send_counts, send_displs, PARTICLE, recv_buff, recv_counts, recv_displs, PARTICLE, MPI_COMM_WORLD);
	if(VERBOSE_LEVEL == PARTICLES_LEVEL){
		MPI_Barrier(MPI_COMM_WORLD);
		std::cout << "All processes have received data\n";
		std::cout << "Process " << rank <<" received " << total_recv_counts << std::endl;
	}

	*(buff_length) = total_recv_counts;

	free(grey_send_ph);
	free(grey_send_buff);
	free(send_counts);
	free(recv_counts);
	free(send_displs);
	free(recv_displs);

	return recv_buff;

}

particle_t* receive_particles(int* buff_length, int n_proc, particle_t* particles, int total_bins, int rank, int n,
	double bin_size, int bins_per_row, int bins_per_proc, MPI_Datatype PARTICLE){

	particle_ph* total_bins_particles_ph;
	particle_t *send_buff, *recv_buff ;
	int *send_displs, *send_counts;
	int recv_count;

	if(rank == 0) // global bins setup from process 0
	{
		total_bins_particles_ph = (particle_ph *) malloc(total_bins * sizeof(particle_ph));
		reset_particles_placeholders(total_bins_particles_ph, total_bins);
		// assign particles to bins
		assign_particles_to_ph(particles, total_bins_particles_ph, n, bins_per_row, bins_per_proc, bin_size);


		if (VERBOSE_LEVEL > PARTICLES_LEVEL) {
			show_placeholders(total_bins_particles_ph, total_bins, bins_per_row);
			MPI_Barrier(MPI_COMM_WORLD);
			for (int i = 0; i < total_bins; i++) {
				std::cout << "Bin " << i << " with " << total_bins_particles_ph[i].size << std::endl;
			}
		}
		send_displs = (int*) malloc( (n_proc+1) * sizeof(int) );
		send_displs[0] = 0;

		send_counts = (int*) malloc( n_proc * sizeof(int) );
		for(int i = 0; i< n_proc;i++)
			send_counts[i] = 0;

		send_buff  = (particle_t*) malloc( n * sizeof(particle_t) );
		int particles_copied = 0;
		for (int i = 0; i < total_bins; i++) {
			particle_t* c_p = total_bins_particles_ph[i].first;
			while(c_p)
			{
				send_counts[(int)c_p->proc_id]++;
				pt_copy(send_buff + particles_copied, c_p);
				particles_copied++;
				c_p = c_p->next;
			}
		}
		for(int i = 1; i< n_proc+1;i++) {
			send_displs[i] = send_displs[i - 1] + send_counts[i - 1];

		}
		if(VERBOSE_LEVEL == PARTICLES_LEVEL){
			for(int i = 0; i< n_proc;i++) {
				std::cout << "Size at " << i << " = " << send_counts[i] << std::endl;
				std::cout << "Offset at " << i << " = " << send_displs[i] << std::endl;
			}
		}
	}
	// let know to other process the number of particles to receive
	MPI_Scatter(send_counts, 1, MPI_INT, &recv_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

	recv_buff = (particle_t*) malloc( recv_count * sizeof(particle_t) );

	MPI_Scatterv(send_buff, send_counts, send_displs, PARTICLE, recv_buff, recv_count, PARTICLE, 0, MPI_COMM_WORLD);

	*(buff_length) = recv_count;
	if(VERBOSE_LEVEL == PARTICLES_LEVEL){
		MPI_Barrier(MPI_COMM_WORLD);
		std::cout << "Process " << rank << " with " << *(buff_length) << " particles " << std::endl;
		for(int i = 0; i < *(buff_length); i++){
			std::cout << "Received particle for process " << recv_buff[i].proc_id << " for bin " << recv_buff[i].global_bin_id << std::endl;
		}
	}
	if(rank == 0){
		free(total_bins_particles_ph);
		free(send_buff);
		free(send_displs);
		free(send_counts);
	}
	return recv_buff;
}

// populates buff_length
bin_t* organize_and_send_bins( int* buff_length, int n_proc, int bins_per_proc, int total_bins, int rank,
	int bins_per_row, MPI_Datatype BIN){
	int *bins_offsets, *bins_per_proc_sizes;
	bin_t* global_bins;
	int recv_count;


	if(rank == 0){
		bins_per_proc_sizes = (int*) malloc( n_proc * sizeof(int) );
		int remaining_bins = total_bins;
		for( int i = 0; i < n_proc; i++ ) {
			if(remaining_bins >= bins_per_proc) {
				bins_per_proc_sizes[i] = bins_per_proc;

			}
			else if(remaining_bins < bins_per_proc && remaining_bins > 0) {
				bins_per_proc_sizes[i] = remaining_bins;

			}
			else if (remaining_bins <= 0) {
				bins_per_proc_sizes[i] = 0;
			}
			//std::cout << "Process " << i << " assigned " << bins_per_proc_sizes[i] << " bins " << std::endl;
			remaining_bins -= bins_per_proc;
		}

		bins_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
		bins_offsets[0] = 0;
		for( int i = 1; i < n_proc + 1; i++ ) {
			bins_offsets[i] = bins_per_proc_sizes[i - 1] + bins_offsets[i - 1];
			//std::cout << "offset at "<< i << " -> " << bins_offsets[i] << std::endl;
		}

		global_bins = (bin_t*) malloc( total_bins * sizeof(bin_t));
		reset_bins(global_bins, total_bins);
		link_bins(bins_per_row, global_bins);
		//show_bins(global_bins, total_bins, bins_per_row);
	}
	// process 0 sends the size of data to be recevied by everyone else
	MPI_Scatter(bins_per_proc_sizes, 1, MPI_INT, &recv_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if(VERBOSE_LEVEL == BINS_LEVEL) {
		MPI_Barrier(MPI_COMM_WORLD);
		if(rank == 0) std::cout << "All processes receive que number of bins to handle\n";
		std::cout << "Process " << rank << " will handle " << recv_count << " bins " << std::endl;
		MPI_Barrier(MPI_COMM_WORLD);
	}
	// allocate data to be received
	bin_t* recv_buff = (bin_t*) malloc( recv_count * sizeof(bin_t));
	// scatters the bins from process 0
	MPI_Scatterv(global_bins, bins_per_proc_sizes, bins_offsets, BIN, recv_buff, recv_count, BIN, 0, MPI_COMM_WORLD);

	*(buff_length) = recv_count;

	if(VERBOSE_LEVEL == BINS_LEVEL) {
		MPI_Barrier(MPI_COMM_WORLD);
		if(rank == 0) std::cout << "All processes receive the actual bins to handle\n";
		std::cout << "Process " << rank << " up to " << *(buff_length) <<"\n";
		show_bins(recv_buff, recv_count, bins_per_row);
	}


	if( rank == 0) {
		free(bins_offsets);
		free(bins_per_proc_sizes);
		free(global_bins);
	}

	return recv_buff;
}

#endif

