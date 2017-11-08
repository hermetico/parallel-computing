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
#define VERBOSE_LEVEL 6


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
	to->next = from->next;
}

int get_proc_from_bin(int bin_id, int bins_per_proc){
	return (int)floor(bin_id / bins_per_proc);
}

int get_local_bin_from_global_bin(int bin_id, int bins_per_proc){
	return bin_id % bins_per_proc;
}

void reset_particles_placeholders(particle_ph* bins_particles, int local_nbins){
	for(int i = 0; i < local_nbins; i++) {
		bins_particles[i].first = NULL;
		bins_particles[i].size = 0;
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

void show_bins(particle_ph* ph, int num_bins, int bins_per_row){
	std::cout << std::endl;
	std::cout << std::endl;
	for(int y = 0; y < num_bins; y++ )
	{
		std::cout <<  ph[y].size << ",   " ;
		if((y + 1) % bins_per_row == 0) std::cout << std::endl;

	}
	std::cout << std::endl;
}

// returns recv_buff and buff_length
void send_and_receive_grey_area_particles(particle_t* recv_buff, int* buff_length, int n_proc, int nlocal,
particle_t* local_particles, int bins_per_row, int local_nbins, int rank, MPI_Datatype PARTICLE){

	particle_ph* grey_send_ph;
	int *send_counts, *recv_counts, *send_displs, *recv_displs;
	particle_t* grey_send_buff;

	grey_send_ph  = (particle_ph*) malloc( n_proc * sizeof(particle_ph) );
	reset_particles_placeholders(grey_send_ph, n_proc);
	send_counts = (int*) malloc(n_proc * sizeof(int));
	for(int i = 0; i< n_proc; i++){
		send_counts[i] = 0;
	}
	recv_counts = (int*) malloc(n_proc * sizeof(int));
	send_displs = (int*) malloc(1 + n_proc * sizeof(int));
	recv_displs = (int*) malloc(1 + n_proc * sizeof(int));

	int total_send_counts = 0;
	// collect info about data to be sent
	for(int i = 0; i < nlocal; i++){
		int bin_id = local_particles[i].global_bin_id;
		// needs to go up?
		if(bin_id + bins_per_row >= local_nbins * (rank + 1) && rank < n_proc - 1)
		{
			int recv_id = rank + 1;
			//pt_copy(&grey_send_buff[particle_per_proc * recv_id + send_counts[recv_id]], &local_particles[i]);
			local_particles[i].next = grey_send_ph[recv_id].first;
			grey_send_ph[recv_id].first = &local_particles[i];
			grey_send_ph[recv_id].size++;
			send_counts[recv_id]++;
			total_send_counts++;
			std::cout << "Particle at process " << rank << " sending it to " <<recv_id;
		}
		// needs to go down?
		if(bin_id - bins_per_row < local_nbins * (rank + 1) && rank != 0)
		{
			int recv_id = rank - 1;
			local_particles[i].next = grey_send_ph[recv_id].first;
			grey_send_ph[recv_id].first = &local_particles[i];
			grey_send_ph[recv_id].size++;
			send_counts[recv_id]++;
			total_send_counts++;
			std::cout << "Particle at process " << rank << " sending it to " <<recv_id <<std::endl;
		}
	}

	// organize data to be sent
	grey_send_buff  = (particle_t*) malloc( total_send_counts *  sizeof(particle_t) );
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
	if(VERBOSE_LEVEL == PARTICLES_LEVEL) {
		MPI_Barrier(MPI_COMM_WORLD);
		std::cout << "Process " << rank << " grey particles assigned " << std::endl;
	}
	// prepare the offsets
	send_displs[0] = 0;
	for(int i = 1; i< n_proc+1;i++) {
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
	for(int i = 1; i< n_proc+1;i++) {
		total_recv_counts +=recv_counts[i - 1];
		recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
	}
	recv_buff  = (particle_t*) malloc( total_recv_counts * sizeof(particle_t) );
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

}


#endif

