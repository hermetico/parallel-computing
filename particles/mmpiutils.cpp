#ifndef __MY_MPI_UTILS__
#define __MY_MPI_UTILS__
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "common.h"

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
#endif

