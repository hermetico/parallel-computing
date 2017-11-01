#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include <iostream>
#include "omp.h"

using namespace std;

// the cutoff
#define cutoff  0.01

// the size of the grid
extern double size;
typedef struct bin_t bin_t;
struct bin_t
{
	// bin pointers
	bin_t* top = NULL;
	bin_t* bottom = NULL;
	bin_t* left = NULL;
	bin_t* right = NULL;
	bin_t* top_left = NULL;
	bin_t* top_right = NULL;
	bin_t* bottom_left = NULL;
	bin_t* bottom_right = NULL;

	// particles pointers
	particle_t* first = NULL;
	particle_t* last = NULL;
	particle_t* new_ones_first = NULL;
	particle_t* new_ones_last = NULL;
};

void show_bins(bin_t* bins, int bins_per_row){
	cout << endl;
	cout << endl;
	for(int y = 0; y < bins_per_row; y++ )
	{
		for(int x = 0; x < bins_per_row; x++)
		{
			//TODO change this
			int num_particles = 0; //bins[y * bins_per_row + x].size;
			cout << num_particles << ",   " ;
		}
		cout << endl;
	}
}

void apply_forces_linked_particles(particle_t* a_particle, particle_t* b_particle, double* dmin, double* davg,  int* navg)
{
	while(b_particle)
	{

		apply_force(*(a_particle), *(b_particle), dmin, davg, navg);
		b_particle = b_particle->next;
	}
}

void notify_bin(bin_t* bin, particle_t* particle)
{
	//adds the particle to the new ones

	particle->next = NULL;
	if(!bin->new_ones_first){
		bin->new_ones_first = particle;
		bin->new_ones_last = particle;
	}else{
		particle_t* tmp = bin->new_ones_last;
		bin->new_ones_last = particle;
		tmp->next = bin->new_ones_last;
	}
}

int get_bin_id(int bins_per_row, double bin_size,  double x, double y){
	int binx = (int) ceil(x / bin_size) - 1;
	int biny = (int) ceil(y / bin_size) - 1;

	return  bins_per_row * biny + binx;
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{   
	int navg,nabsavg=0,numthreads; 
	double dmin, absmin=1.0,davg,absavg=0.0;
	
	if( find_option( argc, argv, "-h" ) >= 0 )
	{
		printf( "Options:\n" );
		printf( "-h to see this help\n" );
		printf( "-n <int> to set number of particles\n" );
		printf( "-o <filename> to specify the output file name\n" );
		printf( "-s <filename> to specify a summary file name\n" ); 
		printf( "-no turns off all correctness checks and particle output\n");   
		return 0;
	}

	int n = read_int( argc, argv, "-n", 1000 );
	char *savename = read_string( argc, argv, "-o", NULL );
	char *sumname = read_string( argc, argv, "-s", NULL );

	FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
	FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;	  

	particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
	set_size( n );
	init_particles( n, particles );

	float bin_size = cutoff * 5;
	int total_bins = ceil((size * size) / (bin_size * bin_size));
	int bins_per_row = ceil(sqrt(total_bins));

	total_bins = bins_per_row * bins_per_row;

	bin_t* bins = (bin_t*) malloc( total_bins * sizeof(bin_t));
	omp_lock_t* bin_locks = (omp_lock_t*) malloc(total_bins * sizeof(omp_lock_t));

    for (int i = 0; i < total_bins; ++i)
		omp_init_lock(bin_locks + i);

	//
	//  simulate a number of time steps
	//
	double simulation_time = read_timer( );

	#pragma omp parallel private(dmin) 
	{

	numthreads = omp_get_num_threads();

	#pragma omp for
	for(int y = 0; y < bins_per_row; y++ )
	{
		for(int x = 0; x < bins_per_row; x++)
		{
			bin_t new_bin;
			bins[y * bins_per_row + x] = new_bin;
		}
	}

	// link bins
	#pragma omp for collapse(2)
	for(int y = 0; y < bins_per_row; y++ )
	{
		for(int x = 0; x < bins_per_row; x++)
		{
			bin_t* c_bin = &bins[y * bins_per_row + x];
			if( y > 0)
			{
				c_bin->bottom = &bins[(y-1) * bins_per_row + x];

				if (x > 0)
					c_bin->bottom_left = &bins[(y - 1) * bins_per_row + (x - 1)];

				if (x < bins_per_row - 1)
					c_bin->bottom_right = &bins[(y - 1) * bins_per_row + (x + 1)];
			}

			if( y < bins_per_row - 1)
			{
				c_bin->top = &bins[(y+1) * bins_per_row + x];

				if (x > 0)
					c_bin->top_left = &bins[(y + 1) * bins_per_row + (x - 1)];

				if (x < bins_per_row - 1)
					c_bin->top_right = &bins[(y + 1) * bins_per_row + (x + 1)];
			}

			if (x > 0)
				c_bin->left = &bins[y * bins_per_row + (x - 1)];

			if (x < bins_per_row - 1)
				c_bin->right = &bins[y * bins_per_row + (x + 1)];
		}
	}
	// fill bins with particles
	#pragma omp for
	for(int i = 0; i < n; i++){
		//TODO check edge case particle position 0.0


		int particle_owner = get_bin_id(bins_per_row, bin_size, particles[i].x, particles[i].y);

		particles[i].next = NULL;
		bin_t* c_bin = &bins[particle_owner];

		omp_set_lock(&bin_locks[particle_owner]);
		if(!c_bin->first){
			c_bin->first = &particles[i];
			c_bin->last = &particles[i];
		}else{
			particle_t* tmp = c_bin->last;
			c_bin->last = &particles[i];
			tmp->next = c_bin->last;
		}
		omp_unset_lock(&bin_locks[particle_owner]);

	}

	for( int step = 0; step < NSTEPS; step++ )
	{
		#pragma omp master
		{
			navg = 0;
			davg = 0.0;
		}

		dmin = 1.0;


		//
		//  compute all forces
		//
		#pragma omp for reduction (+:navg) reduction(+:davg)
		for(int y = 0; y < bins_per_row; y++ )
		{
			for(int x = 0; x < bins_per_row; x++)
			{
				bin_t* c_bin = &bins[y * bins_per_row + x];

				particle_t* c_particle = c_bin->first;
				while(c_particle)
				{

					c_particle->ax = 0;
					c_particle->ay = 0;

					// same bin
					apply_forces_linked_particles(c_particle, c_bin->first, &dmin, &davg, &navg);

					if(c_bin->top)
						apply_forces_linked_particles(c_particle, c_bin->top->first, &dmin, &davg, &navg);
					if(c_bin->bottom)
						apply_forces_linked_particles(c_particle, c_bin->bottom->first, &dmin, &davg, &navg);
					if(c_bin->left)
						apply_forces_linked_particles(c_particle, c_bin->left->first, &dmin, &davg, &navg);
					if(c_bin->right)
						apply_forces_linked_particles(c_particle, c_bin->right->first, &dmin, &davg, &navg);
					if(c_bin->top_left)
						apply_forces_linked_particles(c_particle, c_bin->top_left->first, &dmin, &davg, &navg);
					if(c_bin->top_right)
						apply_forces_linked_particles(c_particle, c_bin->top_right->first, &dmin, &davg, &navg);
					if(c_bin->bottom_left)
						apply_forces_linked_particles(c_particle, c_bin->bottom_left->first, &dmin, &davg, &navg);
					if(c_bin->bottom_right)
						apply_forces_linked_particles(c_particle, c_bin->bottom_right->first, &dmin, &davg, &navg);

					c_particle = c_particle->next;
				}

			}
		}
		
		
		//
		//  move particles
		//
		#pragma omp for nowait
		for( int i = 0; i < n; i++ ) 
			move( particles[i] );
  
		if( find_option( argc, argv, "-no" ) == -1 ) 
		{
		  //
		  //  compute statistical data
		  //
		  #pragma omp master
		  if (navg) { 
			absavg += davg/navg;
			nabsavg++;
		  }

		  #pragma omp critical
		  if (dmin < absmin) absmin = dmin;
		
		  //
		  //  save if necessary
		  //
		  #pragma omp master
		  if( fsave && (step%SAVEFREQ) == 0 )
			  save( fsave, n, particles );
		}

		// check for particles that are now outside the bin
		#pragma omp for
		for( int i  = 0; i < total_bins; i++)
		{
			int owner;

			//FIRST particle
			// check for the first particle
			particle_t* first = bins[i].first;
			while(first == bins[i].first && first){
				owner = get_bin_id(bins_per_row, bin_size, first->x, first->y);
				if(owner != i){
					//unlink first particle
					bins[i].first = first->next;
					omp_set_lock(&bin_locks[owner]);
					notify_bin(&bins[owner], first);
					omp_unset_lock(&bin_locks[owner]);
				}
				// check first again
				first = first->next;
			}

			// MIDDLE particles and last
			// now we can look ahead one particle
			particle_t* prev = bins[i].first;
			particle_t* current;
			while(prev){
				current = prev->next;
				if(!current) break;
				owner = get_bin_id(bins_per_row, bin_size, current->x, current->y);

				//checks if a particle corresponds to a new bin
				if(i != owner)
				{
					// unlink the current particle
					prev->next = current->next;
					omp_set_lock(&bin_locks[owner]);
					notify_bin(&bins[owner], current);
					omp_unset_lock(&bin_locks[owner]);
				}

				if(!prev->next)// is it the last one now?
				{
					bins[i].last = prev;
					prev = NULL;
				}
				else
				{
					// update the previous one
					prev = prev->next;
				}
			}

			// if current == NULL, prev is the last one
			bins[i].last = prev;

		}
		#pragma omp for nowait
		// marge particles list
		for( int i  = 0; i < total_bins; i++)
		{
			if(bins[i].new_ones_first && bins[i].last){
				particle_t* tmp = bins[i].last;
				tmp->next = bins[i].new_ones_first;
				bins[i].last = bins[i].new_ones_last;
				bins[i].new_ones_first = NULL;
				bins[i].new_ones_last = NULL;
			}else if(bins[i].new_ones_first)
			{
				bins[i].first = bins[i].new_ones_first;
				bins[i].last = bins[i].new_ones_last;
				bins[i].new_ones_first = NULL;
				bins[i].new_ones_last = NULL;
			}

		}
	}
}
	simulation_time = read_timer( ) - simulation_time;
	
	printf( "n = %d,threads = %d, simulation time = %g seconds", n,numthreads, simulation_time);

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
		fprintf(fsum,"%d %d %g\n",n,numthreads,simulation_time);

	//
	// Clearing space
	//
	if( fsum )
		fclose( fsum );

	free( particles );
	free( bins );
	if( fsave )
		fclose( fsave );
	
	return 0;
}
