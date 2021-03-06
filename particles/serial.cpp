#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "common.h"




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

	// particles pointers
	particle_t* first;
};


void apply_forces_linked_particles(particle_t* a_particle, particle_t* b_particle, double* dmin, double* davg,  int* navg)
{
	while(b_particle)
	{
		apply_force(*(a_particle), *(b_particle), dmin, davg, navg);
		b_particle = b_particle->next;
	}
}


int get_bin_id(int bins_per_row, double bin_size,  double x, double y){
	int binx = floor(x / bin_size);
	int biny = floor(y / bin_size);

	return  bins_per_row * biny + binx;
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{	
	int navg,nabsavg=0;
	double davg,dmin, absmin=1.0, absavg=0.0;

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
	
	FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
	FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;

	particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
	set_size( n );
	init_particles( n, particles );



    double bin_size = cutoff * 1.7;
	int bins_per_row = ceil(size / bin_size);
	int total_bins = bins_per_row * bins_per_row;

	bin_t* bins = (bin_t*) malloc( total_bins * sizeof(bin_t));

    for(int y = 0; y < total_bins; y++ )
	{
		bin_t new_bin;
		new_bin.top = -1;
		new_bin.bottom = -1;
		new_bin.left = -1;
		new_bin.right = -1;
		new_bin.top_left = -1;
		new_bin.top_right = -1;
		new_bin.bottom_left = -1;
		new_bin.bottom_right = -1;
		new_bin.first = NULL;
		bins[y] = new_bin;
	}

	// link bins
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

	//
	//  simulate a number of time steps
	//
	double simulation_time = read_timer( );

    // asign particles initially


	for( int step = 0; step < NSTEPS; step++ )
	{
		navg = 0;
		davg = 0.0;
		dmin = 1.0;




		for(int i = 0; i < n; i++){
			int particle_owner = get_bin_id(bins_per_row, bin_size, particles[i].x, particles[i].y);
			particles[i].next = bins[particle_owner].first;
			bins[particle_owner].first = &particles[i];
		}
		
		// compute forces
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

					if(c_bin->top != -1)
						apply_forces_linked_particles(c_particle, bins[c_bin->top].first, &dmin, &davg, &navg);
					if(c_bin->bottom != -1)
						apply_forces_linked_particles(c_particle, bins[c_bin->bottom].first, &dmin, &davg, &navg);
					if(c_bin->left != -1)
						apply_forces_linked_particles(c_particle, bins[c_bin->left].first, &dmin, &davg, &navg);
					if(c_bin->right != -1)
						apply_forces_linked_particles(c_particle, bins[c_bin->right].first, &dmin, &davg, &navg);
					if(c_bin->top_left != -1)
						apply_forces_linked_particles(c_particle, bins[c_bin->top_left].first, &dmin, &davg, &navg);
					if(c_bin->top_right != -1)
						apply_forces_linked_particles(c_particle, bins[c_bin->top_right].first, &dmin, &davg, &navg);
					if(c_bin->bottom_left != -1)
						apply_forces_linked_particles(c_particle, bins[c_bin->bottom_left].first, &dmin, &davg, &navg);
					if(c_bin->bottom_right != -1)
						apply_forces_linked_particles(c_particle, bins[c_bin->bottom_right].first, &dmin, &davg, &navg);

					c_particle = c_particle->next;
				}

			}
		}

		//
		//  move particles
		//
		for( int i = 0; i < n; i++ ) 
			move( particles[i] );

		if( find_option( argc, argv, "-no" ) == -1 )
		{
		  //
		  // Computing statistical data
		  //
		  if (navg) {
			absavg +=  davg/navg;
			nabsavg++;
		  }
		  if (dmin < absmin) absmin = dmin;
		
		  //
		  //  save if necessary
		  //
		  if( fsave && (step%SAVEFREQ) == 0 )
			  save( fsave, n, particles );
		}

		// reset bins
		for(int y = 0; y < total_bins; y++ )
			bins[y].first = NULL;
	}
	simulation_time = read_timer( ) - simulation_time;

	//printf( "n = %d, simulation time = %g seconds", n, simulation_time);
	printf( "%d\t%g", n, simulation_time);


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
		fprintf(fsum,"%d %g\n",n,simulation_time);
 
	//
	// Clearing space
	//
	if( fsum )
		fclose( fsum );	
	free( particles );
    free(bins);
	if( fsave )
		fclose( fsave );
	
	return 0;
}
