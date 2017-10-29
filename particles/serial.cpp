#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <iostream>
#include "common.h"

#include <vector>

using namespace std;

// the cutoff
#define cutoff  0.01

// the size of the grid
extern double size;
typedef struct bin_t bin_t;
struct bin_t
{
	bin_t* top = NULL;
	bin_t* bottom = NULL;
	bin_t* left = NULL;
	bin_t* right = NULL;
	bin_t* top_left = NULL;
	bin_t* top_right = NULL;
	bin_t* bottom_left = NULL;
	bin_t* bottom_right = NULL;
	vector<particle_t*> particles;
};

void show_bins(vector<bin_t> bins, int bins_per_row){
	cout << endl;
	cout << endl;
	for(int y = 0; y < bins_per_row; y++ )
	{
		for(int x = 0; x < bins_per_row; x++)
		{
			int num_particles = bins[y * bins_per_row + x].particles.size();
			cout << num_particles << ",   " ;
		}
		cout << endl;
	}
}


void apply_forces(particle_t* a_particle, vector<particle_t*> particles, double* dmin, double* davg,  int* navg)
{
	for(int i=0; i< particles.size(); i++)
	{

		apply_force(*(a_particle), *(particles[i]), dmin, davg, navg);
	}
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


	vector<bin_t> bins;
	int total_bins = ceil((size * size) / (cutoff * cutoff));
	int bins_per_row = ceil(sqrt(total_bins));
	for(int y = 0; y < bins_per_row; y++ )
	{
		for(int x = 0; x < bins_per_row; x++)
		{
			bin_t new_bin;
			//new_bin->particles = vector<particle_t*>;
			bins.push_back(new_bin);
		}
	}

	// link bins
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

	//
	//  simulate a number of time steps
	//
	double simulation_time = read_timer( );
	
	for( int step = 0; step < NSTEPS; step++ )
	{
		navg = 0;
		davg = 0.0;
		dmin = 1.0;

		// reset bins
		for(int i = 0; i< bins.size(); i++)
		{
			bins[i].particles.clear();
		}

		// bind particles to bins
		for(int i = 0; i < n; i++){
			int binx = ceil(particles[i].x / cutoff) - 1;
			int biny = ceil(particles[i].y / cutoff) - 1;
			bins[bins_per_row * biny + binx].particles.push_back(&particles[i]);
		}

		//show_bins(bins, bins_per_row);


		// compute forces
		for(int y = 0; y < bins_per_row; y++ )
		{
			for(int x = 0; x < bins_per_row; x++)
			{
				bin_t* c_bin = &bins[y * bins_per_row + x];
				for(int i = 0; i < c_bin->particles.size(); i++)
				{

					particle_t* c_particle = c_bin->particles[i];
					c_particle->ax = 0;
					c_particle->ay = 0;
					// same bin
					apply_forces(c_particle, c_bin->particles, &dmin, &davg, &navg);

					if(c_bin->top)
						apply_forces(c_particle, c_bin->top->particles, &dmin, &davg, &navg);
					if(c_bin->bottom)
						apply_forces(c_particle, c_bin->bottom->particles, &dmin, &davg, &navg);
					if(c_bin->left)
						apply_forces(c_particle, c_bin->left->particles, &dmin, &davg, &navg);
					if(c_bin->right)
						apply_forces(c_particle, c_bin->right->particles, &dmin, &davg, &navg);
					if(c_bin->top_left)
						apply_forces(c_particle, c_bin->top_left->particles, &dmin, &davg, &navg);
					if(c_bin->top_right)
						apply_forces(c_particle, c_bin->top_right->particles, &dmin, &davg, &navg);
					if(c_bin->bottom_left)
						apply_forces(c_particle, c_bin->bottom_left->particles, &dmin, &davg, &navg);
					if(c_bin->bottom_right)
						apply_forces(c_particle, c_bin->bottom_right->particles, &dmin, &davg, &navg);

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
	}
	simulation_time = read_timer( ) - simulation_time;
	
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
		fprintf(fsum,"%d %g\n",n,simulation_time);
 
	//
	// Clearing space
	//
	if( fsum )
		fclose( fsum );	
	free( particles );
	if( fsave )
		fclose( fsave );
	
	return 0;
}
