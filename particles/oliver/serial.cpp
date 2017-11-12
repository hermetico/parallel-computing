#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"

#define density 0.0005
#define binsize  0.01

typedef struct bin_struct
{
  struct particle_t* bin;
} bin_struct;


//  benchmarking program
int main( int argc, char **argv )
{   
    int navg, nabsavg = 0;
    double davg, dmin, absmin = 1.0, absavg = 0.0;
    if(find_option(argc, argv, "-h") >= 0)
    {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set the number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        printf("-s <filename> to specify a summary file name\n");
        printf("-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int(argc, argv, "-n", 1000);

    char *savename = read_string(argc, argv, "-o", NULL);
    char *sumname = read_string(argc, argv, "-s", NULL);
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;

    particle_t* particles = (particle_t*) malloc(n * sizeof(particle_t));
    set_size(n);
	double size = sqrt(density * n);
    init_particles(n, particles);
    
    //  simulate a number of time steps
	int k = floor(size / binsize);
	int g = k * k;
	int j;
	particle_t* p;
	particle_t* q;
	bin_struct* bins = (bin_struct*) malloc(g * sizeof(bin_struct));
	int* bin_sizes = (int*) malloc(g * sizeof(int));
    double simulation_time = read_timer();
    for(int step = 0; step < NSTEPS; ++step)
	{
		navg = 0;
        davg = 0.0;
		dmin = 1.0;

		
		// bins initially empty
		for (int i = 0; i < g; ++i)
			bin_sizes[i] = 0;


		// assign particles to bins
		for (int i = 0; i < n; ++i)
		{
			p = particles + i;
			p -> ax = 0;
			p -> ay = 0;
			j = min((*p).x / binsize, k - 1) * k + min((*p).y / binsize, k - 1);
			p -> next = bins[j].bin;
			bins[j].bin = p;
			bin_sizes[j] += 1;
		}


		// compute forces
		for (int j = 0; j < g; ++j)
		{
			p = bins[j].bin;
			//apply forces to particles in bin j
			for(int x = 0; x < bin_sizes[j]; ++x) 
			{	
				if (j - k - 1 >= 0 && (j % k) != 0)
				{
					q = bins[j - k - 1].bin; // NW
					for(int y = 0; y < bin_sizes[j - k - 1]; ++y) 
					{
						apply_force(*p, *q, &dmin, &davg, &navg);
						q =  (*q).next;
					}
				}
				if (j - k >= 0) 
				{
					q = bins[j - k].bin; // W
					for(int y = 0; y < bin_sizes[j - k]; ++y)
					{
						apply_force(*p, *q, &dmin, &davg, &navg);
						q =  (*q).next;
					}
				}
				if (j - k + 1 >= 0 && (j % k) != k - 1) 
				{
					q = bins[j - k + 1].bin; // SW
					for(int y = 0; y < bin_sizes[j - k + 1]; ++y)
					{
						apply_force(*p, *q, &dmin, &davg, &navg);
						q =  (*q).next;
					}
				}
				if (j - 1 >= 0 && (j % k) != 0) 
				{
					q = bins[j - 1].bin; // N
					for(int y = 0; y < bin_sizes[j - 1]; ++y)
					{
						apply_force(*p, *q, &dmin, &davg, &navg);
						q =  (*q).next;
					}
				}
				q = bins[j].bin; // C
				for(int y = 0; y < bin_sizes[j]; ++y) 
				{
					apply_force(*p, *q, &dmin, &davg, &navg);
					q =  (*q).next;
				}
				if (j + 1 < g && (j % k) != k - 1) 
				{
					q = bins[j + 1].bin; // S
					for(int y = 0; y < bin_sizes[j + 1]; ++y)
					{
						apply_force(*p, *q, &dmin, &davg, &navg);
						q =  (*q).next;
					}
				}
				if (j + k - 1 < g && (j % k) != 0) 
				{
					q = bins[j + k - 1].bin; // NE
					for(int y = 0; y < bin_sizes[j + k - 1]; ++y)
					{
						apply_force(*p, *q, &dmin, &davg, &navg);
						q =  (*q).next;
					}
				}
				if (j + k < g) 
				{
					q = bins[j + k].bin; // E
					for(int y = 0; y < bin_sizes[j + k]; ++y)
					{
						apply_force(*p, *q, &dmin, &davg, &navg);
						q =  (*q).next;
					}
				}
				if (j + k + 1 < g && (j % k) != k - 1) 
				{
					q = bins[j + k + 1].bin; // SE
					for(int y = 0; y < bin_sizes[j + k + 1]; ++y)
					{
						apply_force(*p, *q, &dmin, &davg, &navg);
						q =  (*q).next;
					}
				}
				p = (*p).next;
			}
		}
		

		// move particles
		for (int i = 0; i < n; ++i)
		{
			move(particles[i]);
		}		

		
        if(find_option( argc, argv, "-no" ) == -1)
        {
          	// Computing statistical data
          	if (navg) 
			{
            	absavg +=  davg/navg;
            	nabsavg++;
          	}
          	if (dmin < absmin) 
				absmin = dmin;
		
          	//  save if necessary
          	if(fsave && (step%SAVEFREQ) == 0)
              	save(fsave, n, particles);
        }
    }


	free(bins);
	free(bin_sizes);
    simulation_time = read_timer( ) - simulation_time;
    printf( "n = %d, simulation time = %g seconds", n, simulation_time);
    if(find_option( argc, argv, "-no" ) == -1)
    {
		if (nabsavg) absavg /= nabsavg;
		// The minimum distance absmin between 2 particles during the run of the simulation
		// A correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
		// A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
		// The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
		printf(", absmin = %lf, absavg = %lf", absmin, absavg);
		if (absmin < 0.4) printf("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
		if (absavg < 0.8) printf("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");    
    // Printing summary data
    if(fsum) 
        fprintf(fsum,"%d %g\n",n,simulation_time);
    // Clearing space
    if(fsum)
        fclose(fsum);    
    free(particles);
    if(fsave)
    	fclose(fsave);
    return 0;
}

