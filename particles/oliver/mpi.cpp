#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"

#define binsize 0.01
#define density 0.0005

typedef struct node 
{
  particle_t * p;
  struct node * next;
} node_t;

// linked list functionality
//{
void print_list(node_t * head) 
{
  node_t * current = head;
  while (current != NULL) 
  {
    printf("(%f,%f)\n", current->p->x, current->p->y);
    current = current->next;
  }
}

void push(node_t ** head, particle_t * p) 
{
  node_t * new_node;
  new_node = (node_t*) malloc(sizeof(node_t));
  new_node->p = p;
  new_node->next = *head;
  *head = new_node;
}

particle_t* pop(node_t ** head) 
{
  particle_t* retval;
  node_t * next_node = NULL;
  next_node = (*head)->next;
  retval = (*head)->p;
  free(*head);
  *head = next_node;
  return retval;
}

void delete_list(node_t ** head) 
{
  node_t * next_node = NULL;
  while (*head != NULL)
  {
    next_node = (*head)->next;
    free(*head);
    *head = next_node;
  }
}
//}

// functions mapping particles to their respecitve processor, bin or local bin
//{
int proc_num(particle_t p, int k, int w, int h, int m)
{
  return (floor(min(floor(p.x / binsize), k - 1) / w)) * m + floor(min(floor(p.y / binsize), k - 1) / h);
}

int bin_num(particle_t p, int k)
{
  return (min(floor(p.x / binsize), k - 1)) * k + min(floor(p.y / binsize), k - 1);
}

int bin_num_local(particle_t p, int k, int w, int h)
{
  return (min(floor(p.x / binsize), k - 1) % w) * h + (min(floor(p.y / binsize), k - 1) % h);
}
//}


int main(int argc, char **argv)
{
  // variables for checking correctness
  //{
  int navg, nabsavg = 0;
  double dmin, absmin = 1.0, davg, absavg = 0.0;
  double rdavg, rdmin;
  int rnavg;
  //}
  
  //  process command line parameters
  //{
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
  int n_particles = read_int( argc, argv, "-n", 1000 );
  char *savename = read_string( argc, argv, "-o", NULL );
  char *sumname = read_string( argc, argv, "-s", NULL );
  //}
  
  // set up MPI
  //{
  int n_proc, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //}
  
  //  allocate generic resources
  //{
  FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
  FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;
  //}
  
  // define particle datatype
  //{
  MPI_Datatype PARTICLE;
  MPI_Type_contiguous(6, MPI_DOUBLE, &PARTICLE);
  MPI_Type_commit(&PARTICLE);
  //}
  
  // define required variables and initialise
  //{
  particle_t* all_particles = (particle_t*) malloc(n_particles * sizeof(particle_t)); // TO REMOVE
  // instance size parameters
  double size = sqrt(density * n_particles);
  int k = floor(size / binsize);
  int m, n;
  m = n = floor(sqrt(n_proc));
  while ((m + 1) * n <= n_proc)
    m++;
  int w = 1 + ((k - 1) / n);
  int h = 1 + ((k - 1) / m);
  // define a communicator to exclude unused processors
  /*
  MPI_Comm working_processes;
  if (rank < m * n)
  {
    MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &working_processes);
  } else {
    MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, rank, &working_processes);
    MPI_Finalize();
    return 0;
  }
  */
  // local particles
  particle_t* particles_local;
  int part_count_local;
  node_t** bins_local = (node_t**) malloc(w * h * sizeof(node_t*));
  for (int i = 0; i < w * h; ++i)
    bins_local[i] = NULL;
  node_t* current_part;
  node_t* current_nb;
  int b;
  // variables for communication
  particle_t* sendbuf;
  int* dest_proc;
  int* sendcounts = (int*) malloc(n_proc * sizeof(int));
  int* sdispls = (int*) malloc(n_proc * sizeof(int));
  int* recvcounts = (int*) malloc(n_proc * sizeof(int));
  int* rdispls = (int*) malloc(n_proc * sizeof(int));
  int* pos_send = (int*) malloc(n_proc * sizeof(int));
  // ghostzone variables
  int west_ghostzone_size, east_ghostzone_size, north_ghostzone_size, south_ghostzone_size;
  int nw_ghostzone_size, sw_ghostzone_size, ne_ghostzone_size, se_ghostzone_size;
  int ghostzone_size;
  particle_t* west_ghostzone_buf;
  particle_t* east_ghostzone_buf;
  particle_t* north_ghostzone_buf;
  particle_t* south_ghostzone_buf;
  particle_t* nw_ghostzone_buf;
  particle_t* sw_ghostzone_buf;
  particle_t* ne_ghostzone_buf;
  particle_t* se_ghostzone_buf;
  particle_t* ghostzone;
  MPI_Request w_req;
  MPI_Request e_req;
  MPI_Request n_req;
  MPI_Request s_req;
  MPI_Request nw_req;
  MPI_Request sw_req;
  MPI_Request ne_req;
  MPI_Request se_req;
  MPI_Status status;
  //}
  
  // initialise instance and prepare distribution
  //{
  set_size(n_particles);
  if (rank == 0)
  {
    init_particles(n_particles, all_particles);
    
    sendbuf = (particle_t*) malloc(n_particles * sizeof(particle_t));
    for (int i = 0; i < n_proc; ++i)
    {
      sendcounts[i] = 0;
      pos_send[i] = 0;  
    }
    dest_proc = (int*) malloc(n_particles * sizeof(int));
    for (int i = 0; i < n_particles; ++i)
    {
      int dest = proc_num(all_particles[i], k, w, h, m);
      sendcounts[dest] ++;
      dest_proc[i] = dest;
    }
    int stmp = 0;
    for (int i = 0; i < n_proc; ++i)
    {
      sdispls[i] = stmp;
      stmp += sendcounts[i];
    }
    sendbuf = (particle_t*) malloc(n_particles * sizeof(particle_t));
    for (int i = 0; i < n_particles; ++i)
    {
      sendbuf[sdispls[dest_proc[i]] + pos_send[dest_proc[i]]] = all_particles[i];
      pos_send[dest_proc[i]]++;
    }
    free(dest_proc);
  }
  //}
  
  // distribution of the particles to the respective processors
  //{
  MPI_Scatter(sendcounts, 1, MPI_INT, &part_count_local, 1, MPI_INT, 0, MPI_COMM_WORLD);
  particles_local = (particle_t*) malloc(part_count_local * sizeof(particle_t));
  MPI_Scatterv(sendbuf, sendcounts, sdispls, PARTICLE, particles_local, part_count_local, PARTICLE, 0, MPI_COMM_WORLD);
  if(rank == 0)
  {
    free(sendbuf);
  }
  //}
  
  // start simulation
  int STEPCOUNT = NSTEPS;
  double simulation_time = read_timer();
  for (int step = 0; step < STEPCOUNT; ++step)
  { 
    // reset values
    //{
    navg = 0;
    dmin = 1.0;
    davg = 0.0;
    west_ghostzone_size = east_ghostzone_size = north_ghostzone_size = south_ghostzone_size = 0;
    nw_ghostzone_size = sw_ghostzone_size = ne_ghostzone_size = se_ghostzone_size = 0;
    //}
    
    // put particles into bins
    //{
    for (int i = 0; i < part_count_local; ++i)
    {
      b = bin_num_local(particles_local[i], k, w, h);
      push(bins_local + b, particles_local + i);
      if (b < h)
        west_ghostzone_size++;
      if (b >= w * h - h)
        east_ghostzone_size++;
      if (b % h == 0)
        south_ghostzone_size++;
      if (b % h == h - 1)
        north_ghostzone_size++;
      if (b == 0)
        sw_ghostzone_size++;
      if (b == h - 1)
        nw_ghostzone_size++;
      if (b == w * h - h)
        se_ghostzone_size++;
      if (b == w * h - 1)
        ne_ghostzone_size++;
    }
    //}
    
    
    // gather all particles and save
    //{
    MPI_Allgather(&part_count_local, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
    int tmp = 0;
    for (int i = 0; i < n_proc; ++i)
    {
      rdispls[i] = tmp;
      tmp += recvcounts[i];
    }
    MPI_Allgatherv(particles_local, part_count_local, PARTICLE, all_particles, recvcounts, rdispls, PARTICLE, MPI_COMM_WORLD);
    // save current step if necessary (slightly different semantics than in other codes)
    if(find_option(argc, argv, "-no") == -1)
      if(fsave && (step%SAVEFREQ) == 0){
        save(fsave, n_particles, all_particles);
	}
    //}
    
    
    // transmit particles in ghostzone
    //{
    if (rank < m * n) 
    {
    if (rank >= m)
    {
      // west ghostzone needs to be transmit
      west_ghostzone_buf = (particle_t*) malloc(west_ghostzone_size * sizeof(particle_t));
      b = 0;
      for (int y = 0; y < h; ++y)
      {
        current_part = bins_local[y];
        while (current_part != NULL)
        {
          west_ghostzone_buf[b] = *(current_part->p);
          b++;
          current_part = current_part->next;
        }
      }
      MPI_Isend(west_ghostzone_buf, west_ghostzone_size, PARTICLE, rank - m, 0, MPI_COMM_WORLD, &w_req);
    }
    if (rank < m * n - m)
    {
      // east ghostzone needs to be transmit
      east_ghostzone_buf = (particle_t*) malloc(east_ghostzone_size * sizeof(particle_t));
      b = 0;
      for (int y = 0; y < h; ++y)
      {
        current_part = bins_local[w * h - h + y];
        while (current_part != NULL)
        {
          east_ghostzone_buf[b] = *(current_part->p);
          b++;
          current_part = current_part->next;
        }
      }
      MPI_Isend(east_ghostzone_buf, east_ghostzone_size, PARTICLE, rank + m, 0, MPI_COMM_WORLD, &e_req);
    }
    if (rank % m != 0)
    {
      // south ghostzone needs to be transmit
      south_ghostzone_buf = (particle_t*) malloc(south_ghostzone_size * sizeof(particle_t));
      b = 0;
      for (int x = 0; x < w; ++x)
      {
        current_part = bins_local[x * h];
        while (current_part != NULL)
        {
          south_ghostzone_buf[b] = *(current_part->p);
          b++;
          current_part = current_part->next;
        }
      }
      MPI_Isend(south_ghostzone_buf, south_ghostzone_size, PARTICLE, rank - 1, 0, MPI_COMM_WORLD, &s_req);
    }
    if (rank % m != m - 1)
    {
      // north ghostzone needs to be transmit
      north_ghostzone_buf = (particle_t*) malloc(north_ghostzone_size * sizeof(particle_t));
      b = 0;
      for (int x = 0; x < w; ++x)
      {
        current_part = bins_local[x * h + h - 1];
        while (current_part != NULL)
        {
          north_ghostzone_buf[b] = *(current_part->p);
          b++;
          current_part = current_part->next;
        }
      }
      MPI_Isend(north_ghostzone_buf, north_ghostzone_size, PARTICLE, rank + 1, 0, MPI_COMM_WORLD, &n_req);
    }
    if (rank >= m && rank % m != 0)
    {
      // southwest ghostzone needs to be transmit
      sw_ghostzone_buf = (particle_t*) malloc(sw_ghostzone_size * sizeof(particle_t));
      b = 0;
      current_part = bins_local[0];
      while (current_part != NULL)
      {
        sw_ghostzone_buf[b] = *(current_part->p);
        b++;
        current_part = current_part->next;
      }
      MPI_Isend(sw_ghostzone_buf, sw_ghostzone_size, PARTICLE, rank - m - 1, 0, MPI_COMM_WORLD, &sw_req);
    }
    if (rank >= m && rank % m != m - 1)
    {
      // northwest ghostzone needs to be transmit
      nw_ghostzone_buf = (particle_t*) malloc(nw_ghostzone_size * sizeof(particle_t));
      b = 0;
      current_part = bins_local[h - 1];
      while (current_part != NULL)
      {
        nw_ghostzone_buf[b] = *(current_part->p);
        b++;
        current_part = current_part->next;
      }
      MPI_Isend(nw_ghostzone_buf, nw_ghostzone_size, PARTICLE, rank - m + 1, 0, MPI_COMM_WORLD, &nw_req);
    }
    if (rank < m * n - m && rank % m != 0)
    {
      // southeast ghostzone needs to be transmit
      se_ghostzone_buf = (particle_t*) malloc(se_ghostzone_size * sizeof(particle_t));
      b = 0;
      current_part = bins_local[w * h - h];
      while (current_part != NULL)
      {
        se_ghostzone_buf[b] = *(current_part->p);
        b++;
        current_part = current_part->next;
      }
      MPI_Isend(se_ghostzone_buf, se_ghostzone_size, PARTICLE, rank + m - 1, 0, MPI_COMM_WORLD, &se_req);
    }
    if (rank < m * n - m && rank % m != m - 1)
    {
      // northeast ghostzone needs to be transmit
      ne_ghostzone_buf = (particle_t*) malloc(ne_ghostzone_size * sizeof(particle_t));
      b = 0;
      current_part = bins_local[w * h - 1];
      while (current_part != NULL)
      {
        ne_ghostzone_buf[b] = *(current_part->p);
        b++;
        current_part = current_part->next;
      }
      MPI_Isend(ne_ghostzone_buf, ne_ghostzone_size, PARTICLE, rank + m + 1, 0, MPI_COMM_WORLD, &ne_req);
    }
    }
    //}
    
    // apply forces I - interaction with particles not in the ghostzone
    //{
    if (rank < m * n)
    {
    // inner bins
    for (int x = 1; x < w - 1; ++x)
      for (int y = 1; y < h - 1; ++y)
      {
        b = x * h + y;
        current_part = bins_local[b];
        while (current_part != NULL)
        {
          current_part->p->ax = 0;
          current_part->p->ay = 0;
          // C
          current_nb = bins_local[b];
          while (current_nb != NULL)
          {
            apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
            current_nb = current_nb->next;
          }
          // S
          current_nb = bins_local[b - 1];
          while (current_nb != NULL)
          {
            apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
            current_nb = current_nb->next;
          }
          // N
          current_nb = bins_local[b + 1];
          while (current_nb != NULL)
          {
            apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
            current_nb = current_nb->next;
          }
          // W
          current_nb = bins_local[b - h];
          while (current_nb != NULL)
          {
            apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
            current_nb = current_nb->next;
          }
          // E
          current_nb = bins_local[b + h];
          while (current_nb != NULL)
          {
            apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
            current_nb = current_nb->next;
          }
          // SW
          current_nb = bins_local[b - h - 1];
          while (current_nb != NULL)
          {
            apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
            current_nb = current_nb->next;
          }
          // NW
          current_nb = bins_local[b - h + 1];
          while (current_nb != NULL)
          {
            apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
            current_nb = current_nb->next;
          }
          // SE
          current_nb = bins_local[b + h - 1];
          while (current_nb != NULL)
          {
            apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
            current_nb = current_nb->next;
          }
          // NE
          current_nb = bins_local[b + h + 1];
          while (current_nb != NULL)
          {
            apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
            current_nb = current_nb->next;
          }
          current_part = current_part->next;
        }
      }
    // west border
    for (int y = 1; y < h - 1; ++y)
    {
      b = y;
      current_part = bins_local[b];
      while (current_part != NULL)
      {
        current_part->p->ax = 0;
        current_part->p->ay = 0;
        // C
        current_nb = bins_local[b];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // S
        current_nb = bins_local[b - 1];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // N
        current_nb = bins_local[b + 1];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // E
        current_nb = bins_local[b + h];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // SE
        current_nb = bins_local[b + h - 1];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // NE
        current_nb = bins_local[b + h + 1];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        current_part = current_part->next;
      }
    }
    // east border
    for (int y = 1; y < h - 1; ++y)
    {
      b = w * h - h + y;
      current_part = bins_local[b];
      while (current_part != NULL)
      {
        current_part->p->ax = 0;
        current_part->p->ay = 0;
        // C
        current_nb = bins_local[b];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // S
        current_nb = bins_local[b - 1];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // N
        current_nb = bins_local[b + 1];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // W
        current_nb = bins_local[b - h];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // SW
        current_nb = bins_local[b - h - 1];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // NW
        current_nb = bins_local[b - h + 1];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        current_part = current_part->next;
      }
    }
    // south border
    for (int x = 1; x < w - 1; ++x)
    {
      b = x * h;
      current_part = bins_local[b];
      while (current_part != NULL)
      {
        current_part->p->ax = 0;
        current_part->p->ay = 0;
        // C
        current_nb = bins_local[b];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // N
        current_nb = bins_local[b + 1];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // W
        current_nb = bins_local[b - h];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // E
        current_nb = bins_local[b + h];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // NW
        current_nb = bins_local[b - h + 1];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // NE
        current_nb = bins_local[b + h + 1];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        current_part = current_part->next;
      }
    }
    // north border
    for (int x = 1; x < w - 1; ++x)
    {
      b = x * h + h - 1;
      current_part = bins_local[b];
      while (current_part != NULL)
      {
        current_part->p->ax = 0;
        current_part->p->ay = 0;
        // C
        current_nb = bins_local[b];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // S
        current_nb = bins_local[b - 1];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // W
        current_nb = bins_local[b - h];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // E
        current_nb = bins_local[b + h];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // SW
        current_nb = bins_local[b - h - 1];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        // SE
        current_nb = bins_local[b + h - 1];
        while (current_nb != NULL)
        {
          apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
          current_nb = current_nb->next;
        }
        current_part = current_part->next;
      }
    }
    // southwest corner
    b = 0;
    current_part = bins_local[b];
    while (current_part != NULL)
    {
      current_part->p->ax = 0;
      current_part->p->ay = 0;
      // C
      current_nb = bins_local[b];
      while (current_nb != NULL)
      {
        apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
        current_nb = current_nb->next;
      }
      // N
      current_nb = bins_local[b + 1];
      while (current_nb != NULL)
      {
        apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
        current_nb = current_nb->next;
      }
      // E
      current_nb = bins_local[b + h];
      while (current_nb != NULL)
      {
        apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
        current_nb = current_nb->next;
      }
      // NE
      current_nb = bins_local[b + h + 1];
      while (current_nb != NULL)
      {
        apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
        current_nb = current_nb->next;
      }
      current_part = current_part->next;
    }
    // northwest corner
    b = h - 1;
    current_part = bins_local[b];
    while (current_part != NULL)
    {
      current_part->p->ax = 0;
      current_part->p->ay = 0;
      // C
      current_nb = bins_local[b];
      while (current_nb != NULL)
      {
        apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
        current_nb = current_nb->next;
      }
      // S
      current_nb = bins_local[b - 1];
      while (current_nb != NULL)
      {
        apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
        current_nb = current_nb->next;
      }
      // E
      current_nb = bins_local[b + h];
      while (current_nb != NULL)
      {
        apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
        current_nb = current_nb->next;
      }
      // SE
      current_nb = bins_local[b + h - 1];
      while (current_nb != NULL)
      {
        apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
        current_nb = current_nb->next;
      }
      current_part = current_part->next;
    }
    // southeast corner
    b = w * h - h;
    current_part = bins_local[b];
    while (current_part != NULL)
    {
      current_part->p->ax = 0;
      current_part->p->ay = 0;
      // C
      current_nb = bins_local[b];
      while (current_nb != NULL)
      {
        apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
        current_nb = current_nb->next;
      }
      // N
      current_nb = bins_local[b + 1];
      while (current_nb != NULL)
      {
        apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
        current_nb = current_nb->next;
      }
      // W
      current_nb = bins_local[b - h];
      while (current_nb != NULL)
      {
        apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
        current_nb = current_nb->next;
      }
      // NW
      current_nb = bins_local[b - h + 1];
      while (current_nb != NULL)
      {
        apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
        current_nb = current_nb->next;
      }
      current_part = current_part->next;
    }
    // northeast corner
    int b = w * h - 1;
    current_part = bins_local[b];
    while (current_part != NULL)
    {
      current_part->p->ax = 0;
      current_part->p->ay = 0;
      // C
      current_nb = bins_local[b];
      while (current_nb != NULL)
      {
        apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
        current_nb = current_nb->next;
      }
      // S
      current_nb = bins_local[b - 1];
      while (current_nb != NULL)
      {
        apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
        current_nb = current_nb->next;
      }
      // W
      current_nb = bins_local[b - h];
      while (current_nb != NULL)
      {
        apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
        current_nb = current_nb->next;
      }
      // SW
      current_nb = bins_local[b - h - 1];
      while (current_nb != NULL)
      {
        apply_force(*(current_part->p), *(current_nb->p), &dmin, &davg, &navg);
        current_nb = current_nb->next;
      }
      current_part = current_part->next;
    }
    }
    //}
    
    // apply forces II - interaction with particles from the ghostzone
    //{
    if (rank < m * n)
    {
    if (rank >= m)
    {
      // west ghostzone exists - receive it
      MPI_Probe(rank - m, 0, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, PARTICLE, &ghostzone_size);
      ghostzone = (particle_t*) malloc(ghostzone_size * sizeof(particle_t));
      MPI_Recv(ghostzone, ghostzone_size, PARTICLE, rank - m, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // compute interaction of particles in this ghostzone
      for (int y = 0; y < h; ++y)
      {
        current_part = bins_local[y];
        while (current_part != NULL)
        {
          for (int j = 0; j < ghostzone_size; ++j)
          {
            apply_force(*(current_part->p), ghostzone[j], &dmin, &davg, &navg);
          }
          current_part = current_part->next;
        }
      }
      free(ghostzone);
    }
    if (rank < m * n - m)
    {
      // east ghostzone exists - receive it
      MPI_Probe(rank + m, 0, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, PARTICLE, &ghostzone_size);
      ghostzone = (particle_t*) malloc(ghostzone_size * sizeof(particle_t));
      MPI_Recv(ghostzone, ghostzone_size, PARTICLE, rank + m, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // compute interaction of particles in this ghostzone
      for (int y = 0; y < h; ++y)
      {
        current_part = bins_local[w * h - h + y];
        while (current_part != NULL)
        {
          for (int j = 0; j < ghostzone_size; ++j)
          {
            apply_force(*(current_part->p), ghostzone[j], &dmin, &davg, &navg);
          }
          current_part = current_part->next;
        }
      }
      free(ghostzone);
    }
    if (rank % m != 0)
    {
      // south ghostzone exists - receive it
      MPI_Probe(rank - 1, 0, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, PARTICLE, &ghostzone_size);
      ghostzone = (particle_t*) malloc(ghostzone_size * sizeof(particle_t));
      MPI_Recv(ghostzone, ghostzone_size, PARTICLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // compute interaction of particles in this ghostzone
      for (int x = 0; x < w; ++x)
      {
        current_part = bins_local[x * h];
        while (current_part != NULL)
        {
          for (int j = 0; j < ghostzone_size; ++j)
          {
            apply_force(*(current_part->p), ghostzone[j], &dmin, &davg, &navg);
          }
          current_part = current_part->next;
        }
      }
      free(ghostzone);
    }
    if (rank % m != m - 1)
    {
      // north ghostzone exists - receive it
      MPI_Probe(rank + 1, 0, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, PARTICLE, &ghostzone_size);
      ghostzone = (particle_t*) malloc(ghostzone_size * sizeof(particle_t));
      MPI_Recv(ghostzone, ghostzone_size, PARTICLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // compute interaction of particles in this ghostzone
      for (int x = 0; x < w; ++x)
      {
        current_part = bins_local[x * h + h - 1];
        while (current_part != NULL)
        {
          for (int j = 0; j < ghostzone_size; ++j)
          {
            apply_force(*(current_part->p), ghostzone[j], &dmin, &davg, &navg);
          }
          current_part = current_part->next;
        }
      }
      free(ghostzone);
    }
    if (rank >= m && rank % m != 0)
    {
      // southwest ghostzone exists - receive it
      MPI_Probe(rank - m - 1, 0, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, PARTICLE, &ghostzone_size);
      ghostzone = (particle_t*) malloc(ghostzone_size * sizeof(particle_t));
      MPI_Recv(ghostzone, ghostzone_size, PARTICLE, rank - m - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // compute interaction of particles in this ghostzone
      current_part = bins_local[0];
      while (current_part != NULL)
      {
        for (int j = 0; j < ghostzone_size; ++j)
        {
          apply_force(*(current_part->p), ghostzone[j], &dmin, &davg, &navg);
        }
        current_part = current_part->next;
      }
      free(ghostzone);
    }
    if (rank >= m && rank % m != m - 1)
    {
      // northwest ghostzone exists - receive it
      MPI_Probe(rank - m + 1, 0, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, PARTICLE, &ghostzone_size);
      ghostzone = (particle_t*) malloc(ghostzone_size * sizeof(particle_t));
      MPI_Recv(ghostzone, ghostzone_size, PARTICLE, rank - m + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // compute interaction of particles in this ghostzone
      current_part = bins_local[h - 1];
      while (current_part != NULL)
      {
        for (int j = 0; j < ghostzone_size; ++j)
        {
          apply_force(*(current_part->p), ghostzone[j], &dmin, &davg, &navg);
        }
        current_part = current_part->next;
      }
      free(ghostzone);
    }
    if (rank < m * n - m && rank % m != 0)
    {
      // southeast ghostzone exists - receive it
      MPI_Probe(rank + m - 1, 0, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, PARTICLE, &ghostzone_size);
      ghostzone = (particle_t*) malloc(ghostzone_size * sizeof(particle_t));
      MPI_Recv(ghostzone, ghostzone_size, PARTICLE, rank + m - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // compute interaction of particles in this ghostzone
      current_part = bins_local[w * h - h + 1];
      while (current_part != NULL)
      {
        for (int j = 0; j < ghostzone_size; ++j)
        {
          apply_force(*(current_part->p), ghostzone[j], &dmin, &davg, &navg);
        }
        current_part = current_part->next;
      }
      free(ghostzone);
    }
    if (rank < m * n - m && rank % m != m - 1)
    {
      // northeast ghostzone exists - receive it
      MPI_Probe(rank + m + 1, 0, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, PARTICLE, &ghostzone_size);
      ghostzone = (particle_t*) malloc(ghostzone_size * sizeof(particle_t));
      MPI_Recv(ghostzone, ghostzone_size, PARTICLE, rank + m + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // compute interaction of particles in this ghostzone
      current_part = bins_local[w * h - 1];
      while (current_part != NULL)
      {
        for (int j = 0; j < ghostzone_size; ++j)
        {
          apply_force(*(current_part->p), ghostzone[j], &dmin, &davg, &navg);
        }
        current_part = current_part->next;
      }
      free(ghostzone);
    }
    }
    //}
    
    // compute values to check correctness
    //{
    if(find_option(argc, argv, "-no") == -1)
    {
      MPI_Reduce(&davg, &rdavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&navg, &rnavg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&dmin, &rdmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
      if (rank == 0)
      {
        // computing statistical data
        if (rnavg) 
        {
          absavg += rdavg / rnavg;
          nabsavg++;
        }
        if (rdmin < absmin) 
          absmin = rdmin;
      }
    }
    //}
    
    // free memory for buffers
    //{
    if (rank < m * n)
    {
    if (rank >= m)
    {
      MPI_Wait(&w_req, MPI_STATUS_IGNORE);
      free(west_ghostzone_buf);
    }
    if (rank < m * n - m)
    {
      MPI_Wait(&e_req, MPI_STATUS_IGNORE);
      free(east_ghostzone_buf);
    }
    if (rank % m != 0)
    {
      MPI_Wait(&s_req, MPI_STATUS_IGNORE);
      free(south_ghostzone_buf);
    }
    if (rank % m != m - 1)
    {
      MPI_Wait(&n_req, MPI_STATUS_IGNORE);
      free(north_ghostzone_buf);
    }
    if (rank >= m && rank % m != 0)
    {
      MPI_Wait(&sw_req, MPI_STATUS_IGNORE);
      free(sw_ghostzone_buf);
    }
    if (rank >= m && rank % m != m - 1)
    {
      MPI_Wait(&nw_req, MPI_STATUS_IGNORE);
      free(nw_ghostzone_buf);
    }
    if (rank < m * n - m && rank % m != 0)
    {
      MPI_Wait(&se_req, MPI_STATUS_IGNORE);
      free(se_ghostzone_buf);
    }
    if (rank < m * n - m && rank % m != m - 1)
    {
      MPI_Wait(&ne_req, MPI_STATUS_IGNORE);
      free(ne_ghostzone_buf);
    }
    }
    //}
     
    // move particles and empty bins while doing so
    //{
    for (int i = 0; i < w * h; ++i)
    {
      while (bins_local[i] != NULL)
      {
        move(*pop(bins_local + i));
      }
    }
    //}
    
    // create send buffer
    // consider removing array dest_proc to save allocation cost in each iteration at the cost of recomputing position. 
    // the latter might be cheaper.
    //{
    for (int i = 0; i < n_proc; ++i)
    {
      sendcounts[i] = 0;
      pos_send[i] = 0;  
    }
    dest_proc = (int*) malloc(part_count_local * sizeof(int));
    for (int i = 0; i < part_count_local; ++i)
    {
      int dest = proc_num(particles_local[i], k, w, h, m);
      sendcounts[dest] ++;
      dest_proc[i] = dest;
    }
    int stmp = 0;
    for (int i = 0; i < n_proc; ++i)
    {
      sdispls[i] = stmp;
      stmp += sendcounts[i];
    }
    sendbuf = (particle_t*) malloc(part_count_local * sizeof(particle_t));
    for (int i = 0; i < part_count_local; ++i)
    {
      sendbuf[sdispls[dest_proc[i]] + pos_send[dest_proc[i]]] = particles_local[i];
      pos_send[dest_proc[i]]++;
    }
    free(dest_proc);
    //}
    
    // transmit sizes
    //{
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
    //}
    
    // update variables
    //{
    int rtmp = 0;
    for (int i = 0; i < n_proc; ++i)
    {
      rdispls[i] = rtmp;
      rtmp += recvcounts[i];
    }
    part_count_local = rtmp;
    free(particles_local);
    particles_local = (particle_t*) malloc(part_count_local * sizeof(particle_t));
    //}
    
    // exchange particles
    //{
    MPI_Alltoallv(sendbuf, sendcounts, sdispls, PARTICLE, particles_local, recvcounts, rdispls, PARTICLE, MPI_COMM_WORLD);
    free(sendbuf);
    //}
  }
  simulation_time = read_timer() - simulation_time;
  
  // print results
  //{
  if (rank == 0) 
  {  
    //printf("n = %d, simulation time = %g seconds", n_particles, simulation_time);
    printf( "%d\t%g", n, simulation_time);
    if(find_option( argc, argv, "-no") == -1)
    {
      if (nabsavg) 
        absavg /= nabsavg;
      //  -The minimum distance absmin between 2 particles during the run of the simulation
      //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
      //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
      //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
      printf(", absmin = %lf, absavg = %lf", absmin, absavg);
      if (absmin < 0.4) 
        printf("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
      if (absavg < 0.8) 
        printf("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");
    // Printing summary data
    if(fsum)
      fprintf(fsum, "%d %d %g\n", n_particles, n_proc, simulation_time);
  }
  //}
  
  // free resources
  //{
  free(particles_local);
  free(sendcounts);
  free(sdispls);
  free(recvcounts);
  free(rdispls);
  free(pos_send);
  //}
  
  MPI_Finalize();
  
  return 0;
}


// printf("size = %f, k = %d, m = %d, n = %d, w = %d, h = %d\n", size, k, m, n, w, h);    
