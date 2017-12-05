#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "comm.h"
#include "matrix.h"



inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }

void print_mat(int m, int n, double* a)
{
	for (int row = 0; row < m; row++)
	{
		for(int col = 0; col < n; col++)
			printf("%f     ", a[row+col*m]);
		printf("\n");
	}
}

/* Print a header for results output */
void results_header()
{
	printf("Dims  No. Proc.  Avg. RT / Dev. (Eff.) Speedup\n");
}

/* Print the stats for 1 run */
void write_result(int full_dim, int procs, double rt, double dev, double eff, double speedup)
{
	printf("%-5i\t%-10i\t%-5.5f / %-5.5f\t%-5.5f\t%-5.5f\n", full_dim, procs, rt, dev, eff, speedup);
}

/* Average and standard deviation */
double average(int count, double *list, double *dev)
{
	int i;
	double sum = 0.0, avg;

	for (i = 0; i < count; i++)
	{
		sum += list[i];
	}

	avg = sum/(double)count;

	if (dev != 0)
	{
		sum = 0.0;
		for (i = 0; i < count; i++)
		{
			sum += (list[i] - avg)*(list[i] - avg);
		}

		*dev = sqrt(sum/(double)count);
	}

	return avg;
}

static void core_dummy(int m, int n, int k, double* A, double* B, double* C)
{
	double cij;
	/* For each row i of A */
	for (int i = 0; i < m; ++i)
		/* For each column j of B */
		for (int j = 0; j < n; ++j)
		{
			cij = 0;
			/* Compute C(i,j) */
			for (int l = 0; l < k; ++l)
				cij += A[i + l * m] * B[l + j * k];
			C[i + j * m] = cij;
		}
}


int main(int argc, char **argv)
{
	/* Statistics */
	double startTime = 0.0, endTime = 0.0, avg, dev; /* Timing */
	double times[10]; /* Times for all runs */

	MPI_Init(&argc, &argv);

	/* Get MPI process stats */
	int rank, p, q, n;
	//MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* Get parameters */
	if (argc == 3)
	{
		/* Get number of processes */
		p = atoi(argv[1]);
		q = pow(p, 1/3.);

		/* Get maximum matrix dimension */
		n = atoi(argv[2]);
	}
	else
	{
		printf("Wrong number of parameters\n");
		exit(-1);
	}

	/* Write header */
	if (rank == 0)
		results_header();

	/* Compute grid coordinates and create communicators */
	int coords[3];
	coords[0] = (rank / q) % q;
	coords[1] = rank % q;
	coords[2] = rank / (q * q);
	// this assumes that 0 <= rank < p, this must be given
	MPI_Comm comm_i;
	MPI_Comm_split(MPI_COMM_WORLD, coords[2] * q + coords[1], coords[0], &comm_i);
	MPI_Comm comm_j;
	MPI_Comm_split(MPI_COMM_WORLD, coords[2] * q + coords[0], coords[1], &comm_j);
	MPI_Comm comm_k;
	MPI_Comm_split(MPI_COMM_WORLD, coords[0] * q + coords[1], coords[2], &comm_k);
	MPI_Comm comm_0;
	MPI_Comm_split(MPI_COMM_WORLD, coords[2], coords[0] * q + coords[1], &comm_0);
	int rank_i, rank_j, rank_k;
	MPI_Comm_rank(comm_i, &rank_i);
	MPI_Comm_rank(comm_j, &rank_j);
	MPI_Comm_rank(comm_k, &rank_k);
	MPI_Status status;


	/* compute optimal time */

	double optimal_time = (2. * pow(n, 3) ) / (8.4 * pow(10, 9));
	/* Make full matrices */
	double* A;
	double* B;
	double* C;
	int check[4];
	double c_ab;
	if (rank == 0)
	{
		srand48(time(NULL));
		A = (double*) malloc(n * n * sizeof(double));
		B = (double*) malloc(n * n * sizeof(double));
		for(int i = 0; i < n * n; ++i)
		{
			A[i] = drand48();
			B[i] = drand48();
		}
		/* Compute a random entry to check correctness later */
		// Note that this entry is not chosen uniformly at random to ease the indexing, but it is close
		check[0] = lrand48() % q; // i
		check[1] = lrand48() % q; // j
		check[2] = lrand48() % (n / q + (check[0] < (n % q))); // x
		check[3] = lrand48() % (n / q + (check[1] < (n % q))); // y
		c_ab = 0;
		int a = (check[0] * (n / q) + min(check[0], n % q) + check[2]);
		int b = (check[1] * (n / q) + min(check[1], n % q) + check[3]);
		for (int k = 0; k < n; ++k)
			c_ab += A[k * n + a] * B[b * n + k]; // add A_ak * B_kb
	}

	/* Distribute full matrices to bottom layer */
	int n_subi_orig, n_subj_orig;
	n_subi_orig = n / q + (coords[0] < (n % q));
	n_subj_orig = n / q + (coords[1] < (n % q));
	double* A_sub_orig;
	double* B_sub_orig;
	double* C_sub_orig;
	double* sendbuf;
	int* sendcounts;
	int* displs;
	if (coords[2] == 0)
	{
		A_sub_orig = (double*) malloc(n_subi_orig * n_subj_orig * sizeof(double));
		B_sub_orig = (double*) malloc(n_subi_orig * n_subj_orig * sizeof(double));
		C_sub_orig = (double*) malloc(n_subi_orig * n_subj_orig * sizeof(double));
		sendcounts = (int*) malloc(q * q * sizeof(int));
		displs = (int*) malloc(q * q * sizeof(int));
		for (int i = 0; i < q; ++i)
			for (int j = 0; j < q; ++j)
			{
				sendcounts[i * q + j] = (n / q + (i < (n % q))) * (n / q + (j < (n % q)));
			}
		displs[0] = 0;
		for (int i = 1; i < q * q; ++i)
			displs[i] = displs[i - 1] + sendcounts[i - 1];
	}

	if (rank == 0)
	{
		sendbuf = (double*) malloc(n * n * sizeof(double));
		/* Prepare matrix A */
		for (int i = 0; i < q; ++i)
			for (int j = 0; j < q; ++j)
				for (int x = 0; x < n / q + (i < (n % q)); ++x)
					for (int y = 0; y < n / q + (j < (n % q)); ++y)
						sendbuf[i * n * ((n / q) + min(i, n % q)) + j * ((n / q) + (i < (n % q)))  * ((n / q) + min(j, n % q)) + y * ((n / q) + (i < (n % q))) + x] =
								A[(j * (n / q) + min(j, n % q) + y) * n + i * (n / q) + min(i, n % q) + x];
	}
	/* Distribute matrix A */
	if (coords[2] == 0)
		MPI_Scatterv(sendbuf, sendcounts, displs, MPI_DOUBLE, A_sub_orig, n_subi_orig * n_subj_orig, MPI_DOUBLE, 0, comm_0);

	if (rank == 0)
	{
		/* Prepare matrix B */
		for (int i = 0; i < q; ++i)
			for (int j = 0; j < q; ++j)
				for (int x = 0; x < n / q + (i < (n % q)); ++x)
					for (int y = 0; y < n / q + (j < (n % q)); ++y)
						sendbuf[i * n * ((n / q) + min(i, n % q)) + j * ((n / q) + (i < (n % q)))  * ((n / q) + min(j, n % q)) + y * ((n / q) + (i < (n % q))) + x] =
								B[(j * (n  / q) + min(j, n % q) + y) * n + i * (n / q) + min(i, n % q) + x];
	}
	/* Distribute matrix B */
	if (coords[2] == 0)
	{
		MPI_Scatterv(sendbuf, sendcounts, displs, MPI_DOUBLE, B_sub_orig, n_subi_orig * n_subj_orig, MPI_DOUBLE, 0, comm_0);
	}
	if (rank == 0)
	{
		free(A);
		free(B);
		free(sendbuf);
	}

	/* Allocate matrices */
	// Coordinates of the local matrix for computation
	int n_subi_A = n / q + (coords[0] < (n % q));
	int n_subj_A = n / q + (coords[2] < (n % q));
	int n_subi_B = n / q + (coords[2] < (n % q));
	int n_subj_B = n / q + (coords[1] < (n % q));
	A = (double*) malloc(n_subi_A * n_subj_A * sizeof(double));
	B = (double*) malloc(n_subi_B * n_subj_B * sizeof(double));
	C = (double*) malloc(n_subi_A * n_subj_B * sizeof(double));
	if (coords[2] == 0 && coords[1] == 0)
		memcpy(A, A_sub_orig, n_subi_orig * n_subj_orig * sizeof(double));
	if (coords[2] == 0 && coords[0] == 0)
		memcpy(B, B_sub_orig, n_subi_orig * n_subj_orig * sizeof(double));

	/* Run each config 10 times */
	for (int k = 0; k < 10; k++)
	{
		/* Start timer */
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0)
			startTime = MPI_Wtime();

		/* Distribute matrices (one-to-one comm) */
		if (coords[2] == 0)
		{
			/* (i, j, 0) sends A to (i, j, j) (unless j = 0) */
			if (coords[1] != 0)
				MPI_Send(A_sub_orig, n_subi_orig * n_subj_orig, MPI_DOUBLE, q * q * coords[1] + q * coords[0] + coords[1], 0, MPI_COMM_WORLD);
			/* (i, j, 0) sends B to (i, j, i) (unless i = 0) */
			if (coords[0] != 0)
				MPI_Send(B_sub_orig, n_subi_orig * n_subj_orig, MPI_DOUBLE, q * q * coords[0] + q * coords[0] + coords[1], 0, MPI_COMM_WORLD);
		}
		if (coords[1] == coords[2])
			/* (i, j, j) receives A from (i, j, 0) (unless j = 0) */
			if (coords[1] != 0)
				MPI_Recv(A, n_subi_orig * n_subj_orig, MPI_DOUBLE, q * coords[0] + coords[1], 0, MPI_COMM_WORLD, &status);

		if (coords[0] == coords[2])
			/* (i, j, i) receives B from (i, j, 0) (unless i = 0) */
			if (coords[0] != 0)
				MPI_Recv(B, n_subi_orig * n_subj_orig, MPI_DOUBLE, q * coords[0] + coords[1], 0, MPI_COMM_WORLD, &status);

		/* Distribute matrices (respective one-to-"all" comm) */
		MPI_Bcast(A, n_subi_A * n_subj_A, MPI_DOUBLE, coords[2], comm_j);
		MPI_Bcast(B, n_subi_B * n_subj_B, MPI_DOUBLE, coords[2], comm_i);

		/* Multiply matrices */
		// TODO replace nested loop by call to library function
		//core_dummy(n_subi_A, n_subj_B, n_subj_A, A, B, C);
		matrix_mult(n_subi_A, n_subj_B, n_subj_A, A, B, C);

		/* Collect results ("all" to one reduction) */
		MPI_Reduce(C, C_sub_orig, n_subi_A * n_subj_B, MPI_DOUBLE, MPI_SUM, 0, comm_k);

		/* End timer */
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0)
		{
			endTime = MPI_Wtime();
			times[k] = endTime - startTime;
		}

		/* Reset matrices */
		free(A);
		free(B);
		A = (double*) malloc(n_subi_A * n_subj_A * sizeof(double));
		B = (double*) malloc(n_subi_B * n_subj_B * sizeof(double));
		if (coords[2] == 0 && coords[1] == 0)
			memcpy(A, A_sub_orig, n_subi_orig * n_subj_orig * sizeof(double));
		if (coords[2] == 0 && coords[0] == 0)
			memcpy(B, B_sub_orig, n_subi_orig * n_subj_orig * sizeof(double));
	}

	/* collect matrices on node (0, 0, 0) and check entry there */
	double* C_complete;
	if (rank == 0)
		C_complete = (double*) malloc(n * n * sizeof(double));
	if (coords[2] == 0)
		MPI_Gatherv(C_sub_orig, n_subi_orig * n_subj_orig, MPI_DOUBLE, C_complete, sendcounts, displs, MPI_DOUBLE, 0, comm_0);
	if (rank == 0)
	{
		if (C_complete[check[0] * n * ((n / q) + min(check[0], n % q)) + check[1] * ((n / q) + (check[0] < (n % q))) * ((n / q) + min(check[1], n % q)) + check[3] * ((n / q) + (check[0] < (n % q))) + check[2]] != c_ab)
		{
			printf("The checked entry did not coincide. Please ignore if the below values do not differ significantly.\n");
			printf("c_ab = %.20f, result = %.20f\n", c_ab, C_complete[check[0] * n * ((n / q) + min(check[0], n % q)) + check[1] * ((n / q) + (check[0] < (n % q))) * ((n / q) + min(check[1], n % q)) + check[3] * ((n / q) + (check[0] < (n % q))) + check[2]]);
		}
		free(C_complete);
	}

	/* Destroy matrices */
	free(A);
	free(B);
	if (coords[2] == 0)
	{
		free(A_sub_orig);
		free(B_sub_orig);
		free(sendcounts);
		free(displs);
	}

	/* Print stats */
	if (rank == 0)
	{

		avg = average(10, times, &dev);

		double  speedup = optimal_time / avg;
		double eff = speedup / p;

		write_result(n, q * q * q, avg, dev, eff, speedup);
	}

	/* Exit program */
	MPI_Finalize();

	return 0;
}