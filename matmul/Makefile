# on Edision we will benchmark you against the default vendor-tuned BLAS. The compiler wrappers handle all the linking. If you wish to compare with other BLAS implementations, check the NERSC documentation.
# This makefile is intended for the GNU C compiler. To change compilers, you need to type something like: "module swap PrgEnv-pgi PrgEnv-gnu" See the NERSC documentation for available compilers.

CC = cc 
OPT = -O3
CFLAGS = -g -Wall -std=gnu99 $(OPT) -mavx
LDFLAGS = -Wall
# librt is needed for clock_gettime

# EDISON
LDLIBS = -lrt

# NOTEBOOK
LDLIBS = -lrt -lblas -llapack

targets = benchmark benchmark-plot short-benchmark
objects = dgemm.o benchmark-plot.o short-benchmark.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark : benchmark.o dgemm.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-plot : benchmark-plot.o dgemm.o
	$(CC) -o $@ $^ $(LDLIBS)
short-benchmark : short-benchmark.o dgemm.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
