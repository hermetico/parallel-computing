#
# on Franklin:
# type "module load acml" to use ACML (required)
# type "module swap PrgEnv-pgi PrgEnv-gnu" to use GNU compiler
# type "module swap PrgEnv-pgi PrgEnv-pathscale" to use PathScale compiler
#
# On other machines you might need to use other library than ACML
# this could be MKL on an Intel platform or ESSL on an IBM platform.
#
#CC = CC
CC = g++
#LIBS = -lacml
#LIBS = $(shell pkg-config cblas --libs)

#multi-thread cblas for IMADA i5 3.2 GHz machine
#use the following line only with a hand compiled atlas
#LIBS = -L/home/hermetico/opt/atlas/lib -lptcblas -latlas -lpthread

#otherwise use
#LIBS = -lcblas -latlas -lpthread

#single-thread cblas for IMADA i5 3.2 GHz machine
#LIBS = -L/home/daniel/atlas/lib -lcblas -latlas

#not optimized Ubuntu version
LIBS= -lcblas



#not optimized Ubuntu version
#LIBS= -lcblas

CFLAGS = -O3 -mavx

all:	benchmark-naive benchmark-blocked benchmark-blas benchmark detailed-benchmark-blocked detailed-benchmark detailed-benchmark-blas detailed-benchmark-naive tests

benchmark-naive: benchmark.o dgemm-naive.o
	$(CC) -o $@ benchmark.o dgemm-naive.o $(LIBS)

benchmark-blocked: benchmark.o dgemm-blocked.o
	$(CC) -o $@ benchmark.o dgemm-blocked.o $(LIBS)

benchmark-blas: benchmark.o dgemm-blas.o
	$(CC) -o $@ benchmark.o dgemm-blas.o $(LIBS)

benchmark: benchmark.o dgemm.o
	$(CC) -o $@ benchmark.o dgemm.o $(LIBS)

detailed-benchmark-blocked: detailed_benchmark.o dgemm-blocked.o
	$(CC) -o $@ detailed_benchmark.o dgemm-blocked.o $(LIBS)

detailed-benchmark: detailed_benchmark.o dgemm.o
	$(CC) -o $@ detailed_benchmark.o dgemm.o $(LIBS)

detailed-benchmark-blas: detailed_benchmark.o dgemm-blas.o
	$(CC) -o $@ detailed_benchmark.o dgemm-blas.o $(LIBS)

detailed-benchmark-naive: detailed_benchmark.o dgemm-naive.o
	$(CC) -o $@ detailed_benchmark.o dgemm-naive.o $(LIBS)

tests: tests.o dgemm.o
	$(CC) -o $@ tests.o dgemm.o $(LIBS)


%.o: %.cpp
	$(CC) -c $(CFLAGS) $<

%.o: %.c
	$(CC) -c $(CFLAGS) $<

clean:
	rm -f benchmark-naive benchmark-blocked benchmark-blas benchmark detailed-benchmark-blas detailed-benchmark-blocked detailed-benchmark detailed-benchmark-naive tests *.o
