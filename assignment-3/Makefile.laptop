# Makefile for DNS-algorithm skeleton
# Made for Franklin
# For other machines, replace CC = cc with
# mpicc or other appropriate compiler

# pc
CC = mpicc
LIBS = -lm -lrt -lblas -llapack

# Edison hpc
#CC = CC
#LIBS = -lrt



CFLAGS = -O3


#dnsmat: main.o matrix.o comm.o
#	$(CC) $(CFLAGS) -o dnsmat main.o matrix.o comm.o $(LIBS)

dnsmat: main.o matrix.o
	$(CC) $(CFLAGS) -o dnsmat main.o matrix.o $(LIBS)

%.o: %.c
	$(CC) -c $(CFLAGS) $<

clean:
	rm -f dnsmat *.o
