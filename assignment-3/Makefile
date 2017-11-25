# Makefile for DNS-algorithm skeleton
# Made for Franklin
# For other machines, replace CC = cc with
# mpicc or other appropriate compiler


CC = mpicc # mpich compiler on laptop


#CC = cc
CFLAGS = -O3
LIBS = 

dnsmat: main.o matrix.o comm.o
	$(CC) $(CFLAGS) -o dnsmat main.o matrix.o comm.o $(LIBS)

%.o: %.c
	$(CC) -c $(CFLAGS) $<

clean:
	rm -f dnsmat *.o
