# Makefile for DNS-algorithm skeleton
# Made for Franklin
# For other machines, replace CC = cc with
# mpicc or other appropriate compiler


# Edison hpc
CC = cc
LIBS = -lrt




CFLAGS = -g -Wall  -std=gnu99 -O3


#dnsmat: main.o matrix.o comm.o
#	$(CC) $(CFLAGS) -o dnsmat main.o matrix.o comm.o $(LIBS)

dnsmat: main.o matrix.o
	$(CC) $(CFLAGS) -o dnsmat main.o matrix.o $(LIBS)

%.o: %.c
	$(CC) -c $(CFLAGS) $<

clean:
	rm -f dnsmat *.o
