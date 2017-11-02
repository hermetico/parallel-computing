CC = g++
MPICC = mpic++
TARGETS = hello_world send_receive

all: ${TARGETS}

hello_world: mpihello.cpp
	$(MPICC) -o $@ mpihello.cpp

send_receive: mpisendrecv.cpp
	$(MPICC) -o $@ mpisendrecv.cpp

clean:
	rm -f ${TARGETS} *.o
