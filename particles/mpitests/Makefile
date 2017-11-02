CC = g++
MPICC = mpic++
TARGETS = hello_world send_receive ring checkstatus

all: ${TARGETS}

hello_world: mpihello.cpp
	$(MPICC) -o $@ mpihello.cpp

send_receive: mpisendrecv.cpp
	$(MPICC) -o $@ mpisendrecv.cpp

ring:  mpiring.cpp
	$(MPICC) -o $@ mpiring.cpp

checkstatus: mpicheckstatus.cpp
	$(MPICC) -o $@ mpicheckstatus.cpp


clean:
	rm -f ${TARGETS} *.o
