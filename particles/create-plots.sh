#!/bin/bash

PLOT=serial.eps

gnuplot << GPLOT
set terminal postscript eps enhanced color
set xlabel "log(n)"
set ylabel "log(seconds)"
set output "$PLOT"

plot "data-serial.txt" using (log(\$1)/log(2)):(log(\$2)/log(2)) title "Serial implementation"
GPLOT

PLOT=openmp.eps
gnuplot << GPLOT
set terminal postscript eps enhanced color
set xlabel "log(n)"
set ylabel "log(seconds)"
set output "$PLOT"

plot "data-openmp.txt" using (log(\$1)/log(2)):(log(\$2)/log(2)) title "Openmp 1 thread",
    "data-openmp-2.txt" using (log(\$1)/log(2)):(log(\$2)/log(2)) title "Openmp 2 threads",
    "data-openmp-4.txt" using (log(\$1)/log(2)):(log(\$2)/log(2)) title "Openmp 4 threads",
    "data-openmp-6.txt" using (log(\$1)/log(2)):(log(\$2)/log(2)) title "Openmp 6 threads",
    "data-openmp-12.txt" using (log(\$1)/log(2)):(log(\$2)/log(2)) title "Openmp 12 threads",
    "data-openmp-16.txt" using (log(\$1)/log(2)):(log(\$2)/log(2)) title "Openmp 16 threads",
    "data-openmp-24.txt" using (log(\$1)/log(2)):(log(\$2)/log(2)) title "Openmp 24 threads"
GPLOT

PLOT=mpi.eps
gnuplot << GPLOT
set terminal postscript eps enhanced color
set xlabel "log(n)"
set ylabel "log(seconds)"
set output "$PLOT"

plot "data-mpi.txt" using (log(\$1)/log(2)):(log(\$2)/log(2)) title "MPI 1 thread",
    "data-mpi-2.txt" using (log(\$1)/log(2)):(log(\$2)/log(2)) title "MPI 2 threads",
    "data-mpi-4.txt" using (log(\$1)/log(2)):(log(\$2)/log(2)) title "MPI 4 threads",
    "data-mpi-6.txt" using (log(\$1)/log(2)):(log(\$2)/log(2)) title "MPI 6 threads",
    "data-mpi-12.txt" using (log(\$1)/log(2)):(log(\$2)/log(2)) title "MPI 12 threads",
    "data-mpi-18.txt" using (log(\$1)/log(2)):(log(\$2)/log(2)) title "MPI 18 threads",
    "data-mpi-24.txt" using (log(\$1)/log(2)):(log(\$2)/log(2)) title "MPI 24 threads"
GPLOT