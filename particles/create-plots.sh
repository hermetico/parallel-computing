#!/bin/bash
PLOT=serial.eps

gnuplot << GPLOT
set terminal postscript eps enhanced color
set xlabel "log(n)"
set ylabel "log(seconds)"
set output "$PLOT"

plot "data-serial.txt" using (log(\$1)/log(2)):2 title "Serial implementation", \
     "$BFSDATAFILE" using (log(\$1)/log(2)):2 title "BFS", \
     "$DFSDATADIRFILE" using (log(\$1)/log(2)):2 title "DFS, alpha = 0.75"
GPLOT