#Readme file

1. Binning version (OK)
2. OpenMP binning version(OK)
3. MPI
    1. Distribute bins among proceses (KO)
        1. Map processes and local bins from global bin id (Double ckeck this)
    3. Distribute grey bins among processes
        1. Map grey bins to processes *other* processes
    4. Distribute particles among processes