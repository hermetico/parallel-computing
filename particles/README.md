# Readme file

1. Binning version (OK)
2. OpenMP binning version(OK)
3. MPI
    1. Distribute bins among proceses (OK)
        1. Map processes and local bins from global bin id (OK)
    3. Distribute grey bins among processes  (OK)
        1. Map grey bins to processes *other* processes  (OK)
    4. Distribute particles among processes  (OK)
        1. assign particles to bins
    5. Distribute grey particles among processes
    6. Apply force
    7. Move
    8. Distribute again particles among processes
    9. Assign particles to bins
    10. Distribute again grey particles among processes


### Things to check

1. Check apply force is correct
2. Check new distributed particles are assign correctly to bins
3. Important, check that particles can go up and down using the grey link