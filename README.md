# practathon22
Practathon 2022 (Track: Beginner)
Solved ☑️

## Problem Statement

Given a set of data `D` of 10 million vectors/arrays/points (which in this case are generated randomly) having 100
dimensions (features).

Task: Find the top 10 nearest (closest) neighbours to the given query (which is another vector of dimensions 100) from
the data `D` in < 1 second.

## Usage Instructions

Ensure the following Python libraries have been installed:
```
Numpy
Pandas
Scipy
```

### Brute Force approach
Then, to execute the brute force approach which is running on only some toy data of size 10K (each still in
100-dimensional space), one can execute the following command:-
```shell
python3 brute_force.py
```

### KD Tree approach
Also, to execute the newly added KD Tree approach which is running on the actual data of size 10M (each in
100-dimensional space), one can execute the following command:-
```shell
python3 kdtree.py
```

___

`*` If you wish to see the o/p that was generated on a particular run, 3 `.txt` files have been added. These files not
only contain the results, but also the query execution times.
They are: -
- `brute_force_op_on_toy_data.txt` -> The output of the 1st approach on the smaller toy data
- `kdtree_op_on_toy_data.txt` -> The output of the 2nd approach on the smaller toy data (can be used to verify
correctness from previous results)
- `kdtree_op_on_actual_data.txt` -> The output of the 2nd approach on the actual data (in sub-second times) ☑️

