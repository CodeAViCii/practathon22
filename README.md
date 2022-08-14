# practathon22
Practathon 2022 (Track: Beginner)

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

Then, to execute the brute force approach which is running on only some toy data of size 10K (each still in
100-dimensional space), one can execute the following command:-
```shell
python3 brute_force.py
```
