import time
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

np.random.seed(3)  # to ensure all the runs are repeatable

"""
## Actual Data (10M X 100)
20 Queries
"""

print(
    '''
    K-D Tree approach on Actual Data of size (10M X 100)
    
    Initializing the data, followed by some preprocessing (creating the tree)...
    '''
)

actual_data_size = (int(1e7), 100)  # 10M X 100

# points uniformly distributed in the range [-100,100] in 100-dimensional space
actual_data = np.random.rand(actual_data_size[0], actual_data_size[1]) * 200 - 100

print('Actual Data:-')
print(actual_data)

print(
    '''
    
    Done.
    
    Creating 20 random query points...
    '''
)

actual_queries = np.random.rand(20, actual_data_size[1]) * 200 - 100

print('Actual Queries:-')
print(actual_queries)

# # Optionally this data can be written to a file, but with the random seed specified the results are repeatable
# # so this step is not necessary

# actual_data.to_csv('actual_data.csv')
# actual_queries.to_csv('actual_queries.csv')


tree = KDTree(actual_data, leafsize=actual_data_size[0] + 1)

all_times = []
results = []
index = 1

for query in actual_queries:
    start_time_for_query = time.time()

    # K-D Tree approach on actual_data to find 10 nearest points
    # (uncomment print statement to see the indices and distances of the results)
    distances, ndx = tree.query([query], k=10)

    # print(distances, ndx)
    # print(actual_data[ndx])

    end_time_for_query = time.time()

    nearest_neighbours = pd.DataFrame(actual_data[ndx][0])
    # print(nearest_neighbours)

    nearest_neighbours['query'] = index
    results.append(nearest_neighbours)

    time_taken = end_time_for_query - start_time_for_query

    print('\n\nTime taken for query', index, ':', time_taken, '\n\n')
    index += 1

    all_times.append(time_taken)

all_times = pd.Series(all_times)

results = pd.concat(results)
results['index'] = results.index
results.set_index(['query', 'index'], inplace=True)

print('Results of all 20 queries:-')
print(results)

# results.to_csv('actual_results.csv')

print('\n\nAll runs complete. Execution time details over 20 queries :-')
print(pd.Series([np.median(all_times), np.mean(all_times), np.std(all_times), np.min(all_times), np.max(all_times)],
                index=['Median', 'Mean', 'Standard deviation', 'Minima', 'Maxima']))
