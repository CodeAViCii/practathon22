import time
import numpy as np
import pandas as pd
from scipy.spatial import distance

np.random.seed(3)  # to ensure all the runs are repeatable

"""
## Toy Data (10K X 100)
10 Queries
"""

print(
    '''
    Brute force approach on Toy Data of size (10K X 100)
    
    Initializing the data, followed by some preprocessing...
    '''
)

toy_data_size = (int(1e4), 100)  # 10K X 100

# points uniformly distributed in the range [-100,100] in 100-dimensional space
toy_data = pd.DataFrame(np.random.rand(toy_data_size[0], toy_data_size[1]) * 200 - 100)

print('Toy Data:-')
print(toy_data)

print(
    '''
    
    Done.
    
    Creating 10 random query points...
    '''
)

toy_queries = pd.DataFrame(np.random.rand(10, toy_data_size[1]) * 200 - 100)

print('Toy Queries:-')
print(toy_queries)


# # Optionally this data can be written to a file, but with the random seed specified the results are repeatable
# # so this step is not necessary

# toy_data.to_csv('toy_data.csv')
# toy_queries.to_csv('toy_queries.csv')


# Brute-force approach on toy_data (uncomment print statement to see the indices and distances of the results)
def perform_suggestions_task_bf(data, query):
    # print('data',data,sep='\n')
    # print('query',query,sep='\n')

    euclidean_dist = data.apply(lambda p: distance.euclidean(p, query), axis=1)
    smallest_euclidean_distances = euclidean_dist.nsmallest(10).sort_values()
    # print(smallest_euclidean_distances)
    result = data.iloc[smallest_euclidean_distances.index].reset_index(drop=True)

    return result


all_times = []
results = []
index = 1

for each_query in toy_queries.itertuples(index=False):
    start_time_for_query = time.time()

    nearest_neighbours = perform_suggestions_task_bf(toy_data, each_query)

    end_time_for_query = time.time()

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

print('Results of all 10 queries:-')
print(results)

# results.to_csv('toy_results.csv')

print('\n\nAll runs complete. Execution time details over 10 queries :-')
print(pd.Series([np.median(all_times), np.mean(all_times), np.std(all_times), np.min(all_times), np.max(all_times)],
                index=['Median', 'Mean', 'Standard deviation', 'Minima', 'Maxima']))
