# %%
import numpy as np
import mmh3
from math import log, ceil 

# %%
class BloomFilter:
    def __init__(self, num_rows: int, false_pos_rate: float):
        # optimal values determined by wikipedia equations
        self.bit_arr_size = ceil(-num_rows * log(false_pos_rate) / (log(2)**2))
        self.num_hash_fns = ceil(self.bit_arr_size * log(2)/ num_rows)
        print(f"Creating filter of size {self.bit_arr_size} with {self.num_hash_fns} hash fns")
        self.bit_arr = np.zeros(self.bit_arr_size, dtype=bool)
        
    def _get_bit_arr_idx(self, hash_algo_seed: int, row: np.ndarray) -> int:
        row_hash = mmh3.hash(row.tobytes(), hash_algo_seed) 
        return row_hash % self.bit_arr_size 

    def add(self, row: np.ndarray) -> None:
        """
        From wikipedia:
        To add an element, feed it to each of the k hash functions to get k array positions. Set the bits at all these positions to 1.
        """
        for hash_algo_seed in range(self.num_hash_fns):
            bit_idx = self._get_bit_arr_idx(hash_algo_seed, row)
            self.bit_arr[bit_idx] = True

    def __contains__(self, row: np.ndarray) -> bool:
        for seed in range(self.num_hash_fns):
            bit_idx = self._get_bit_arr_idx(seed, row)
            if not self.bit_arr[bit_idx]:
                return False
        return True

# %%
def deduplicate(data: np.array, false_pos_rate: float) -> np.array:
    n = data.shape[0]
    bf = BloomFilter(n, false_pos_rate)
    unique_rows = []
    num_false_positives = 0
    for idx, row in enumerate(data):
        if row in bf:
            # avoiding false positives by scanning our actual dataset
            prev_rows_matching_curr = np.all(data[:idx] == row, axis=1)
            row_truly_already_present = np.any(prev_rows_matching_curr)
            if row_truly_already_present:
                continue
            else:
                num_false_positives += 1
        # no false negatives, so if absent from bf then it's unique
        unique_rows.append(row)
        bf.add(row)
    print(f"False positive rate {num_false_positives / n}")
    return np.array(unique_rows)        

# %%
total_num_rows = 100_000
num_dups = 100
all_data = np.random.randint(0, 50, size=(total_num_rows, 7))
assert len(all_data) == len(np.unique(all_data, axis=0)), "Initial data has duplicates; rerun"
rows_to_duplicate = np.random.choice(total_num_rows-2, num_dups, replace=False)
all_data[rows_to_duplicate] = all_data[-1]
data_deduped_by_np = np.unique(all_data, axis=0)
assert total_num_rows - len(data_deduped_by_np) == num_dups

# %%
false_pos_rate = 0.01
deduped_by_bf = deduplicate(all_data, false_pos_rate)
assert np.allclose(
    np.sort(deduped_by_bf, axis=0),
    np.sort(data_deduped_by_np, axis=0)
)