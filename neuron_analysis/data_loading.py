from pathlib import Path
import h5py
import numpy as np
import gc

def get_neuron_files_for_session(folder: str, session_date: str, area: str = None):
    folder = Path(folder)
    all_files = folder.glob("*_neu.h5")
    return sorted(
        f for f in all_files
        if f.name.startswith(session_date)
        and (area is None or f"_{area.lower()}_" in f.name)
    )

def build_spike_matrix_from_files(file_list, t_start=0, t_stop=1000):
    all_counts = []
    engagement_vector = None

    for path in sorted(file_list):
        with h5py.File(path, "r", driver="core", backing_store=False) as f:
            spikes = f["data/sp_samples"][:, t_start:t_stop][...]
            sample_id = f["data/sample_id"][...]
        gc.collect()

        all_counts.append(np.count_nonzero(spikes > 0, axis=1))
        if engagement_vector is None:
            engagement_vector = (sample_id != 0).astype(int)

    X = np.stack(all_counts, axis=0)   # shape (neurons, trials)
    return X.T, engagement_vector      # transpose for sklearn


# Function to get t_max from files
def get_tmax_from_files(file_list):
    if not file_list:
        raise FileNotFoundError("No neuron files provided to determine t_max.")
    
    with h5py.File(file_list[0], "r") as f:
        t_max = f["data/sp_samples"].shape[1]
    return t_max
