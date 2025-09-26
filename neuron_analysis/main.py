import os
import numpy as np
from datetime import datetime
from neuron_analysis.data_loading import get_neuron_files_for_session, build_spike_matrix_from_files, get_tmax_from_files
#from neuron_analysis.decoding import sliding_window_auc
from neuron_analysis.visualization import plot_auc_vs_time, best_window

DATA_FOLDER = '/Users/arnofeinstein/Documents/DESU Data Science/Tutored Project/neurons'
SESSION_DATE = "2022-11-22" 
AREA = None
BASE_OUTPUT_FOLDER = "/Users/arnofeinstein/Documents/Les études/PhD/DESU Data Science/Tutored Project/neurons/output"

def main():
    files = get_neuron_files_for_session(DATA_FOLDER, SESSION_DATE, AREA)
    print(f"{len(files)} neuron files loaded.")

    t_max = get_tmax_from_files(files)
    print(f"Detected t_max = {t_max} ms for session {SESSION_DATE}")

    # windows, centers, aucs = sliding_window_auc(
    #     files, build_spike_matrix_from_files, t_max=t_max
    # )

    # Create timestamped output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(BASE_OUTPUT_FOLDER, f"{SESSION_DATE}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    # Save .npy data
    np.save(os.path.join(output_folder, f"{SESSION_DATE}_windows.npy"), windows)
    np.save(os.path.join(output_folder, f"{SESSION_DATE}_centers.npy"), centers)
    np.save(os.path.join(output_folder, f"{SESSION_DATE}_aucs.npy"), aucs)
    print(f"Data saved in {output_folder}")

    # Save CSV summary
    import csv
    csv_path = os.path.join(output_folder, f"{SESSION_DATE}_summary.csv")
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["window_start", "window_end", "center", "auc"])
        for (t0, t1), c, auc in zip(windows, centers, aucs):
            writer.writerow([t0, t1, c, auc])
    print(f"CSV summary saved to {csv_path}")

    # Save plot
    plot_auc_vs_time(centers, aucs, SESSION_DATE, output_folder=output_folder)

    # Print best window
    win, auc = best_window(windows, aucs)
    print(f"Best window: {win} | AUC = {auc:.3f}")

if __name__ == "__main__":
    main()


# alignment_utils

import numpy as np
from compdec_v4pfclip_01.utils import config_task
from compdec_v4pfclip_01.utils.trials import align_trials


def patched_align_on(self, select_block, event, time_before, error_type, stim_loc="both"):
    """
    Aligns spike samples to a specified event for selected trials, optionally filtering by block, stimulus location, and error type.

    Parameters
    ----------
    select_block : int
        The block number to select trials from (1 or 2).
    event : str
        The event name to align trials on.
    time_before : int or float
        The time (in samples) before the event to align the spike data.
    error_type : int or None
        The error type to filter trials. If None, no error filtering is applied.
    stim_loc : str, optional
        The stimulus location filter. Options are "both", "in", "out", "contra", "ipsi". Default is "both".

    Returns
    -------
    aligned_sp : np.ndarray
        The spike samples aligned to the specified event for valid trials.
    complete_mask : np.ndarray
        Boolean mask indicating which trials in the original data are included in the alignment.
    """
    if select_block == 1:
        code = config_task.EVENTS_B1[event]
        block_mask = (self.block == 1)
        if stim_loc == "both":
            loc_mask = np.ones_like(block_mask, dtype=bool)
        elif stim_loc == "in":
            loc_mask = (self.rf_loc == 1)
        elif stim_loc == "out":
            loc_mask = (self.rf_loc == -1)
        elif stim_loc == "contra":
            loc_mask = (self.pos_code == 1)
        elif stim_loc == "ipsi":
            loc_mask = (self.pos_code == -1)
        else:
            raise KeyError(f"Invalid stim_loc value: {stim_loc}")
    elif select_block == 2:
        code = config_task.EVENTS_B2[event]
        block_mask = (self.block == 2)
        loc_mask = np.ones_like(block_mask, dtype=bool)
    else:
        raise ValueError(f"Invalid block: {select_block}")

    mask = block_mask & loc_mask
    if error_type is not None:
        mask &= (self.trial_error == error_type)

    sp_samples_m = self.sp_samples[mask]
    code_nums = self.code_numbers[mask]
    code_times = self.code_samples[mask]

    shifts = []
    valid_trials_idx = []
    for i, (codes, times) in enumerate(zip(code_nums, code_times)):
        idx = np.where(codes == code)[0]
        if len(idx) > 0:
            t_event = times[idx[0]]
            shifts.append(int(t_event - time_before))
            valid_trials_idx.append(i)

    shifts = np.array(shifts)
    sp_selected = sp_samples_m[valid_trials_idx]
    aligned_sp = align_trials.indep_roll(sp_selected, -shifts, axis=1)

    trial_indices_global = np.arange(len(self.sp_samples))[mask]
    complete_mask = np.isin(np.arange(len(self.sp_samples)), trial_indices_global[valid_trials_idx])

    return aligned_sp, complete_mask


def patched_get_align_on(self, params, inplace=False, delete_att=None):
    """
    Aligns and processes spike data based on provided parameters.

    For each parameter dictionary in `params`, this method:
    - Calls the internal `_align_on` method to obtain spike data and mask.
    - Trims spike data to the specified time window.
    - Constructs attribute names based on block and event information.
    - Optionally sets the processed data as attributes on the instance (`inplace=True`), or returns a list of dictionaries containing the processed data (`inplace=False`).
    - Optionally deletes specified attributes by setting them to empty arrays.

    Args:
        params (list of dict): List of parameter dictionaries, each containing:
            - 'select_block' (int): Block identifier (1 or 2).
            - 'event' (int): Event identifier.
            - 'time_before' (int): Time before the event to include.
            - 'time_after' (int): Time after the event to include.
            - 'error_type' (any): Error type for alignment.
            - 'loc' (any): Stimulus location.
        inplace (bool, optional): If True, sets processed data as attributes on the instance. If False, returns processed data. Defaults to False.
        delete_att (list of str, optional): List of attribute names to delete (set to empty arrays). Defaults to None.

    Returns:
        self or list of dict: Returns self if `inplace` is True, otherwise returns a list of dictionaries containing processed spike data, mask, and time_before for each parameter set.

    Raises:
        ValueError: If an invalid block is specified in the parameters.
    """
    aligned = []
    for it in params:
        sp, mask = self._align_on(
            select_block=it["select_block"],
            event=it["event"],
            time_before=it["time_before"],
            error_type=it["error_type"],
            stim_loc=it["loc"],
        )
        endt = it["time_before"] + it["time_after"]
        if it["select_block"] == 1:
            att_name = f"{config_task.EVENTS_B1_SHORT[it['event']]}_{it['loc']}"
        elif it["select_block"] == 2:
            att_name = f"{config_task.EVENTS_B2_SHORT[it['event']]}_{it['loc']}"
        else:
            raise ValueError("Invalid block")

        sp_trimmed = np.array(sp[:, :endt], dtype=np.int8)
        mask_bool = np.array(mask, dtype=bool)
        t_before = np.array(it["time_before"], dtype=np.int32)

        if inplace:
            setattr(self, f"sp_{att_name}", sp_trimmed)
            setattr(self, f"mask_{att_name}", mask_bool)
            setattr(self, f"time_before_{att_name}", t_before)
        else:
            aligned.append({
                f"sp_{att_name}": sp_trimmed,
                f"mask_{att_name}": mask_bool,
                f"time_before_{att_name}": t_before
            })

    if delete_att:
        for a in delete_att:
            if hasattr(self, a):
                setattr(self, a, np.array([]))

    return self if inplace else aligned


def get_align_params(loc="both", event="sample_on", pre=200, post=1500):
    return [{
        "select_block": 1,
        "event": event,
        "time_before": pre,
        "time_after": post,
        "loc": loc,
        "error_type": 0,
    }]


def make_aligned_builder(event="sample_on", pre=200, post=1500, loc="both", random_state=42):
    """
    Creates a builder function to generate aligned spike count matrices for a list of neuron data objects.

    Parameters
    ----------
    event : str, optional
        The event name to align spikes to (default is "sample_on").
    pre : int, optional
        Number of time bins before the event to include (default is 200).
    post : int, optional
        Number of time bins after the event to include (default is 1500).
    loc : str, optional
        Location specifier for mask and spike attributes (default is "both").
    random_state : int, optional
        Random seed for reproducibility (default is 42).

    Returns
    -------
    build_matrix : function
        A function that takes a list of neuron data objects and optional time window arguments,
        and returns:
            X : np.ndarray
                2D array of spike counts, shape (n_trials, n_neurons).
            y : np.ndarray
                1D array of binary labels for each trial, shape (n_trials,).

    Raises
    ------
    ValueError
        If there are no common trials among the neurons, or if the specified time window exceeds
        the available spike data length.
    """
    suffix = f"son_{loc}"
    t_stop_global = pre + post

    def build_matrix(neurons_list, t_start=None, t_stop=None):
        if t_start is None:
            t_start = 0
        if t_stop is None:
            t_stop = t_stop_global

        trial_ids_all = []
        for nd in neurons_list:
            mask = getattr(nd, f"mask_{suffix}")
            trial_ids = np.where(mask)[0]
            trial_ids_all.append(set(trial_ids))

        common_trial_ids = sorted(set.intersection(*trial_ids_all))
        if len(common_trial_ids) == 0:
            raise ValueError("Aucun essai en commun entre les neurones !")

        all_counts = []
        for nd in neurons_list:
            mask = getattr(nd, f"mask_{suffix}")
            spikes = getattr(nd, f"sp_{suffix}")
            trial_ids = np.where(mask)[0]
            id_map = {tid: i for i, tid in enumerate(trial_ids)}
            local_ids = [id_map[tid] for tid in common_trial_ids if tid in id_map]
            spikes_common = spikes[local_ids]

            if t_stop > spikes_common.shape[1]:
                raise ValueError(f"t_stop={t_stop} > spike len ({spikes_common.shape[1]})")

            counts = np.count_nonzero(spikes_common[:, t_start:t_stop] > 0, axis=1)
            all_counts.append(counts)

        X = np.stack(all_counts, axis=0).T
        y = (neurons_list[0].sample_id[common_trial_ids] != 0).astype(int)

        assert X.shape[0] == len(y), f"Shape mismatch X: {X.shape}, y: {y.shape}"
        return X, y

    return build_matrix


# Function to patch neuron data class
def patch_neuron_data_class():
    import compdec_v4pfclip_01.utils.structures.neuron_data as neuron_data
    neuron_data.NeuronData._align_on = patched_align_on
    neuron_data.NeuronData.get_align_on = patched_get_align_on
