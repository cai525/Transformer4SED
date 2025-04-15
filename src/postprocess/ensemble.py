import os
import pandas as pd
import numpy as np
import torch


def load_predictions(subfolders):
    """Load all predictions from subfolders into a dictionary."""
    predictions = {}

    # Check if all subfolders contain the same set of files
    all_files = None
    for subfolder in subfolders:
        files = {file for file in os.listdir(subfolder) if file.endswith(".tsv")}
        if all_files is None:
            all_files = files
        else:
            if files != all_files:
                raise ValueError(f"Subfolder {subfolder} contains different set of files.")

    # Load predictions
    for subfolder in subfolders:
        model_name = os.path.basename(subfolder)
        for file in all_files:
            file_path = os.path.join(subfolder, file)
            if file not in predictions:
                predictions[file] = []
            predictions[file].append((model_name, pd.read_csv(file_path, sep='\t')))

    return predictions


def weighted_average_ensemble(predictions, weights):
    """Perform weighted average ensemble on the predictions."""
    ensemble_predictions = {}
    for file, models_dfs in predictions.items():
        dfs = [df for _, df in models_dfs]
        stacked_df = [df.values for df in dfs]
        max_t = max([df.shape[0] for df in dfs])
        max_id = [i for i in range(len(stacked_df)) if stacked_df[i].shape[0] == max_t][0]
        for i, df in enumerate(stacked_df):
            if df.shape[0] < max_t:
                tensor = torch.Tensor(df).transpose(0, 1).unsqueeze(0)
                tensor = torch.nn.functional.interpolate(tensor, scale_factor=max_t / df.shape[0],
                                                         mode="linear").squeeze(0).transpose(0, 1)
                stacked_df[i] = tensor.numpy()
                stacked_df[i][:, :2] = stacked_df[max_id][:, :2]

        stacked_df = np.array(stacked_df)
        weighted_sum = np.tensordot(stacked_df, weights, axes=(0, 0))
        averaged_predictions = np.round(weighted_sum / np.sum(weights), decimals=4)
        ensemble_predictions[file] = pd.DataFrame(averaged_predictions, columns=dfs[0].columns)

    return ensemble_predictions


def save_predictions(predictions, output_folder):
    """Save the ensemble predictions to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file, df in predictions.items():
        output_path = os.path.join(output_folder, file)
        df.to_csv(output_path, sep='\t', index=False)


def ensemble(root, output_dir, model_list, weights):
    assert abs(1.0 - sum(weights) <= 0.0001)
    # Ensure weights match the number of models
    if len(weights) != len(model_list):
        raise ValueError("The number of weights must match the number of models.")
    subfolders = [os.path.join(root, subfolder) for subfolder in model_list]
    predictions = load_predictions(subfolders)
    print("bengin to output weighted scores")
    ensemble_predictions = weighted_average_ensemble(predictions, weights)
    save_predictions(ensemble_predictions, output_dir)


if __name__ == "__main__":
    """ Ensemble the results in the root_dir. Make sure that the structure of the root_dir is like:
        root_dir/
            model1/
                file1.tsv
                file2.tsv
                ...
            model2/
                file1.tsv
                file2.tsv
                ...
            ...
    """
    root_dir = 'ROOT-PATH/exps/dcase2024/ensemble/val/sub'  # Path to the main folder containing subfolders with prediction results
    output_dir = 'ROOT-PATH/exps/dcase2024/ensemble/val/res/0.7-0.3'  # Path to the output folder to save the ensemble results
    model_list = ["cnn-trans", "atst-cnn"]
    weights = np.array([0.7, 0.3])  # Example weights for each model's predictions

    ensemble(root_dir, output_dir, model_list, weights)
