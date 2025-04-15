import argparse
import glob
import os
import warnings
from pathlib import Path
from pprint import pformat

import yaml
from sed_scores_eval import io


expected_event_classes = {
    "Alarm_bell_ringing",
    "Blender",
    "Cat",
    "Dishes",
    "Dog",
    "Electric_shaver_toothbrush",
    "Frying",
    "Running_water",
    "Speech",
    "Vacuum_cleaner",
    "cutlery and dishes",
    "furniture dragging",
    "people talking",
    "children voices",
    "coffee machine",
    "footsteps",
    "large_vehicle",
    "car",
    "brakes_squeaking",
    "cash register beeping",
    "announcement",
    "shopping cart",
    "metro leaving",
    "metro approaching",
    "door opens/closes",
    "wind_blowing",
    "birds_singing",
}


def get_data_yaml(yaml_path):
    with open(yaml_path, 'r', encoding="utf-8") as stream:
        # Problem with tab in files, so making sure there is not at the end
        lines = []
        for a in stream.readlines():
            line = a.split("#")[0]
            line = line.rstrip()
            lines.append(line)

        # Read YAML
        data = yaml.safe_load("\n".join(lines))
    return data


def _validate_general(submission):
    if submission["label"] in ["Cornell_CMU_task4_1"]:
        raise ValueError("Please change the label of the submission with your name")
    for key in ["label", "name", "abbreviation"]:
        if "baseline" in submission[key].lower():
            raise ValueError("Please do not put 'baseline' in your system label, name or abbreviation")


def _validate_authors(list_authors):
    corresponding = False
    for author in list_authors:
        if author.get("corresponding") is not None:
            corresponding = True
        if author.get("firstname") is None or author.get("lastname") is None:
            raise ValueError("An author need to have a first name and a last name")

    if not corresponding:
        raise ValueError("Please put a corresponding author")


def _validate_system(system):
    if not isinstance(system["description"]["input_sampling_rate"], (int, float)):
        raise TypeError("The sampling rate needs to be a number (float or int)")

    ac_feat = system["description"]["acoustic_features"]
    if ac_feat is not None:
        if not isinstance(ac_feat, list):
            assert isinstance(ac_feat, str), "acoustic_features is a string if not a list"
            ac_feat = [ac_feat]
        common_values = ["mfcc", "log-mel energies", "log-mel amplitude", "spectrogram", "CQT", "raw waveform"]
        for ac_f in ac_feat:
            if ac_f.lower() not in common_values:
                warnings.warn(f"Please check you don't have a typo if "
                              f"you use common acoustic features: {common_values}")

    if not isinstance(system["complexity"]["total_parameters"], int):
        raise TypeError("the number of total_parameters needs to be an integer")

    if system["source_code"] == "https://github.com/turpaultn/dcase20_task4/tree/public_branch/baseline":
        raise ValueError("If you do not share your source code, please put '!!null'")


def _validate_ss_system(system):
    if system["ensemble_method_subsystem_count"] is not None:
        if not isinstance(system["ensemble_method_subsystem_count"], (int, float)):
            raise TypeError("The ensemble_method_subsystem_count needs to be a number (float or int)")
    if system["source_code"] == "https://github.com/google-research/sound-separation/tree/master/models/dcase2020_fuss_baseline":
        raise ValueError("If you do not share your source code, please put '!!null'")


def _validate_results(results):
    devtest_results = results["devtest"]
    if not isinstance(devtest_results["desed"]["PSDS1"], float):
        raise TypeError("The PSDS1 on desed devtest set needs to be a float")
    if not isinstance(devtest_results["maestro"]["mPAUC"], float):
        raise TypeError("The mPAUC on maestro devtest set needs to be a float")

    # per_class = results["development_dataset"]["class_wise"]
    # for label in per_class:
    #     if not isinstance(per_class[label]["F-score"], (int, float)):
    #         raise TypeError("The F-score on development set needs to be a float or integer")


def _validate_ss_results(results):
    for dataset in results:
        for result in results[dataset]:
            if not isinstance(results[dataset][result], (int, float)):
                raise TypeError(f"The {result} on {dataset} set needs to be a float or integer")


def validate_yaml(yaml_path):
    system_name = yaml_path.name.split(".")[0]
    dict_data = get_data_yaml(yaml_path)
    assert dict_data["submission"]["label"] == system_name, f"system label in yaml ({dict_data['submission']['label']}) does not match folder name {system_name}"
    _validate_general(dict_data["submission"])
    _validate_authors(dict_data["submission"]["authors"])
    _validate_system(dict_data["system"])
    _validate_results(dict_data["results"])


def validate_system(system_dir):
    system_name = system_dir.name
    assert len(system_name.split("_")) == 4, "System names [{system_name}] must comply with the format: <author_name>_<affiliation>_task4_<system_index> (e.g., Cornell_CMU_task4_1)".format(system_name=system_name)
    yaml_path = system_dir / f"{system_name}.meta.yaml"
    assert yaml_path.exists(), f"{system_dir} does not contain {yaml_path}"
    validate_yaml(yaml_path)
    print(f"{yaml_path} is validated, continuing...")
    for run_idx in range(1, 4):
        output_path = system_dir / f"{system_name}_run{run_idx}.output"
        assert output_path.exists(), f"{system_dir} does not contain {output_path}."
        validate_output(output_path)
        unprocessed_output_path = system_dir / f"{system_name}_run{run_idx}_unprocessed.output"
        if unprocessed_output_path.exists():
            validate_output(unprocessed_output_path)
        else:
            warnings.warn(f"Could not find {unprocessed_output_path}.")
    print(f"{system_dir} is validated, continuing...")


def validate_output(run_output_path):
    tsv_files = sorted(run_output_path.glob("*.tsv"))
    assert len(tsv_files) == 2200, f"{run_output_path} must contain a tsv file for each of the 2200 evaluation clips but only {len(tsv_files)} found."
    example_clip_scores = io.read_sed_scores(tsv_files[1000])
    _, event_classes = io.validate_score_dataframe(example_clip_scores)
    for expected_event_class in expected_event_classes:
        assert expected_event_class in event_classes, f"tsv file doesn't contain expected sound class {expected_event_class}."
    print(f"{run_output_path} is validated, continuing...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default='.', type=str,
                        help="Submission dir to be validated.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).absolute()
    print(input_dir)
    system_count = 0
    for file in sorted(input_dir.iterdir()):
        if file.is_dir():
            validate_system(file)
            system_count += 1
    assert system_count > 0, f"No systems found in {input_dir}."
    assert system_count <= 8, "You must not submit more than 8 systems"

    pdf_files = glob.glob(os.path.join(args.input_dir, "*.pdf"))
    if len(pdf_files) == 0:
        raise IndexError("You need to upload a report in your submission")

    with open(os.path.join(args.input_dir, "validated"), "w") as f:
        f.write("Submission validated")

    print("Submission validated")
