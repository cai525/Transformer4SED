# AudioSet Strong

The AudioSet Strong dataset is an enhanced version of the original AudioSet, offering fine-grained, temporally precise annotations of sound events.

---

## 1. Key Features

- **Strong Labels**:  
  Unlike the original AudioSet which only provides weak labels (i.e., presence of sound events without precise timestamps), this version includes accurate **start and end times** for each sound event within a clip.

- **Annotation Process**:  
  Human annotators marked every distinguishable sound event using a hierarchical vocabulary (AudioSet Ontology), ultimately producing **456 distinct labels**.

- **Dataset Composition**:  
  - 16,996 annotated clips from the **evaluation set**  
  - 103,463 annotated clips from the **training set**  
  - A total of **934,821 sound events** annotated in the training set

---

## 2. Description of Meta Files

### 2.1 Train & Validation

The original metadata files for training and validation are stored in the `train` and `val` directories, respectively:

- According to the [dataset maintainers' discussion](https://github.com/audioset/ontology/issues/9), the training set includes 447 sound event classes, and the validation set includes 416.

- The intersection of event classes between train and validation sets contains 407 classes. The [val_407](./val_407/) folder contains validation annotations filtered to retain only those 407 categories.

- The file [`validation_no_overlapped.tsv`](./val/validation_no_overlapped.tsv) merges overlapping events in the validation set to avoid issues during PSDS score computation, which may fail if events overlap.

---

### 2.2 Label Enhancement via AudioSet Ontology

Not all labels in the AudioSet Strong dataset are originally included in the AudioSet Ontology. To ensure consistency and ontology compatibility, we re-partitioned the dataset:

- All classes in the filtered dataset are guaranteed to exist in the AudioSet Ontology；
- All event types in the validation set also exist in the training set

Then, we enhanced the labels by adding all ancestor categories (hypernyms) from the ontology to each event label. After this process, the following subsets were created:

- [train.tsv](./hierarchical/train.tsv):  Contains 407 classes, 93,927 audio clips — all event types are covered by the AudioSet Ontology.

- [dropped_train.tsv](./hierarchical/dropped_train.tsv):  The remainder of the original training set excluding the above train.tsv. Typically unused in experiments.

- [val.tsv](./hierarchical/val.tsv): Contains 381 classes, 14,203 audio clips — all events are guaranteed to exist in the training set.

- [dropped_val.tsv](./hierarchical/dropped_val.tsv): Complementary subset of the original validation set, containing 2,697 audio clips and 37 unique event types not present in val.tsv.  

## 2.3 AS-partial
The event duration distribution in AudioSet Strong exhibits a typical long-tail pattern. In the experiments, we classify sound events into common and rare classes based on their total duration in the training set, using a threshold of six minutes, where 99 events with a duration of less than six minutes are designated as rare, while the remaining events are categorized as common. The divided result is saved in [./state.json](./state.json) 

For open-vocabulary experiments,we follow the partial-label evaluation strategy, treating common classes as base classes, while keeping rare classes  as novel classes.The model is trained only on common classes with all rare class labels removed to ensure zero exposure, while evaluation is conducted on the full label set including both common and rare classes.This setting is denoted as AS-partial.