# Reviewer Feedback Fixes

## 1. Title and multiple attack types

### What the current model actually does
The bundled trained model is **binary classification: BENIGN vs DDoS**. The raw file used by the project is the DDoS-specific CICIDS2017 capture slice, so other attack labels are not present for the model to learn.

### What was changed
The project title and dashboard language now use **Network Traffic Classification & Attack Detection** and explicitly state: **Current benchmark = BENIGN vs DDoS; multi-attack extension requires the combined CICIDS2017 attack files.** This avoids implying that the present model already classifies every attack family.

The expanded CICIDS2017 dataset contains multiple attack families distributed across separate capture files, including FTP-Patator, SSH-Patator, several DoS variants, Web attacks, Infiltration, Bot and PortScan in addition to DDoS.

## 2. 99.99% accuracy

The reported 99.9955% is **internally consistent** with the supplied confusion matrix: 19,019 true negatives + 25,601 true positives out of 44,622 holdout flows gives 99.9955% accuracy.

The project no longer presents this as real-world accuracy. It is labeled as an **internal group-aware holdout benchmark**. The important limitation is that the source does not contain a timestamp/session/flow-ID field, so the project cannot verify generalization to a different capture session or contemporary network environment.

A bundled 6,000-flow sanity check also remains highly separable, but it is explicitly marked as a sample-level check, not external validation.

## 3. Why two features dominate

The model still receives 30 selected features. The current XGBoost native importance assigns about **98.67%** of the total importance to the top two features.

This concentration is partly explained by redundant feature definitions. In the bundled sample, examples include:

- `Total Length of Fwd Packets` = `Subflow Fwd Bytes`
- `Subflow Bwd Bytes` = `Total Length of Bwd Packets`
- `Avg Bwd Segment Size` = `Bwd Packet Length Mean`
- `Fwd Header Length` = `Fwd Header Length.1`
- `Avg Fwd Segment Size` = `Fwd Packet Length Mean`
- `Subflow Fwd Packets` = `Total Fwd Packets`

With redundant copies available, a tree can repeatedly split on one copy and give the other little or no native importance. That does **not** mean the other traffic features are irrelevant. The mutual-information ranking shows many additional features carrying signal.

### Presentation wording to use
> "The model uses 30 traffic features. Two packet-size/flow-volume variables dominate the XGBoost gain for this dataset, while the remaining features provide supporting signal. Because CICFlowMeter contains redundant feature definitions, native tree importance can concentrate on one member of a duplicate pair. We therefore treat feature importance as model- and dataset-specific, not as a claim that only two measurements matter in every network."
