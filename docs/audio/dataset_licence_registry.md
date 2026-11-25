# PathPilot Audio ML — Dataset Acquisition & Licence Registry
**Owner:** Debajyoti (Audio ML Lead)  
**Last Audited:** 20 November 2026  
**Status:** Audited & Verified for Prototype Demonstration  

---

## 1. Regulatory & Licence Governance
The PathPilot project operates under strict data provenance and licence compliance standards. No raw audio files, copyrighted samples, or proprietary recordings are committed directly to the Git repository.

Every dataset integrated into the audio cue ML subsystem has been audited for:
1. Legal licence terms and commercial vs non-commercial restrictions.
2. Original academic attribution requirements.
3. Integrity checksum verification.
4. Data partitioning hygiene (zero data leakage across splits).

---

## 2. Audited Datasets

### A. ESC-50: Dataset for Environmental Sound Classification
- **Dataset ID:** `DATASET_ESC50_V2`
- **Curator / Author:** Karol J. Piczak
- **Official Source URL:** https://github.com/karolpiczak/ESC-50
- **Version:** 2.0.0
- **Licence:** Creative Commons Attribution-NonCommercial 3.0 (CC BY-NC 3.0)
- **Permitted Use:** Non-commercial research, academic evaluation, and physical prototype demonstrations.
- **Commercial Caveat:** The CC BY-NC licence prohibits commercial deployment or sale of models directly derived from ESC-50 without explicit commercial licensing from original Freesound authors. Any future commercial production release must transition exclusively to CC0/commercial-cleared sources.
- **Attribution Requirement:**  
  *Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification. Proceedings of the 23rd ACM International Conference on Multimedia, pp. 1015–1018.*
- **Storage Location (Outside Git):** `data/raw/audio/esc50/`
- **Audit Findings:** 2,000 clips in 5 pre-arranged folds. Cleanly mapped classes: `screaming` -> `screaming_shouting`, `siren` -> `alarm_siren`, `glass_breaking` -> `glass_breaking`.

---

### B. UrbanSound8K
- **Dataset ID:** `DATASET_URBANSOUND8K_V1`
- **Curators / Authors:** Justin Salamon, Christopher Jacoby, and Juan Pablo Bello
- **Official Source URL:** https://urbansounddataset.weebly.com/urbansound8k.html
- **Version:** 1.0.0
- **Licence:** Creative Commons Attribution-NonCommercial 3.0 (CC BY-NC 3.0)
- **Permitted Use:** Research, teaching, and experimental prototyping.
- **Commercial Caveat:** CC BY-NC restriction applies. Must not be bundled into a paid proprietary app.
- **Attribution Requirement:**  
  *Salamon, J., Jacoby, C., and Bello, J. P. (2014). A Dataset and Taxonomy for Urban Sound Research. 22nd ACM International Conference on Multimedia, pp. 1041–1044.*
- **Storage Location (Outside Git):** `data/raw/audio/urbansound8k/`
- **Audit Findings:** 8,732 urban sound excerpts across 10 classes in 10 predefined folds. Folds guarantee no slice from the same recording appears across different folds. Mapped classes: `siren` -> `alarm_siren`, `car_horn` -> `alarm_siren`, ambient sounds -> `background_normal`. Note: `gun_shot` is excluded as outside target urban transport taxonomy.

---

### C. FSD50K: Freesound Dataset 50K
- **Dataset ID:** `DATASET_FSD50K_V1`
- **Curators / Authors:** Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, and Xavier Serra (MTG-UPF)
- **Official Source URL:** https://zenodo.org/record/4060432
- **Version:** 1.0.0
- **Licence:** Mixed per-clip Freesound Licences (Creative Commons CC0, CC-BY 3.0, CC-BY 4.0, CC-BY-NC).
- **Permitted Use:** Requires clip-level allowlist filtering.
- **Commercial Caveat:** Only clips licensed under CC0 (Public Domain) or CC-BY (with author attribution) are cleared for commercial unconstrained prototypes. All CC-BY-NC clips are quarantined or marked for non-commercial evaluation only.
- **Attribution Requirement:**  
  *Fonseca, E., Favory, X., Pons, J., Font, F., and Serra, X. (2021). FSD50K: an open dataset of human-labeled sound events. IEEE/ACM Transactions on Audio, Speech, and Language Processing.*
- **Storage Location (Outside Git):** `data/raw/audio/fsd50k/`
- **Audit Findings:** 51,197 audio clips with verified sound ontology. Rigorous allowlist in `ml/audio/registry/licence_manifest.csv` tracks each clip's licence.

---

### D. AudioSet (Evaluation Reference Subset)
- **Dataset ID:** `DATASET_AUDIOSET_V1`
- **Curator / Author:** Google Research (Gemmeke et al.)
- **Official Source URL:** https://research.google.com/audioset/
- **Licence:** Annotations CC BY 4.0; underlying audio hosted on YouTube under YouTube Terms of Service.
- **Permitted Use:** Evaluation ontology cross-reference and benchmark comparison only.
- **Redistribution Policy:** Raw YouTube audio cannot be redistributed in PathPilot repositories or datasets. Only external fetch scripts with authorised API keys are permitted.
- **Storage Location (Outside Git):** `data/raw/audio/audioset/`

---

## 3. Storage & Manifest Verification
The full audit register is stored in machine-readable CSV format:
- `ml/audio/registry/dataset_acquisition_register.csv`
- `ml/audio/registry/licence_manifest.csv`

All clips loaded into training or testing are verified against `included == TRUE` before preprocessing.
