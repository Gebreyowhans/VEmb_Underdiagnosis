# CXR_Embedding_Fairness , Underdiagnosis bias mitigation with expert foundation model's representation

## Dataset access:

Both the MIMIC-CXR and CheXpert datasets, in their image and vector embedding forms, are publicly available under respective data use agreements.

MIMIC-CXR image  dataset is available at: https://physionet.org/content/mimic-cxr/2.0.0/

MIMIC-CXR embeddings  dataset is available at: https://physionet.org/content/image-embeddings-mimic-cxr/1.0/

CheXpert image dataset is available at: https://stanfordmlgroup.github.io/competitions/chexpert/

CheXpert embeddings dataset can be requested by filling out the [CXR Foundation Access Form](https://docs.google.com/forms/d/e/1FAIpQLSek0P-JSwSfonIiZJlz7gOTbL0lugsDug0FUnMhS1zVzpEKlg/viewform) from the authors of the CXR foundation model.

Access to both datasets requires user registration and the signing of a data use agreement. Only the MIMIC-CXR dataset requires the completion of an additional credentialing process. After following these procedures, the MIMIC-CXR data is available through PhysioNet (https://physionet.org/). The race/ethnicities and insurance type of the patients are not provided directly with the download of the MIMIC-CXR dataset. However, this data is available through merging the patient IDs in MIMIC-CXR with subject IDs in MIMIC-IV (https://physionet.org/content/mimiciv/0.4/) datasets, using the patient and admissions tables. Access to MIMIC-IV requires a similar procedure as MIMIC-CXR and the same credentialing process is applicable for both datasets. 

----------------------------------------------------------------------------------------------------------------------------
## Reproducing the results:

### Model Configuration : CXR_Emb

This part uses YAML configuration files located in the `./configs` directory to handle training, testing, and inference. Update the paths and parameters in these `.yaml` files to match your setup before execution.

Model Configuration files for this `CXR_Emb` are located in the `./configs` configs directory under CXR_Emb/Training and evalaution  folder. 
---
#### Training
To train the model, use the following command. Replace ***.yaml with the name of your specific configuration file, and execute for different seeds. The seed numbers we used are [19,31,38,47,77]

```bash
`python main.py fit -c ./configs/***_config.yaml `

```

`***_config.yaml` should be replaced with actual config file name.

---

#### Testing
After training, the model can be tested  by specifying the path to the saved checkpoint and the configuration file used for training:

```bash
`python main.py test -c ./configs/***_config.yaml --ckpt_path ./path_to_saved_checkpoint `
```
Use the same YAML config used during training.

---

#### 🔄 About `prediction_on`

The `prediction_on` setting (in config) select which split (`train`, `val`, or `test`) the `predict` command runs on. It save prediction and will save them in the result folder:

| Value   | Files Saved                                     |
|---------|-------------------------------------------------|
| `train` | `probabilities_train.npy`, `labels_train.npy`   |
| `val`   | `probabilities_val.npy`, `labels_val.npy`       |
| `test`  | `probabilities_test.npy`, `labels_test.npy`     |

Make sure `save_probabilities_path` is set to enable saving.

You can override it on the CLI:

```bash
python main.py predict -c ./configs/your_config.yaml --ckpt_path ./path_to_checkpoint.ckpt
```

---

#### ⚙️ Notes

- Use `ConcatDataset` to train across MIMIC + CheXpert.
- Adjust `batch_size`, `num_workers`, and `transform` as needed.
- `prediction_on` determines the split used for prediction and what files are saved.

## 📈 Inference / Prediction

Run inference and optionally save probabilities + labels:

```bash
python main.py predict -c ./configs/***r_config.yaml --ckpt_path ./path_to_checkpoint.ckpt
```

#### Underdiagnosis analysis

The underdiagnosis analysis for the `CXR_Emb` dataset is available in the `CXR_Emb/Fairness` directory. This directory includes five subfolders, each corresponding to a different dataset variant. For each scenario, we calculate the False Positive Rate (FPR) across multiple random seeds, compute the average FPR over five runs with 95% confidence intervals, and visualize the results using the `FPR_Visualizations_**.ipynb` notebook, where `**` should be replaced with the relevant dataset name (e.g., `MIMIC` for the MIMIC dataset).

### Model For : BMC_Emb
 
## Training and testing 

The training and testing scripts for each scenario are located in the `**_Biomedclip.py` files within the `BMC_Emb` directory. This directory contains three subfolders corresponding to the `CheXpert`, `MIMIC`, and `ALL` datasets. Replace `**` with the appropriate dataset name (e.g., MIMIC for the MIMIC dataset).


#### Underdiagnosis analysis

The underdiagnosis analysis for the BMC_Emb dataset is available in the three subfolders within the `BMC_Emb` directory. For each scenario, the False Positive Rate (FPR) is computed across different random seeds. We then calculate the average FPR over five runs, along with 95% confidence intervals, and visualize the results using the `FPR_Visualizations_**.ipynb` notebook, where  `**` should be replaced with the corresponding dataset name (e.g., `MIMIC` for the MIMIC dataset).


### Race and Sex classifiers:

The race and sex classifiers for CXR_Emb, including both image-based and embedding-based models, are located in the `CXR_Emb` folder. Configuration files for the image-based models can be found under the `./configs` directory.


### Final remark:
 We are not able to share the trained model and the true label and predicted label CSV files of the test set due to the data-sharing agreement. However, we have provided the random seed, and the code. Then, the true label and predicted label CSV files and trained models can be generated by users who have downloaded the data from the original source following the procedure that is described in the “Data access” session.



