# Task-Perspective-Het

## 📌 Overview

This repository contains the official implementation of the paper:

[**Redefining non-IID Data in Federated Learning for Computer Vision Tasks: Migrating from Labels to Embeddings for Task-Specific Data Distributions**](https://doi.org/10.48550/arXiv.2503.14553)\
Kasra Borazjani, Payam Abdisarabshali, Naji Khosravan, Seyyedali Hosseinalipour\
ArXiv Preprint, 2025

Federated Learning (FL) enables decentralized model training while preserving data privacy, but its performance is significantly impacted by the **non-IID nature** of data across clients. Traditionally, non-IID settings in FL have been defined based on **label distribution skew**. However, in this work, we demonstrate that this definition oversimplifies real-world heterogeneity, particularly in computer vision tasks beyond classification.

We introduce a **task-specific perspective** on data heterogeneity, leveraging **embedding-based distributions** rather than label distributions to model non-IID data. By extracting **task-specific data embeddings** using pre-trained deep neural networks, we redefine FL data heterogeneity at a more fundamental level. Our methodology clusters data points in embedding space and distributes them among clients using a **Dirichlet allocation strategy**, better reflecting the diversity encountered in real-world FL applications.

Through extensive experiments, we show that current FL approaches **overestimate performance** when relying solely on label distribution skew. We provide new benchmark performance measures and highlight open research directions to further advance FL in non-IID settings.

### 🔥 Key Contributions

- We challenge the conventional **label-based definition** of non-IID data in FL and propose a **task-specific embedding-based perspective**.
- We introduce a **new data partitioning methodology** that clusters data in embedding space, resulting in a more realistic non-IID distribution.
- We evaluate multiple FL algorithms under this setting and show that existing methods **struggle more than previously assumed**, exposing an important gap in the literature.
- We establish **new benchmark performance measures** for FL models under embedding-based heterogeneity and outline key research challenges.

---

## 📂 Repository Structure

```
task-perspective-het/
│── datasets/            # Dataset and preprocessing scripts
│── models/              # Model implementations
│── network/             # Training and evaluation scripts
│── utils/               # Helper functions and utilities
│── results/             # Logs, figures, and saved models
│── README.md            # This file
│── requirements.txt     # Required dependencies
│── extract_datapoints_for_experiments.py    # Extracts the datapoints that will be used in the experiments
|── extract_embeddings.py # Used to extract embeddings from trained centralized models
|── losses.py # Defining the losses used in training the tasks
|── model_initialization # Used to create the initial model weights used for each task
|── test_fed.py  # Testing the federated performance of a specific task
|── train_centralized_task_model.py # Training each task's centralized model to extract embeddings from
```

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/somerandomusernamelab/task-perspective-het.git
cd task-perspective-het
pip install -r requirements.txt
```

You will also have to install the CLIP module from the source using (prerequisites are included in the requirements.txt file):

```bash
pip install git+https://github.com/openai/CLIP.git
```



## 📊 Dataset

We have used the Taskonomy dataset in our experiments. To see how to download the dataset, please refer to the manual available [here](https://github.com/StanfordVL/taskonomy/tree/master/data). You can also use the alternative sample we have provided below to recreate our results:

```bash
# Make sure everything is installed
sudo apt-get install aria2
pip install 'omnidata-tools' # Just to make sure it's installed

# Install the 'debug' subset of the Replica and Taskonomy components of the dataset
omnitools.download rgb normal class_scene keypoints3d depth_euclidean reshading segment_semantic \
  --components taskonomy \
  --subset tiny \
  --name <insert name> --email <insert email> \
  --dest ./taskonomy_dataset/ --agree-all
```

## 🏋️ Training

To manually train the models, you should start with exporting the scenes from the Taskonomy dataset that are going to be used to train both the centralized and the federated models. To do so, you should run the following script:

```bash
python extract_datapoints_for_experiments.py\
    --client_file_name <name of the csv file> \
    --data_path <path to downloaded data> \
    --taskonomy_clients_to_include <number of scenes from taskonomy to include> \
    --scene_class_file_name <name of the scene class file>
```

The `client_file_name` argument points to the name of the csv file that includes the name of the scenes inside the taskonomy subset downloaded (tiny/medium/full/fullplus) which can be accessed [here](https://github.com/StanfordVL/taskonomy/raw/master/data/assets/splits_taskonomy.zip). The `taskonomy_clients_to_include` refers to the number of scenes which are going to be used in the experiments. We have used the first _three_ scenes in our experiments. The `scene_class_file_name` is referring to the label-encoded scene class labels which is included in this repository for reference as `places_class_names.txt`. However, it should be moved into the folder that the data is in (e.g, the `./taskonomy_dataset/` folder if you have used the exact code as provided to download the dataset).

To train the centralized models, run the following code:

```bash
python train_centralized_task_model.py \
--task <task_name> --n_epochs <number_of_epochs> --batch_size <batch_size> --use_accelerator True --init_lr 1e-2 --data_path <path_to_dataset>
```

After training the centralized models, you should extract the embeddings and cluster them using the following code

```bash
python extract_embeddings.py \
--task <task_name> --n_clusters <number_of_clusters> \
--observe_performance True --from_checkpoint True \
--load_model_dir <path_to_checkpoint>
```

You can then test the FL performance using the code

```bash
python test_fed.py \
--alpha <alpha> --task <task_name> --n_rounds <number_of_global_rounds> --n_clusters <number_of_clusters> --type <embedding_based/class_based> \
--n_clients <number_of_clients> --sgd_per_epoch <number_of_sgd_per_global_round> \
--init_lr <initial_learning_rate> --assignment_files_path <path_to_assignment_file>
```

We recommend using a `--init_lr` of 1e-5 for the classification task and 1e-1 for all the other tasks. Also, depending on the dataset and configuration, training time may vary. We recommend using a **GPU-enabled environment** for optimal performance. To do so, you can add the `--use_accelerator True` argument to the commands to run each script when applicable.


## 📜 Citation

If you find this work useful, please consider citing:

```bibtex
@article{borazjani25redefining,
  author = {Kasra Borazjani, Payam Abdisarabshali, Naji Khosravan, Seyyedali Hosseinalipour},
  title = {Redefining non-IID Data in Federated Learning for Computer Vision Tasks: Migrating from Labels to Embeddings for Task-Specific Data Distributions},
  journal = {ArXiv Preprint},
  year = {2025},
  doi = {https://doi.org/10.48550/arXiv.2503.14553}
}
```

## 📌 Contact

For questions, feel free to open an issue or contact: kasrabor@buffalo.edu
