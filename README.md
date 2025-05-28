# PlantIF: Multimodal Semantic Interactive Fusion via Graph Learning for Plant Disease Diagnosis

## Environments

- **Set up our environment**

  ```bash
  conda env create -f environment.yml
  conda activate plantif
  ```

## Training

- **Step1** Enter the directory of the experiment

  ```bash
  cd /path/to/experiment
  ```

- **Step2** Download the PlantDM Datasets(Upload after the paper is accepted)

  ```
  The PlantDM dataset will be opened after the paper is accepted
  ```

- **Step3** Run the code of training

  ```python
  python train.py
  ```

## Testing

- **Step1** Download the PlantIF_test datasets

  ```
  https://drive.google.com/uc?export=download&id=1hNxfOG7CeMo2LSraU8XEG6bIWba3OLlj
  ```

- **Step2** Download PlantIF_weight weight

  ```
  https://drive.google.com/uc?export=download&id=11OZQyiA5uPW4C-uNTaPpYZtt7pSYRhIH
  ```

- **Step3** Move PlantIF_test and PlantIF_weight

  - **step3.1** Move `PlantIF_test` to `datasets` Folder

  - **step3.2** Move `PlantIF_weight` to `checkpoints` Folder

- **Step4** Run the code of testing

  ```python
  python test.py
  ```
