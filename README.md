# CPDM: Boosting CT to PET Translation with Domain-Knowledge-Guided Diffusion Model

**Authors:** Dac Thai Nguyen, Trung Thanh Nguyen, Huu Tien Nguyen, Thanh Trung Nguyen, Huy Hieu Pham, Phi Le Nguyen.

***
CPDM utilizes a Brownian Bridge process-based diffusion model to directly learn the translation process from the CT domain to the PET domain, thereby reducing the stochasticity typically encountered in generative models.

![img](resources/CPDM_architecture.png)

## Requirements
```commandline
cond env create -f environment.yml
conda activate CPDM
```

## Data preparation
The path of paired image dataset should be formatted as:
```yaml
your_dataset_path/train/A  # training reference
your_dataset_path/train/B  # training ground truth
your_dataset_path/val/A  # validating reference
your_dataset_path/val/B  # validating ground truth
your_dataset_path/test/A  # testing reference
your_dataset_path/test/B  # testing ground truth
```

## Train and test Segmentation Model for Object Detection in CT image
### Train your Segmentation Model
Specity your checkpoint path to save model and dataset path in <font color=violet><b>train_segmentation_model.py</b></font>. Run below command to train model.
```commandline
python train_segmentation_model.py
```
### Test your Segmentation Model
Specity your checkpoint path, dataset path and sampling path in <font color=violet><b>test_segmentation_model.py</b></font>. Run below command for sampling and saving results to your path.
```commandline
python test_segmentation_model.py
```
Note that you can modify this code for training, validation or testing sampling.

## Train and test CPDM
### Specify your configuration file
Modify the configuration file based on our templates in <font color=violet><b>configs/Template-CPDM.yaml</b></font>. Don't forget to specify your VQGAN checkpoint path, dataset path and corresponding training and validation/testing sampling path of your Segmentation Model.

Note that you need to train your VQGAN (https://github.com/CompVis/taming-transformers) and sample results of Segmentation Model before starting training CPDM.
### Run
Specity your shell file based on our templates in <font color=violet><b>configs/Template-shell.sh</b></font>. Run below command to train or test model.
```commandline
sh shell/your_shell.sh
```

## Acknowledgement
Our code is implemented based on Brownian Bridge Diffusion Model (https://github.com/xuekt98/BBDM)  

### Citation
If you find this code useful for your research, please cite the following paper:

```
@inproceedings{nguyent2024CPDM,
  title={CPDM: Boosting CT to PET Translation with Domain-Knowledge-Guided Diffusion Model},
  author={Dac Thai Nguyen, Trung Thanh Nguyen, Huu Tien Nguyen, Thanh Trung Nguyen, Huy Hieu Pham, Phi Le Nguyen},
  year={2024}
}
```