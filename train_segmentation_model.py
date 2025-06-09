from datasets.SegmentedPETCTDataset_png import SegmentedPETCTDataset
import segmentation_models_pytorch as smp
import argparse
import omegaconf 
import matplotlib.pyplot as plt
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

CHECKPOINT_PATH = 'checkpoints'
DATA_PATH = '/kaggle/input/autopet-png'
IMAGE_SIZE = 256
CT_MAX = 1
PET_MAX = 1
BATCH_SIZE = 16

class SegmentationModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        
        # Initialize lists to store outputs for each epoch
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # normalize image here
        # image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch[0]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch, "train")
        self.training_step_outputs.append(output)
        return output

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()  # Clear the list for next epoch

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()  # Clear the list for next epoch

    def test_step(self, batch, batch_idx):
        output = self.shared_step(batch, "test")
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()  # Clear the list for next epoch

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def get_image_paths_from_dir(fdir):
    flist = os.listdir(fdir)
    flist.sort()
    image_paths = []
    for i in range(0, len(flist)):
        fpath = os.path.join(fdir, flist[i])
        if os.path.isdir(fpath):
            image_paths.extend(get_image_paths_from_dir(fpath))
        else:
            image_paths.append(fpath)
    return image_paths

def get_dataset_by_stage(data_path, stage, image_size, ct_max_pixel, pet_max_pixel, flip):
    ct_paths = get_image_paths_from_dir(os.path.join(data_path, f'{stage}/A'))
    pet_paths = get_image_paths_from_dir(os.path.join(data_path, f'{stage}/B'))

    return SegmentedPETCTDataset(ct_paths, pet_paths, image_size, ct_max_pixel, pet_max_pixel, flip)
    
def main():
    train_dataset = get_dataset_by_stage(DATA_PATH, 'train', (IMAGE_SIZE, IMAGE_SIZE), CT_MAX, PET_MAX, True)
    val_dataset = get_dataset_by_stage(DATA_PATH, 'val', (IMAGE_SIZE, IMAGE_SIZE), CT_MAX, PET_MAX, False)
    test_dataset = get_dataset_by_stage(DATA_PATH, 'test', (IMAGE_SIZE, IMAGE_SIZE), CT_MAX, PET_MAX, False)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model_name = "Unet"
    encoder_name = "resnet50"
    
    # Check for existing checkpoints
    checkpoint_dir = '/kaggle/working/CPDM/checkpoints/Unet/lightning_logs/version_0/checkpoints'
    checkpoint_path = None
    start_epoch = 0

    if os.path.exists(checkpoint_dir):
        # Look for the last checkpoint
        last_checkpoint = os.path.join(checkpoint_dir, "last.ckpt")
        if os.path.exists(last_checkpoint):
            checkpoint_path = last_checkpoint
            print(f"Found last checkpoint: {last_checkpoint}")
            # For last.ckpt, we need to load it to get the epoch
            try:
                # Load just the checkpoint metadata to extract epoch
                ckpt = torch.load(last_checkpoint, map_location='cpu')
                start_epoch = ckpt.get('epoch', 0)
                print(f"Last checkpoint is from epoch {start_epoch}")
            except Exception as e:
                print(f"Warning: Could not extract epoch from checkpoint: {e}")
                # If we can't extract the epoch, assume it's the latest
                start_epoch = 57  # Based on your final-epoch filename
        else:
            # If no last.ckpt, look for checkpoints in the format final-epoch-epoch=XX.ckpt
            final_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("final-epoch-") and f.endswith(".ckpt")]
            if final_checkpoints:
                latest_final = os.path.join(checkpoint_dir, final_checkpoints[-1])
                checkpoint_path = latest_final
                # Extract epoch number from filename using regex to find the number after "epoch="
                import re
                epoch_match = re.search(r'epoch=(\d+)', final_checkpoints[-1])
                if epoch_match:
                    start_epoch = int(epoch_match.group(1))
                print(f"Found final checkpoint: {latest_final} (Epoch: {start_epoch})")
            else:
                # Look for periodic checkpoints in the format epoch-epoch=XX.ckpt
                periodic_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch-") and f.endswith(".ckpt")]
                if periodic_checkpoints:
                    # Sort by epoch number
                    periodic_checkpoints.sort(key=lambda x: int(re.search(r'epoch=(\d+)', x).group(1)))
                    latest_periodic = os.path.join(checkpoint_dir, periodic_checkpoints[-1])
                    checkpoint_path = latest_periodic
                    # Extract epoch number from filename
                    epoch_match = re.search(r'epoch=(\d+)', periodic_checkpoints[-1])
                    if epoch_match:
                        start_epoch = int(epoch_match.group(1))
                    print(f"Found periodic checkpoint: {latest_periodic} (Epoch: {start_epoch})")
                else:
                    # Look for best checkpoints in the format best-epoch=XX-valid_dataset_iou=X.XX.ckpt
                    best_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("best-") and f.endswith(".ckpt")]
                    if best_checkpoints:
                        # Sort by epoch number
                        best_checkpoints.sort(key=lambda x: int(re.search(r'epoch=(\d+)', x).group(1)))
                        latest_best = os.path.join(checkpoint_dir, best_checkpoints[-1])
                        checkpoint_path = latest_best
                        # Extract epoch number from filename
                        epoch_match = re.search(r'epoch=(\d+)', best_checkpoints[-1])
                        if epoch_match:
                            start_epoch = int(epoch_match.group(1))
                        print(f"Found best checkpoint: {latest_best} (Epoch: {start_epoch})")
    
    # Initialize model - either fresh or from checkpoint
    if checkpoint_path:
        print(f"Resuming training from checkpoint at epoch {start_epoch}")
        model = SegmentationModel(model_name, encoder_name, in_channels=1, out_classes=1)
        
        trainer = pl.Trainer(
            accelerator="gpu", 
            devices=1,
            max_epochs=100,
            default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
            resume_from_checkpoint=checkpoint_path,  # This tells PyTorch Lightning to resume training
            callbacks=[
                # Same callbacks as before
                ModelCheckpoint(
                    save_weights_only=True, 
                    mode="max", 
                    monitor="valid_dataset_iou",
                    filename="best-{epoch:02d}-{valid_dataset_iou:.2f}",
                    save_top_k=1
                ),
                ModelCheckpoint(
                    save_weights_only=True,
                    every_n_epochs=20,
                    filename="epoch-{epoch:02d}",
                    save_top_k=-1
                ),
                ModelCheckpoint(
                    save_weights_only=True,
                    save_last=True,
                    filename="final-epoch-{epoch:02d}"
                ),
                LearningRateMonitor("epoch"),
            ],
        )
    else:
        print("Starting training from scratch")
        model = SegmentationModel(model_name, encoder_name, in_channels=1, out_classes=1)
        
        trainer = pl.Trainer(
            accelerator="gpu", 
            devices=1,
            max_epochs=100,
            default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
            callbacks=[
                # Same callbacks as before
                ModelCheckpoint(
                    save_weights_only=True, 
                    mode="max", 
                    monitor="valid_dataset_iou",
                    filename="best-{epoch:02d}-{valid_dataset_iou:.2f}",
                    save_top_k=1
                ),
                ModelCheckpoint(
                    save_weights_only=True,
                    every_n_epochs=20,
                    filename="epoch-{epoch:02d}",
                    save_top_k=-1
                ),
                ModelCheckpoint(
                    save_weights_only=True,
                    save_last=True,
                    filename="final-epoch-{epoch:02d}"
                ),
                LearningRateMonitor("epoch"),
            ],
        )

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    trainer.fit(
        model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=valid_dataloader,
    )
    
if __name__ == '__main__':
    main()
    