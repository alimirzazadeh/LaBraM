import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from spectrogram_cnn import TUABBaselineDataset, SpectrogramCNN, TUEVBaselineDataset
import torchmetrics
from tqdm import tqdm
from ipdb import set_trace as bp
import numpy as np
import random
import argparse

class MulticlassMetrics:
    """
    Metrics class for multiclass classification tasks.
    Computes AUROC, Loss, Balanced Accuracy, Cohen's Kappa, and Weighted F1.
    """
    
    def __init__(self, num_classes, device='cuda'):
        """
        Initialize metrics for multiclass classification.
        
        Args:
            num_classes (int): Number of classes in the classification task
            device (str): Device to run metrics on ('cuda' or 'cpu')
        """
        self.num_classes = num_classes
        self.device = device
        
        self.init_metrics()
    
    def init_metrics(self):
        """Initialize all metric calculators."""
        self.metrics_dict = {
            "balanced_acc": torchmetrics.Accuracy(
                task='multiclass',
                num_classes=self.num_classes,
                average='macro'
            ).to(self.device),
            
            "kappa": torchmetrics.CohenKappa(
                task='multiclass',
                num_classes=self.num_classes
            ).to(self.device),
            
            "f1_weighted": torchmetrics.classification.MulticlassF1Score(
                num_classes=self.num_classes,
                average='weighted'
            ).to(self.device),
            
            "auroc": torchmetrics.AUROC(
                task="multiclass",
                num_classes=self.num_classes,
                average='macro'
            ).to(self.device),
            
            "auprc": torchmetrics.AveragePrecision(
                task="multiclass",
                num_classes=self.num_classes,
                average='macro'
            ).to(self.device),
        }
    
    def update(self, predictions, labels):
        """
        Update metrics with new predictions and labels.
        
        Args:
            predictions (torch.Tensor): Raw logits/predictions of shape (batch_size, num_classes)
            labels (torch.Tensor): Ground truth labels of shape (batch_size,)
        """
        # Convert logits to class predictions for accuracy, kappa, and f1
        if predictions.shape[1] > 1:
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = (predictions > 0)
        labels = labels.long()
        
        # Update each metric
        self.metrics_dict["balanced_acc"].update(pred_classes, labels)
        self.metrics_dict["kappa"].update(pred_classes, labels)
        self.metrics_dict["f1_weighted"].update(pred_classes, labels)
        self.metrics_dict["auroc"].update(predictions, labels)
        self.metrics_dict["auprc"].update(predictions, labels)
    def compute(self, loss=None):
        """
        Compute all metrics and return as dictionary.
        
        Args:
            loss (float, optional): Loss value to include in metrics
            
        Returns:
            dict: Dictionary containing all computed metrics
        """
        results = {
            "balanced_accuracy": self.metrics_dict["balanced_acc"].compute().item(),
            "cohens_kappa": self.metrics_dict["kappa"].compute().item(),
            "weighted_f1": self.metrics_dict["f1_weighted"].compute().item(),
            "auroc": self.metrics_dict["auroc"].compute().item(),
            "auprc": self.metrics_dict["auprc"].compute().item(),
        }
        
        if loss is not None:
            results["loss"] = loss
        
        return results
    
    def reset(self):
        """Reset all metrics to initial state."""
        for metric in self.metrics_dict.values():
            metric.reset()
    
    def compute_and_reset(self, loss=None):
        """
        Compute metrics, get results, then reset.
        Convenience method for end-of-epoch evaluation.
        
        Args:
            loss (float, optional): Loss value to include in metrics
            
        Returns:
            dict: Dictionary containing all computed metrics
        """
        results = self.compute(loss=loss)
        self.reset()
        return results


def logger(writer, metrics, phase, epoch_index):

    for key, value in metrics.items():
        ## check for 2 class multiclass
        if type(value)!= float and len(value.shape) > 0 and value.shape[0] == 2:
            value = value[1]
        elif type(value)!= float and len(value.shape) > 0 and value.shape[0] > 2:
            bp()
        writer.add_scalar("%s/%s"%(phase, key), value, epoch_index)
    writer.flush()
    

def train_epoch(args, model, loader, optimizer, loss_fn, device, metrics, scheduler, epoch):
    model.train()
    running_loss = 0.0
    
    step = epoch * len(loader)
    print('Step: ', step, 'Learning rate: ', optimizer.param_groups[0]['lr'])
    for X, Y in tqdm(loader):
        X = X.to(device)
        Y = Y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        # Y = Y.float().unsqueeze(1)  # for BCEWithLogitsLoss
        loss = loss_fn(outputs, Y)
        loss.backward()
        optimizer.step()
        metrics.update(outputs, Y)
        running_loss += loss.item() * X.size(0)
        
        step += 1

        if epoch < args.epochs * args.lr_warmup_prop:
            lr = args.lr * (step / (args.epochs * len(loader) * args.lr_warmup_prop))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            if scheduler is None:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, int(args.epochs * len(loader) * (1-args.lr_warmup_prop)), eta_min=0)
            else:
                scheduler.step() 
        
    
    epoch_loss = running_loss / len(loader.dataset)
    results = metrics.compute_and_reset(loss=epoch_loss)
    
    return epoch_loss, results, scheduler

def validate_epoch(model, loader, loss_fn, device, metrics):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for X, Y in tqdm(loader):
            X = X.to(device)
            Y = Y.to(device).long()
            outputs = model(X)
            # Y = Y.to(device).long()
            loss = loss_fn(outputs, Y)
            metrics.update(outputs, Y)
            running_loss += loss.item() * X.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    results = metrics.compute_and_reset(loss=epoch_loss)
    return epoch_loss, results

def main(args):
    batch_size = args.batch_size
    num_workers = args.num_workers
    lr = args.lr
    epochs = args.epochs
    model_type = args.model_type
    if args.dataset == 'TUAB':
        args.num_classes = 2
    elif args.dataset == 'TUEV':
        args.num_classes = 6
    num_classes = args.num_classes
    window_length = args.window_length
    resolution = args.resolution
    exp_name = f'{model_type}_{num_classes}_classes_lr_{lr}_bs_{batch_size}_epochs_{epochs}_cosine_annealing_{args.dataset}_window_{window_length}_resolution_{resolution}_resolutionfactor_{args.resolution_factor}_stride_{args.stride_length}_bw_{args.bandwidth}_{'multitaper' if args.multitaper else 'stft'}_{'load_spec_true' if args.load_spec_true else 'load_spec_recon' if args.load_spec_recon else ''}_{str(args.lr_warmup_prop) + '_warmup'}_{args.seed}'
    print(exp_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=f'/data/scratch/alimirz/2025/EEG_FM/{args.dataset}_V2/{exp_name}')
    
    
    if args.dataset == 'TUAB':
        trainset = TUABBaselineDataset(args, mode='train', window_length=window_length, resolution=resolution)
        valset = TUABBaselineDataset(args, mode='val', window_length=window_length, resolution=resolution)
        testset = TUABBaselineDataset(args, mode='test', window_length=window_length, resolution=resolution)
    elif args.dataset == 'TUEV':
        trainset = TUEVBaselineDataset(args, mode='train', window_length=window_length, resolution=resolution, stride_length=args.stride_length)
        valset = TUEVBaselineDataset(args, mode='val', window_length=window_length, resolution=resolution)
        testset = TUEVBaselineDataset(args, mode='test', window_length=window_length, resolution=resolution)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    num_channels = 19 if args.load_spec_true or args.load_spec_recon else 23
    if args.dataset == 'TUAB':
        if args.load_spec_true or args.load_spec_recon:
            data_length = 8 
        else:
            data_length = 10
    elif args.dataset == 'TUEV':
        data_length = 5
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    model = SpectrogramCNN(model=model_type, num_classes=num_classes, dataset=args.dataset, num_channels=num_channels, data_length=data_length)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = None
    loss_fn = nn.CrossEntropyLoss()
    metrics = MulticlassMetrics(num_classes=num_classes, device=device)

    for epoch in tqdm(range(epochs)):
        val_loss, val_results = validate_epoch(model, val_loader, loss_fn, device, metrics)
        logger(writer, val_results, 'val', epoch)
        test_loss, test_results = validate_epoch(model, test_loader, loss_fn, device, metrics)
        logger(writer, test_results, 'test', epoch)
        train_loss, train_results, scheduler = train_epoch(args, model, train_loader, optimizer, loss_fn, device, metrics, scheduler, epoch)
        # Get current learning rate and add to metrics
        current_lr = optimizer.param_groups[0]['lr']
        train_results['lr'] = current_lr
        logger(writer, train_results, 'train', epoch)
        # scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return train_results, val_results, test_results
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='TUEV', choices=['TUAB', 'TUEV'])
    # parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--model_type', type=str, default='conv1d', choices=['conv1d', 'conv2d', 'resnet'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--window_length', type=int, default=5)
    parser.add_argument('--resolution_factor', type=int, default=1)
    parser.add_argument('--stride_length', type=int, default=1)
    parser.add_argument('--bandwidth', type=float, default=-1)
    parser.add_argument('--multitaper', type=bool, default=False)
    parser.add_argument('--load_spec_true', type=bool, default=False)
    parser.add_argument('--load_spec_recon', type=bool, default=False)
    parser.add_argument('--lr_warmup_prop', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.resolution = 0.2
    main(args)
