import os
import pprint
import argparse
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pandas as pd
import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, Resize
import torch.nn.functional as F
import clip
from model import CLIP, Bottleneck, AttentionPool2d
from simple_tokenizer import SimpleTokenizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import wandb

from train import train_main, load_data, load_clip, preprocess_text, preprocess_text_bert
from zero_shot import run_cxr_zero_shot, run_zero_shot, run_softmax_eval, make_true_labels
from eval import evaluate

# Initialize WandB
wandb.login()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='/home/vault/iwi5/iwi5207h/new/CheXzero/dataset/training_fold_1.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--txt_filepath', type=str, default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/training/training_fold_1.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--val_filepath', type=str, default='/home/vault/iwi5/iwi5207h/new/CheXzero/dataset/validation_fold_1.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--label_filepath', type=str, default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/training/train_fold_1.csv', help="Directory to load labels from.")
    parser.add_argument('--val_label_filepath', type=str, default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/validation/valid_fold_1.csv', help="Directory to load labels from.")
    parser.add_argument('--val_txt_filepath', type=str, default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/validation/validation_text_fold_1.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--cxr_true_labels_path', type=str, default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/validation/valid_fold_1.csv', help="Directory to load true labels from.")
    parser.add_argument('--rare_filepath', type=str, default='/home/vault/iwi5/iwi5207h/new/CheXzero/dataset/bal_low_count_diseases_training.h5', help="Directory to load rare labels from.")
    parser.add_argument('--rare_txt_filepath', type=str, default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/bal_low_count_training_present_report.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--rare_label_filepath', type=str, default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/low_count_diseases_training_balanced.csv', help="Directory to load labels from.")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=500)
    parser.add_argument('--best_model_dir', type=str, default="/home/vault/iwi5/iwi5207h/metric_learning_ds_new/checkpoints/best_model/", help="Directory to save the best model.")
    parser.add_argument('--save_dir', type=str, default="/home/vault/iwi5/iwi5207h/metric_learning_ds_new/checkpoints", help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--model_name', type=str, default="fold_1_batch_64", help="Name of the model.")
    parser.add_argument('--plot_dir', type=str, default="/home/vault/iwi5/iwi5207h/metric_learning_ds_new/plots/", help="Directory to save the plots.")
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau', help="Type of learning rate scheduler to use.")
    parser.add_argument('--patience', type=int, default=3, help="Patience for ReduceLROnPlateau scheduler.")
    parser.add_argument('--factor', type=float, default=0.1, help="Factor by which the learning rate will be reduced.")
    args = parser.parse_args()
    return args

def model_pipeline(config, verbose=0):
    wandb.init(project="k_fold_batch", config=config, name="batch_64")
    
    # Make the model, data, and optimization problem
    model, train_loader, rare_loader, eval_loader, val_loader, device, c_criterion, b_criterion, optimizer, scheduler = make(config)

    # Train the model
    train(model, train_loader, rare_loader, eval_loader, val_loader, device, c_criterion, b_criterion, optimizer, scheduler, config, resume=True, use_metric_loss=True)

    # Save model
    model_path = os.path.join(config.save_dir, str(config.model_name), 'checkpoint.pt')
    torch.save(model, model_path)
    
    wandb.save(model_path)
    wandb.finish()
    
    if verbose: 
        print(model)
    return model

def make(config): 
    pretrained = True
    train_loader, device = load_data(config.cxr_filepath, config.txt_filepath, config.label_filepath, batch_size=config.batch_size, pretrained=pretrained, column="impression")
    eval_loader, _ = load_data(config.val_filepath, config.val_txt_filepath, config.val_label_filepath, batch_size=config.batch_size, pretrained=pretrained, column="impression", type='eval')
    val_loader, _ = load_data(config.val_filepath, config.val_txt_filepath, config.val_label_filepath, batch_size=config.batch_size, pretrained=pretrained, column="impression", type='test')
    model = load_clip(model_path="/home/woody/iwi5/iwi5207h/Chexzero checkpoints/best_64_5e-05_original_22000_0.864.pt", pretrained=pretrained, context_length=config.context_length)
    model.to(device)
    model = model.to(torch.float32)
    rare_loader= train_loader
    print('Model loaded and moved to device.')

    label_columns = ['Adenopathy', 'Atelectasis', 'Azygos Lobe', 'Calcification of the Aorta', 
                     'Cardiomegaly', 'Clavicle Fracture', 'Consolidation', 'Edema', 'Emphysema', 
                     'Enlarged Cardiomediastinum', 'Fibrosis', 'Fissure', 'Fracture', 'Granuloma', 
                     'Hernia', 'Hydropneumothorax', 'Infarction', 'Infiltration', 'Kyphosis', 
                     'Lobar Atelectasis', 'Lung Lesion', 'Lung Opacity', 'Mass', 'Nodule', 
                     'Normal', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening', 
                     'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax', 
                     'Pulmonary Embolism', 'Pulmonary Hypertension', 'Rib Fracture', 
                     'Round(ed) Atelectasis', 'Subcutaneous Emphysema', 'Support Devices', 
                     'Tortuous Aorta', 'Tuberculosis']
    
    class_weights = compute_class_weights(config.label_filepath, label_columns).to(device)
    c_criterion = nn.CrossEntropyLoss().to(device)
    b_criterion = nn.BCEWithLogitsLoss(weight=class_weights).to(device)
    
    if config.optimizer == "adam": 
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0001)
    elif config.optimizer == "sgd": 
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    
    if config.scheduler == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config.patience, factor=config.factor)
    
    return model, train_loader, rare_loader, eval_loader, val_loader, device, c_criterion, b_criterion, optimizer, scheduler

def compute_class_weights(label_filepath, label_columns):
    labels_df = pd.read_csv(label_filepath)
    labels_df = labels_df[label_columns]
    class_counts = labels_df.sum(axis=0).values
    total_samples = len(labels_df)
    epsilon = 1e-6
    class_counts = class_counts + epsilon
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(class_weights, dtype=torch.float32)

def freeze_vit_layers(model, strategy='gradual'):
    """
    Strategically freeze/unfreeze ViT and text transformer layers for fine-tuning on rare classes.
    """
    def set_requires_grad(module, requires_grad=False, attention_only=False):
        if attention_only:
            for name, param in module.named_parameters():
                param.requires_grad = requires_grad if 'attn' in name or 'mlp' in name else False
        else:
            for param in module.parameters():
                param.requires_grad = requires_grad

    for param in model.parameters():
        param.requires_grad = False

    if strategy == 'gradual':
        num_layers = len(model.visual.transformer.resblocks)
        for i in range(num_layers - 4, num_layers):
            set_requires_grad(model.visual.transformer.resblocks[i], True)
        
        if hasattr(model.visual, 'attnpool'):
            set_requires_grad(model.visual.attnpool, True)
        
        if hasattr(model.visual, 'ln_post'):
            set_requires_grad(model.visual.ln_post, True)
        if hasattr(model.visual, 'proj'):
            model.visual.proj.requires_grad = True
            
    elif strategy == 'attention_only':
        for block in model.visual.transformer.resblocks:
            set_requires_grad(block, True, attention_only=True)
        if hasattr(model.visual, 'attnpool'):
            set_requires_grad(model.visual.attnpool, True)
            
    elif strategy == 'minimal':
        if hasattr(model.visual, 'ln_post'):
            set_requires_grad(model.visual.ln_post, True)
        if hasattr(model.visual, 'proj'):
            model.visual.proj.requires_grad = True
        if hasattr(model.visual, 'attnpool'):
            set_requires_grad(model.visual.attnpool, True)

    if hasattr(model, 'logit_scale'):
        model.logit_scale.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")

    return model

def train(model, train_loader, rare_loader, eval_loader, val_loader, device, c_criterion, b_criterion, optimizer, scheduler, config, resume=False, use_metric_loss=False): 
    model.train()
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    wandb.watch(model, log="all")
    
    if not os.path.exists(model_save_dir): 
        os.makedirs(model_save_dir)

    batch_ct = 0
    start_epoch = 1
    report_freq = config.log_interval
    highest_val_auc = 0

    train_losses = []
    val_losses = []
    val_aucs = []
    val_loss = 0
    
    csv_file = os.path.join(model_save_dir, f'{config.model_name}_metrics.csv')
    eval_results, val_auc, val_loss = validate(model, eval_loader, val_loader, device, c_criterion, b_criterion, config.context_length, config.cxr_true_labels_path, config, use_metric_loss)
    
    if resume and os.path.exists(model_save_dir):
        checkpoint_files = [file for file in os.listdir(model_save_dir) if file.endswith('.pt')]
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            checkpoint_path = os.path.join(model_save_dir, checkpoint_files[-1])
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            model.to(device)
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from checkpoint: {checkpoint_path} at epoch {start_epoch}")

            if os.path.exists(csv_file):
                with open(csv_file, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        train_losses.append(float(row[1]))
                        val_losses.append(float(row[2]))
                        val_aucs.append(float(row[3]))
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = checkpoint['optimizer']['param_groups'][0]['lr']

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        running_loss = 0.0

        for data in tqdm(train_loader, desc=f'Epoch {epoch}/{config.epochs}'):
            images = data['img']
            texts = data['txt']
            labels = data["label"]
            texts = preprocess_text(texts, model)
            total_loss = train_batch(images, texts, labels, model, device, c_criterion, b_criterion, optimizer, use_metric_loss)
            batch_ct += 1
            running_loss += total_loss.item()
            
            if (batch_ct % report_freq) == 0:
                avg_train_loss = running_loss / report_freq
                train_log(avg_train_loss, epoch)
                train_losses.append(avg_train_loss)
                wandb.log({"Train Loss": avg_train_loss})
                running_loss = 0.0
            
            if (batch_ct % config.save_interval) == 0:
                model_path = os.path.join(model_save_dir, f"checkpoint_{batch_ct}.pt")
                print(f"Saved checkpoint to: {model_path}")
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                }
                save(checkpoint, model_path)

                checkpoints = sorted([ckpt for ckpt in os.listdir(model_save_dir) if ckpt.startswith("checkpoint_")])
                if len(checkpoints) > 3:  # Keep only last 3 checkpoints
                    os.remove(os.path.join(model_save_dir, checkpoints[0]))

            if (batch_ct % config.val_interval) == 0:
                eval_results, val_auc, val_loss = validate(model, eval_loader, val_loader, device, c_criterion, b_criterion, config.context_length, config.cxr_true_labels_path, config, use_metric_loss)
                val_losses.append(val_loss)
                val_aucs.append(val_auc)

                if val_auc > highest_val_auc:
                    highest_val_auc = val_auc
                    save_best_model(model, optimizer, epoch, val_auc, config.best_model_dir)

                wandb.log({"Val AUC": val_auc, "Val Loss": val_loss})
                with open(csv_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, avg_train_loss, val_loss, val_auc])
        
        if config.scheduler == 'reduce_on_plateau':
            scheduler.step(val_loss)
    
    print('Training complete.')

def train_batch(images, texts, labels, model, device, c_criterion, b_criterion, optimizer, use_metric_loss=False):
    model.train()
    optimizer.zero_grad()
    
    if use_metric_loss:
        logits_per_image, logits_per_text, metric_loss_value = model(images, texts, labels, use_metric_loss=True)
    else:
        logits_per_image, logits_per_text = model(images, texts)
        metric_loss_value = 0

    batch_size = images.shape[0]
    c_labels = torch.arange(batch_size).to(device)
    loss_img = c_criterion(logits_per_image, c_labels)
    loss_txt = c_criterion(logits_per_text, c_labels)
    contrastive_loss_value = (loss_img + loss_txt) / 2
    total_loss = contrastive_loss_value + metric_loss_value
    total_loss.backward()
    optimizer.step()

    return total_loss

def validate(model, eval_loader, val_loader, device, c_criterion, b_criterion, context_length, cxr_true_labels_path, config, use_metric_loss=False):
    model.eval()
    total_loss = 0
    all_images = []
    all_texts = []
    all_image_features = torch.tensor([])
    all_text_features = torch.tensor([])
    all_labels = torch.tensor([])
    
    pair_template = ("{}", "no {}")
    cxr_labels = ['Adenopathy', 'Atelectasis', 'Azygos Lobe', 'Calcification of the Aorta', 
                     'Cardiomegaly', 'Clavicle Fracture', 'Consolidation', 'Edema', 'Emphysema', 
                     'Enlarged Cardiomediastinum', 'Fibrosis', 'Fissure', 'Fracture', 'Granuloma', 
                     'Hernia', 'Hydropneumothorax', 'Infarction', 'Infiltration', 'Kyphosis', 
                     'Lobar Atelectasis', 'Lung Lesion', 'Lung Opacity', 'Mass', 'Nodule', 
                     'Normal', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening', 
                     'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax', 
                     'Pulmonary Embolism', 'Pulmonary Hypertension', 'Rib Fracture', 
                     'Round(ed) Atelectasis', 'Subcutaneous Emphysema', 'Support Devices', 
                     'Tortuous Aorta', 'Tuberculosis']
    
    model.to(device)
    model = model.to(torch.float32)
    with torch.no_grad():
        for batch in val_loader:
            images, texts, labels = batch['img'], batch['txt'], batch['label']
            if isinstance(images, list):
                images = torch.stack(images)
        
            texts = preprocess_text(texts, model)
            if use_metric_loss:
                logits_per_image, logits_per_text, metric_loss_value = model(images, texts, labels, use_metric_loss=True)
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
            else:
                logits_per_image, logits_per_text = model(images, texts)
                metric_loss_value = 0
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
                
            all_image_features = torch.cat((all_image_features, image_features.cpu().detach()), dim=0)
            all_text_features = torch.cat((all_text_features, text_features.cpu().detach()), dim=0)
            all_labels = torch.cat((all_labels, labels.cpu().detach()), dim=0)

            batch_size = images.shape[0]
            contrastive_labels = torch.arange(batch_size).to(device)
            loss_img = c_criterion(logits_per_image, contrastive_labels)
            loss_txt = c_criterion(logits_per_text, contrastive_labels)
            c_loss = (loss_img + loss_txt) / 2
            loss = c_loss + metric_loss_value
            total_loss += loss.item()

        combined_features = torch.cat([all_image_features, all_text_features], dim=1)
        if combined_features.shape[1] > 10:  
            n_components = min(10, combined_features.shape[1], combined_features.shape[0])
            pca = PCA(n_components=n_components)
            reduced_features = pca.fit_transform(combined_features.cpu().detach().numpy())
        else:
            reduced_features = combined_features.cpu().detach().numpy()

        perplexity = min(50, max(5, reduced_features.shape[0] // 10))
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
        tsne_features = tsne.fit_transform(reduced_features)

        # Log TSNE plot to WandB
        plt.figure(figsize=(12, 12))
        scatter = plt.scatter(
            tsne_features[:, 0], tsne_features[:, 1], 
            c=all_labels.argmax(axis=1), cmap='tab20', alpha=0.7
        )
        plt.colorbar(scatter)
        plt.title(f'TSNE Visualization of CLIP Features')
        plt.xlabel('TSNE Dimension 1')
        plt.ylabel('TSNE Dimension 2')

        # Log TSNE plot to WandB
        wandb.log({"TSNE Visualization": wandb.Image(plt)})
        plt.close()

        # Log PCA plot to WandB
        plt.figure(figsize=(12, 12))
        scatter = plt.scatter(
            reduced_features[:, 0], reduced_features[:, 1], 
            c=all_labels.argmax(axis=1), cmap='tab20', alpha=0.7
        )
        plt.colorbar(scatter)
        plt.title(f'PCA Visualization of CLIP Features')
        plt.xlabel('PCA Dimension 1')
        plt.ylabel('PCA Dimension 2')

        # Log PCA plot to WandB
        wandb.log({"PCA Visualization": wandb.Image(plt)})
        plt.close()

        avg_val_loss = total_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss}')
        
    # Get predictions and evaluate
    y_pred = run_softmax_eval(model, eval_loader, cxr_labels, pair_template, context_length)
    test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)
    test_true = test_true[:len(y_pred)]
    eval_results = evaluate(y_pred, test_true, cxr_labels)
    
    # Log confusion matrix to WandB
    confusion_matrix = wandb.plot.confusion_matrix(
        y_true=test_true.argmax(axis=1),
        preds=y_pred.argmax(axis=1),
        class_names=cxr_labels,
        title="Confusion Matrix"
    )
    wandb.log({"Confusion Matrix": confusion_matrix})
    
    # Log evaluation results for each label
    eval_results_dict = eval_results.to_dict(orient='list')
    for label in cxr_labels:
        auc_key = f"{label}_auc"
        if auc_key in eval_results_dict:
            auc_value = eval_results_dict[auc_key][0]
            wandb.log({auc_key: auc_value})
    
    val_auc = eval_results.filter(like='_auc').mean(axis=1).mean()
    
    return eval_results, val_auc, avg_val_loss

def plot_metrics(train_losses, val_losses, val_aucs, config):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    if val_losses:
        plt.plot(val_losses, label='Val Loss', color='red')
    if val_aucs:
        plt.plot(val_aucs, label='Val AUC', color='green')

    plt.xlabel('Batch Steps')
    plt.ylabel('Metrics')
    plt.title(f'{config.model_name} Training/Validation Metrics')
    plt.legend()
    
    plot_path = os.path.join(config.plot_dir, f'{config.model_name}_metrics_plot.png')
    plt.savefig(plot_path)
    plt.show()

def save(checkpoint, path):
    torch.save(checkpoint, path)

def save_best_model(model, optimizer, epoch, val_auc, model_save_dir):
    best_model_path = os.path.join(model_save_dir, "fold_1_batch_64.pt")
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_auc': val_auc,
    }
    save(checkpoint, best_model_path)
    print(f"Best model saved at {best_model_path}")

def train_log(avg_train_loss, epoch):
    print(f'Epoch {epoch}: Train Loss = {avg_train_loss}')

if __name__ == "__main__":
    args = parse_args()
    model = model_pipeline(args)