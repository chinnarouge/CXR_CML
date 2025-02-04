import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from PIL import Image
import h5py
import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import sys
sys.path.append("../..")
import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer
import torchvision.models as models

class CXRDataset(data.Dataset):
    def __init__(self, img_path, txt_path, label_path, column="report", size=None, transform=None):
        super().__init__()
        if size != None:
            self.img_dset = h5py.File(img_path, "r")["cxr"][:size]
            self.txt_dset = pd.read_csv(txt_path)[column][:size]
            self.label_dset = pd.read_csv(label_path)[
                ['Adenopathy', 'Atelectasis', 'Azygos Lobe', 'Calcification of the Aorta', 
                     'Cardiomegaly', 'Clavicle Fracture', 'Consolidation', 'Edema', 'Emphysema', 
                     'Enlarged Cardiomediastinum', 'Fibrosis', 'Fissure', 'Fracture', 'Granuloma', 
                     'Hernia', 'Hydropneumothorax', 'Infarction', 'Infiltration', 'Kyphosis', 
                     'Lobar Atelectasis', 'Lung Lesion', 'Lung Opacity', 'Mass', 'Nodule', 
                     'Normal', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening', 
                     'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax', 
                     'Pulmonary Embolism', 'Pulmonary Hypertension', 'Rib Fracture', 
                     'Round(ed) Atelectasis', 'Subcutaneous Emphysema', 'Support Devices', 
                     'Tortuous Aorta', 'Tuberculosis']
            ][:size]
        else:
            self.img_dset = h5py.File(img_path, "r")["cxr"]
            self.txt_dset = pd.read_csv(txt_path)[column]
            self.label_dset = pd.read_csv(label_path)[
                ['Adenopathy', 'Atelectasis', 'Azygos Lobe', 'Calcification of the Aorta', 
                     'Cardiomegaly', 'Clavicle Fracture', 'Consolidation', 'Edema', 'Emphysema', 
                     'Enlarged Cardiomediastinum', 'Fibrosis', 'Fissure', 'Fracture', 'Granuloma', 
                     'Hernia', 'Hydropneumothorax', 'Infarction', 'Infiltration', 'Kyphosis', 
                     'Lobar Atelectasis', 'Lung Lesion', 'Lung Opacity', 'Mass', 'Nodule', 
                     'Normal', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening', 
                     'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax', 
                     'Pulmonary Embolism', 'Pulmonary Hypertension', 'Rib Fracture', 
                     'Round(ed) Atelectasis', 'Subcutaneous Emphysema', 'Support Devices', 
                     'Tortuous Aorta', 'Tuberculosis']
            ][:size]
        self.transform = transform

    def __len__(self):
        return len(self.txt_dset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.img_dset[idx]  # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        txt = self.txt_dset[idx]  # python str
        if type(txt) == type(float("nan")):  # capture the case of empty "Impression" sections
            txt = " "

        img = torch.from_numpy(img)  # torch, (3, 320, 320)
        label_row = self.label_dset.iloc[idx]
        label_tensor = torch.tensor(label_row.values, dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        sample = {"img": img, "txt": txt, "label": label_tensor}

        return sample

def load_data(cxr_filepath, txt_filepath, label_filepath, batch_size=4, column="report", pretrained=False, verbose=False, type="train"):
    if torch.cuda.is_available():
        dev = "cuda:0"
        cuda_available = True
        print("Using CUDA.")
    else:
        dev = "cpu"
        cuda_available = False
        print("Using cpu.")

    device = torch.device(dev)

    if cuda_available:
        torch.cuda.set_device(device)

    if pretrained:
        input_resolution = 224
        transform = Compose(
            [
                Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
                Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
            ]
        )
        print("Interpolation Mode: ", InterpolationMode.BICUBIC)
        print("Finished image transforms for pretrained model.")
    else:
        input_resolution = 320
        transform = Compose(
            [
                Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            ]
        )
        print("Finished image transforms for clip model.")

    torch_dset = CXRDataset(
        img_path=cxr_filepath,
        txt_path=txt_filepath,
        label_path=label_filepath,
        column=column,
        transform=transform,
    )
    #torch_dset = torch.utils.data.Subset(torch_dset, range(10))
    if verbose:
        for i in range(len(torch_dset)):
            sample = torch_dset[i]
            plt.imshow(sample["img"][0])
            plt.show()
            print(i, sample["img"].size(), sample["txt"])
            if i == 3:
                break

    loader_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}

    if type == "test":
        loader_params = {"batch_size": batch_size, "shuffle": False, "num_workers": 0}
    if type == "eval":
        loader_params = {"shuffle": False, "num_workers": 0}

    data_loader = data.DataLoader(torch_dset, **loader_params)
    return data_loader, device

def load_clip(model_path=None, pretrained=False, context_length=77):
    """
    FUNCTION: load_clip
    -------------------------------
    This function loads in a model with the CLIP model
    architecture.

    args:
        * model_path (optional) - path to model weights that the model
        will be initialized with
        * pretrained (optional) - if True, will load the pretrained
        CLIP model
        * context_length (optional) - length of the maximum number of
        tokens that can be inputted into the CLIP model
    """

    params = {
        "embed_dim": 768,
        "image_resolution": 320,
        "vision_layers": 12,
        "vision_width": 768,
        "vision_patch_size": 16,
        "context_length": context_length,
        "vocab_size": 49408,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12,
    }

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if pretrained:
        # load clip pre-trained model
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        print("Loaded in pretrained model.")
        vit = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)

        # Copy ViT weights to CLIP vision encoder
        model.visual.load_state_dict(vit.state_dict(), strict=False)
        print("Loaded ImageNet pretrained weights into CLIP vision encoder.")
    else:
        model = CLIP(**params)
        print("Loaded in clip model.")

    # if a model_path is provided, load in weights to backbone
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    return model

def preprocess_text_bert(texts, model):
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized",
        trust_remote_code=True  # Allowing remote code execution for this tokenizer
    )
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=model.context_length,
        return_tensors="pt",
    )
    # We should return both input_ids and attention_mask for BERT
    return {
        "input_ids": encoded["input_ids"].to(model.device),
        "attention_mask": encoded["attention_mask"].to(model.device),
    }

def preprocess_text(texts, model):
    #     if model.context_length is None:
    #         model = model.module

    _tokenizer = SimpleTokenizer()
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), model.context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > model.context_length:
            tokens = tokens[: model.context_length]
            tokens[model.context_length - 1] = eot_token
        result[i, : len(tokens)] = torch.tensor(tokens)
    return result

def make(config, cxr_filepath, txt_filepath, model_path=None):
    """
    FUNCTION: make
    ---------------------------------
    This function makes the model, the data loader, loss and optimizer.

    args:
        * config - dict, configuration of experiment
        * cxr_filepath - string, filepath to chest x-ray images
        * txt_filepath - string, filepath to corresponding text reports
        * model_path - string, filepath to previously trained model
    """
    data_loader, device = load_data(
        cxr_filepath,
        txt_filepath,
        batch_size=config.batch_size,
        pretrained=config.pretrained,
        column=config.column,
    )
    model = load_clip(
        model_path=model_path,
        pretrained=config.pretrained,
        context_length=config.context_length,
    )
    model.to(device)
    print("Model on Device.")

    # make the optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # todo: incorporate - torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    return model, data_loader, device, criterion, optimizer

def train_main(
    cxr_filepath,
    txt_filepath,
    hyperparams,
    output_path,
    model_path=None,
    pretrained=False,
):
    """
    args:
        * cxr_filpath- str filepath to cxr images
        * txt_filepath- str filepath to text reports
        * hyperparams- dictionary with the following hyperparams:
        `batch_size`, `criterion`, `learning_rate`, `momentum`, `epochs`
        * output_path- str filepath to where the trained model will be saved
        * model_path- str filepath to model that will be used as baseline model for training.
        If not provided, a model will be trained from scratch
        * pretrained- whether or not the clip model was pretrained with generic images
    This function is the main train function for CXR-CLIP.
    """

    # unpack `hyperparams`
    batch_size = hyperparams["batch_size"]
    criterion = hyperparams["criterion"]
    learning_rate = hyperparams["learning_rate"]
    momentum = hyperparams["momentum"]
    epochs = hyperparams["epochs"]

    # load input cxr + report data
    data_loader, device = load_data(
        cxr_filepath, txt_filepath, batch_size=batch_size, pretrained=pretrained
    )
    model = load_clip(model_path=model_path, pretrained=pretrained)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_clip(model, data_loader, device, criterion, optimizer, epochs, output_path)
    return model