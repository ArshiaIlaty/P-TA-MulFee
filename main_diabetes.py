from great_diabetes import *
from classifier_diabetes import *
from gan_diabetes import *
import torch
import random
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AdamW,
)
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils import format_row
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def main():
    # Set specific GPU - use GPU 1 if available, otherwise use CPU
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            torch.cuda.set_device(1)  # Use GPU 1
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    csv_path = "diabetes.csv"
    model_name = "gpt2"
    save_path = "./gpt2_finetuned_diabetes"
    classifier_save_path = "./classifier_diabetes.pth"
    N = 2
    total_epoch_num = 1

    df = pd.read_csv(csv_path)
    for epoch in range(total_epoch_num):
        print("----- EPOCH: ", epoch)
        print("LLM Training..")
        if epoch != 0:
            train_great(csv_path, save_path, save_path)
        else:
            train_great(csv_path, model_name, save_path)
        print("Classifier Training..")
        df = classifier_train(
            csv_pth=csv_path,
            N=N,
            model_path=save_path,
            model_name="distilbert-base-uncased",
            classifier_save_pth=classifier_save_path,
        )

        classifier = TextClassifier(model_name="distilbert-base-uncased").to(device)
        classifier.load_state_dict(torch.load(classifier_save_path))
        classifier.eval()

        model = AutoModelForCausalLM.from_pretrained(save_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        print("GAN Training..")
        train_gpt2_with_gan(df, model, tokenizer, classifier)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        print("LLM Finetuning..")
        train_great(csv_path, save_path, save_path)
        print("Data Generating..")
        classifier_train(
            csv_pth=csv_path,
            N=N,
            model_path=save_path,
            model_name=model_name,
            classifier_save_pth=classifier_save_path,
            write_csv=True,
        )


if __name__ == "__main__":
    main()
