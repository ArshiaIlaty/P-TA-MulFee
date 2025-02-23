from great import *
from classifier import *
from gan import *
import torch
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AdamW
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils import format_row


def main():
    csv_path = "heloc.csv"
    output_path="heloc_generated.csv"
    model_name = 'gpt2'
    save_path="./gpt2_finetuned"
    classifier_save_path="./classifier.pth"
    N=2
    total_epoch_num=10

    df = pd.read_csv(csv_pth)
    for _ in range(total_epoch_num):
        train_great(csv_path, model_name, save_path)
        classifier_train(csv_pth=csv_path, N = N, model_path=save_path, model_name=model_name, classifier_save_pth=classifier_save_path)
        
        classifier = TextClassifier(model_name=model_name).cuda()
        classifier.load_state_dict(torch.load("./classifier.pth"))
        classifier.eval()
        
        model = AutoModelForCausalLM.from_pretrained(save_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        train_gpt2_with_gan(df, model, tokenizer, classifier)
        model.save_pretrained(save_pth)
        tokenizer.save_pretrained(save_pth)


    def fill_missing_values(input_text, missing_slots):
        tokens = input_text.split(", ")
        new_tokens = tokens[:]
    
        for idx, col_name in missing_slots.items():
            prompt = f"{col_name} is"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            
            with torch.no_grad():
                output = model.generate(input_ids, max_length=input_ids.shape[1] + 5, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_p=0.9)
            
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            generated_value = generated_text.replace(f"{col_name} is", "").strip().split(",")[0]
            new_tokens[idx] = f"{col_name} is {generated_value}"
    
        return ", ".join(new_tokens)
    
    model = AutoModelForCausalLM.from_pretrained(save_path).cuda()
    model.eval()  
    df["formatted_text"] = df.apply(format_row, axis=1)
    
    df["corrupted_text"] = df["formatted_text"].apply(lambda x: remove_random_values(x, num_remove=N))
    df["repaired_text"] = df["corrupted_text"].apply(lambda x: fill_missing_values(x[0], x[1]))

    df.to_csv(output_path)


if __name__=="__main__":
    main()