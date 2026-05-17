import torch
from torch.utils.data import Dataset


class EminemLirics(Dataset):
    def __init__(self, file_path='data/data.txt', tokenizer, max_length=64):
        self.examples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        kupletz = raw_text.split('\n\n')

        print("dataset building 1\3")

        for sent in kupletz:
                sent = sent.strip()

                if not sent or len(sent) <=5:
                     continue
                
                prompt = "Write a rap verse in the style of Eminem:\n"
                full_text = f"{prompt}{sent}{tokenizer.eos_token}"
                
                # 4. Превращаем текст в тензоры
                tokenized = tokenizer(
                    full_text,
                    padding="max_length",  # Добиваем паддингами до max_length
                    truncation=True,      # Обрезаем, если куплет вдруг гигантский
                    max_length=max_length,
                    return_tensors="pt"   # Сразу возвращаем как тензоры PyTorch
                )
                
                # Извлекаем тензоры из батча размерности [1, seq_len] -> [seq_len]
                item = {key: val.squeeze(0) for key, val in tokenized.items()}
                
                # Наш старый знакомый трюк: для Causal LM метки (labels) — это точная копия входных ID
                item["labels"] = item["input_ids"].clone()
                
                self.examples.append(item)

    def __len__(self):
         return len(self.examples)
    
    def __getitem__(self, key):
        return self.examples[key]
    

