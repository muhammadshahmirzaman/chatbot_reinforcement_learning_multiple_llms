


import os
import torch
import json
from transformers import AutoTokenizer
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained('./opt-125m')
        self.max_length = 512

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        State = "Question:{} Answer:{}"
        item = self.data[index]
        init_source = State.format(item['instruction'] + item['input'], " ")
        target = item['output'] if 'output' in item else None
        source = item['instruction'] + item['input']
        encoding = self.tokenizer(init_source, max_length=self.max_length, padding='max_length', truncation=True,
                                  return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'source': source,
            'target': target,
        }

    def get_example(self, index):
        return self.data[index]


def load_data(data_path, args, max_size=None):
    """
    Load jsonl/json data with a couple of helpful checks:
    - Resolve alternative filenames (e.g. *_1.jsonl) if the requested path
      does not exist.
    - Fail fast with a clear error when no examples are found.
    """
    assert data_path, "data_path is not specified"

    resolved_path = data_path
    if not os.path.exists(resolved_path):
        # Common typo: missing `_1` suffix in the provided datasets
        candidate = data_path.replace(".jsonl", "_1.jsonl")
        if os.path.exists(candidate):
            resolved_path = candidate
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from {resolved_path}")

    if resolved_path.endswith('.jsonl'):
        with open(resolved_path, encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
    elif resolved_path.endswith('.json0127') or resolved_path.endswith('.json'):
        with open(resolved_path, 'r', encoding="utf-8") as fin:
            data = json.load(fin)
    else:
        raise ValueError(f"Unknown data format for {resolved_path}")

    if max_size is not None and max_size > 0:
        data = data[:max_size]
    examples = []
    for i, example in enumerate(data):
        new_item = {}
        # Handle different field names
        new_item['id'] = example.get('id', str(i))
        
        # Map source -> instruction if needed
        if 'instruction' in example:
            new_item['instruction'] = example['instruction']
        elif 'source' in example:
            new_item['instruction'] = example['source']
        else:
            new_item['instruction'] = example.get('prompt', '')
            
        new_item['input'] = example.get('input', '')
        
        # Map target -> output if needed
        if 'output' in example:
            new_item['output'] = example['output']
        elif 'target' in example:
            new_item['output'] = example['target']
        elif 'response' in example:
            new_item['output'] = example['response']
        else:
            new_item['output'] = ''
            
        examples.append(new_item)
    if not examples:
        raise ValueError(f"No examples loaded from {resolved_path}")

    return examples



if __name__ == '__main__':
    pass