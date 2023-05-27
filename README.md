# Falcon-7B Instruct Transformer Model

This repository contains a Python script utilizing the transformer-based model `tiiuae/falcon-7b-instruct` for text generation tasks. The script applies the model to a given text input and prints out the generated text. The text generation model is useful in various applications such as chatbots, storytelling, completion of sentences, and many more.

## Requirements

The script requires Python 3.6+ and the following Python packages:

- `transformers`
- `torch`

You can install these packages using pip:

```bash
pip install torch transformers
```

## Usage

1. Import the required modules:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
```

2. Load the pre-trained model and its associated tokenizer from HuggingFace Model Hub:

```python
model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, revision='main', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, revision='main', trust_remote_code=True).to('cuda')
```

3. Create a pipeline for text generation:

```python
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

4. Set your input text:

```python
input_text = "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"
```

5. Encode the input text and move it to GPU:

```python
input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')
```

6. Generate text using the pipeline:

```python
sequences = text_gen_pipeline(
    input_ids=input_ids,
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
```

7. Print the generated text:

```python
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
```

## Notes

- The model used in this script is moved to the GPU for processing (`model.to('cuda')`), and the encoded input tensors are also moved to the GPU (`input_ids.to('cuda')`). Ensure you have a GPU with enough memory, and CUDA installed for this to work.
- The `trust_remote_code` parameter is set to `True` while loading the model and tokenizer which indicates that the remote code can be trusted and will be executed if it exists.
- The script uses `torch_dtype=torch.bfloat16` for the pipeline which can potentially reduce memory usage but could result in less precision compared to the default `float32` data type.
- The `max_length` parameter in the `text_gen_pipeline` method controls the maximum length of the generated text. You may adjust it according to your needs.
- The `do_sample` parameter, when set to `True`, uses sampling for text generation. You can set it to `False` to use greedy decoding instead.
- The `top_k` parameter controls the number of highest probability vocabulary tokens to keep for top-k-filtering.
- The `num_return_sequences` parameter controls the number of independently computed returned sequences for each element in the batch.
- The `eos_token_id` parameter specifies the end of sequence token. When this token is encountered, text generation is stopped.
