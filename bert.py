import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "lmsys/vicuna-13b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Define your input text
input_text = "The weather is [MASK]."

# Step 3: Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Step 4: Find the position of the mask token
mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

# Step 5: Generate logits for the masked token
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

# Step 6: Get the predicted token id with the highest probability
predicted_token_index = torch.argmax(logits[0, mask_token_index, :]).item()

# Step 7: Convert the predicted token id back to its corresponding token
predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_index])[0]

# Step 8: Replace the mask token with the predicted token
input_ids[0, mask_token_index] = predicted_token_index

# Step 9: Decode the tokenized input to get the completed text
completed_text = tokenizer.decode(input_ids[0])

print("Completed Text:", completed_text)
