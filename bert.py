# from transformers import BertTokenizer, BertForMaskedLM
# import torch

# # Step 1: Load the BERT model and tokenizer
# model_name = 'bert-base-uncased'  # or any other BERT variant
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForMaskedLM.from_pretrained(model_name)

# # Step 2: Define your input text
# input_text = "The weather is [MASK]."

# # Step 3: Tokenize the input text
# tokenized_input = tokenizer.encode(input_text, return_tensors="pt")

# # Step 4: Generate output text
# with torch.no_grad():
#     outputs = model(tokenized_input)
#     predictions = outputs.logits

# # Get the predicted token IDs for the masked token
# masked_index = tokenized_input.squeeze().tolist().index(tokenizer.mask_token_id)
# predicted_index = torch.argmax(predictions[0, masked_index]).item()

# # Convert the predicted token ID back to a token
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

# # Replace the [MASK] token with the predicted token to get the output text
# output_text = input_text.replace('[MASK]', predicted_token)

# print("Input text:", input_text)
# print("Output text:", output_text)
import torch

# Step 1: Load the LLAMA model and tokenizer
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

model = "lmsys/vicuna-13b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model)

# Step 2: Define your input text
input_text = "The weather is [MASK]."

# Step 3: Tokenize the input text
tokenized_input = tokenizer(input_text, return_tensors="pt")

# Step 4: Generate output text
with torch.no_grad():
    outputs = model(**tokenized_input)
    predictions = outputs.logits

# Get the predicted token IDs for the masked token
masked_index = torch.where(tokenized_input["input_ids"] == tokenizer.mask_token_id)[1].tolist()
predicted_index = torch.argmax(predictions[0, masked_index]).item()

# Convert the predicted token ID back to a token
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

# Replace the [MASK] token with the predicted token to get the output text
output_text = input_text.replace("[MASK]", predicted_token)

print("Input text:", input_text)
print("Output text:", output_text)
