from transformers import BertForQuestionAnswering, BertTokenizer
import os
# Remove cached model files
model_cache_dir = './cache'
if os.path.exists(model_cache_dir):
    shutil.rmtree(model_cache_dir)

# Download the model again
try:
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased', cache_dir=model_cache_dir)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=model_cache_dir)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Failed to load model or tokenizer: {e}")
