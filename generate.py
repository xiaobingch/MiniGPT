from model.gpt_model import GPTModel
from configs.config import GPT_CONFIG_124M
from utils.model_inference import generate, token_ids_to_text, text_to_token_ids
from utils.load_gpt2_weights import load_gpt2_weights_into_model
import torch
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
encode = text_to_token_ids("I HAD always thought Jack Gisburn", tokenizer).to(device)

model = GPTModel(GPT_CONFIG_124M)
model_path = 'model.pth'
model.load_state_dict(torch.load(model_path, weights_only=True))
# model_path = 'chapters/pytorch_model.bin' # openAI gpt2_124M_weights
# load_gpt2_weights_into_model(model, model_path) #加载openAI gpt2_124M_weights
model.eval()

torch.manual_seed(123)
token_ids = generate(
    model=model, 
    idx=encode, 
    max_new_tokens=50, 
    context_size=GPT_CONFIG_124M['context_length'],
    # temperature=1.5,
    # top_k=25
)

print("output text:\n", token_ids_to_text(token_ids, tokenizer))