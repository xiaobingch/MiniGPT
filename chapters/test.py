from configs.config import GPT_CONFIG_124M
from model.gpt_model import GPTModel
import torch

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

inputs = torch.tensor([[16833,  3626,  6100],       # ["every effort moves"]
        [   40,  1107,   588]])                     # ["I really like"]

targets = torch.tensor([[ 3626,  6100,   345],      #[" effort moves you"]
        [ 1107,   588, 11311]])                     #[" really like chocolate"]
        
with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)

text_idx = 0
targets_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
print(targets_probas_1)