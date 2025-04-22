from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as T
import torch.nn.functional as F

class LMReward:

    def __init__(self, model = 'gpt2'):
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype=T.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def str_loglikelihood(self, str1, str2):

        #Tokenizing first half to define boundary
        str1_tokens = self.tokenizer(str1, return_tensors="pt", padding=True)    
        str1_length = str1_tokens['input_ids'].shape[1]    

        strings = [str1 + str for str in str2]
        inputs = self.tokenizer(strings, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
       
        with T.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[:, :-1, :]
       
        labels = inputs['input_ids'][:, 1:]
        attention_masks = inputs['attention_mask'][:, 1:]  
       
        log_probs = F.log_softmax(logits, dim=-1)
        labels_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        str1_mask = T.ones_like(attention_masks)
        str1_mask[:, :str1_length-1] = 0.

        mask = attention_masks * str1_mask

        str_log_probs = (mask * labels_log_probs).sum(dim = 1)

        return str_log_probs