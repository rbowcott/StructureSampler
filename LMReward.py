from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

class LMReward:

    def __init__(self, model = "gpt2"):
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def str_loglikelihood(self, string):

        inputs = self.tokenizer(string, return_tensors="pt", padding = True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
       
        with torch.no_grad():
            outputs = self.model(**inputs)

        #Remove prediction after <|endoftext|>  
        logits = outputs.logits[:, :-1, :]
       
        #Shift labels & mask to the left to correspond to predictions
        labels = inputs['input_ids'][:, 1:]
        attention_masks = inputs['attention_mask'][:, 1:]  
       
        log_probs = F.log_softmax(logits, dim=-1)
        labels_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        str_log_probs = (attention_masks * labels_log_probs).sum(dim = 1)
               
        return str_log_probs
    
    def str_avgloglikelihood(self, string):

        inputs = self.tokenizer(string, return_tensors="pt", padding = True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
       
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[:, :-1, :]
       
        labels = inputs['input_ids'][:, 1:]
        attention_masks = inputs['attention_mask'][:, 1:]  
       
        log_probs = F.log_softmax(logits, dim=-1)
        labels_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        str_log_probs = (attention_masks * labels_log_probs).mean(dim = 1)
               
        return str_log_probs