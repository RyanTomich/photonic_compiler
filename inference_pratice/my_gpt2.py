# Inference using packages

# Differen gpt2 model options
# - gpt2: This is the "small" version of GPT-2. It has 124 million self.parameters. 768 context size, 12 decode blocks
# - gpt2-medium: This is the "medium" version of GPT-2. It has 355 million self.parameters.
# - gpt2-large: This is the "large" version of GPT-2. It has 774 million self.parameters. 1280 context size 36 decode blocks
# - gpt2-xl: This is the "extra large" version of GPT-2. It has 1.5 billion self.parameters.

# @article{radford2019language,
#   title={Language Models are Unsupervised Multitask Learners},
#   author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
#   year={2019}
# }

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torchinfo import summary
import torch
import torch.nn.functional as F

def make_transformer_gpt2_inference():
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True, activation_function = 'gelu') # loading gpt2 from transformers library
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # loading gpt2 tokenizer from transformers library
    print(gpt2)
    # print(gpt2.config)

    input_ids = torch.ones((1, 11), dtype=torch.long)
    model_summary = summary(
        gpt2,
        input_data=input_ids,
        depth=6,
        verbose=2,
        col_names=["input_size", "output_size", "num_params", "trainable"],  # Custom columns
        col_width=20,
        row_settings=["var_names"],
        dtypes=[torch.long],
        device="cpu"
    )
    # print(model_summary)

def transformer_gpt2_inference(prompt, model, tokenizer):
# https://huggingface.co/docs/transformers/en/model_doc/gpt2
    prompt = "my favorite music is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    # gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token # set the padding token
    # input_ids = gpt2_tokenizer.encode(input_text, return_tensors='pt') # tokenize input
    # output = gpt2.generate(input_ids, max_length=max) # run inference
    # generated_text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True) # decode output tokens

    print(gen_text)


# Inference using Numpy
import numpy as np
import heapq
import random
import copy


class MyGPT2():
    def __init__(self, parameters):
        self.parameters = parameters
        self.emd_cache = np.empty((0, 768)) # embedding size

    def get_model_arcatecture(self, model):
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            ans = param.numpy()
            if 'h.' not in name: # each h.# refers to a transformer blocks
                # print(f'{name}: {param.shape}')
                pass

        for i in range(36):
            counter = 0
            for name, param in state_dict.items():
                ans = param.numpy()
                if 'h.'+ str(i)+ '.' in name and i == 0: # each h.# refers to a transformer block
                    # print(f'{name}: {ans.shape}')
                    counter +=1
            # print(f'h.{i}: {counter}')

    def torch_to_numpy(self, tensor): # not nessessarry?
        if tensor.is_cuda:
            tensor = tensor.cpu()
        numpy_array = tensor.numpy()
        return numpy_array.copy()

    def softmax(self, vec, temperature = 1):
        max_val = np.max(vec)
        exp = np.exp((vec - max_val)/ temperature)
        norm_vec = exp/np.sum(exp)
        assert 0.975 < np.sum(norm_vec) < 1.025
        return norm_vec

    def log_softmax(self, vec, epsilon=1e-05):
        max_val = np.max(vec)
        exp = np.exp(vec - max_val)
        log_sum_exp = max_val + np.log(np.sum(exp))
        return vec - log_sum_exp

    # activation functions
    def gelu(self, x):
        return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x, 3))))

    def ReLU(self, x):
        # x(np_array) clip negitive activation
        return np.maximum(0, x)

    # Transformer functions
    def embed(self, tok):
        '''
        creat embeding matrix (token, token embeding vector 768)
        tok(np_array): 1d array of toek encodings
        paramaters(dict): dictionary maping names to tensors
        '''
        # word token embeddings
        tok_emb = self.parameters['transformer.wte.weight'][tok,:]
        # print(f'{tok_emb}: {tok_emb.shape}')
        self.emd_cache = np.vstack((self.emd_cache, tok_emb))
        tok_emb = self.emd_cache

        # word position embeddings
        sequence_length = tok_emb.shape[0]
        position_ids = np.arange(sequence_length) #indicies
        position_emb = self.parameters['transformer.wpe.weight'][position_ids,:]

        assert tok_emb.shape == position_emb.shape
        return tok_emb + position_emb

    def layer_norm(self, x, gamma, beta, epsilon=1e-5):
        '''
        for 1D vectors
        layer batch normalization
        x(np_array): array to normalize
        gamma(np_array): scailing paramater vector
        beta(np_array): offset paramater vector
        epsilon(float): div_0_error prevention
        '''
        # print(x.shape)
        u = np.mean(x, axis=-1, keepdims=True)
        # s = np.mean(np.square(x-u))
        s = np.var(x, axis=-1, keepdims=True)
        x = (x - u) / np.sqrt(s + epsilon)
        return x * gamma + beta

    def self_attn(self, emb, block_num, attn_heads = 12):
        '''
        attention block. 12 heads per block
        emb(np_matrix): (tokens, Embedding Size 768)
        paramaters(dict): dictionary maping names to tensors
        block_num: current head
        '''
        # attn
        attn_weights = self.parameters['transformer.h.'+ str(block_num) + '.attn.c_attn.weight']
        attn_bias = self.parameters['transformer.h.'+ str(block_num) + '.attn.c_attn.bias']

        context_matrix = np.empty((0,emb.shape[1]))
        tok_k_matrx = [] # each index is a token
        for tok_index, tok in enumerate(emb):
            qvk_vec = np.split(tok @ attn_weights + attn_bias, 3, axis=0)
            Q_m, K_m, V_m = [s.reshape(attn_heads,int(attn_weights.shape[1]/3/attn_heads)) for s in qvk_vec]
            assert Q_m.shape == K_m.shape == V_m.shape == (12, 64)
            tok_k_matrx.append(K_m)

            masked_k = np.empty((0,K_m.shape[1]))

            context_vec = np.array([])
            for head in range(attn_heads):

                masked_k = np.empty((0,K_m.shape[1]))
                for prev_tok in range(tok_index):
                    masked_k = np.vstack((masked_k, tok_k_matrx[prev_tok][head]))
                assert masked_k.shape[1] == 64

                score_vec = Q_m[head] @ masked_k.T

                sub_context_vec = np.full((1 ,K_m.shape[1]), 0.0)
                for tok, score in enumerate(score_vec):
                    sub_context_vec += score * K_m[head]
                context_vec = np.append(context_vec, sub_context_vec)

            context_vec_norm = self.softmax(context_vec, temperature = 0.9)
            context_matrix = np.vstack((context_matrix, context_vec_norm.reshape(1, -1)))

        # projection
        weights = self.parameters['transformer.h.'+ str(block_num) + '.attn.c_proj.weight']
        bias = self.parameters['transformer.h.'+ str(block_num) + '.attn.c_proj.bias']
        context_proj = (context_matrix @ weights) + bias
        return context_proj

    def matrix_self_attn(self, emb, block_num, attn_heads = 12, mask = None):
        '''
        attention block. 12 heads per block
        emb(np_matrix): (tokens, Embedding Size 768)
        paramaters(dict): dictionary maping names to tensors
        '''
        def reshape_weights(x, w, b):
            *start, nx = x.shape
            nf = w.shape[-1]
            new_x = np.reshape(x, (-1, nx))
            new_w = np.reshape(w, (-1, nf))
            a = np.matmul(new_x, new_w) + b
            return(np.reshape(a, start + [nf]))

        def split_heads(x, attn_heads):
            *start, m = x.shape
            a = np.reshape(x, start + [attn_heads, m//attn_heads]) # matrix to tensor with heads dimention
            return np.transpose(a, [1, 0, 2]) #[heads, sequence, features]

        def merge_heads(x):
            x = np.transpose(x, [1, 0, 2]) #[sequence, heads, features]
            *start, a, b = x.shape
            return np.reshape(x, start + [a*b])

        # attn
        attn_weights = self.parameters['transformer.h.'+ str(block_num) + '.attn.c_attn.weight']
        attn_bias = self.parameters['transformer.h.'+ str(block_num) + '.attn.c_attn.bias']
        ln_1 = np.apply_along_axis(lambda x: self.layer_norm(x, attn_weights, attn_bias), axis=1, arr=emb)

        c = reshape_weights(emb, attn_weights, attn_bias)
        q,k,v = map(lambda x: split_heads(x, attn_heads), np.split(c, 3, axis=-1))
        assert q.shape == k.shape == v.shape

        # To validate query vector
        if block_num == 0:
            # print(emb.shape)
            # print(emb)
            # print(conv1.shape)
            # print(conv1)
            # print(q[0].shape)
            # print(q[0])
            pass


        # multi_headed_attn
        w = np.matmul(q, np.transpose(k, (0, 2, 1)))
        w = w * 1/np.sqrt(np.float32(v.shape[-1]))

        *start, nd, ns = w.shape
        attn_score_mask = w * mask
        attn_score_norm = np.apply_along_axis(lambda x: self.softmax(x, temperature = 0.9), axis=1, arr=attn_score_mask) # (1024, 1024)
        a = np.matmul(attn_score_norm, v)

        a = merge_heads(a)

        weights = self.parameters['transformer.h.'+ str(block_num) + '.attn.c_proj.weight']
        bias = self.parameters['transformer.h.'+ str(block_num) + '.attn.c_proj.bias']
        context_proj = (np.matmul(a, weights)) + bias
        return context_proj

    def mlp(self, emb, block_num):
        '''
        2 layer multi layer perceptron with gelu activation
        emb(np_matrix): (tokens, Embedding Size 768)
        paramaters(dict): dictionary maping names to tensors
        block_num: current head
        '''
        weights = self.parameters['transformer.h.'+ str(block_num) + '.mlp.c_fc.weight']
        bias = self.parameters['transformer.h.'+ str(block_num) + '.mlp.c_fc.bias']
        embl1 = np.matmul(emb, weights) + bias

        embl1 = self.gelu(embl1)

        weights = self.parameters['transformer.h.'+ str(block_num) + '.mlp.c_proj.weight']
        bias = self.parameters['transformer.h.'+ str(block_num) + '.mlp.c_proj.bias']
        return np.matmul(embl1, weights) + bias

    def top_k(self, k, vec):
        largest = heapq.nlargest(k, range(len(vec)), vec.take)
        # print(gpt2_tokenizer.decode(largest, skip_special_tokens=True)) # see words its picking from.
        probs = np.array([vec[i] for i in largest])
        probs = probs / np.sum(probs) # normalize after the selection
        assert 0.975 < np.sum(probs) < 1.025
        # print(np.max(probs))
        return random.choices(largest, weights=probs, k=1)[0]

    def decode_block(self, emb, block_num, mask = None):
        '''
        runs decode block with ln_1 -> attn -> ln_2 -> mlp
        emb (np_array): (tokens, Embedding Size 768)
        paramaters(dict): dictionary maping names to tensors
        block_num: current head
        '''

        weights = self.parameters['transformer.h.'+ str(block_num) + '.ln_1.weight']
        bias = self.parameters['transformer.h.'+ str(block_num) + '.ln_1.bias']
        # emb_norm1 = self.layer_norm(emb, weights, bias)    # ln_1 normalization
        ln_1 = np.apply_along_axis(lambda x: self.layer_norm(x, weights, bias), axis=1, arr=emb)

        if block_num == 0:
            print(ln_1.shape)
            print(ln_1)

        context_matrix = self.matrix_self_attn(emb_norm1, block_num, mask = mask)
        # context_matrix = self_attn(emb_norm1, block_num)

        context_matrix = context_matrix + emb # Residual Connection

        weights = self.parameters['transformer.h.'+ str(block_num) + '.ln_2.weight']
        bias = self.parameters['transformer.h.'+ str(block_num) + '.ln_2.bias']
        emb_norm2 = self.layer_norm(context_matrix, weights, bias, epsilon=1e-5)     # ln_2 normalization

        emb_mlp = self.mlp(emb_norm2, block_num)

        emb_mlp = context_matrix + emb_mlp # Residual Connection
        return emb_mlp

    def next_token(self, tok, decode_block = 12):
        '''
        Generates the next token in sequence
        tok (np_array): 1D token encodigns
        self.parameters(dict): dictionary maping names to tensors
        '''
        emb = self.embed(tok) #(tokens, Embedding Size 768)

        # Padding?
        # tokenizer.pad_token = tokenizer.eos_token
        # tok = tokenizer.encode(prompt, return_tensors='np', padding='max_length', truncation=True, max_length=max_token_len).squeeze()
        # prompt_tok_index = np.where(tok == tokenizer.eos_token_id)[0][0]

        mask = np.full((emb.shape[0], emb.shape[0]), float(1))
        mask[np.triu_indices_from(mask, k=1)] = float(-1e4)

        # EMB MATCHES hidden_states


        block_result = copy.deepcopy(emb)
        for block_num in range(decode_block):
            block_result = self.decode_block(block_result, block_num, mask = mask) # (tokens, Embedding Size 768)

        weights = self.parameters['transformer.ln_f.weight']
        bias = self.parameters['transformer.ln_f.bias']
        head_norm = self.layer_norm(block_result, weights, bias)  # ln_f

        # lm_head
        weights = self.parameters['lm_head.weight'] # (50257, 768)
        logit_matrix = np.matmul(head_norm, weights.T)

        # apply softmax to last words logit
        last_logit_distrabution = self.softmax(logit_matrix[-1], temperature = 0.9)
        return self.top_k(40, last_logit_distrabution)

    def generate(self, prompt, tokenizer, max_token_len = 100):
        '''
        creates generation feedback loop
        prompt(srt)
        start_dict(dict): name: paramaters
        '''

        tok = tokenizer.encode(prompt, return_tensors='np').squeeze()
        ans = np.array([])
        for _ in range(max_token_len):
            if ans.size == 0: #is empty
                next_tok = self.next_token(tok)
                ans = np.append(tok, next_tok)
            else:
                next_tok = self.next_token(ans[-1])
                ans = np.append(ans, next_tok)


        tok = tokenizer.decode(ans, skip_special_tokens=True)
        return tok
