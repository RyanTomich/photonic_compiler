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

# For extraction
import model_trace as trace


# Generic Transformer Library
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torchinfo import summary
import torch
def transformer_gpt2_model(model_name):
    gpt2 = GPT2LMHeadModel.from_pretrained(model_name, output_attentions=True, activation_function = 'gelu', ) # loading gpt2 from transformers library
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name) # loading gpt2 tokenizer from transformers library
    return gpt2, gpt2_tokenizer

def transformer_gpt2_inference(prompt, model, tokenizer, do_sample = False, temperature = 1, max_length = 100):
    # https://huggingface.co/docs/transformers/en/model_doc/gpt2
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(input_ids, do_sample=do_sample, temperature=temperature, max_length=max_length)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    return (gen_text)

def transformer_model_info(model, config = False, summary = False ):
    print(model)
    if config:
        print(model.config)
    if summary:
        input_ids = torch.ones((1, 11), dtype=torch.long)
        model_summary = summary(
            model,
            input_data=input_ids,
            depth=6,
            verbose=2,
            col_names=["input_size", "output_size", "num_params", "trainable"],  # Custom columns
            col_width=20,
            row_settings=["var_names"],
            dtypes=[torch.long],
            device="cpu"
        )
        print(model_summary)

def get_parameters(model):
    state_dict = model.state_dict()
    parameters = {}
    for name, val in state_dict.items():
        parameters[name] = np.round(val.numpy().astype(np.float32), 4)
    return parameters

# Inference using Numpy
import numpy as np
import heapq
import random

@trace.for_all_methods(trace.catch_name)
class NpGPT2():
    def __init__(self, parameters, tokenizer, decode_blocks = 12, attn_heads = 12, logit_strat = 'greedy', temperature = 1, embedding_size = 768):
        self.parameters = parameters
        self.tokenizer = tokenizer
        self.decode_blocks = decode_blocks
        self.attn_heads = attn_heads
        self.logit_strat = logit_strat
        self.temperature = temperature
        self.embedding_size = embedding_size

        self.key_cache = [np.empty((0, self.embedding_size)) for _ in range(decode_blocks)]
        self.value_cache = [np.empty((0, self.embedding_size)) for _ in range(decode_blocks)]
        self.emd_cache = np.empty((0, self.embedding_size))

    def get_model_architecture(self, model):
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            ans = param.numpy()
            if 'h.' not in name: # each h.# refers to a transformer blocks
                print(f'{name}: {param.shape}')
                pass

        for i in range(36):
            counter = 0
            for name, param in state_dict.items():
                ans = param.numpy()
                if 'h.'+ str(i)+ '.' in name and i == 0: # each h.# refers to a transformer block
                    print(f'{name}: {ans.shape}')
                    counter +=1
            # print(f'h.{i}: {counter}')

    def torch_to_numpy(self, tensor): # not nessessarry?
        if tensor.is_cuda:
            tensor = tensor.cpu()
        numpy_array = tensor.numpy()
        return numpy_array.copy()

    def softmax(self, matrix, temperature = 1):
        def vec_softmax(vec):
            trace.file_write(f'row_softamx', f'{vec.shape}')
            temperature = self.temperature
            max_val = np.max(vec)
            exp = np.exp((vec - max_val)/ temperature)
            norm_vec = exp/np.sum(exp)
            assert 0.975 < np.sum(norm_vec) < 1.025
            return norm_vec
        if len(matrix.shape) == 1:
            matrix = matrix.reshape (1, -1)
        return np.apply_along_axis(lambda x: vec_softmax(x), axis=-1, arr=matrix)


    # activation functions
    def gelu(self, x):
        return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x, 3))))

    # Transformer functions
    def embed(self, tok):
        '''
        creat embeding matrix (token, token embeding vector 768)
        tok(np_array): 1d array of toek encodings
        paramaters(dict): dictionary maping names to tensors
        '''
        # word token embeddings
        tok_emb = self.parameters['transformer.wte.weight'][tok,:]

        # word position embeddings
        if self.emd_cache.size == 0:
            sequence_length = tok_emb.shape[0]
            position_ids = np.arange(sequence_length) #indicies
        else:
            position_ids = self.emd_cache.shape[0]

        position_emb = self.parameters['transformer.wpe.weight'][position_ids,:]

        emb = tok_emb + position_emb
        trace.file_write(f'emb_add',f'{tok_emb.shape} + {position_emb.shape}')
        self.emd_cache = np.vstack((self.emd_cache, emb))
        return np.asarray(emb).reshape(1, -1) if emb.ndim == 1 else np.asarray(emb)

    def layer_norm(self, x, gamma, beta, epsilon=1e-5):
        '''
        for 1D vectors
        layer batch normalization
        x(np_array): array to normalize
        gamma(np_array): scailing paramater vector
        beta(np_array): offset paramater vector
        epsilon(float): div_0_error prevention
        '''
        u = np.mean(x, axis=-1, keepdims=True)
        s = np.var(x, axis=-1, keepdims=True)
        x = (x - u) / np.sqrt(s + epsilon)

        if len(gamma.shape) == 1:
            trace.file_write(f'layer_norm_*mult', f'{x.shape}*{gamma.shape}+{beta.shape}')
            return x * gamma + beta
        else:
            trace.file_write(f'layer_norm_@mult', f'{x.shape}@{gamma.shape}+{beta.shape}')
            return x @ gamma + beta

    def matrix_self_attn(self, emb, block_num, mask = None):
        '''
        attention block. 12 heads per block
        emb(np_matrix): (tokens, Embedding Size 768)
        paramaters(dict): dictionary maping names to tensors
        '''
        def split_heads(x):
            *start, m = x.shape
            a = np.reshape(x, start + [self.attn_heads, m//self.attn_heads]) # matrix to tensor with heads dimention
            return np.transpose(a, [1, 0, 2]) #[heads, sequence, features]

        def merge_heads(x):
            x = np.transpose(x, [1, 0, 2]) #[sequence, heads, features]
            *start, a, b = x.shape
            return np.reshape(x, start + [a*b])

        # attn
        attn_weights = self.parameters['transformer.h.'+ str(block_num) + '.attn.c_attn.weight']
        attn_bias = self.parameters['transformer.h.'+ str(block_num) + '.attn.c_attn.bias']
        ln_1 = emb @ attn_weights + attn_bias  # (t, 2304)
        trace.file_write(f'ln_1',f'{emb.shape}@{attn_weights.shape}+{attn_bias.shape}')

        query, key, value = np.hsplit(ln_1,3)

        self.key_cache[block_num] = np.vstack( (self.key_cache[block_num], key) )
        self.value_cache[block_num] = np.vstack( (self.value_cache[block_num],value) )

        Q = split_heads(query)
        K = split_heads(self.key_cache[block_num])
        V = split_heads(self.value_cache[block_num])

        # multi_headed_attn
        K = np.transpose(K, (0, 2, 1))
        w = Q @ K
        trace.file_write(f'Q@K',f'{Q.shape}@{K.shape}')

        w = w * 1/np.sqrt(np.float32(V.shape[-1]))

        attn_score_mask = w + mask
        trace.file_write(f'app_mask',f'{w.shape}+{mask.shape}')

        attn_score_norm = self.softmax(attn_score_mask)

        attn_output = attn_score_norm @ V
        trace.file_write(f'attn_score@V',f'{attn_score_norm.shape}@{V.shape}')

        attn_output = merge_heads(attn_output)

        weights = self.parameters['transformer.h.'+ str(block_num) + '.attn.c_proj.weight']
        bias = self.parameters['transformer.h.'+ str(block_num) + '.attn.c_proj.bias']
        context_proj = attn_output @ weights + bias
        trace.file_write(f'context_proj',f'{attn_output.shape}@{weights.shape}+{bias.shape}')

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
        trace.file_write(f'embl1',f'{emb.shape}@{weights.shape}+{bias.shape}')
        embl1 = emb @ weights + bias

        embl1 = self.gelu(embl1)

        weights = self.parameters['transformer.h.'+ str(block_num) + '.mlp.c_proj.weight']
        bias = self.parameters['transformer.h.'+ str(block_num) + '.mlp.c_proj.bias']
        trace.file_write(f'embl1',f'{embl1.shape}@{weights.shape}+{bias.shape}')
        return embl1 @ weights + bias

    def top_k(self, logits, k=50):
        '''
        Selects top k proabilitie, renormaliaes them and selects a token id based
        on those proabilities
        logits(np.array): matrix of logits
        k(int): top num to be selected
        '''
        vec = self.softmax(logits[-1])
        largest = heapq.nlargest(k, range(len(vec)), vec.take)
        probs = np.array([vec[i] for i in largest])
        probs = probs / np.sum(probs) # normalize after the selection
        # assert 0.975 < np.sum(probs) < 1.025
        return random.choices(largest, weights=probs, k=1)[0]

    def decode_block(self, emb, block_num, mask = None):
        '''
        runs decode block with ln_1 -> attn -> ln_2 -> mlp
        emb (np_array): (tokens, Embedding Size)
        block_num: current head
        mask(np.array): attention mask to use
        '''
        weights = self.parameters['transformer.h.'+ str(block_num) + '.ln_1.weight']
        bias = self.parameters['transformer.h.'+ str(block_num) + '.ln_1.bias']
        ln_1 = self.layer_norm(emb, weights, bias)

        context_matrix = self.matrix_self_attn(ln_1, block_num, mask = mask)

        context_matrix = context_matrix + emb # Residual Connection
        trace.file_write(f'residual',f'{context_matrix.shape}+{emb.shape}')

        weights = self.parameters['transformer.h.'+ str(block_num) + '.ln_2.weight']
        bias = self.parameters['transformer.h.'+ str(block_num) + '.ln_2.bias']
        ln_2 = self.layer_norm(context_matrix, weights, bias)

        emb_mlp = self.mlp(ln_2, block_num)

        emb_mlp = context_matrix + emb_mlp # Residual Connection
        trace.file_write(f'residual',f'{context_matrix.shape}+{emb_mlp.shape}')
        return emb_mlp

    def next_token(self, tok):
        '''
        Generates the next token in sequence
        tok (np_array): 1D token encodigns
        '''
        emb = self.embed(tok) #(tokens, Embedding Size 768)

        mask = np.full((emb.shape[0], emb.shape[0]), float(0))
        mask_value = np.finfo(np.float32).min
        mask[np.triu_indices_from(mask, k=1)] = mask_value
        trace.file_write(f'make_mask',f'{mask.shape}')


        block_result = emb
        for block_num in range(self.decode_blocks):
            block_result = self.decode_block(block_result, block_num, mask = mask)

        weights = self.parameters['transformer.ln_f.weight']
        bias = self.parameters['transformer.ln_f.bias']
        ln_f = self.layer_norm(block_result, weights, bias)

        weights = self.parameters['lm_head.weight']
        logit_matrix = ln_f @ weights.T
        trace.file_write(f'logit_matrix',f'{ln_f.shape}@{weights.T.shape}')

        # Logit selection
        logit_wrappers = {'greedy': lambda x: np.argmax(x[-1]),
            'top_k': self.top_k}

        return logit_wrappers[self.logit_strat](logit_matrix)

    def generate(self, prompt, max_token_len = 100):
        '''
        creates generation feedback loop
        prompt(srt)
        tokenizer
        max_token_len(int): length of resulting sequence
        '''

        tok = self.tokenizer.encode(prompt, return_tensors='np').squeeze()
        trace.file_write(f'encode', f'{tok.shape}')
        ans = np.array([])
        for i in range(max_token_len-tok.shape[0]):
            print('.' * i)
            if ans.size == 0: #is empty
                next_tok = self.next_token(tok)
                ans = np.append(tok, next_tok)
            else:
                next_tok = self.next_token(ans[-1])
                ans = np.append(ans, next_tok)


        tok = self.tokenizer.decode(ans, skip_special_tokens=True)
        trace.file_write(f'decode', f'{ans.shape}')
        return tok
