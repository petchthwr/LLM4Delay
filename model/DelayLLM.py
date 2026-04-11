from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from math import sqrt
warnings.simplefilter(action='ignore', category=FutureWarning)

def merge_and_pad_known_max(tensor_list, pad_value=float('nan')):
    """
    tensor_list: list of tensors [tensor_1, tensor_2, ..., tensor_k]
                 each tensor has shape (B, N_i, D)
    pad_value: value to pad
    """
    B = tensor_list[0].shape[0]
    D = tensor_list[0].shape[2]

    # Get max valid length of each tensor over batch
    max_valid_lens = []
    cleaned = []

    for tensor in tensor_list:
        mask = ~torch.isnan(tensor).any(dim=-1)  # (B, N_i) valid mask
        cleaned.append([tensor[b][mask[b]] for b in range(B)])
        max_valid_lens.append(mask.sum(dim=1).max().item())  # scalar

    N_max = sum(max_valid_lens)

    merged = []
    for b in range(B):
        merged_b = torch.cat([x[b] for x in cleaned], dim=0)  # concat valid rows
        padded = F.pad(merged_b, (0, 0, 0, N_max - merged_b.shape[0]), value=pad_value)
        merged.append(padded)

    return torch.stack(merged)  # (B, N_max, D)
def load_llm_model(model_name: str, token: str, rope_factor: float = 2.0, use_bfloat16: bool = False):
    dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    # Detect if rope_scaling is needed and safe to pass
    rope_scaling = None
    supports_rope = False

    if "phi-4" in model_name.lower():
        rope_scaling = {
            "type": "longrope",
            "short_factor": [rope_factor] * 48,
            "long_factor": [rope_factor] * 48
        }
        supports_rope = True
    elif any(name in model_name.lower() for name in ["llama", "qwen", "deepseek"]):
        rope_scaling = {
            "type": "dynamic",
            "factor": rope_factor
        }
        supports_rope = True

    # Load config (only pass rope_scaling if supported)
    if supports_rope:
        config = AutoConfig.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True,
            rope_scaling=rope_scaling
        )
    else:
        config = AutoConfig.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
        token=token
    )

    return tokenizer, model

"""
TimeLLM: Time-series Patching and Reprogramming Layer
M. Jin, S. Wang, L. Ma, Z. Chu, J. Y. Zhang, X. Shi, P.-Y. Chen, Y. Liang, Y.-F. Li, S. Pan, and Q. Wen,
“Time-LLM: Time series forecasting by reprogramming large language models,”
in International Conference on Learning Representations (ICLR), 2024.
"""
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input):
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        if (x.shape[-1] - self.patch_len) % self.stride != 0:
            pad_len = self.stride - ((x.shape[-1] - self.patch_len) % self.stride)
            x = F.pad(x, (0, pad_len), "constant", 0)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x), n_vars
class TimeLLM_z_projection(nn.Module):
    def __init__(self, x_emb_dim, llm_emb_dim, word_embedding_weights):
        super().__init__()
        self.x_enc_dim = x_emb_dim
        self.llm_emb_dim = llm_emb_dim

        # Layers
        self.word_embeddings = word_embedding_weights
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 100
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.patch_embedding = PatchEmbedding(x_emb_dim, 96, 48, 0.1)
        self.reprogramming_layer = ReprogrammingLayer(
            d_model=x_emb_dim, n_heads=8, d_keys=x_emb_dim // 8, d_llm=llm_emb_dim, attention_dropout=0.1
        )
        self.max_context_length = 2048

    def forward(self, x_enc): # z is a tensor of shape (B, T_max, z_dim)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            B, T, _ = x_enc.shape
            source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
            x_enc = x_enc.permute(0, 2, 1).contiguous()
            enc_out, nvar = self.patch_embedding(x_enc.to(torch.bfloat16))
            enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
            enc_out = enc_out.reshape(B, nvar * enc_out.shape[1], -1)
            if enc_out.shape[1] > self.max_context_length:
                enc_out = enc_out[:, :self.max_context_length, :]
        return enc_out

"""
AutoTimes: Time-series Segment Embedding
Y. Liu, G. Qin, X. Huang, J. Wang, and M. Long, 
“Autotimes: Autoregressive time series forecasters via large language models,”
in Advances in Neural Information Processing Systems, 
A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds.,
vol. 37. Curran Associates, Inc., 2024, pp. 122 154–122 184.
"""
class AutoTimes_MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''

    def __init__(self,
                 f_in,
                 f_out,
                 hidden_dim=256,
                 hidden_layers=2,
                 dropout=0.1,
                 activation='tanh'):
        super(AutoTimes_MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim),
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers - 2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]

        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y
class AutoTimes_z_projection(nn.Module):
    def __init__(self, token_len, llm_emb_dim, use_amp: bool = True):
        super().__init__()
        self.token_len = token_len
        if self.token_len <= 0:
            raise ValueError(f"token_len must be > 0. Got {token_len}")
        self.llm_emb_dim = llm_emb_dim

        self.encoder = AutoTimes_MLP(f_in=token_len, f_out=llm_emb_dim)
        self.use_amp = use_amp
        self.max_context_length = 2048

    def forward(self, x_enc):
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.float16):
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            bs, _, n_vars = x_enc.shape
            x_enc = x_enc.permute(0, 2, 1) # x_enc: [bs x nvars x seq_len]
            x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1) # x_enc: [bs * nvars x seq_len]
            if x_enc.shape[-1] % self.token_len != 0: # Ensure divisibility
                pad_len = self.token_len - (x_enc.shape[-1] % self.token_len)
                x_enc = F.pad(x_enc, (0, pad_len), "constant", 0)
            fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len) # fold_out: [bs * n_vars x token_num x token_len]
            token_num = fold_out.shape[1] # No "mix" option for our case
            times_embeds = self.encoder(fold_out) # [bs * n_vars x token_num x hidden_dim_of_llama]
            times_embeds = times_embeds.reshape(bs, n_vars * token_num, -1) # [bs x n_vars * token_num x hidden_dim_of_llama]
            if times_embeds.shape[1] > self.max_context_length:
                times_embeds = times_embeds[:, :self.max_context_length, :]
        return times_embeds

class LlamaDelayPredModel(nn.Module):
    def __init__(self, model_name, token, rope_factor=2.0, z_dim=320, task='dt_airspace', ablation_tag=None):
        super().__init__()
        self.task = task
        self.ablation_tag = ablation_tag

        # LLM model backbone
        self.llm_model_name = model_name
        self.tokenizer, self.model = load_llm_model(model_name, token, rope_factor)
        print(self.model)

        # Core model selection
        if hasattr(self.model, "model"):
            core_model = self.model.model
        elif hasattr(self.model, "transformer"):
            core_model = self.model.transformer
        elif hasattr(self.model, "gpt_neox"):
            core_model = self.model.gpt_neox
        else:
            raise ValueError(f"Unsupported model structure for {self.llm_model_name}")

        # Embedding extraction
        self.embed_tokens = (
            core_model.embed_tokens
            if hasattr(core_model, "embed_tokens")
            else core_model.wte
            if hasattr(core_model, "wte")
            else core_model.embed_in
            if hasattr(core_model, "embed_in")
            else None
        )

        if self.embed_tokens is None:
            raise ValueError(f"Could not locate embedding layer for {self.llm_model_name}")

        # Freeze layers based on user input
        self._freeze(self.model, True)
        self._freeze(self.embed_tokens, True)

        # Set padding token id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # z projection layer
        if self.ablation_tag == 'TimeLLM':
            self.z_projection = TimeLLM_z_projection(
                x_emb_dim=16,
                llm_emb_dim=self.model.config.hidden_size,
                word_embedding_weights=self.embed_tokens.weight
            )
        elif self.ablation_tag == 'AutoTimes':
            self.z_projection = AutoTimes_z_projection(
                token_len=96,
                llm_emb_dim=self.model.config.hidden_size,
                use_amp=True
            )
        else:
            self.z_dim = z_dim
            self.z_projection = nn.Sequential(
                nn.Linear(self.z_dim, self.model.config.hidden_size),
                nn.GELU(),
                nn.Dropout(0.35),  # Use a small dropout for the projection layer
                nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
                nn.GELU(),
                nn.Dropout(0.35),
            )

        # Model output head (MLP)
        self.output_dims = 1
        self.linear_proj = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.model.config.hidden_size // 2, self.output_dims),
        ).to(torch.float32)

        self.unavailable_txt_prompt = ('Textual information is unavailable for this sample; '
                                       'therefore, no general flight information, weather reports (including METAR and TAF), '
                                       'or operational notices (NOTAM) are provided as part of the prediction context. '
                                       'In this setting, all forms of textual data that would normally describe flight conditions, '
                                       'environmental factors, or airspace constraints are intentionally omitted. '
                                       'As a result, the prediction must be performed without access to any supplementary descriptive information, '
                                       'relying solely on the remaining non-textual inputs.')

    def _freeze(self, module, freeze):
        if freeze:
            for param in module.parameters():
                param.requires_grad = False

    def embed_language(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)  # (B, S, E)
        pad_mask = (input_ids == self.tokenizer.pad_token_id) # (B, S)
        hidden_states[pad_mask] = float('nan')
        return hidden_states

    def embed_traj_tokens(self, z): # z is a tensor of shape (B, T_max, z_dim)
        nan_mask = z.isnan().any(dim=-1) # (B, T_max)
        z[nan_mask] = 0
        z = self.z_projection(z)  # (B, T_max, E)
        z[nan_mask] = torch.nan
        return z

    def get_standard_trajectory_prompt(self, ablation_tag=None):
        before_focus = (
            "Airspace is described using three trajectory types. This embedding is for the focus trajectory: {"
        )
        before_active = (
            "} These embeddings are for other active trajectories: {"
        )
        before_affecting = (
            "} These embeddings are for past or inactive trajectories that may still matter: {"
        )
        closing_prompt = (
            " "
        )
        if self.task == 'dt_airspace':
            closing_prompt += "The predicted total time spent in the airspace is: "
        else:
            raise ValueError(f"Unknown task: {self.task}")

        if ablation_tag == 'exclude_focusing':
            before_focus = (
                "The airspace condition is described using three types of trajectory embeddings. "#, representing aircraft that are currently within, previously operated within, or otherwise associated with the airspace. "
                "The the focusing trajectory is unavailable denoted by empty brackets. {"
            )
        if ablation_tag == 'exclude_active':
            before_active = (
                "} The active trajectory is unavailable denoted by empty brackets. {"
            )
        if ablation_tag == 'exclude_affecting':
            before_affecting = (
                "} The affecting trajectory is unavailable denoted by empty brackets. {"
            )

        return before_focus, before_active, before_affecting, closing_prompt

    def tokenize_embed_repeat(self, prompt, repeat):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt', truncation=True, add_special_tokens=False).to(self.embed_tokens.weight.device)
        embedded_prompt = self.embed_language(tokenized_prompt['input_ids'])
        embedded_prompt = embedded_prompt.repeat(repeat, 1, 1)
        return embedded_prompt

    def arrange_z(self, z_focusing, z_active, z_affecting, ablation_tag=None):
        B = z_focusing.shape[0]

        # raise error for unknown ablation_tag
        if ablation_tag not in ['exclude_focusing', 'exclude_active', 'exclude_affecting', 'exclude_trajectory', 'exclude_text', None]:
            raise ValueError(f"Unknown ablation_tag: {ablation_tag}")

        # If ablation_tag is 'exclude_trajectory', return unavailable trajectory prompt
        if ablation_tag == 'exclude_trajectory':
            unavailable_traj_prompt = 'All trajectory data is unavailable! {' # unvailable trajectory prompt
            emb_z = self.tokenize_embed_repeat(unavailable_traj_prompt, B)
            return emb_z

        # Get standard trajectory prompt
        before_focus, before_active, before_affecting, _ = self.get_standard_trajectory_prompt(ablation_tag=ablation_tag)

        # Trajectory embeddings
        z_focusing = self.tokenize_embed_repeat(before_focus, B) if ablation_tag == 'exclude_focusing' \
            else torch.cat((self.tokenize_embed_repeat(before_focus, B), z_focusing), dim=1)

        z_active = self.tokenize_embed_repeat(before_active, B) if ablation_tag == 'exclude_active' \
            else torch.cat((self.tokenize_embed_repeat(before_active, B), z_active), dim=1)

        z_affecting = self.tokenize_embed_repeat(before_affecting, B) if ablation_tag == 'exclude_affecting' \
            else torch.cat((self.tokenize_embed_repeat(before_affecting, B), z_affecting), dim=1)

        # Merge and pad trajectory components
        emb_z = merge_and_pad_known_max([z_focusing, z_active, z_affecting])

        return emb_z # Shape: (B, N_max, E); Padded with NaN

    def convert_nan_vector_to_pad_token(self, x):
        B, S, E = x.shape
        device = x.device

        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id.")

        pad_token_id = self.tokenizer.pad_token_id
        emb_pad = self.embed_tokens(torch.tensor([pad_token_id], device=device, dtype=torch.long))  # (1, E)
        emb_pad = emb_pad.view(1, 1, E).expand(B, S, E)  # (B, S, E)

        nan_mask = torch.isnan(x).all(dim=-1, keepdim=True)  # Detect NaN vectors (B, S, 1)
        x = torch.where(nan_mask, emb_pad, x) # Replace NaN rows with pad embedding

        # Attention mask → 1 = real token, 0 = padding in Hugging Face convention; we invert the nan_mask for this
        attn_mask = (~nan_mask).squeeze(-1).long()  # (B, S)

        return x, attn_mask

    def forward(self, flight_prompt, z):
        dtype = next(self.parameters()).dtype

        # Language & Trajectory Embedding
        x = self.embed_language(flight_prompt) if self.ablation_tag != 'exclude_text' \
            else self.tokenize_embed_repeat(self.unavailable_txt_prompt, flight_prompt.shape[0])

        # Three additional ablation tags: LLMTIME, TimeLLM, AutoTimes if not these three process normally
        if self.ablation_tag == 'LLMTIME':
            xz = x # traj array are tokenized prompt since the data loader z is empty tensor
        elif self.ablation_tag == 'TimeLLM' or self.ablation_tag == 'AutoTimes':
            z = z['focusing']
            z = self.z_projection(z)  # (B, N, E)
            opening_prompt = self.tokenize_embed_repeat('Airspace is defined by 3 concatenated trajectory types:', flight_prompt.shape[0])
            closing_prompt = self.tokenize_embed_repeat('The predicted total time spent in the airspace is: ', flight_prompt.shape[0])
            xz = merge_and_pad_known_max([x, opening_prompt, z, closing_prompt])  # (B, S, E) # Merge & pad language & trajectory embeddings
        else:
            closing_prompt = self.tokenize_embed_repeat('The predicted total time spent in the airspace is: ', flight_prompt.shape[0])
            z_focusing = self.embed_traj_tokens(z['focusing'])  # (B, N1, E)
            z_active = self.embed_traj_tokens(z['active'])  # (B, N2, E)
            z_affecting = self.embed_traj_tokens(z['affecting'])  # (B, N3, E)
            z_focusing = F.dropout(z_focusing, p=0.3) if self.training else z_focusing
            z_active = F.dropout(z_active, p=0.3) if self.training else z_active
            z_affecting = F.dropout(z_affecting, p=0.3) if self.training else z_affecting
            z = self.arrange_z(z_focusing, z_active, z_affecting, ablation_tag=self.ablation_tag)  # (B, N, E)
            xz = merge_and_pad_known_max([x, z, closing_prompt])  # (B, S, E) # Merge & pad language & trajectory embeddings

        # Replace NaN vectors with PAD token embedding & build attention mask
        xz, attn_mask = self.convert_nan_vector_to_pad_token(xz)  # (B, S, E), (B, S)

        # For TimeLLM and AutoTimes, cut it max valid length
        if self.ablation_tag in ['TimeLLM', 'AutoTimes', 'LLMTIME']:
            valid_len = attn_mask.sum(dim=1).max().item()
            xz = xz[:, :valid_len, :] # (B, valid_len, E)
            attn_mask = attn_mask[:, :valid_len] # (B, valid_len)

        xz = xz.to(device=xz.device, dtype=dtype)
        position_ids = torch.arange(xz.size(1), device=xz.device).unsqueeze(0).expand(xz.size(0), -1)  # (B, S)

        # When debug on server
        if self.ablation_tag == 'LLMTIME':
            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    if hasattr(self.model, "gpt_neox"): # Pythia / GPT-NeoX
                        out = self.model.gpt_neox(inputs_embeds=xz, attention_mask=attn_mask, position_ids=position_ids, use_cache=False, return_dict=True)
                    else: # Llama-like + Qwen-like (your existing path)
                        out = self.model.model(inputs_embeds=xz, attention_mask=attn_mask, position_ids=position_ids, use_cache=False, return_dict=True)
                    xz = out.last_hidden_state
        else:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                if hasattr(self.model, "gpt_neox"):  # Pythia / GPT-NeoX
                    out = self.model.gpt_neox(inputs_embeds=xz, attention_mask=attn_mask, position_ids=position_ids, use_cache=False, return_dict=True)
                else:  # Llama-like + Qwen-like (your existing path)
                    out = self.model.model(inputs_embeds=xz,attention_mask=attn_mask, position_ids=position_ids, use_cache=False, return_dict=True)
                xz = out.last_hidden_state

        # Take last valid token based on attention_mask
        last_valid_index = attn_mask.sum(dim=1).clamp(min=1) - 1  # (B,)
        last_token = torch.gather(xz, dim=1, index=last_valid_index.view(-1, 1, 1).expand(-1, 1, xz.size(-1)))  # (B, 1, E)

        # Final projection
        xz = self.linear_proj(last_token.squeeze(1).float())  # (B, output_dim)
        xz = xz.view(-1)
        return xz

    def backbone_passing(self, xz, attn_mask, position_ids):
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                if hasattr(self.model, "gpt_neox"):  # Pythia / GPT-NeoX
                    out = self.model.gpt_neox(inputs_embeds=xz, attention_mask=attn_mask, position_ids=position_ids, use_cache=False, return_dict=True)
                else:  # Llama-like + Qwen-like
                    out = self.model.model(inputs_embeds=xz, attention_mask=attn_mask, position_ids=position_ids, use_cache=False, return_dict=True)
                xz = out.last_hidden_state
        return xz

    def inference_with_H(self, flight_prompt, z, replace_mode="pad_token"):
        self.eval()
        with torch.inference_mode():
            # Prepare Input Embeddings
            x = self.embed_language(flight_prompt)  # (B, Lx, D)
            z = self.arrange_z(
                self.embed_traj_tokens(z['focusing']),
                self.embed_traj_tokens(z['active']),
                self.embed_traj_tokens(z['affecting']),
                ablation_tag=self.ablation_tag
            )
            closing_prompt = self.tokenize_embed_repeat(
                'The predicted total time spent in the airspace is: ',
                flight_prompt.shape[0]
            )
            xz = merge_and_pad_known_max([x, z, closing_prompt])
            xz, attn_mask = self.convert_nan_vector_to_pad_token(xz)

            # Crop to max valid length
            valid_len = attn_mask.sum(dim=1).max().item()
            xz = xz[:, :valid_len, :] # (B, S, E)
            attn_mask = attn_mask[:, :valid_len] # (B, S)
            position_ids = torch.arange(xz.size(1), device=xz.device).unsqueeze(0).expand(xz.size(0), -1)  # (B, S)

            summary, per_sample = self.get_prompt_part_ranges_from_xz_batch(xz, attn_mask.sum(dim=1).tolist())

            # Inference for original input
            y_hidden = self.backbone_passing(xz, attn_mask, position_ids)
            last_idx = attn_mask.sum(dim=1).clamp(min=1) - 1
            last_token = torch.gather(y_hidden, dim=1, index=last_idx.view(-1, 1, 1).expand(-1, 1, xz.size(-1)))
            y_base = self.linear_proj(last_token.squeeze(1).float()).view(-1)  # (B,)

            # ----- Sensitivity matrix -----
            B, Lxz, D = xz.shape
            H = torch.zeros(B, Lxz, device=x.device, dtype=torch.float32)

            # optional replacement baseline
            if replace_mode == "mean":
                # (B, 1, D)
                replacement_vec = x.mean(dim=1, keepdim=True)
            elif replace_mode == "zero":
                replacement_vec = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
            elif replace_mode == "pad_token":
                pad_token_id = self.tokenizer.pad_token_id
                replacement_vec = self.embed_tokens(torch.tensor([pad_token_id], device=x.device, dtype=torch.long)).view(1, 1, D).expand(B, 1, D)
            else:
                raise ValueError("replace_mode must be 'zero' or 'mean' or 'pad_token'")

            for j in range(Lxz):
                xz_pert = xz.clone()
                xz_pert[:, j:j + 1, :] = replacement_vec
                y_hidden_pert = self.backbone_passing(xz_pert, attn_mask, position_ids)
                last_token_pert = torch.gather(y_hidden_pert, dim=1, index=last_idx.view(-1, 1, 1).expand(-1, 1, xz.size(-1)))
                y_pert = self.linear_proj(last_token_pert.squeeze(1).float()).view(-1)  # (B,)
                delta = y_pert - y_base
                H[:, j] = delta.abs()

        return y_base, H, summary

    def find_pattern_in_embedding_sequence(self, seq, pat, atol=1e-6, rtol=1e-5):
        """
        seq: (L, D)
        pat: (P, D)
        return: start index or None
        """
        pat = pat.to(device=seq.device, dtype=seq.dtype)

        L, D = seq.shape
        P, Dp = pat.shape
        if P > L or D != Dp:
            return None

        for i in range(L - P + 1):
            if torch.allclose(seq[i:i + P], pat, atol=atol, rtol=rtol):
                return i
        return None

    def get_prompt_part_ranges_from_xz_batch(self, xz, valid_lens, atol=1e-6, rtol=1e-5):
        if isinstance(valid_lens, torch.Tensor):
            valid_lens = valid_lens.tolist()

        B = xz.shape[0]
        # embed patterns using your helper
        metar_pat = self.tokenize_embed_repeat(self.tokenizer.decode([1277, 275, 1055, 27]), B)
        taf_pat = self.tokenize_embed_repeat(self.tokenizer.decode([9006, 275, 1055, 27]), B)
        notam_pat = self.tokenize_embed_repeat(self.tokenizer.decode([84, 27]), B)
        focusing_pat = self.tokenize_embed_repeat("This embedding is for the focus trajectory: {", B)
        active_pat = self.tokenize_embed_repeat("} These embeddings are for other active trajectories: {", B)
        prior_pat = self.tokenize_embed_repeat(
            "} These embeddings are for past or inactive trajectories that may still matter: {", B)
        closing_pat = self.tokenize_embed_repeat("The predicted total time spent in the airspace is: ", B)

        per_sample = []
        for b in range(B):
            seq = xz[b, :valid_lens[b]]  # (Lv, D)

            metar_begin = self.find_pattern_in_embedding_sequence(seq, metar_pat[b], atol=atol, rtol=rtol)
            taf_begin = self.find_pattern_in_embedding_sequence(seq, taf_pat[b], atol=atol, rtol=rtol)
            notam_begin = self.find_pattern_in_embedding_sequence(seq, notam_pat[b], atol=atol, rtol=rtol)
            focusing_begin = self.find_pattern_in_embedding_sequence(seq, focusing_pat[b], atol=atol, rtol=rtol)
            active_begin = self.find_pattern_in_embedding_sequence(seq, active_pat[b], atol=atol, rtol=rtol)
            prior_begin = self.find_pattern_in_embedding_sequence(seq, prior_pat[b], atol=atol, rtol=rtol)
            closing_begin = self.find_pattern_in_embedding_sequence(seq, closing_pat[b], atol=atol, rtol=rtol)

            flight_info_begin = 0
            flight_info_end = metar_begin

            metar_end = taf_begin
            taf_end = notam_begin
            notam_end = focusing_begin
            focusing_end = active_begin
            active_end = prior_begin
            prior_end = closing_begin
            closing_end = valid_lens[b]

            per_sample.append({
                "flight_info_begin": flight_info_begin,
                "flight_info_end": flight_info_end,

                "metar_begin": metar_begin,
                "metar_end": metar_end,

                "taf_begin": taf_begin,
                "taf_end": taf_end,

                "notam_begin": notam_begin,
                "notam_end": notam_end,

                "focusing_begin": focusing_begin,
                "focusing_end": focusing_end,

                "active_begin": active_begin,
                "active_end": active_end,

                "prior_begin": prior_begin,
                "prior_end": prior_end,

                "closing_begin": closing_begin,
                "closing_end": closing_end,
            })

        def minmax(begin_key, end_key):
            begins = [d[begin_key] for d in per_sample if d[begin_key] is not None]
            ends = [d[end_key] for d in per_sample if d[end_key] is not None]
            if len(begins) == 0 or len(ends) == 0:
                return None, None
            return min(begins), max(ends)

        summary = {
            "min_flight_info_begin": minmax("flight_info_begin", "flight_info_end")[0],
            "max_flight_info_end": minmax("flight_info_begin", "flight_info_end")[1],

            "min_metar_begin": minmax("metar_begin", "metar_end")[0],
            "max_metar_end": minmax("metar_begin", "metar_end")[1],

            "min_taf_begin": minmax("taf_begin", "taf_end")[0],
            "max_taf_end": minmax("taf_begin", "taf_end")[1],

            "min_notam_begin": minmax("notam_begin", "notam_end")[0],
            "max_notam_end": minmax("notam_begin", "notam_end")[1],

            "min_focusing_begin": minmax("focusing_begin", "focusing_end")[0],
            "max_focusing_end": minmax("focusing_begin", "focusing_end")[1],

            "min_active_begin": minmax("active_begin", "active_end")[0],
            "max_active_end": minmax("active_begin", "active_end")[1],

            "min_prior_begin": minmax("prior_begin", "prior_end")[0],
            "max_prior_end": minmax("prior_begin", "prior_end")[1],

            "min_closing_begin": minmax("closing_begin", "closing_end")[0],
            "max_closing_end": minmax("closing_begin", "closing_end")[1],
        }

        return summary, per_sample