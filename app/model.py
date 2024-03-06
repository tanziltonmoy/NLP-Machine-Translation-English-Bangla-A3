import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer


# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

SRC_LANGUAGE = 'en'  # Source language code (English).
TRG_LANGUAGE = 'bn'  # Target language code (Bengali).
token_transform = {}
vocab_transform = {}

# Configure tokenizers for the target and source languages using spaCy models.
token_transform[TRG_LANGUAGE] = get_tokenizer('spacy', language='xx_ent_wiki_sm')  # Target language tokenizer
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')  # Source language tokenizer

class AttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0  # Ensure the hidden dim is divisible by the number of heads

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        # Define linear layers for Q, K, and V
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        # Output linear layer
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        # Define linear layers for additive attention
        self.W1 = nn.Linear(self.head_dim, self.head_dim)  # Transform hi (K)
        self.W2 = nn.Linear(self.head_dim, self.head_dim)  # Transform s (Q)
        self.v = nn.Linear(self.head_dim, 1)  # Projection vector v

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear transformations for Q, K, V
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Reshape and permute for multi-head attention
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)


        # Ensure the reshaping and operations align correctly
        transformed_Q = self.W1(Q.unsqueeze(3))  # [batch_size, n_heads, key_len, head_dim]
        transformed_K = self.W2(K.unsqueeze(2))  # [batch_size, n_heads, query_len, head_dim]
        # Since the error mentions a mismatch, double-check the shapes of K and Q before transformation

        # The broadcasting should work without unsqueeze if dimensions are aligned correctly
        # If unsqueeze is used, it should be to introduce a compatible dimension for broadcasting
        energy = torch.tanh(transformed_Q + transformed_K)
        # print("energy shape before squeeze(-1):", energy.shape)
        energy = self.v(energy).squeeze(-1)
        # print("energy shape after squeeze(-1):", energy.shape)

        # Apply mask (if provided) to ignore certain positions in the attention scores
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # Compute attention weights by applying softmax to the energy scores
        attention = torch.softmax(energy, dim=-1)

        # Apply the attention weights to the value vector V
        x = torch.matmul(self.dropout(attention), V)

        # Reshape and project the output back to the original hid_dim
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)

        return x, attention



class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x = [batch size, src len, hid dim]
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x



class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)
        self.self_attention       = AttentionLayer(hid_dim, n_heads, dropout, device)
        self.feedforward          = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout              = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len]   #if the token is padding, it will be 1, otherwise 0
        _src, _ = self.self_attention(src, src, src, src_mask)
        src     = self.self_attn_layer_norm(src + self.dropout(_src))
        #src: [batch_size, src len, hid dim]

        _src    = self.feedforward(src)
        src     = self.ff_layer_norm(src + self.dropout(_src))
        #src: [batch_size, src len, hid dim]

        return src


   
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 500):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers        = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                           for _ in range(n_layers)])
        self.dropout       = nn.Dropout(dropout)
        self.scale         = torch.sqrt(torch.FloatTensor([hid_dim])).to(self.device)

    def forward(self, src, src_mask):

        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len    = src.shape[1]

        pos        = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos: [batch_size, src_len]

        src        = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        #src: [batch_size, src_len, hid_dim]

        for layer in self.layers:
            src = layer(src, src_mask)
        #src: [batch_size, src_len, hid_dim]

        return src



class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm  = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)
        self.self_attention       = AttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention    = AttentionLayer(hid_dim, n_heads, dropout, device)
        self.feedforward          = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout              = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg     = self.self_attn_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]

        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg             = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]
        #attention = [batch_size, n heads, trg len, src len]

        _trg = self.feedforward(trg)
        trg  = self.ff_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]

        return trg, attention



class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 500):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers        = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                            for _ in range(n_layers)])
        self.fc_out        = nn.Linear(hid_dim, output_dim)
        self.dropout       = nn.Dropout(dropout)
        self.scale         = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len    = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos: [batch_size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        #trg: [batch_size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        #trg: [batch_size, trg len, hid dim]
        #attention: [batch_size, n heads, trg len, src len]

        output = self.fc_out(trg)
        #output = [batch_size, trg len, output_dim]

        return output, attention




class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        # store the input parameters so we can retrive them later for model inferencing
        self.params = {'encoder': encoder, 'decoder': decoder,
                       'src_pad_idx': src_pad_idx, 'trg_pad_idx': trg_pad_idx}
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):

        #src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        #trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        #trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        #trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask
        #trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):

        #src = [batch size, src len]
        #trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)
        #enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]

        return output, attention



def translate_sentence(sentence, src_vocab_transform, trg_vocab_transform, model, device, max_len=50):
    """
    Translates a given sentence from the source language to the target language using a specified model.
    
    Parameters:
    - sentence (str): The sentence in the source language to be translated.
    - src_vocab_transform (dict): A mapping from source language tokens to indices.
    - trg_vocab_transform (dict): A mapping from target language tokens to indices.
    - model (Seq2SeqTransformer): The trained sequence-to-sequence model for translation.
    - device (torch.device): The device (CPU or GPU) on which the computation will be performed.
    - max_len (int): The maximum length of the translated sentence.
    
    Returns:
    - str: The translated sentence in the target language.
    """
    model.eval()  # Switch the model to evaluation mode.

    # Tokenize the input sentence and add start-of-sentence (<sos>) and end-of-sentence (<eos>) tokens.
    tokens = ['<sos>'] + [token.lower() for token in token_transform[SRC_LANGUAGE](sentence)] + ['<eos>']
    src_indexes = [src_vocab_transform[token] for token in tokens]

    # Convert the token indices to a tensor and add a batch dimension.
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # Generate the source mask for the input tensor.
    src_mask = model.make_src_mask(src_tensor)

    # Forward pass through the encoder.
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
        trg_indexes = [trg_vocab_transform['<sos>']]  # Initialize the target sequence with <sos> token.

        for _ in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)

            # Decode the target tensor to generate the next token in the sequence.
            output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)

            # Break the loop if the end-of-sentence token is generated.
            if pred_token == trg_vocab_transform['<eos>']:
                break

    # Convert the target indices back to tokens and join them to form the translated sentence.
    trg_tokens = [trg_vocab_transform.get_itos()[i] for i in trg_indexes]

    return ' '.join(trg_tokens[1:-1])  # Exclude the <sos> and <eos> tokens in the final sentence.