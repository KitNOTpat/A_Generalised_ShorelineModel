 # model defenition

import torch
import torch.nn as nn
import numpy as np
import math 
import torch.nn.init as init

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import matplotlib.pyplot as plt
import copy

# #####################################################
# #####################################################

class TFT(nn.Module):
    '''
    Temporal Fusion Transformer Model (Encoder: LSTM & Self ATTN)
    '''

    def __init__(self, settings):

        super(TFT, self).__init__()

        self.nan_token = -9999

        # architecture
        self.use_staticEnrichment = True
        self.use_sequentialProcessing = settings['use_sequentialProcessing']
        self.use_multiheadAttention = settings['use_multiheadAttention']
        
        self.historical_N = len(settings['dynamic_inputs'] or []) + (1 if settings['use_prev_target'] else 0) + 2 # +2 for time encoding, +1 for the pre target

        self.future_N = len(settings['dynamic_inputs'] or []) + 2 # +2 for time encoding
        self.num_static_N = len(settings['numerical_static_inputs'])
        self.cat_static_N = len(settings['catagorical_static_inputs'] or [])

        self.d_model = settings['d_model']  # Hidden dimension for LSTM
        self.n_heads = settings['attention_heads']
        self.lstm_layers = settings['lstm_layers']

        self.L = settings['sequence_length']
        self.B = settings['batch_size']
        self.output_dim = 3  # Number of Quantiles to predict...
        self.pred_len = settings['output_length']  # Prediction length
        self.dropout = settings['neuron_dropout']
        

        # =====================
        # Static Embedding/Enrichment Networks...
        # =====================
        
        self.static_encoder_enrichment = GatedResidualNetwork(input_dim=self.num_static_N + self.cat_static_N,
                                                        d_model=self.d_model,
                                                        output_dim=self.d_model,
                                                        dropout=self.dropout)
        self.static_encoder_sequential_cell_init = copy.deepcopy(self.static_encoder_enrichment)
        self.static_encoder_sequential_state_init = copy.deepcopy(self.static_encoder_enrichment)

        self.static_enrichment_grn_attn = GatedResidualNetwork(input_dim=self.d_model,
                                                          d_model=self.d_model,
                                                          output_dim=self.d_model,
                                                          dropout=self.dropout)
        self.static_enrichment_grn_past = GatedResidualNetwork(input_dim=self.historical_N,
                                                               d_model=self.d_model,
                                                               output_dim=self.d_model,
                                                               dropout=self.dropout)
        self.static_enrichment_grn_future = GatedResidualNetwork(input_dim=self.future_N,
                                                               d_model=self.d_model,
                                                               output_dim=self.d_model,
                                                               dropout=self.dropout)

        # =====================
        # Sequential Processing Networks...
        # =====================

        # LSTM encoder
        self.lstm_encoder = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, 
                                    num_layers=self.lstm_layers,
                                    dropout=self.dropout,
                                    batch_first=True)
        # LSTM encoder
        self.lstm_decoder = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, 
                                    num_layers=self.lstm_layers,
                                    dropout=self.dropout,
                                    batch_first=True)
        
        # =====================
        # Global Attention Networks...
        # =====================
        
        self.multihead_attention =  nn.MultiheadAttention(embed_dim=self.d_model,
                                                          num_heads=self.n_heads,
                                                          batch_first = True)
        
        self.positional_encoding = PositionalEncoding(self.d_model)

        # =====================
        # Output Networks...
        # =====================
    
        self.PositionWiseFF = GatedResidualNetwork(input_dim=self.d_model,
                                                   d_model=self.d_model,
                                                   output_dim=self.d_model,
                                                   dropout=self.dropout)

        self.transect_heads = nn.ModuleList([
                                    nn.Linear(self.d_model, 1) for _ in range(self.output_dim)
                                ])

        # =====================
        # Other Layers...
        # =====================
        self.layer_dropout = nn.Dropout(self.dropout)
        self.Gate = LogisticGate(self.d_model)

#####################################################
#####################################################
   
    def forward(self, x_past, x_future, Xs_num, Xs_cat):
        
        x_static_num, x_static_cat = Xs_num, Xs_cat
        x_static = torch.cat((x_static_num,x_static_cat), dim=-1)

        # Clone x_past & x_future to avoid modifying original tensors used for masking
        x_past_no_token = x_past.clone()
        x_past_no_token[x_past_no_token == -9999] = 0  # Replace NaN tokens with 0

        x_future_no_token = x_future.clone()
        x_future_no_token[x_future_no_token == -9999] = 0  # Replace NaN tokens with 0

        # =========== Transform all input channels ==============

        c_enrichment, c_seq_cell, c_seq_hidden = self.get_static_encoders(x_static)

        # embedded_input = torch.cat([x_past_emb, x_future_emb], dim=1)

        enriched_sequence_past = self.static_enrichment(
                                                    x = x_past_no_token, 
                                                    enrichment = c_enrichment, 
                                                    layer = 'past')
        enriched_sequence_future = self.static_enrichment(
                                                    x = x_future_no_token, 
                                                    enrichment = c_enrichment, 
                                                    layer = 'future')
        
        enriched_sequence = torch.cat([enriched_sequence_past, enriched_sequence_future], dim=1)

        # =========== Locality Enhancement - Sequential Processing ==============
        if self.use_sequentialProcessing:
            lstm_output, decoder_output = self.sequential_processing(
                                                    x = enriched_sequence,
                                                    residual_ = enriched_sequence,
                                                    hidden_states = c_seq_hidden,
                                                    cell_states = c_seq_cell)
        
        # =========== Global Enhancement - Multi-Head Attention =================
        if self.use_multiheadAttention:
            # Static enrichment, 2nd layer
            mha_input = lstm_output if self.use_sequentialProcessing else enriched_sequence

            enriched_sequence = self.static_enrichment(x = mha_input, 
                                                       enrichment = c_enrichment, 
                                                       layer = 'attn')
            
            if not self.use_sequentialProcessing:
                enriched_sequence = self.positional_encoding(enriched_sequence)

            output = self.global_enhancement(x_past, x_future, enriched_sequence)
        
        else:
            output = lstm_output[:, self.L:, :]

        # =========== Position Wise FF ==============

        if self.use_sequentialProcessing:
            PwFF_residual = decoder_output  # Will still throw an error if not defined
        else:
            PwFF_residual = enriched_sequence[:, self.L:, :]

        PwFF_output = self.PositionWiseFF(output)
        PwFF_output = self.Gate(PwFF_output, residual = PwFF_residual) # skip connection w/ decoder output
  
        # Apply each linear head separately and concatenate results
        model_output = torch.cat([self.transect_heads[i](PwFF_output) for i in range(self.output_dim)], dim=-1)

        return model_output
    
#####################################################
#####################################################

    def get_static_encoders(self, selected_static: torch.tensor):
        """
        This method processes the variable selection results for the static data, yielding signals which are designed
        to allow better integration of the information from static metadata.
        Each of the resulting signals is generated using a separate GRN, and is eventually wired into various locations
        in the temporal fusion decoder, for allowing static variables to play an important role in processing.

        c_seq_hidden & c_seq_cell will be used both for local processing of temporal features
        c_enrichment will be used for enriching temporal features with static information.
        """
        c_enrichment = self.static_encoder_enrichment(selected_static)
        c_seq_hidden = self.static_encoder_sequential_state_init(selected_static)
        c_seq_cell = self.static_encoder_sequential_cell_init(selected_static)

        return c_enrichment, c_seq_cell, c_seq_hidden
    
#####################################################
#####################################################

    def static_enrichment(self, x, enrichment, layer):
        
        num_samples, num_temporal_steps, _ = x.shape
        time_distributed_context = enrichment.unsqueeze(1).repeat(1,num_temporal_steps, 1)

        flattened_input = self.stack_time_steps_along_batch(x)
        time_distributed_context = self.stack_time_steps_along_batch(time_distributed_context)

        if layer == 'past':
            enriched_output = self.static_enrichment_grn_past(
                                                            flattened_input,
                                                            static_context=time_distributed_context)
        if layer == 'future':
            enriched_output = self.static_enrichment_grn_future(
                                                            flattened_input,
                                                            static_context=time_distributed_context)
        elif layer == 'attn':
            enriched_output = self.static_enrichment_grn_attn(
                                                            flattened_input,
                                                            static_context=time_distributed_context)

        enriched_output = enriched_output.view(num_samples, -1, self.d_model)

        return enriched_output
    
#####################################################
#####################################################

    def sequential_processing(self, x, residual_, hidden_states, cell_states):

        enc_input = x[:,:self.L,:]
        dec_input = x[:,self.L:,:]
        
        hidden_states = hidden_states.unsqueeze(0).repeat(self.lstm_layers, 1, 1)
        cell_states = cell_states.unsqueeze(0).repeat(self.lstm_layers, 1, 1)

        encoder_output, (h_e, c_e) = self.lstm_encoder(enc_input, (hidden_states, cell_states))
        decoder_output, _ = self.lstm_decoder(dec_input, (h_e, c_e))

        lstm_output = torch.cat([encoder_output, decoder_output], dim=1)
        lstm_output = self.layer_dropout(lstm_output)

        gated_lstm_output = self.Gate(lstm_output, residual = residual_)

        return gated_lstm_output, decoder_output
    
#####################################################
#####################################################

    def global_enhancement(self, x_past, x_future, enriched_sequence):

        # Generate mask 
        attn_mask = self.generate_attn_mask(x_past, x_future)
        attn_mask = attn_mask[:, self.L:, :]
    
        Q = enriched_sequence[:, self.L:, :]  # Decoder queries
        K = enriched_sequence
        V = enriched_sequence  # Values remain the same

        mha_output, _ = self.multihead_attention(Q, K, V, attn_mask=attn_mask)
       
        # we slice here because we are only interested in the future values...
        # ususally in other implementaions there is customized interpertable MHA - so they are able to execute MHA with a pre cut Q
        # this is the best I could think of for now, otherwise custom implementation needed as this is less efficient.
  
        mha_output = self.layer_dropout(mha_output)
        mha_output = self.Gate(mha_output, residual = Q)
        
        return mha_output

#####################################################
#####################################################

    def generate_attn_mask(self, x_past, x_future):

        L = x_past.shape[1]
        S = x_future.shape[1]

        # Detect NaNs separately in x_past and x_future
        nan_mask_past = (x_past == self.nan_token).any(dim=-1)  # Shape: (B, L) - True where NaN
        nan_mask_future = (x_future == self.nan_token).any(dim=-1)  # Shape: (B, S) - True where NaN

        # Concatenate NaN masks along the time dimension → Shape: (B, L+S)
        nan_mask_combined = torch.cat([nan_mask_past, nan_mask_future], dim=1)  # Shape: (B, L+S)

        # Step 3: Expand NaN mask to match attention shape (B, L+S, L+S)
        nan_mask = nan_mask_combined.unsqueeze(1).expand(-1, L+S, -1)  # Shape: (B, L+S, L+S)

        # Create causal mask (upper triangular matrix)
        causal_mask = torch.triu(torch.ones(L+S, L+S, dtype=torch.bool), diagonal=1).to(nan_mask.device)  # (L+S, L+S)
        # # Merge both masks (final binary mask)
        attn_mask = nan_mask | causal_mask  # Logical OR → Mask where either NaN or causal
      

        attn_mask = attn_mask.unsqueeze(1)  # (B, 1, L, S)
        attn_mask = attn_mask.repeat(1, self.n_heads, 1, 1)  # (B, H, L, S)
        attn_mask = attn_mask.view(-1, attn_mask.shape[2], attn_mask.shape[3])  # (B*H, L, S)
        
        return attn_mask  # Boolean mask for PyTorch MHA

    
#####################################################
#####################################################
    
    @staticmethod
    def stack_time_steps_along_batch(temporal_signal: torch.tensor) -> torch.tensor:
        """
        This method gets as an input a temporal signal [num_samples x time_steps x num_features]
        and stacks the batch dimension and the temporal dimension on the same axis (dim=0).

        The last dimension (features dimension) is kept as is, but the rest is stacked along dim=0.
        """
        return temporal_signal.view(-1, temporal_signal.size(-1))

#####################################################
#####################################################

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x shape: (B, L, N)
        attn_output, _ = self.mha(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout1(attn_output))  # Add & Norm

        # Feedforward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))  # Add & Norm

        return x

#####################################################
#####################################################

class LogisticGate(nn.Module):
    def __init__(self, d_model):
        super(LogisticGate, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, residual):

        x = self.sigmoid(x)
        x = self.layer_norm(x + residual)

        return x

#####################################################
#####################################################

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, d_model, output_dim, dropout):
        super(GatedResidualNetwork, self).__init__()

        self.skip_residual_transform = (input_dim == output_dim)
        self.transform_residual = nn.Linear(input_dim, output_dim)
        
        self.Dense1 = nn.Linear(input_dim, d_model)
        self.Dense2 = nn.Linear(d_model, output_dim)
        
        self.GLU = nn.Sequential(
            nn.Linear(output_dim, output_dim),  # Linear layer
            nn.Sigmoid())

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.ELU = nn.ELU()
    
    def forward(self, x, static_context=None):
 
        if self.skip_residual_transform:
            residual_ = x
        else:
            residual_  = self.transform_residual(x)
            
        x = self.Dense1(x)
        
        if static_context is not None:
            x = x + static_context

        x = self.ELU(x)
        x = self.Dense2(x)
        x = self.dropout(x)

        gate = self.GLU(x)  
        x = (gate * x) + residual_  #Add
        x = self.layer_norm(x) #Norm
        
        return x

#####################################################
#####################################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    

#####################################################
#####################################################