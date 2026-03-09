import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from Layers import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphCodeBERT(nn.Module):
    def __init__(self, encoder, config, tokenizer, args, nl_encoder=None):
        super(GraphCodeBERT, self).__init__()
        self.encoder = encoder
        self.nl_encoder = nl_encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.w_embeddings = self.encoder.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.graphEmb = GraphEmbedding(feature_dim_size=768, hidden_size=256, dropout=config.hidden_dropout_prob)
        self.query = 0
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # NL投影层：768 → 256，与graph_emb维度对齐
        self.nl_projector = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.classifier = PredictionClassification(config, args, input_size= 1280)
        
        # ���������� RoBERTa ǰ n �㣨���綳��ǰ 6 �㣩
        n_freeze_layers = 8  # �� ��ɸ���ʵ�������0=ȫ�ſ���12=ȫ����
        if n_freeze_layers > 0:
            # ���� embeddings����ѡ��ͨ��������
            #for param in self.encoder.embeddings.parameters():
                #param.requires_grad = False

            # ���� Transformer ��������ǰ n ��
            for layer in self.encoder.encoder.layer[:n_freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        if self.nl_encoder is not None:
            n_freeze_nl = 8
            for layer in self.nl_encoder.encoder.layer[:n_freeze_nl]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, inputs_ids=None, attn_mask=None, position_idx=None, labels=None, ast_adj=None, cfg_adj=None, pdg_adj=None, node_features=None, node_mask=None, nl_input_ids=None, nl_attn_mask=None):
        g_emb = self.graphEmb(node_features.to(device).float(), ast_adj.to(device).float(), cfg_adj.to(device).float(), pdg_adj.to(device).float(), node_mask.to(device).float())
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)

        inputs_embeddings = self.encoder.embeddings.word_embeddings(inputs_ids)

        vec = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask, position_ids=position_idx)[0][:, 0, :]
        # 3. NL embedding
        if self.nl_encoder is not None and nl_input_ids is not None:
            nl_hidden = self.nl_encoder(
                input_ids=nl_input_ids,
                attention_mask=nl_attn_mask
            )[0]  # [B, nl_seq_len, 768]
            nl_mask_expanded = nl_attn_mask.float().unsqueeze(-1)  # [B, nl_seq_len, 1]
            nl_vec = (nl_hidden * nl_mask_expanded).sum(dim=1) / nl_mask_expanded.sum(dim=1).clamp(min=1e-9)  # [B, 768]
            nl_vec = self.nl_projector(nl_vec)  # [B, 256]

            # 4. 三路拼接
            combined = torch.cat([vec, g_emb, nl_vec], dim=1)  # [B, 1280]
        else:
            # 没有NL时退化为两路，便于消融实验
            combined = torch.cat([vec, g_emb], dim=1)  # [B, 1024]

        combined = self.dropout(combined)
        outputs = self.classifier(combined)
        logits = outputs.squeeze(-1)

        if labels is not None:
            labels = labels.float()
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


def distill_loss(logits, knowledge, temperature=10.0):
    loss = F.kl_div(F.log_softmax(logits/temperature), F.softmax(knowledge /
                    temperature), reduction="batchmean") * (temperature**2)
    # Equivalent to cross_entropy for soft labels, from https://github.com/huggingface/transformers/blob/50792dbdcccd64f61483ec535ff23ee2e4f9e18d/examples/distillation/distiller.py#L330

    return loss