from modeling_bart import VLBart
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import spacy
# from pykeen.models import TransE
# from pykeen.datasets import ConceptNet
nlp = spacy.load("en_core_web_sm")
def extract_important_words(batch_texts):
    important_words_batch = []
    
    for text in batch_texts:
        if not isinstance(text, str):
            raise ValueError(f"Expected string, got {type(text)}")
        doc = nlp(text)
        important_words = [token.text.lower() for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ"}]
        important_words_batch.append(important_words)
    
    return important_words_batch

def process_features(question_feats1):
    processed_feats = []

    for q in question_feats1:
        length = q.shape[0]

        if length < 32:
            # Pad with zeros if length < 32
            q_padded = F.pad(q, (0, 32 - length))
            processed_feats.append(q_padded)

        elif length == 32:
            # No change if length == 32
            processed_feats.append(q)

        else:
            # If length > 32
            remainder = length % 32
            if remainder != 0:
                # Pad to the next multiple of 32
                pad_size = 32 - remainder
                q = F.pad(q, (0, pad_size))
            
            # Reshape and average chunks of 32
            # Convert to float before reshaping and averaging
            q_reshaped = q.float().view(-1, 32)
            q_mean = torch.mean(q_reshaped, dim=0)
            processed_feats.append(q_mean)

    # Stack all processed features
    return torch.stack(processed_feats)


class SimpleTransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super(SimpleTransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, embed_dim)  # Output projection
        self.linear_layer = nn.Linear(2048+32,2048)

    def forward(self, img_feats, question_feats):
        # Repeat question features V_L times
        B, V_L, D = img_feats.shape
        question_feats = question_feats.unsqueeze(1).expand(-1, V_L, -1)  # Shape: (B, V_L, D)
        
        # Concatenate along feature dimension
        combined_feats_1 = torch.cat((img_feats, question_feats), dim=-1)  # Shape: (B, V_L, 2*D)
        combined_feats = self.linear_layer(combined_feats_1)
        
        # Pass through transformer decoder
        tgt = combined_feats.permute(1, 0, 2)  # Transformer expects (seq_len, batch, dim)
        memory = torch.zeros_like(tgt)  # Placeholder for memory (modify if needed)
        output = self.transformer_decoder(tgt, memory)  # Shape: (V_L, B, D)
        
        return self.fc(output.permute(1, 0, 2))  # Shape: (B, V_L, D)


class VLBartWebQA_QA(VLBart):
    def __init__(self, config):
        super().__init__(config)
        self.decoder1 = SimpleTransformerDecoder(embed_dim=2048, num_heads=8, ff_dim=4096, num_layers=1)
        self.decoder2 = SimpleTransformerDecoder(embed_dim=2048, num_heads=8, ff_dim=4096, num_layers=1)
        # Freeze all parameters of the base model (VLBart)
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze only the parameters of the SimpleTransformerDecoder layers
        for param in self.decoder1.parameters():
            param.requires_grad = True

        for param in self.decoder2.parameters():
            param.requires_grad = True

    # def train_step(self, batch):
    #     device = next(self.parameters()).device
    #     input_ids = batch['input_ids'].to(device)
    #     B = len(input_ids)
    #     V_L = batch['vis_feats'].size(2)
    #     vis_feats = batch['vis_feats'].to(device).view(B, 2*V_L, 2048)
    #     vis_pos = batch['boxes'].to(device).view(B, 2*V_L, 4)

    #     lm_labels = batch["target_ids"].to(device)


    #     img_order_ids = [0] * V_L + [1] * V_L
    #     img_order_ids = torch.tensor(img_order_ids, dtype=torch.long, device=device)
    #     img_order_ids = img_order_ids.view(1, 2*V_L).expand(B, -1)

    #     obj_order_ids = torch.arange(V_L, dtype=torch.long, device=device)
    #     obj_order_ids = obj_order_ids.view(1, 1, V_L).expand(B, 2, -1).contiguous().view(B, 2*V_L)

        
    #     output = self(
    #         input_ids=input_ids,
    #         vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
    #         labels=lm_labels,
    #         return_dict=True
    #     )

    #     assert 'loss' in output

    #     lm_mask = (lm_labels != -100).float()
    #     B, L = lm_labels.size()

    #     loss = output['loss']

    #     loss = loss.view(B, L) * lm_mask

    #     loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

    #     loss = loss.mean()

    #     result = {
    #         'loss': loss
    #     }

    #     # logits = output['logits'].detach()[:, 1]
    #     # logits = logits.view(B, self.lm_head.out_features)
    #     # true_logit = logits[:, self.true_id]
    #     # false_logit = logits[:, self.false_id]

    #     # pred_true = true_logit > false_logit
    #     # pred_true = pred_true.long().cpu().numpy()
    #     # result['pred_ans_id'] = pred_true

    #     return result
    
    def train_step(self, batch):
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        B = len(input_ids)
        V_L = batch['vis_feats'].size(2)
        img1_feats = batch['vis_feats'][:, 0, :].to(device)  # Shape: (B, V_L, 2048)
        img2_feats = batch['vis_feats'][:, 1, :].to(device)  # Shape: (B, V_L, 2048)
        question_feats1 = batch['input_ids'].to(device)  # Shape: (B, D)
        #question_feats = torch.stack([torch.mean(q.view(-1, 32), dim=0) if q.shape[0] >= 32 else F.pad(q, (0, 32 - q.shape[0])) for q in question_feats1])
        question_feats = process_features(question_feats1)

        sent = batch['sent']
        #important_words = extract_important_words(sent)
        # out_dict["imp_words"] = important_words


                # Load the Countries dataset (pre-split)
        # dataset = ConceptNet()
        # triples_factory = dataset.training  # This is already a TriplesFactory object

        # # Get entity-to-ID mapping
        # entity_to_id = triples_factory.entity_to_id

        # # Load a pretrained TransE model using the Countries dataset
        # model = TransE(triples_factory=triples_factory)


        # # Identify important words (should ideally be extracted using NLP, hardcoded for now)
        # important_words = batch['imp_words']

        # # Extract entity embeddings for important words if they exist in the dataset
        # entity_embeddings = {
        #     word: model.entity_representations[0](torch.tensor([entity_to_id[word]]))
        #     for word in important_words if word in entity_to_id
        # }

        # print(entity_embeddings)


        
        out1 = self.decoder1(img1_feats, question_feats)  # Shape: (B, V_L, 2048)
        out2 = self.decoder2(img2_feats, question_feats)  # Shape: (B, V_L, 2048)
        fused_feats = torch.cat((out1, out2), dim=1)  # Shape: (B, 2*V_L, 2048)
        
        vis_pos = batch['boxes'].to(device).view(B, 2*V_L, 4)
        lm_labels = batch["target_ids"].to(device)
        
        img_order_ids = torch.tensor([0] * V_L + [1] * V_L, dtype=torch.long, device=device)
        img_order_ids = img_order_ids.view(1, 2*V_L).expand(B, -1)
        
        obj_order_ids = torch.arange(V_L, dtype=torch.long, device=device)
        obj_order_ids = obj_order_ids.view(1, 1, V_L).expand(B, 2, -1).contiguous().view(B, 2*V_L)
        
        output = self(
            input_ids=input_ids,
            vis_inputs=(fused_feats, vis_pos, img_order_ids, obj_order_ids),
            labels=lm_labels,
            return_dict=True
        )

        assert 'loss' in output
        lm_mask = (lm_labels != -100).float()
        B, L = lm_labels.size()
        loss = output['loss'].view(B, L) * lm_mask
        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)
        loss = loss.mean()
        
        return {'loss': loss}



    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        B = len(input_ids)
        V_L = batch['vis_feats'].size(2)
        img1_feats = batch['vis_feats'][:, 0, :].to(device)  # Shape: (B, V_L, 2048)
        img2_feats = batch['vis_feats'][:, 1, :].to(device)  # Shape: (B, V_L, 2048)
        question_feats1 = batch['input_ids'].to(device)  # Shape: (B, D)
        #question_feats = torch.stack([torch.mean(q.view(-1, 32), dim=0) if q.shape[0] >= 32 else F.pad(q, (0, 32 - q.shape[0])) for q in question_feats1])
        question_feats = process_features(question_feats1)

        out1 = self.decoder1(img1_feats, question_feats)  # Shape: (B, V_L, 2048)
        out2 = self.decoder2(img2_feats, question_feats)  # Shape: (B, V_L, 2048)
        fused_feats = torch.cat((out1, out2), dim=1)  # Shape: (B, 2*V_L, 2048)

        vis_pos = batch['boxes'].to(device).view(B, 2*V_L, 4)

        img_order_ids = [0] * V_L + [1] * V_L
        img_order_ids = torch.tensor(img_order_ids, dtype=torch.long, device=device)
        img_order_ids = img_order_ids.view(1, 2*V_L).expand(B, -1)

        obj_order_ids = torch.arange(V_L, dtype=torch.long, device=device)
        obj_order_ids = obj_order_ids.view(1, 1, V_L).expand(B, 2, -1).contiguous().view(B, 2*V_L)


        decoder_input_ids = torch.tensor(
            [self.config.decoder_start_token_id, self.config.bos_token_id],
            dtype=torch.long, device=device).unsqueeze(0).expand(B, 2)

        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(fused_feats, vis_pos, img_order_ids, obj_order_ids),
            decoder_input_ids=decoder_input_ids,
            early_stopping=True,
            **kwargs,
        )


        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)


        result = {}
        result['pred'] = generated_sents

        return result