import torch
from torch import nn

class CAN_loss(nn.Module):
    def __init__(self, train_config):
        super().__init__()
        # define Loss setting
        self.using_graylabel = train_config['exp_setting']['using_graylabel']
        self.using_contra = train_config['exp_setting']['using_contra']
        self.using_weight_decay = train_config['exp_setting']['using_weight_decay']
        # define contrastive loss decay setting
        self.decay_start = train_config['step']['total_step'] // train_config['step']['contra_deacy_start']
        self.decay_end = train_config['step']['total_step'] // train_config['step']['contra_deacy_end']
        
        self.target_weight = train_config['model']['target_contra_loss_weight']
        self.init_weight = train_config['model']['init_contra_loss_weight']
        
        self.ce = nn.CrossEntropyLoss()
        self.cs = nn.CosineSimilarity()
        
    def update_contra_weight(self, step):
        if step < self.decay_start:
            return self.init_weight
        if step > self.decay_end:
            return self.target_weight
        else:
            return self.init_weight - (self.init_weight - self.target_weight) * ((step - self.decay_start) / self.decay_end - self.decay_start)
        
    def _forward(self, emo_out, emo_label, sim, contrastive_label):

        if self.using_graylabel:
            emo_loss = (1 / sum(self.cs(emo_out, emo_label)))
        else:
            emo_loss = self.ce(emo_out, emo_label)
            
        if self.using_contra:
            contra_loss = self.ce(sim, contrastive_label)    
        else:
            contra_loss = 0.0
        return emo_loss, contra_loss

    def forward(self, emo_out, emo_label, sim, contrastive_label, step):
        
        emo_loss, contra_loss = self._forward(emo_out, emo_label, sim, contrastive_label)
        if self.using_weight_decay and self.using_contra:
            weight = self.update_contra_weight(step)
            return (1 - weight) * emo_loss, weight * contra_loss, weight
        else:
            return (1 - self.target_weight) * emo_loss, self.target_weight * contra_loss, self.target_weight