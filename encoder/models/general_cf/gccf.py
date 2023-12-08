import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class GCNLayer(nn.Module):
    def __init__(self, latdim):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(init(t.empty(latdim, latdim)))

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds) # @ self.W (Performs better without W)

class GCCF(BaseModel):
    def __init__(self, data_handler):
        super(GCCF, self).__init__(data_handler)

        self.adj = data_handler.torch_adj
        
        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.gcnLayers = nn.Sequential(*[GCNLayer(self.embedding_size) for i in range(self.layer_num)])
        self.is_training = True
    
    def forward(self, adj=None):
        if adj is None:
            adj = self.adj
        if not self.is_training:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = t.concat(embeds_list, dim=-1)
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:], embeds_list[-1]
    
    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds, _ = self.forward(self.adj)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        loss = bpr_loss + reg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        return loss, losses

    # def _predict_all_wo_mask(self, ancs):
    #     user_embeds, item_embeds = self.forward(self.adj)
    #     pck_users = ancs
    #     pck_user_embeds = user_embeds[pck_users]
    #     full_preds = pck_user_embeds @ item_embeds.T
    #     return full_preds

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _ = self.forward(self.adj)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds