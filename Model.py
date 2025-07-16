import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
from torch_scatter import scatter_sum, scatter_softmax
import math

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
	def __init__(self, handler):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
		self.eEmbeds = nn.Parameter(init(torch.empty(args.entity_n, args.latdim)))
		self.rEmbeds = nn.Parameter(init(torch.empty(args.relation_num, args.latdim)))

		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])
		self.rgat = RGAT(args.latdim, args.layer_num_kg, args.mess_dropout_rate)

		self.kg_dict = handler.kg_dict
		self.edge_index, self.edge_type = self.sampleEdgeFromDict(self.kg_dict, triplet_num=args.triplet_num)
	
	def getEntityEmbeds(self):
		return self.eEmbeds

	def getUserEmbeds(self):
		return self.uEmbeds
				
	def forward(self, adj, mess_dropout=True, kg=None):
		if kg == None:
			hids_KG = self.rgat.forward(self.eEmbeds, self.rEmbeds, [self.edge_index, self.edge_type], mess_dropout)
		else:
			hids_KG = self.rgat.forward(self.eEmbeds, self.rEmbeds, kg, mess_dropout)
						
		embeds = torch.concat([self.uEmbeds, hids_KG[:args.item, :]], axis=0)
		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst)

		return embeds[:args.user], embeds[args.user:]
	
	def sampleEdgeFromDict(self, kg_dict, triplet_num=None):
		sampleEdges = []
		for h in kg_dict:
			t_list = kg_dict[h]
			if triplet_num != -1 and len(t_list) > triplet_num:
				sample_edges_i = random.sample(t_list, triplet_num)
			else:
				sample_edges_i = t_list
			for r, t in sample_edges_i:
				sampleEdges.append([h, t, r])
		return self.getEdges(sampleEdges)
	
	def getEdges(self, kg_edges):
		graph_tensor = torch.tensor(kg_edges)
		index = graph_tensor[:, :-1]
		type = graph_tensor[:, -1]
		return index.t().long().cuda(), type.long().cuda()
	
class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return torch.spmm(adj, embeds)
		
class RGAT(nn.Module):
	def __init__(self, latdim, n_hops, mess_dropout_rate=0.4):
		super(RGAT, self).__init__()
		self.mess_dropout_rate = mess_dropout_rate
		self.W = nn.Parameter(init(torch.empty(size=(2*latdim, latdim)), gain=nn.init.calculate_gain('relu')))

		self.leakyrelu = nn.LeakyReLU(0.2)
		self.n_hops = n_hops
		self.dropout = nn.Dropout(p=mess_dropout_rate)

	def agg(self, entity_emb, relation_emb, kg):
		edge_index, edge_type = kg
		head, tail = edge_index
		a_input = torch.cat([entity_emb[head], entity_emb[tail]], dim=-1)
		e_input = torch.multiply(torch.mm(a_input, self.W), relation_emb[edge_type]).sum(-1)
		e = self.leakyrelu(e_input)
		e = scatter_softmax(e, head, dim=0, dim_size=entity_emb.shape[0])
		agg_emb = entity_emb[tail] * e.view(-1, 1)
		agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=entity_emb.shape[0])
		agg_emb = agg_emb + entity_emb
		return agg_emb
		
	def forward(self, entity_emb, relation_emb, kg, mess_dropout=True):
		entity_res_emb = entity_emb
		for _ in range(self.n_hops):
			entity_emb = self.agg(entity_emb, relation_emb, kg)
			if mess_dropout:
				entity_emb = self.dropout(entity_emb)
			entity_emb = F.normalize(entity_emb)

			entity_res_emb = args.res_lambda * entity_res_emb + entity_emb
		return entity_res_emb