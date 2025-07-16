import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model
from Denoise import Denoise
from DualviewDiff import DualviewDiffusion
from DataHandler import DataHandler
import numpy as np
from torch import nn
import pickle
from Utils.Utils import *
import os
import random

from VAE import VAE


class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        log('Model Initialized')

        recallMax = 0
        ndcgMax = 0
        bestEpoch = 0

        for ep in range(0, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                if (reses['Recall'] > recallMax):
                    recallMax = reses['Recall']
                    ndcgMax = reses['NDCG']
                    bestEpoch = ep
                log(self.makePrint('Test', ep, reses, tstFlag))
            print()
        print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax)

    def prepareModel(self):
        self.model1 = Model(self.handler).cuda()
        self.model2 = Model(self.handler).cuda()

        print(args.entity_n)
        self.vae = VAE(args.entity_n, args.entity_n).cuda()  # 要么换一下
        # self.vae = VAE(49545, args.entity_n).cuda()# 要么换一下
        self.optvae = torch.optim.Adam(self.vae.parameters(), lr=args.lr, weight_decay=0)
        self.opt1 = torch.optim.Adam(self.model1.parameters(), lr=args.lr, weight_decay=0)
        self.opt2 = torch.optim.Adam(self.model2.parameters(), lr=args.lr, weight_decay=0)

        self.noisy_diffusion_model = DualviewDiffusion(args.noise_scale, args.noise_min, args.noise_max,
                                                        args.steps).cuda()
        self.sound_diffusion_model = DualviewDiffusion(args.noise_scale, args.noise_min, args.noise_max,
                                                        args.steps).cuda()

        out_dims = eval(args.dims) + [args.entity_n]  # 1000 128
        in_dims = out_dims[::-1]

        self.noisy_denoise_model = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
        self.sound_denoise_model = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()

        self.noisy_denoise_opt = torch.optim.Adam(self.noisy_denoise_model.parameters(), lr=args.lr, weight_decay=0)
        self.sound_denoise_opt = torch.optim.Adam(self.sound_denoise_model.parameters(), lr=args.lr, weight_decay=0)

    def trainEpoch(self):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epRecLoss, epClLoss = 0, 0, 0
        epDiLoss, epUKLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        vaeLoader = self.handler.diffusionLoader1
        diffusionLoader = self.handler.diffusionLoader

        log('')
        log('Start to learn noise')
        criterion = nn.MSELoss()  # It is suitable for the reconstruction of embedded data
        for i, batch in enumerate(vaeLoader):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.cuda(), batch_index.cuda()
            # print(batch_item.shape)
            self.optvae.zero_grad()
            recon_x, mu, logvar = self.vae(batch_item)

            MSE = criterion(recon_x, batch_item)  # Calculate the reconstruction error
            # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
            # noise_loss = MSE + 0.01 * KLD  # 0.01 control KL divergence
            noise_loss = MSE

            noise_loss.backward()
            self.optvae.step()
            log('train noise Step %d/%d' % (i, vaeLoader.dataset.__len__() // args.batch), save=False, oneline=True)

        log('')

        for i, batch in enumerate(diffusionLoader):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

            ui_matrix = self.handler.ui_matrix
            iEmbeds = self.model1.getEntityEmbeds().detach()
            uEmbeds = self.model1.getUserEmbeds().detach()

            self.sound_denoise_opt.zero_grad()

            diff_loss, ukgc_loss = self.sound_diffusion_model.training_losses(self.sound_denoise_model, batch_item,
                                                                              ui_matrix, uEmbeds, iEmbeds, batch_index,
                                                                              self.vae)
            loss = diff_loss.mean() * (1 - args.e_loss) + ukgc_loss.mean() * args.e_loss

            epDiLoss += diff_loss.mean().item()
            epUKLoss += ukgc_loss.mean().item()

            loss.backward()

            self.sound_denoise_opt.step()

            log('Diffusion Step %d/%d' % (i, diffusionLoader.dataset.__len__() // args.batch), save=False, oneline=True)

        log('')
        log('Start to re-build kg1')

        with torch.no_grad():
            denoised_edges = []
            h_list = []
            t_list = []

            for _, batch in enumerate(diffusionLoader):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.cuda(), batch_index.cuda()
                denoised_batch = self.sound_diffusion_model.p_sample(self.sound_denoise_model, batch_item,
                                                                     args.sampling_steps)
                top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)
                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]):
                        h_list.append(batch_index[i])
                        t_list.append(indices_[i][j])

            edge_set = set()
            for index in range(len(h_list)):
                edge_set.add((int(h_list[index].cpu().numpy()), int(t_list[index].cpu().numpy())))
            for index in range(len(h_list)):
                if (int(t_list[index].cpu().numpy()), int(h_list[index].cpu().numpy())) not in edge_set:
                    h_list.append(t_list[index])
                    t_list.append(h_list[index])

            relation_dict = self.handler.relation_dict
            for index in range(len(h_list)):
                try:
                    denoised_edges.append([h_list[index], t_list[index],
                                           relation_dict[int(h_list[index].cpu().numpy())][
                                               int(t_list[index].cpu().numpy())]])
                except Exception:
                    continue
            graph_tensor = torch.tensor(denoised_edges)
            print(graph_tensor.shape)
            index_ = graph_tensor[:, :-1]
            type_ = graph_tensor[:, -1]
            denoisedKG = (index_.t().long().cuda(), type_.long().cuda())

        log('KG1 built!')

        # diff 2
        ######################################

        for i, batch in enumerate(diffusionLoader):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

            ui_matrix = self.handler.ui_matrix
            iEmbeds = self.model2.getEntityEmbeds().detach()
            uEmbeds = self.model2.getUserEmbeds().detach()

            self.noisy_denoise_opt.zero_grad()

            diff_loss, ukgc_loss = self.noisy_diffusion_model.training_losses(self.noisy_denoise_model, batch_item,
                                                                              ui_matrix, uEmbeds, iEmbeds, batch_index)
            loss = diff_loss.mean() * (1 - args.e_loss) + ukgc_loss.mean() * args.e_loss

            epDiLoss += diff_loss.mean().item()
            epUKLoss += ukgc_loss.mean().item()

            loss.backward()

            self.noisy_denoise_opt.step()

            log('Diffusion2 Step %d/%d' % (i, diffusionLoader.dataset.__len__() // args.batch), save=False,
                oneline=True)

        log('')
        log('Start to re-build kg2')

        with torch.no_grad():
            denoised_edges2 = []
            h_list = []
            t_list = []

            for _, batch in enumerate(diffusionLoader):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.cuda(), batch_index.cuda()
                denoised_batch = self.noisy_diffusion_model.p_sample(self.noisy_denoise_model, batch_item,
                                                                     args.sampling_steps)
                top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)
                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]):
                        h_list.append(batch_index[i])
                        t_list.append(indices_[i][j])

            edge_set = set()
            for index in range(len(h_list)):
                edge_set.add((int(h_list[index].cpu().numpy()), int(t_list[index].cpu().numpy())))
            for index in range(len(h_list)):
                if (int(t_list[index].cpu().numpy()), int(h_list[index].cpu().numpy())) not in edge_set:
                    h_list.append(t_list[index])
                    t_list.append(h_list[index])

            relation_dict = self.handler.relation_dict
            for index in range(len(h_list)):
                try:
                    denoised_edges2.append([h_list[index], t_list[index],
                                            relation_dict[int(h_list[index].cpu().numpy())][
                                                int(t_list[index].cpu().numpy())]])
                except Exception:
                    continue
            # change all triplets from denoised_edges1 and denoised_edges2 to tuple, and merge them(union set), finally, transfrom it to list
            union_edges = list(
                set(tuple(edge) for edge in denoised_edges) | set(tuple(edge) for edge in denoised_edges2))

            graph_tensor = torch.tensor(union_edges)
            print(graph_tensor.shape)
            index_ = graph_tensor[:, :-1]
            type_ = graph_tensor[:, -1]
            denoisedKG = (index_.t().long().cuda(), type_.long().cuda())

        log('KG2 built!')
        ##############################################
        with torch.no_grad():
            index_, type_ = denoisedKG
            mask = ((torch.rand(type_.shape[0]) + args.keepRate).floor()).type(torch.bool)
            denoisedKG = (index_[:, mask], type_[mask])
            self.generatedKG = denoisedKG

        # index_, type_ = denoisedKG2
        # mask = ((torch.rand(type_.shape[0]) + args.keepRate).floor()).type(torch.bool)
        # denoisedKG2 = (index_[:, mask], type_[mask])
        # self.generatedKG2 = denoisedKG2

        for i, tem in enumerate(trnLoader):
            ancs, poss, negs = tem
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()

            self.opt1.zero_grad()
            # self.opt2.zero_grad()

            if args.cl_pattern == 0:
                usrEmbeds, itmEmbeds = self.model1(self.handler.torchBiAdj, denoisedKG)
            # usrEmbeds2, itmEmbeds2 = self.model2(self.handler.torchBiAdj, denoisedKG2)
            # usrEmbeds = args.acdscd * usrEmbeds + (1 - args.acdscd) * usrEmbeds2
            # itmEmbeds = args.acdscd * itmEmbeds + (1 - args.acdscd) * itmEmbeds2
            else:
                usrEmbeds, itmEmbeds = self.model1(self.handler.torchBiAdj)
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]
            negEmbeds = itmEmbeds[negs]

            scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
            bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
            regLoss1 = calcRegLoss(self.model1) * args.reg
            regLoss2 = calcRegLoss(self.model2) * args.reg
            regLoss = args.two_reg_weiht * regLoss1 + (1 - args.two_reg_weiht) * regLoss2

            if args.cl_pattern == 0:
                usrEmbeds_kg, itmEmbeds_kg = self.model1(self.handler.torchBiAdj)
            else:
                usrEmbeds_kg, itmEmbeds_kg = self.model1(self.handler.torchBiAdj, denoisedKG)
            # usrEmbeds_kg2, itmEmbeds_kg2 = self.model2(self.handler.torchBiAdj, denoisedKG2)
            # usrEmbeds_kg = args.acdscd * usrEmbeds_kg  + (1- args.acdscd )*usrEmbeds_kg2
            # itmEmbeds_kg = args.acdscd * itmEmbeds_kg  + (1- args.acdscd )*itmEmbeds_kg2
            denoisedKGEmbeds = torch.concat([usrEmbeds, itmEmbeds], axis=0)
            kgEmbeds = torch.concat([usrEmbeds_kg, itmEmbeds_kg], axis=0)

            clLoss = (contrastLoss(kgEmbeds[args.user:], denoisedKGEmbeds[args.user:], poss, args.temp) + contrastLoss(
                kgEmbeds[:args.user], denoisedKGEmbeds[:args.user], ancs, args.temp)) * args.ssl_reg

            loss = bprLoss + regLoss + clLoss

            epLoss += loss.item()
            epRecLoss += bprLoss.item()
            epClLoss += clLoss.item()

            loss.backward()
            self.opt1.step()
            # self.opt2.step()

            log('Step %d/%d: loss = %.3f, regLoss = %.3f' % (i, steps, loss, regLoss), save=False, oneline=True)

        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['recLoss'] = epRecLoss / steps
        ret['clLoss'] = epClLoss / steps
        ret['diLoss'] = epDiLoss / diffusionLoader.dataset.__len__()
        ret['UKGCLoss'] = epUKLoss / diffusionLoader.dataset.__len__()
        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epRecall, epNdcg = [0] * 2
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat

        with torch.no_grad():
            if args.cl_pattern == 0:
                denoisedKG = self.generatedKG
                # denoisedKG2 = self.generatedKG2
                usrEmbeds, itmEmbeds = self.model1(self.handler.torchBiAdj, mess_dropout=False, kg=denoisedKG)
            # usrEmbeds2, itmEmbeds2 = self.model2(self.handler.torchBiAdj, denoisedKG2)
            # usrEmbeds = args.acdscd * usrEmbeds + (1 - args.acdscd) * usrEmbeds2
            # itmEmbeds = args.acdscd * itmEmbeds + (1 - args.acdscd) * itmEmbeds2
            else:
                usrEmbeds, itmEmbeds = self.model1(self.handler.torchBiAdj, mess_dropout=False)

        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()

            allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            _, topLocs = t.topk(allPreds, args.topk)
            recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
            epRecall += recall
            epNdcg += ndcg
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False,
                oneline=True)
        ret = dict()
        ret['Recall'] = epRecall / num
        ret['NDCG'] = epNdcg / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


if __name__ == '__main__':
    seed_it(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.saveDefault = True

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    coach.run()

