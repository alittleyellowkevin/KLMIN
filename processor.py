from turtle import update
import torch
import torch.nn.functional as F
from models.models import KLMIN_model
from tqdm import tqdm
import numpy as np
from metrics.eval_reid import eval_func


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(data, device):
    if data['model_arch'] == '1B':
        model = KLMIN_model(class_num=data['n_classes'], n_branches=["R50"],
                          losses=["ce+tri"], LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
    if data['model_arch'] == 'se1B':
        model = KLMIN_model(class_num=data['n_classes'], n_branches=["SE"],
                          losses=["ce+tri"], LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
    if data['model_arch'] == 'bot1B':
        model = KLMIN_model(class_num=data['n_classes'], n_branches=["BoT"],
                          losses=["ce+tri"], LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
    if data['model_arch'] == '2B':
        model = KLMIN_model(class_num=data['n_classes'], n_branches=["R50", "BoT"],
                          losses=["ce+tri", "ce+tri"], LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    if data['model_arch'] == 'se2B':
        model = KLMIN_model(class_num=data['n_classes'], n_branches=["SE", "BoT"],
                          losses=["ce+tri", "ce+tri"], LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    if data['model_arch'] == 'ff2B':
        model = KLMIN_model(class_num=data['n_classes'], n_branches=["SE-BoT"], losses=["ce+tri", "ce+tri"],
                         LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    if data['model_arch'] == '4B':
        model = KLMIN_model(class_num=data['n_classes'], n_branches=["R50", "R50", "BoT", "BoT"],
                          losses=["ce", "tri", "ce", "tri"], LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    if data['model_arch'] == 'se4B':
        model = KLMIN_model(class_num=data['n_classes'], n_branches=["SE", "BoT", "SE", "BoT"], losses=["ce", "ce", "tri", "tri"],
                        LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    if data['model_arch'] == 'ff4B':
        model = KLMIN_model(class_num=data['n_classes'], n_branches=["se-bot", "se-bot"], losses=["ce", "ce", "tri", "tri"],
                         LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    if data['model_arch'] == '6B':
        model = KLMIN_model(class_num=data['n_classes'], n_branches=["R50", "BoT", "R50","BoT", "R50", "BoT"],
                          losses=["ce+tri", "ce+tri", "ce", "ce", "tri", "tri"],LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    if data['model_arch'] == 'se6B':
        model = KLMIN_model(class_num=data['n_classes'], n_branches=["SE", "BoT", "SE", "BoT", "SE", "BoT"], losses=["ce+tri","ce+tri", "ce", "ce", "tri", "tri"],
                           LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    if data['model_arch'] == 'ff6B':
        model = KLMIN_model(class_num=data['n_classes'], n_branches=["SE-BoT", "SE-BoT", "SE-BoT"], losses=["ce+tri","ce+tri", "ce", "ce", "tri", "tri"],
                           LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])




    return model.to(device)

if __name__ == "__main__":
    model = KLMIN_model(575, n_branches=["se-bot", "se-bot"], losses=["ce", "ce", "tri", "tri"], LAI=True, n_cams=20, n_views=8)
    print(model)
    print(count_parameters(model))

def train_epoch(model, device, dataloader, loss_fn, triplet_loss, optimizer, data, logger, epoch,
                scheduler=None, scaler=False):
    # Set train mode for both the encoder and the decoder
    model.train()
    train_loss = []
    ce_loss_log = []
    triplet_loss_log = []

    ce_loss = data['ce_loss']
    tri_loss = data['tri_loss']

    loss_log = tqdm(total=0, position=1, bar_format='{desc}', leave=True)
    loss_ce_log = tqdm(total=0, position=2, bar_format='{desc}', leave=True)
    loss_triplet_log = tqdm(total=0, position=3, bar_format='{desc}', leave=True)

    n_images = 0
    acc_v = 0
    stepcount = 0
    for image_batch, label, cam, view in tqdm(dataloader, desc='Epoch ' + str(epoch + 1) + ' (%)',
                                              bar_format='{l_bar}{bar:20}{r_bar}'):
        # Move tensor to the proper device
        loss_ce = 0
        loss_t = 0
        optimizer.zero_grad()

        image_batch = image_batch.to(device)
        label = label.to(device)
        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):

                preds, embs, _, _ = model(image_batch, cam, view)
                loss = 0
                #### Losses
                if type(preds) != list:
                    preds = [preds]
                    embs = [embs]
                for i, item in enumerate(preds):
                    loss_ce += ce_loss[i] * loss_fn(item, label)
                for i, item in enumerate(embs):
                    loss_t += tri_loss[i] * triplet_loss(item, label, epoch)

                if data['mean_losses']:
                    loss = loss_ce / len(preds) + loss_t / len(embs)
                else:
                    loss = loss_ce + loss_t
        else:
            preds, embs, ffs, activations = model(image_batch, cam, view)

            loss = 0
            #### Losses
            if type(preds) != list:
                preds = [preds]
                embs = [embs]

            for i, item in enumerate(preds):
                loss_ce += ce_loss[i] * loss_fn(item, label)
            for i, item in enumerate(embs):
                loss_t += tri_loss[i] * triplet_loss(item, label, epoch)

            if data['mean_losses']:
                loss = loss_ce / len(preds) + loss_t / len(embs)
            else:
                loss = loss_ce + loss_t

        ###Training Acurracy
        for prediction in preds:
            acc_v += torch.sum(torch.argmax(prediction, dim=1) == label)
            n_images += prediction.size(0)
        stepcount += 1

        ### backward prop and optimizer step
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()



        loss_log.set_description_str(f'train loss : {loss.data:.3f}')
        loss_ce_log.set_description_str(f'CrossEntropy: {loss_ce.data:.3f}')
        loss_triplet_log.set_description_str(f'Triplet : {loss_t.data:.3f}')

        train_loss.append(loss.detach().cpu().numpy())
        ce_loss_log.append(loss_ce.detach().cpu().numpy())
        triplet_loss_log.append(loss_t.detach().cpu().numpy())

        logger.write_scalars({"Loss/train_total": np.mean(train_loss),
                              "Loss/train_crossentropy": np.mean(ce_loss_log),
                              "Loss/train_triplet": np.mean(triplet_loss_log),
                              "lr/learning_rate": get_lr(optimizer),
                              "Loss/AccuracyTrain": (acc_v / n_images).cpu().numpy()},
                             epoch * len(dataloader) + stepcount,
                             write_epoch=True
                             )

    print('\nTrain ACC (%): ', acc_v / n_images, "\n")

    return np.mean(train_loss), np.mean(ce_loss_log), np.mean(triplet_loss_log)


def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea, galFea])
        print('using GPU to compute original distance')
        distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
                  torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distmat.addmm_(1, -2, feat, feat.t())
        original_dist = distmat.cpu().numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def test_epoch(model, device, dataloader_q, dataloader_g, model_arch, re_rank, writer, epoch, remove_junk=True,
               scaler=False):
    model.eval()
    ###needed lists
    qf = []
    gf = []
    q_camids = []
    g_camids = []
    q_vids = []
    g_vids = []
    q_images = []
    g_images = []

    with torch.no_grad():
        for image, q_id, cam_id, view_id in tqdm(dataloader_q, desc='Query infer (%)',
                                                 bar_format='{l_bar}{bar:20}{r_bar}'):
            image = image.to(device)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, ffs, _ = model(image, cam_id, view_id)
            else:
                _, _, ffs, _ = model(image, cam_id, view_id)

            end_vec = []
            for item in ffs:
                end_vec.append(F.normalize(item))
            qf.append(torch.cat(end_vec, 1))

            q_vids.append(q_id)
            q_camids.append(cam_id)

            if epoch == 119:
                q_images.append(F.interpolate(image, (64, 64)).cpu())

        #### TensorBoard emmbeddings for projector visualization
        if epoch == 119:
            writer.write_embeddings(torch.cat(qf).cpu(), torch.cat(q_vids).cpu(), torch.cat(q_images) / 2 + 0.5, 120,
                                    tag='Query embeddings')

        del q_images

        for image, q_id, cam_id, view_id in tqdm(dataloader_g, desc='Gallery infer (%)',
                                                 bar_format='{l_bar}{bar:20}{r_bar}'):
            image = image.to(device)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, ffs, _ = model(image, cam_id, view_id)
            else:
                _, _, ffs, _ = model(image, cam_id, view_id)

            end_vec = []
            for item in ffs:
                end_vec.append(F.normalize(item))
            gf.append(torch.cat(end_vec, 1))
            g_vids.append(q_id)
            g_camids.append(cam_id)
        del g_images

    qf = torch.cat(qf, dim=0)
    gf = torch.cat(gf, dim=0)
    m, n = qf.shape[0], gf.shape[0]

    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    distmat = torch.sqrt(distmat).cpu().numpy()
    if re_rank:
        distmat_re = re_ranking(qf, gf, k1=80, k2=16, lambda_value=0.3)

        q_camids = torch.cat(q_camids, dim=0).cpu().numpy()
        g_camids = torch.cat(g_camids, dim=0).cpu().numpy()
        q_vids = torch.cat(q_vids, dim=0).cpu().numpy()
        g_vids = torch.cat(g_vids, dim=0).cpu().numpy()

        del qf, gf

        cmc, mAP = eval_func(distmat, q_vids, g_vids, q_camids, g_camids, remove_junk=remove_junk)
        cmc_re, mAP_re = eval_func(distmat_re, q_vids, g_vids, q_camids, g_camids, remove_junk=remove_junk)

        writer.write_scalars({"Accuraccy/CMC1": cmc[0], "Accuraccy/CMC5": cmc[4], "Accuraccy/mAP": mAP}, epoch)
        writer.write_scalars(
            {"Accuraccy/CMC1_re": cmc_re[0], "Accuraccy/CMC5_re": cmc_re[4], "Accuraccy/mAP_re": mAP_re}, epoch)

        return cmc, mAP, cmc_re, mAP_re
    else:
        q_camids = torch.cat(q_camids, dim=0).cpu().numpy()
        g_camids = torch.cat(g_camids, dim=0).cpu().numpy()
        q_vids = torch.cat(q_vids, dim=0).cpu().numpy()
        g_vids = torch.cat(g_vids, dim=0).cpu().numpy()

        del qf, gf

        cmc, mAP = eval_func(distmat, q_vids, g_vids, q_camids, g_camids, remove_junk=remove_junk)
        writer.write_scalars({"Accuraccy/CMC1": cmc[0], "Accuraccy/CMC5": cmc[4], "Accuraccy/mAP": mAP}, epoch)

        return cmc, mAP