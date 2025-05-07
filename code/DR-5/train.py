import torchvision
from torch import nn
import os
import pickle
import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from PIL import Image
import warnings
from utils import *
warnings.filterwarnings("ignore")


print('training...')

print('Random_mode:'+str(random_mode))
repeat_acc_hm_list = []
repeat_acc_m_list = []
repeat_acc_h_list = []

# AI model pretrained 'num_repeats' times repeatedly
for repeat_i in range(num_repeats):
    train_dataset = CustomDataset(f"../../data/DR-5/model_output_{repeat_i}_train.csv")
    test_dataset = CustomDataset(f"../../data/DR-5/model_output_{repeat_i}_test.csv")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, True)
    notrain_loader = torch.utils.data.DataLoader(train_dataset, len(train_dataset), False)
    test_loader = torch.utils.data.DataLoader(test_dataset, len(test_dataset), False)    
    dc_criterion = torch.nn.CrossEntropyLoss().to(device)  
    dc_criterion_all = torch.nn.CrossEntropyLoss(reduction="none").to(device)
    set_seed(1+repeat_i)  
    evaluator_model_last = Evaluator(1280).to(device) 
    for param in evaluator_model_last.parameters():
        param.requires_grad = False
    # calibration
    calibrator = TSCalibratorMAP()
    with torch.no_grad():
        for data in notrain_loader:
            logits, labels, h_labels, feat = data
            human = HumanAgent(labels, h_labels)
    logits = np.log(np.clip(to_numpy_(logits), 1e-10, 1))
    calibrator.fit(logits, to_numpy_(labels))
    # Evaluator repeated training 'num_human_repeats' times
    for human_repeat_id in tqdm(range(num_human_repeats)):
        set_np_seed(repeat_i*10+human_repeat_id+1)
        human.int_random_indices()
        update_sampele_dict = {}
        acc_hm_list = []
        acc_m_list = []
        acc_h_list = []
        for nums_limit in limit_human_pred:
            best_acc_temp = 0
            best_acc_temp_cm = 0
            test_record_temp = []
            test_record_temp_h = []
            test_record_temp_h_all = []
            set_seed(repeat_i*10+1+human_repeat_id)
            evaluator_model = Evaluator(1280).to(device) 
            # Actively collecting limited expert predictions
            if random_mode == True or nums_limit == limit_human_pred[-1]:
                human.make_human_limit(nums_limit)
            else:
                if nums_limit == limit_human_pred[0]:
                    human.make_human_limit(nums_limit) #human.make_alh_limit(sampele_dict, nums_limit)
                else:
                    human.make_alh_limit_plusone(update_sampele_dict, nums_limit)
                update_sampele_dict = human.alh_choose_sample(num_al, limit_human_pred[limit_human_pred.index(nums_limit)+1])
            subset_l_dataset = Subset(train_dataset, human.limit_ex_pred)
            # train evaluator_model
            if len(subset_l_dataset) > 100:
                subset_batch_size = 12
            else:
                subset_batch_size = 5
            subset_loader = torch.utils.data.DataLoader(subset_l_dataset, len(subset_l_dataset), True)
            if nums_limit == limit_human_pred[-1]:
                epochs = 1
            else:
                epochs = 50
            optimizer = torch.optim.Adam(evaluator_model.parameters(), lr = model_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            for epoch in range(epochs):
                evaluator_model.train()
                for data in subset_loader:
                    logits, labels, h_labels, feat = data
                    logits = logits.to(device)
                    labels = labels.to(device)
                    h_labels = h_labels.to(device)
                    feat = feat.to(device)
                    calibrated_model_probs = calibrator.calibrate(logits)
                    h_outputs = evaluator_model(h_labels, feature_input=feat)
                    final_hm_outputs = human.combine_hm_eva(calibrated_model_probs, h_outputs)
                    loss = dc_criterion(final_hm_outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                evaluator_model.eval()
                with torch.no_grad():
                    test_num_m = 0
                    test_num_h = 0
                    test_num_hm = 0
                    for data in test_loader:
                        logits, labels, h_labels, feat = data
                        logits = logits.to(device)
                        labels = labels.to(device)
                        h_labels = h_labels.to(device)
                        feat = feat.to(device)
                        calibrated_model_probs = calibrator.calibrate(logits)
                        h_outputs = evaluator_model(h_labels, feature_input=feat)
                        final_hm_outputs = human.combine_hm_eva(calibrated_model_probs, h_outputs)
                        h_cm_outputs, hm_outputs, h_cm_outputs_all = human.combine_hm(calibrated_model_probs, h_labels)
                        test_num_m += torch.sum(labels.eq(hm_outputs.argmax(dim=-1))).item()
                        test_num_h += torch.sum(labels.eq(h_labels)).item()
                        test_num_hm += torch.sum(labels.eq(final_hm_outputs.argmax(dim=-1))).item()
                acc_m = test_num_m / len(test_dataset)
                acc_h = test_num_h / len(test_dataset)
                acc_hm = test_num_hm/ len(test_dataset)
                test_record_temp.append(acc_hm)
                test_record_temp_h.append(h_cm_outputs)
                test_record_temp_h_all.append(h_cm_outputs_all)
                if best_acc_temp < acc_hm:
                    best_acc_temp = acc_hm
                    evaluator_model_last.load_state_dict(evaluator_model.state_dict())
                if acc_m > best_acc_temp_cm:
                    best_acc_temp_cm = acc_m
            # active collection
            # for the next limit human label
            evaluator_model_last.eval()
            with torch.no_grad():
                acc_list = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
                if nums_limit < limit_human_pred[-2] and random_mode != True:
                    al_sample_list = [item for sublist in update_sampele_dict.values() for item in sublist]
                    al_sample_dataset = Subset(train_dataset, al_sample_list)
                    al_sample_loader = torch.utils.data.DataLoader(al_sample_dataset, len(al_sample_list), False)
                    for sample in al_sample_loader:
                        logits, labels, h_labels, feat = sample
                        logits = logits.to(device)
                        labels = labels#.to(device)
                        h_labels = h_labels.to(device)
                        feat = feat.to(device)
                        calibrated_model_probs = calibrator.calibrate(logits)
                        h_outputs = evaluator_model_last(h_labels, feature_input=feat)
                        final_hm_outputs = human.combine_hm_eva(calibrated_model_probs, h_outputs)
                        # Calculate the difference between the true probability and 0.5
                        entropy = h_outputs[torch.arange(h_outputs.size(0)), labels].cpu()
                        abs_acc = torch.abs(entropy-acc_list[labels])
                        data = np.column_stack((al_sample_list, to_numpy_(torch.tensor(abs_acc)), to_numpy_(labels[:len(al_sample_list)])))
                        sorted_data = {}
                        for category in np.unique(to_numpy_(labels)):
                            sorted_data[category] = data[data[:, 2] == category]
                            sorted_data[category] = sorted(sorted_data[category], key=lambda x: x[1], reverse=True)
                        for category, samples in sorted_data.items():
                            sample_ids = [int(sample[0]) for sample in samples]  
                            update_sampele_dict[int(category)] = sample_ids
            # metrics
            # The final accuracy is the average of the top five best results
            top_5_numbers_with_ids = sorted(enumerate(test_record_temp), key=lambda x: x[1], reverse=True)[:5]
            top_5_numbers = [x[1] for x in top_5_numbers_with_ids]
            top_5_indexes = [x[0] for x in top_5_numbers_with_ids]
            average = sum(top_5_numbers) / len(top_5_numbers)
            # kl divergence
            if nums_limit != limit_human_pred[-1]:
                avg_test_h_all = [test_record_temp_h[i].cpu() for i in top_5_indexes]
                avg_test_h_all = torch.stack(avg_test_h_all)
                avg_test_h_all = avg_test_h_all.mean(dim=0)
                perfect_h_all = [test_record_temp_h_all[i].cpu() for i in top_5_indexes]
                perfect_h_all = torch.stack(perfect_h_all)
                perfect_h_all = perfect_h_all.mean(dim=0)
                kl_div = torch.sum(perfect_h_all * torch.log(perfect_h_all / avg_test_h_all), dim=1)
                average_kl_div = torch.mean(kl_div).item()
            acc_hm_list.append(round(average,4))
            acc_m_list.append(round(average_kl_div,4))
            acc_h_list.append(round(acc_h,4))
        repeat_acc_hm_list.append(acc_hm_list)
        repeat_acc_m_list.append(acc_m_list)
        repeat_acc_h_list.append(acc_h_list)
    print('H_ACC: ' +str([round(float(i),4) for i in np.mean(np.array(repeat_acc_h_list), axis=0)]))
    print('H_STD: ' +str([round(float(i),4) for i in np.std(np.array(repeat_acc_h_list), axis=0)]))
    print('HM_ACC: ' +str([round(float(i),4) for i in np.mean(np.array(repeat_acc_hm_list), axis=0)]))
    print('HM_STD: ' +str([round(float(i),4) for i in np.std(np.array(repeat_acc_hm_list), axis=0)]))


