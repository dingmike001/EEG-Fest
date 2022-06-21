import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import trange
import math
import itertools
import utils
from Network import LSTM, AttentionalLSTM, AttentionTransformer_Ning
from utils import get_lr_scheduler, mean_percentage_error, mean_absolute_percentage_error, dataset_ss
from scipy.io import savemat

# torch.manual_seed(145)
warnings.filterwarnings("ignore")


# train_data, test_data = utils.loaddata('/tmp/pycharm_project_LSTM/Dataset/6_20151121_noon_data_5.mat',
#                                        '/tmp/pycharm_project_LSTM/Dataset/6_20151121_noon_perclos.mat')


# train_data, val_data, test_data = utils.loaddatawithtest('/tmp/pycharm_project_LSTM/Dataset/6_20151121_noon_data_5.mat',
#                                                          '/tmp/pycharm_project_LSTM/Dataset/6_20151121_noon_perclos.mat')


#
# epoch_num = 1000
# channel_size = 17
# data_dimention = 5
# lays_num = 3
# hidden_dimention = 12
# label_size = 1
#
# model = Network.LSTMClsssifier(data_dimention, hidden_dimention, lays_num, label_size, channel_size)
#
# optimizer = optim.Adam(model.parameters(), lr=0.00001)

# lossweight = torch.tensor([1, 1, 1]).float()
# lossF = nn.CrossEntropyLoss(weight=lossweight)
# lossF = nn.CrossEntropyLoss()
# lossF = nn.MSELoss()
# loss_array = []
#
# with trange(epoch_num, unit='iteration', desc='epoch') as pbar:
#     for epoch in range(epoch_num):
#         running_loss = 0
#         for idx, content in enumerate(train_data):
#             inputa = torch.tensor(content['eeg']).float()
#             # labels = torch.tensor(content['label']).long()
#             labels = torch.tensor(content['label']).float()
#             output = model(inputa)
#             optimizer.zero_grad()
#             if labels.shape[0] == 1:
#                 labels = labels.squeeze(0)
#             else:
#                 labels = labels.squeeze()
#             loss = lossF(output, labels)
#
#             # loss = lossF(output, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         loss = running_loss / idx
#         pbar.set_postfix(loss=loss)
#         pbar.n += 1
#         loss_array.append(loss)
#
# plt.plot(loss_array)
# plt.show()
#
# testaccuracy = 0
# drowisinessacc = 0
# drowisiall = 0
# awakeacc = 0
# awakeall = 0
# tiredacc = 0
# tiredall = 0
#
# prict = []
# label = []
# with torch.no_grad():
#     for idx, content in enumerate(val_data):
#         inputa = torch.tensor(content['eeg']).float()
#         # labels = torch.tensor(content['label']).long()
#         labels = content['label']
#         output = model(inputa)
#         output = output[0][0]
#         prict.append(output)
#         label.append(labels)
# inputa = torch.max(output, dim=1)
# inputa = inputa.indices
# inputa = np.array(inputa)
# inputa = inputa[0]
# labels = np.array(labels)[0][0]
#         if labels == 2:
#             drowisiall += 1
#             if inputa == labels:
#                 testaccuracy += 1
#                 drowisinessacc += 1
#         elif labels == 0:
#             awakeall += 1
#             if inputa == labels:
#                 testaccuracy += 1
#                 awakeacc += 1
#         elif labels ==1:
#             tiredall+=1
#             if inputa == labels:
#                 testaccuracy+=1
#                 tiredacc+=1
#         print("test ", idx, " probs:", output, " real label: ", labels, " accuracy number: ", testaccuracy)
# testaccuracy = testaccuracy / (idx + 1)
# awakeacc = awakeacc / awakeall
# tiredacc = tiredacc/tiredall
# if drowisiall ==0:
#     drowisinessacc=0
# else:
#     drowisinessacc = drowisinessacc / drowisiall
# print('Accuracy = ', testaccuracy)
# print("Awake accuracy = ", awakeacc, "Awake number = ", awakeall)
# print("Tired accuracy = ", tiredacc, "Tired number = ", tiredall)
# print('Drowisiness accuracy = ', drowisinessacc, 'drowisiness number = ', drowisiall)

# plt.plot(prict)
# plt.plot(label)
# plt.legend(['prict', 'label'])
# plt.show()


def main(args):
    # train_data, test_data = utils.loaddata('./Dataset/train_de_2subjects.mat',
    #                                        './Dataset/val_de_2subjects_1.mat')
    # train_data, test_data = dataset_ss('./Dataset/seed_vig/1_20151124_noon_2_train.mat',
    #                                    './Dataset/seed_vig/1_20151124_noon_2_val.mat')
    # train_data, test_data = dataset_ss('./Dataset/seed_vig/1_20151124_noon_2_train.mat',
    #                                    './Dataset/seed_vig/1_20151124_noon_2_val.mat')
    train_data, test_data = dataset_ss(args.train_data,args.val_data)


    # train_data, test_data = dataset_ss('./Dataset/val_de_23subjects_1.mat',
    #                                    './Dataset/val_de_23subjects_5.mat')
    scaler = MinMaxScaler()

    # Lists #
    train_losses, train_lossesa, val_losses = list(), list(), list()
    # val_maes, val_mses, val_rmses, val_mapes, val_mpes, val_r2s = list(), list(), list(), list(), list(), list()
    # test_maes, test_mses, test_rmses, test_mapes, test_mpes, test_r2s = list(), list(), list(), list(), list(), list()
    pred_tests, labels = list(), list()

    # Constants #
    best_val_loss = 10000
    best_val_improv = 0

    # Prepare Network #
    # model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size, args.bidirectional)
    model = AttentionTransformer_Ning()
    model.load_state_dict(torch.load('./Model/self_attention/2_20151106_noon.pkl', map_location='cuda'))
    model = model.to('cuda')

    # model = AttentionalLSTM(input_size, qkv, hidden_size, num_layers, output_size, bidirectional)
    # Loss Function #
    criterion = torch.nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_scheduler = get_lr_scheduler(args.lr_scheduler, optim)
    if args.mode == 'train':
        with trange(args.epoch_num, unit='iteration', desc='epoch') as pbar:
            for epoch in range(args.epoch_num):

                for i, content in enumerate(train_data):
                    # Prepare Data #
                    data = torch.tensor(content['eeg']).float()
                    label = torch.tensor(content['label']).float()

                    # Forward Data #
                    pred = model(data)

                    # Calculate Loss #
                    train_loss = criterion(pred, label)

                    # Initialize Optimizer, Back Propagation and Update #
                    optim.zero_grad()
                    train_loss.backward()
                    optim.step()

                    # Add item to Lists #
                    train_losses.append(train_loss.item())

                loss = np.average(train_losses)
                pbar.set_postfix(loss=loss)
                pbar.n += 1
                train_lossesa.append(loss)

                # Learning Rate Scheduler #
                optim_scheduler.step()

        plt.plot(train_lossesa)
        plt.show()

        if args.trainmode == 'valid':
            with torch.no_grad():
                for i, content in enumerate(val_data):
                    # Prepare Data #
                    data = torch.tensor(content['eeg']).float()
                    label = torch.tensor(content['label']).float()

                    # Forward Data #
                    pred_val = model(data)

                    # Calculate Loss #
                    val_loss = criterion(pred_val, label)

                    if args.multi_step:
                        pred_val = np.mean(pred_val.detach().numpy(), axis=1)
                        label = np.mean(label.detach().numpy(), axis=1)
                    else:
                        pred_val, label = pred_val, label

                # Add item to Lists #
                val_losses.append(val_loss.item())

                # Save the model only if validation loss decreased #
                curr_val_loss = np.average(val_losses)

        cur_tra_loss = np.average(train_losses)

        if cur_tra_loss < best_val_loss:
            best_val_loss = min(cur_tra_loss, best_val_loss)
            torch.save(model.state_dict(),
                       os.path.join('/tmp/pycharm_project_LSTM/Model',
                                    'BEST_Model_using_LSTM.pkl'))

            print("Best model is saved!\n")
            print("best_val_loss = ", best_val_loss)
            best_val_improv = 0

        elif cur_tra_loss >= best_val_loss:
            best_val_improv += 1
            print("Best Validation has not improved for {} epochs.\n".format(best_val_improv))





    elif args.mode == 'test':

        # Load the Model Weight #
        model.load_state_dict(
            torch.load(os.path.join('/tmp/pycharm_project_LSTM/Model',
                                    'BEST_Model_using_LSTM.pkl')))

        # Test #
        with torch.no_grad():
            for i, content in enumerate(test_data):

                # Prepare Data #
                data = torch.tensor(content['eeg']).float()
                label = torch.tensor(content['label']).float()

                # Forward Data #
                pred_test = model(data)

                # Convert to Original Value Range #
                # pred_test, label = pred_test.detach().numpy(), label.detach().numpy()
                #
                # pred_test = scaler.inverse_transform(pred_test)
                # label = scaler.inverse_transform(label)

                if args.multi_step:
                    pred_test = np.mean(pred_test, axis=1)
                    label = np.mean(label, axis=1)

                pred_tests += pred_test.tolist()
                labels += label.tolist()

                # Calculate Loss #
                # test_mae = mean_absolute_error(label, pred_test)
                # test_mse = mean_squared_error(label, pred_test, squared=True)
                # test_rmse = mean_squared_error(label, pred_test, squared=False)
                # test_mpe = mean_percentage_error(label, pred_test)
                # test_mape = mean_absolute_percentage_error(label, pred_test)
                # test_r2 = r2_score(label, pred_test)

                # Add item to Lists #
                # test_maes.append(test_mae.item())
                # test_mses.append(test_mse.item())
                # test_rmses.append(test_rmse.item())
                # test_mpes.append(test_mpe.item())
                # test_mapes.append(test_mape.item())
                # test_r2s.append(test_r2.item())

            # Print Statistics #

            # print(" MAE : {:.4f}".format(np.average(test_maes)))
            # print(" MSE : {:.4f}".format(np.average(test_mses)))
            # print("RMSE : {:.4f}".format(np.average(test_rmses)))
            # print(" MPE : {:.4f}".format(np.average(test_mpes)))
            # print("MAPE : {:.4f}".format(np.average(test_mapes)))
            # # print(" R^2 : {:.4f}".format(np.average(test_r2s)))

            print('MAE = ', mean_absolute_error(labels, pred_tests))
            print('R^2 = ', r2_score(labels, pred_tests))

            plt.plot(pred_tests)
            plt.plot(labels)
            plt.legend(['prict', 'label'])
            plt.show()

            # # Plot Figure #
            # plot_pred_test(pred_tests[:args.time_plot], labels[:args.time_plot], args.plots_path, args.feature, model, step)
            #
            # # Save Numpy files #
            pred_tests = np.asarray(pred_tests)
            sizea = pred_tests.shape[0]
            pred_tests = pred_tests.reshape(sizea)
            pred_tests = {'pred_test': pred_tests}
            labels = np.asarray(labels)
            sizea = labels.shape[0]
            labels = labels.reshape(sizea)
            labels = {'label': labels}
            savemat('pred_tests.mat', pred_tests)
            savemat('label.mat', labels)
            # np.save(os.path.join('/tmp/pycharm_project_LSTM/Model',
            #                      '{}_using_{}_TestSet.npy'.format(model.__class__.__name__, step)),
            #         np.asarray(pred_tests))
            # np.save(os.path.join('/tmp/pycharm_project_LSTM/Model',
            #                      'TestSet_using_{}.npy'.format(step)), np.asarray(labels))

    elif args.mode == 'train and test':
        with trange(args.epoch_num, unit='iteration', desc='epoch') as pbar:
            for epoch in range(args.epoch_num):
                model.train()
                train_losses_running = list()
                for i, content in enumerate(train_data):
                    # Prepare Data #
                    data = torch.tensor(content['eeg']).to('cuda').float()
                    label = torch.tensor(content['label']).to('cuda').float()

                    # Forward Data #
                    pred = model(data)

                    # Calculate Loss #
                    train_loss = criterion(pred, label.unsqueeze(1))

                    # Initialize Optimizer, Back Propagation and Update #
                    optim.zero_grad()
                    train_loss.backward()
                    optim.step()

                    # Add item to Lists #
                    train_losses.append(train_loss.item())
                    train_losses_running.append(train_loss.item())

                loss = np.average(train_losses)
                pbar.set_postfix(loss=loss)
                pbar.n += 1
                train_lossesa.append(loss)

                # Learning Rate Scheduler #
                optim_scheduler.step()

        plt.plot(train_lossesa)
        plt.show()
        cur_tra_loss = np.average(train_losses)

        if cur_tra_loss < best_val_loss:
            best_val_loss = min(cur_tra_loss, best_val_loss)
            torch.save(model.state_dict(),
                       os.path.join('./Model',
                                    'BEST_Model_using_LSTM.pkl'))

            print("Best model is saved!\n")
            print("best_val_loss = ", best_val_loss)
            best_val_improv = 0

        elif cur_tra_loss >= best_val_loss:
            best_val_improv += 1
            print("Best Validation has not improved for {} epochs.\n".format(best_val_improv))

        model.load_state_dict(torch.load('./Model/BEST_Model_using_LSTM.pkl', map_location='cuda'))

        model.eval()
        with torch.no_grad():
            for i, content in enumerate(test_data):

                # Prepare Data #
                data = torch.tensor(content['eeg']).to('cuda').float()
                label = torch.tensor(content['label']).to('cuda').float()
                label = label.unsqueeze(1)
                # Forward Data #
                pred_test = model(data)

                if args.multi_step:
                    pred_test = np.mean(pred_test, axis=1)
                    label = np.mean(label, axis=1)

                pred_tests += pred_test.tolist()
                labels += label.tolist()
            predict_out = list(itertools.chain.from_iterable(pred_tests))
            target_out = list(itertools.chain.from_iterable(labels))
            cc = np.corrcoef(predict_out, target_out)
            MSE = np.square(np.subtract(predict_out, target_out)).mean()
            RMSE = math.sqrt(MSE)
            cc = cc[0][1]

            print('MAE = ', mean_absolute_error(target_out, predict_out))
            print('R^2 = ', r2_score(target_out, predict_out))
            print('cc = ', cc)
            print('RMSE = ', RMSE)

            plt.plot(predict_out)
            plt.plot(target_out)
            plt.legend(['prict', 'label'])
            plt.show()

            # # Save mat files #
            # pred_tests = np.asarray(pred_tests)
            # sizea = pred_tests.shape[0]
            # pred_tests = pred_tests.reshape(sizea)
            # pred_tests = {'pred_test': pred_tests}
            # labels = np.asarray(labels)
            # sizea = labels.shape[0]
            # labels = labels.reshape(sizea)
            # labels = {'label': labels}
            # savemat('pred_tests.mat', pred_tests)
            # savemat('label.mat', labels)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=5, help='input_size')
    parser.add_argument('--hidden_size', type=int, default=10, help='hidden_size')
    parser.add_argument('--num_layers', type=int, default=3, help='num_layers')
    parser.add_argument('--output_size', type=int, default=1, help='output_size')
    parser.add_argument('--bidirectional', type=bool, default=False, help='use bidirectional or not')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler',
                        choices=['step', 'plateau', 'cosine'])
    parser.add_argument('--epoch_num', type=int, default=1
                        , help='total epoch')
    parser.add_argument('--multi_step', type=bool, default=False, help='multi-step or not')
    parser.add_argument('--mode', type=str, default='train and test',
                        choices=['train', 'test', 'train and test', 'inference'])
    parser.add_argument('--trainmode', type=str, default='none', choices=['valid', 'none', 'inference'])
    parser.add_argument('--train_data', type=str, default='./Dataset/seed_vig/2_all.mat')
    parser.add_argument('--val_data', type=str, default='./Dataset/seed_vig/2_all.mat')

    config = parser.parse_args()

    main(config)
