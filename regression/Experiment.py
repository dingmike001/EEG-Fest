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

torch.manual_seed(145)
warnings.filterwarnings("ignore")




def main(args):
   
    train_data, test_data = dataset_ss(args.train_data,args.val_data)

    scaler = MinMaxScaler()

    # Lists #
    train_losses, train_lossesa, val_losses = list(), list(), list()
    pred_tests, labels = list(), list()

    # Constants #
    best_val_loss = 10000
    best_val_improv = 0

    # Prepare Network #
    # model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size, args.bidirectional)
    model = AttentionTransformer_Ning()
    model.load_state_dict(torch.load('./Model/self_attention/2_20151106_noon.pkl', map_location='cuda'))
    model = model.to('cuda')

    
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


                if args.multi_step:
                    pred_test = np.mean(pred_test, axis=1)
                    label = np.mean(label, axis=1)

                pred_tests += pred_test.tolist()
                labels += label.tolist()

             

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
