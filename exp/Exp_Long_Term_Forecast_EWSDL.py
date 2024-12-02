from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, adjust_learning_rate_new
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import os
import time
import warnings
import numpy as np
import math
import sys
from tqdm import tqdm
import wandb
warnings.filterwarnings('ignore')



class Exp_Long_Term_Forecast_EWSDL(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_EWSDL, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        ##### 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print("Total parameters:",total_params)
        ##################

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion,is_test = True):
        total_loss = []
        total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # pred = outputs.detach().cpu()
                # true = batch_y.detach().cpu()
                if self.args.model == 'MambaDBF' and is_test == False:
                    # 指数平滑衰减 0.01-->0.005
                    decay_factor = 0.005  
                    self.ratio = np.array([np.exp(-decay_factor * i) for i in range(self.args.pred_len)])
                    self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
                    outputs = outputs * self.ratio
                    batch_y = batch_y * self.ratio
                pred = outputs#.detach().cpu()
                true = batch_y#.detach().cpu()

                loss = criterion(pred, true)

        #         total_loss.append(loss)
        # total_loss = np.average(total_loss)
                total_loss.append(loss.item()*batch_y.shape[0])
                total_samples += batch_y.shape[0]
        total_loss = np.sum(total_loss) /total_samples
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')
        test_data, test_loader = self._get_data(flag='test0')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        c = nn.L1Loss()

        if self.args.lradj == 'TST':
            train_steps = len(train_loader)
            scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        else:
            scheduler = None
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # loss = criterion(outputs, batch_y)
                    if self.args.model == 'MambaDBF':
                        # 指数平滑衰减 0.01-->0.005
                        decay_factor = 0.005  
                        self.ratio = np.array([np.exp(-decay_factor * i) for i in range(self.args.pred_len)])
                        self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
                        outputs = outputs *self.ratio
                        batch_y = batch_y *self.ratio
                        loss = c(outputs, batch_y)
                        
                        


                        use_h_loss = False
                        h_level_range = [4,8,16,24,48,96]
                        h_loss = None
                        if use_h_loss:
                            
                            for h_level in h_level_range:
                                batch,length,channel = outputs.shape
                                # print(outputs.shape)
                                h_outputs = outputs.transpose(-1,-2).reshape(batch,channel,-1,h_level)
                                h_outputs = torch.mean(h_outputs,dim = -1,keepdims = True)
                                h_batch_y = batch_y.transpose(-1,-2).reshape(batch,channel,-1,h_level)
                                h_batch_y = torch.mean(h_batch_y,dim = -1,keepdims = True)
                                h_ratio = self.ratio[:h_outputs.shape[-2],:]
                                # print(h_outputs.shape,h_ratio.shape)
                                h_ouputs_agg = torch.mean(h_outputs,dim = 1,keepdims = True)
                                h_batch_y_agg = torch.mean(h_batch_y,dim = 1,keepdims = True)


                                h_outputs = h_outputs*h_ratio
                                h_batch_y = h_batch_y*h_ratio

                                h_ouputs_agg *= h_ratio
                                h_batch_y_agg *= h_ratio

                                if h_loss is None:
                                    h_loss  = c(h_outputs, h_batch_y)*np.sqrt(h_level) /2 +c(h_ouputs_agg, h_batch_y_agg)*np.sqrt(h_level) /2
                                else:
                                    h_loss = h_loss + c(h_outputs, h_batch_y)*np.sqrt(h_level) /2 +c(h_ouputs_agg, h_batch_y_agg)*np.sqrt(h_level) /2
                            # outputs = 0


                    else:
                        loss = criterion(outputs, batch_y)

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    if h_loss != 0:
                        loss = loss #+ h_loss * 1e-2
                    loss.backward()
                    model_optim.step()
                if self.args.lradj == 'TST':
                    adjust_learning_rate_new(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)
 

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            if self.args.model == 'MambaDBF':
                vali_loss = self.vali(vali_data, vali_loader, c,is_test = False)
                test_loss = self.vali(test_data, test_loader, nn.MSELoss(),is_test = True)
            else:
                test_loss = self.vali(vali_data, vali_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            # test_loss = self.vali(test_data, test_loader, criterion)
            # wandb.log({"Train Loss": train_loss," Vali Loss":vali_loss,"Test loss tmp": test_loss})
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.lradj != 'TST': 
                adjust_learning_rate_new(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader): #enumerate(tqdm(test_loader, desc='Processing', leave=False)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)             
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        # wandb.log({"test mae": mae," test mse":mse})
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

 
        return


