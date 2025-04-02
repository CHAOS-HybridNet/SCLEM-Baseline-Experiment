from sklearn.preprocessing import minmax_scale
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from time import time
from SCWarn.util.dataset import use_mini_batch, apply_sliding_window
from SCWarn.util.corrloss import CorrLoss

class MLSTM(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, batch_size=64):
        super(MLSTM, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.lstm1 = nn.LSTM(input_dim1, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(input_dim2, hidden_dim, batch_first=True)

        self.hiddent1out = nn.Linear(hidden_dim*2, input_dim1)
        self.hiddent2out = nn.Linear(hidden_dim*2, input_dim2)
        # hidden
        # self.hiddenout = nn.Linear(hidden_dim*2, input_dim1+input_dim2)
        self.hiddenout = nn.Linear(hidden_dim*2, 1)

         # 权重初始化
        nn.init.xavier_uniform_(self.hiddenout.weight)
        nn.init.zeros_(self.hiddenout.bias)


    def forward(self, seq1, seq2):
        # 检查 seq1 和 seq2 是否有有效长度
        if seq1.size(1) == 0 or seq2.size(1) == 0:
            raise ValueError(f"Sequence length is zero: seq1 ({seq1.size()}), seq2 ({seq2.size()})")

        # 确保 seq1 和 seq2 的形状匹配 LSTM 的输入维度
        lstm_out1, _ = self.lstm1(seq1.reshape(self.batch_size, -1, self.input_dim1))
        lstm_out2, _ = self.lstm2(seq2.reshape(self.batch_size, -1, self.input_dim2))

        # 调试输出（完成测试后可以删除）
        # print("lstm_out1 shape:", lstm_out1.shape)
        # print("lstm_out2 shape:", lstm_out2.shape)

        # 确保 lstm_out2 的时间步数与 lstm_out1 一致
        if lstm_out2.size(1) < lstm_out1.size(1):
            padding = torch.zeros(lstm_out2.size(0), lstm_out1.size(1) - lstm_out2.size(1), lstm_out2.size(2), device=lstm_out2.device)
            lstm_out2 = torch.cat((lstm_out2, padding), dim=1)

        # 合并两个 LSTM 输出
        shared = torch.cat((lstm_out1, lstm_out2), dim=2)

        # 通过隐藏层计算最终预测
        predict = self.hiddenout(shared)

        # 返回最终预测和 LSTM 最后时间步的输出
        return predict[:, -1, :], (lstm_out1[:, -1, :], lstm_out2[:, -1, :])



    # def forward(self, seq1, seq2):
    #     # lstm_out1, _ = self.lstm1(seq1.view(self.batch_size, -1, self.input_dim1))  # [batch, seq_len, hidden_dim]
    #     # lstm_out2, _ = self.lstm2(seq2.reshape(self.batch_size, -1, self.input_dim2))  # [batch, seq_len, hidden_dim]

    #     lstm_out1, _ = self.lstm1(seq1)  # [batch, seq_len, hidden_dim]
    #     lstm_out2, _ = self.lstm2(seq2)  # [batch, seq_len, hidden_dim]

    #     # 确保 lstm_out2 的时间步与 lstm_out1 一致
    #     if lstm_out2.size(1) < lstm_out1.size(1):
    #         # 如果 lstm_out2 的时间步少于 lstm_out1，则裁剪 lstm_out1
    #         lstm_out1 = lstm_out1[:, :lstm_out2.size(1), :]
    #     elif lstm_out2.size(1) > lstm_out1.size(1):
    #         # 如果 lstm_out2 的时间步多于 lstm_out1，则裁剪 lstm_out2
    #         lstm_out2 = lstm_out2[:, :lstm_out1.size(1), :]

    #     # 拼接 LSTM 输出
    #     shared = torch.cat((lstm_out1, lstm_out2), dim=2)  # [batch, seq_len, hidden_dim*2]

    #     # 通过输出层生成预测
    #     predict = self.hiddenout(shared)  # [batch, seq_len,1]

    #     # 仅返回最后一个时间步的预测结果
    #     predict = predict[:, -1, :]  # [batch,1]

    #     return predict, (lstm_out1[:, -1, :], lstm_out2[:, -1, :])





def train(dataloader, modal, batch_size, n_epoch, lr=0.01):


    input_dim1, input_dim2 = modal[0], modal[1]
    # input_dim1, input_dim2 = 1,16
    # print("input dim1, input dim2", input_dim1, input_dim2)
    model = MLSTM(input_dim1, input_dim2, 48, batch_size)
    loss_function = nn.MSELoss()
    loss_corr = CorrLoss()

    optimizer1 = optim.SGD(model.parameters(), lr=lr)


    for epoch in range(n_epoch):
        t0 = time()
        print("epoch: %d / %d" % (epoch+1, n_epoch))

        loss_sum = 0
        for step, (batch_X, batch_Y) in enumerate(dataloader):
            # print("Shape of input for lstm1:", batch_X[:, :, :input_dim1].shape)
            # print("Shape of input for lstm2:", batch_X[:, :, input_dim1:].shape)
            model.zero_grad()
            #print(batch_X.shape,batch_Y.shape)   #torch.Size([128, 10, 11]) torch.Size([128, 11])
            predicted, (H1, H2) = model(batch_X[:,:, :input_dim1], batch_X[:, :, input_dim1:])
            # loss = loss_function(predicted, batch_Y) + loss_corr(H1, H2)
            loss = loss_function(predicted, batch_Y)
            loss_sum += loss.item()
            if step % 100 == 0:
                print("loss: %f"%(loss_sum / 100))
                loss_sum = 0

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

        print("time: %.2f s" % float(time() - t0))



    return model


# def train(dataloader, modal, batch_size, n_epoch, lr=0.001):  # 降低学习率
#     # 确定输入维度
#     input_dim1, input_dim2 = modal[0], modal[1]  # [1, 10]
#     print("input dim1, input dim2", input_dim1, input_dim2)
    
#     # 初始化模型和损失函数
#     model = MLSTM(input_dim1, input_dim2, 48, batch_size)
#     loss_function = nn.MSELoss()
#     loss_corr = CorrLoss()  # 假设这是自定义相关损失
    
#     optimizer1 = optim.SGD(model.parameters(), lr=lr)

#     model.train()  # 确保模型处于训练模式

#     for epoch in range(n_epoch):
#         t0 = time.time()
#         print("Epoch: %d / %d" % (epoch + 1, n_epoch))

#         loss_sum = 0
#         for step, (batch_X, batch_Y) in enumerate(dataloader):
#             print(f"\nBatch {step + 1}")
#             print("Shape of input for lstm1:", batch_X[:, :, :input_dim1].shape)
#             print("Shape of input for lstm2:", batch_X[:, :, input_dim1:].shape)
            
#             model.zero_grad()
#             # 获取预测输出和隐藏状态
#             predicted, (H1, H2) = model(batch_X[:, :, :input_dim1], batch_X[:, :, input_dim1:])
            
#             # 检查 predicted 和 batch_Y 的形状
#             print("Predicted shape:", predicted.shape)  # [64,1]
#             print("Batch_Y shape:", batch_Y.shape)      # [64,11]

#             # 强行调整 Batch_Y 的形状为 [64,1]
#             try:
#                 # 选择第一个特征作为目标
#                 batch_Y = batch_Y[:, 0].unsqueeze(-1)  # 从 [64,11] 变为 [64,1]
                
#                 # 确保 batch_Y 没有 nan
#                 batch_Y = torch.nan_to_num(batch_Y, nan=0.0)
                
#                 # 确保 predicted 没有 nan
#                 predicted = torch.nan_to_num(predicted, nan=0.0)
                
#                 # 计算 MSE 损失
#                 loss_mse = loss_function(predicted, batch_Y)
                
#                 # 确保 H1 和 H2 没有 nan
#                 H1 = torch.nan_to_num(H1, nan=0.0)
#                 H2 = torch.nan_to_num(H2, nan=0.0)
                
#                 # 计算相关损失
#                 loss_corr_value = loss_corr(H1, H2)
                
#                 # 计算总损失
#                 loss = loss_mse + loss_corr_value
#             except Exception as e:
#                 print("Error in loss calculation:", e)
#                 break

#             print(f"Loss (MSE): {loss_mse.item():.4f}")
#             print(f"Loss (Corr): {loss_corr_value.item():.4f}")
#             print(f"Total Loss: {loss.item():.4f}")

#             loss_sum += loss.item()
            
#             if (step + 1) % 10 == 0:  # 每10个批次打印一次平均损失
#                 print("Average Loss (last 10 batches): %f" % (loss_sum / 10))
#                 loss_sum = 0

#             # 梯度裁剪，防止梯度爆炸
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             optimizer1.zero_grad()
#             loss.backward()
#             optimizer1.step()

#         epoch_time = time.time() - t0
#         print("Time for Epoch %d: %.2f s\n" % (epoch + 1, epoch_time))

#     return model






def predict(model, test_data, seq_len, modal):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.batch_size = 1
    predict_ls = []
    anomaly_scores = []
    loss_function = nn.MSELoss()

    anomaly_scores_per_dim = []

    with torch.no_grad():
        for i in range(seq_len, len(test_data)):
            seq = torch.tensor(test_data[i-seq_len : i], dtype=torch.float32).to(device)
            ground_truth = torch.tensor(test_data[i], dtype=torch.float32).to(device)
            # print(f"seq:{seq}")

            # 填充 NaN 值
            seq = torch.nan_to_num(seq, nan=0.0)
            ground_truth = torch.nan_to_num(ground_truth, nan=0.0)

            num_features = seq.size(1)  # 假设 seq 是你的输入数据
            modal = [num_features // 2, num_features]  # 将特征均分


            print(f"seq shape: {seq.shape}, modal: {modal} at index {i}")

            seq1, seq2 = seq[:, :modal[0]], seq[:, modal[0]:]

            print(f"seq1 shape: {seq1}, seq2 shape: {seq2} at index {i}")

            # if seq1.size(1) == 0 or seq2.size(1) == 0:
            #     # print(f"Invalid sequence lengths: seq1 ({seq1.size()}), seq2 ({seq2.size()}) at index {i}")
            #     continue

            predicted, _ = model(seq1, seq2)

            if predicted.shape != ground_truth.shape:
                ground_truth = ground_truth.unsqueeze(0)  # 添加一个维度
                # ground_truth = ground_truth.expand_as(predicted)

            anomaly_score = loss_function(predicted.view(-1), ground_truth.view(-1))
            predict_ls.append(predicted.tolist())
            anomaly_scores.append(anomaly_score.item())

            dim_scores = np.abs(predicted.cpu().numpy() - ground_truth.cpu().numpy())
            anomaly_scores_per_dim.append(dim_scores)

    anomaly_scores_per_dim = np.array(anomaly_scores_per_dim)

    # 过滤掉 anomaly_scores 中的 NaN 或 inf 值并进行归一化
    anomaly_scores = [score for score in anomaly_scores if not np.isnan(score) and not np.isinf(score)]
    anomaly_scores = np.clip(anomaly_scores, a_min=0, a_max=None)
    if len(anomaly_scores) > 0:
        anomaly_scores = minmax_scale(anomaly_scores)
    else:
        anomaly_scores = np.zeros(len(predict_ls))

    # 检查并处理 anomaly_scores_per_dim
    if anomaly_scores_per_dim.size > 0:
        anomaly_scores_per_dim = np.apply_along_axis(
            lambda x: minmax_scale(np.clip(x, a_min=0, a_max=None)), 
            0, 
            anomaly_scores_per_dim
        )
    else:
        anomaly_scores_per_dim = np.zeros_like(predict_ls)

    return predict_ls, anomaly_scores, anomaly_scores_per_dim




# def predict(model, test_data, seq_len, modal):
#     model.batch_size = 1
#     predict_ls = []
#     # 检查 modal 是否为整数
#     assert isinstance(modal[0], int) and isinstance(modal[1], int), "modal indices must be integers."
#     anomaly_scores = []
#     loss_function = nn.MSELoss()

#     anomaly_scores_per_dim = []
#     model.eval()  # 确保模型处于评估模式
#     with torch.no_grad():
#         for i in range(seq_len, len(test_data)):
#             seq = torch.tensor(test_data[i-seq_len : i]).float()

#             predicted, _ = model(seq[:, :modal[0]], seq[:, modal[0]:])  # [1,1]

#             ground_truth = torch.tensor(test_data[i]).float()  # [11]

#             # 处理 NaN 值
#             ground_truth = torch.nan_to_num(ground_truth, nan=0.0)
#             predicted = torch.nan_to_num(predicted, nan=0.0)

#             # 选择第一个特征作为目标
#             ground_truth = ground_truth[0].unsqueeze(-1)  # 从 [11] 变为 [1]

#             # 确保 predicted 和 ground_truth 的形状一致
#             if predicted.shape != ground_truth.shape:
#                 print(f"Shape mismatch: predicted {predicted.shape}, ground_truth {ground_truth.shape}")
#                 # 根据需要进行调整，这里假设保持前1个
#                 predicted = predicted[:, :ground_truth.size(1), :]
#                 ground_truth = ground_truth[:predicted.size(0), :predicted.size(1)]

#             # 调试信息
#             print(f"Adjusted Predicted shape: {predicted.shape}, Ground truth shape: {ground_truth.shape}")

#             # 计算损失
#             anomaly_score = loss_function(predicted, ground_truth)

#             predict_ls.append(predicted.squeeze(-1).tolist())  # [1]
#             anomaly_scores.append(anomaly_score.item())

#             anomaly_scores_per_dim.append(np.abs(predicted.numpy() - ground_truth.numpy()))

#     anomaly_scores_per_dim = np.array(anomaly_scores_per_dim)
#     # anomaly_scores = minmax_scale(anomaly_scores)

#     return predict_ls, anomaly_scores, anomaly_scores_per_dim




def predict_test(model, train_data, test_data, seq_len, modal):
    model.batch_size = 1
    predict_ls = []
    anomaly_scores = []
    loss_function = nn.MSELoss()

    anomaly_scores_per_dim = []
    with torch.no_grad():
        data = np.concatenate((train_data, test_data), axis=0)
        for i in range(len(test_data)):
            # if i > 9:
            #     seq = torch.tensor(test_data[i-seq_len : i]).float()
            # else:
            length = len(train_data) + i
            seq = torch.tensor(data[length-seq_len : length]).float()

            predicted, _ = model(seq[:, :modal[0]], seq[:, modal[0]:])
            ground_truth = torch.tensor(test_data[i]).float()
            anomaly_score = loss_function(predicted.view(-1), ground_truth.view(-1))

            predict_ls.append(predicted.tolist())
            anomaly_scores.append(anomaly_score.item())
            anomaly_scores_per_dim.append(np.abs(predicted.numpy() - ground_truth.numpy()))

            # print(f"Index {i}: Predicted: {predicted}, Ground Truth: {ground_truth}, Anomaly Score: {anomaly_score}")

    anomaly_scores_per_dim = np.array(anomaly_scores_per_dim)
    # anomaly_scores = minmax_scale(anomaly_scores)
    # print(predict_ls)
    return predict_ls, anomaly_scores, anomaly_scores_per_dim


def get_model_MLSTM(train_data, modal, seq_len=10, batch_size=64, n_epoch=10, lr=0.01):
    seq_dataset, seq_ground_truth = apply_sliding_window(train_data, seq_len=seq_len,flatten=False)
    train_data_loader = use_mini_batch(seq_dataset, seq_ground_truth, batch_size)

    model = train(train_data_loader, modal, batch_size, n_epoch, lr)

    return model

def get_prediction_MLSTM_test(model, train_data, test_data, seq_len, modal):
    predict_ls, scores, dim_scores = predict_test(model, train_data, test_data, seq_len, modal)
    return scores, dim_scores

def get_prediction_MLSTM(model, test_data, seq_len, modal):
    print(f"=======================test_data:{test_data}")
    predict_ls, scores, dim_scores = predict(model, test_data, seq_len, modal)
    return scores, dim_scores

def run_mlstm(train_data, test_data, modal, seq_len=10, batch_size=64, n_epoch=10):
    seq_dataset, seq_ground_truth = apply_sliding_window(train_data, seq_len=seq_len,flatten=False)
    train_data_loader = use_mini_batch(seq_dataset, seq_ground_truth, batch_size)

    # input_dim = train_data_loader.dataset.feature_len
    # print(input_dim)

    model = train(train_data_loader, modal, batch_size, n_epoch)
    predict_ls, scores, dim_scores = predict(model, test_data, seq_len, modal)

    return scores, dim_scores
