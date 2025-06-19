from __future__ import division
from __future__ import print_function
import argparse
import torch_geometric
from pygcn.extract_samples_from_concentration import extract_samples_from_concentration
from pygcn.utils import load_data
from pygcn.models import GAT
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dropout, Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional
from sklearn.metrics import accuracy_score, recall_score, f1_score

print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def graph_reconstruction_loss(output, adj):
    """
    Calculate the graph reconstruction loss. The reconstruction task used here is based on the reconstruction error between the GAT output feature matrix and the original adjacency matrix.
    """
    reconstructed_adj = torch.matmul(output, output.t())  # 使用GCN输出的特征重建图
    loss = F.mse_loss(reconstructed_adj, adj)  # 最小化重建的邻接矩阵与真实邻接矩阵之间的误差
    return loss


def GAT_train(graphs_dict, GAT_model, GAT_optimizer, args, gcn_epochs=1):
    """
    Train each graph separately by concentration and return the GAT feature matrix of the graph at each concentration.
    During the training process, unsupervised graph reconstruction loss is used to optimize the feature matrix of the graph.
    The output is a list, with each element corresponding to a concentration, and each concentration is a tensor of shape (N, 10, 16).
    """
    # Store the optimal GAT feature matrix for each concentration
    gat_output_list = []

    for concentration, graph_data_list in graphs_dict.items():
        concentration_outputs = []

        for graph_data in graph_data_list:
            features = graph_data['features']
            adj = graph_data['adj']

            if isinstance(adj, torch.sparse.Tensor):
                adj = adj.to_dense()

            edge_index = torch_geometric.utils.dense_to_sparse(adj)[0]

            best_output = None
            best_loss = float('inf')

            for epoch in range(gcn_epochs):
                GAT_model.train()
                GAT_optimizer.zero_grad()
                output = GAT_model(features, edge_index)

                loss_train = graph_reconstruction_loss(output, adj)
                loss_train.backward()

                GAT_optimizer.step()

                if loss_train.item() < best_loss:
                    best_loss = loss_train.item()
                    best_output = output.cpu()

            concentration_outputs.append(best_output)

        concentration_tensor = torch.stack(concentration_outputs, dim=0)
        gat_output_list.append(concentration_tensor)

    return gat_output_list


def pad_or_crop_feature_matrices(feature_matrices, target_time_steps, target_nodes):

    padded_matrices = []
    for fm in feature_matrices:
        if len(fm.shape) == 2:
            fm = fm.unsqueeze(1)

        time_steps, nodes, features = fm.shape

        if time_steps < target_time_steps:
            padding = (0, 0, 0, target_time_steps - time_steps)
            fm = F.pad(fm, padding, "constant", 0)
        elif time_steps > target_time_steps:
            fm = fm[:target_time_steps, :, :]

        if nodes < target_nodes:
            fm = F.pad(fm, (0, target_nodes - nodes, 0, 0), "constant", 0)
        elif nodes > target_nodes:
            fm = fm[:, :target_nodes, :]

        padded_matrices.append(fm)

    return padded_matrices



def process_gcn_output(gcn_output_list):
    X = []

    for i, concentration_data in enumerate(gcn_output_list):
        concentration_data_processed = []

        for fm in concentration_data:
            feature_matrix_flat = fm.view(1, -1).detach().numpy()
            concentration_data_processed.append(feature_matrix_flat)

        concentration_data_np = np.array(concentration_data_processed)

        concentration_data_np = concentration_data_np.squeeze(axis=1)

        X.append(concentration_data_np)

    for i, concentration_data in enumerate(X):
        print(f"Concentration {i}: shape = {concentration_data.shape}")

    return X


def plot_performance(history, fold, figure_directory=None):
    xlabel = "Epoch"
    legends = ["Training Accuracy", "Validation Accuracy"]

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"Fold {fold} - Model Accuracy", fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.legend(legends, loc="upper left")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"Fold {fold} - Model Loss", fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.legend(legends, loc="upper left")
    plt.grid()

    plt.tight_layout()
    if figure_directory:
        plt.savefig(f"{figure_directory}/fold_{fold}_performance.png")
    plt.show()

def build_model(input_shape, num_classes):
    """
    Improved GRU model with bidirectional GRU layers, optimized Dropout and learning rate scheduler.
    """
    model = tf.keras.Sequential()

    model.add(Bidirectional(GRU(units=50, return_sequences=True, input_shape=input_shape)))
    model.add(Dropout(0.25))

    model.add(Bidirectional(GRU(units=50, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(GRU(units=50, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(GRU(units=100))
    model.add(Dropout(0.2))

    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(units=16, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(units=num_classes, activation="softmax"))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model

GAT_model = GAT(
    nfeat=2,
    nhid=args.hidden,
    output_dim=16,
    dropout=args.dropout,
    heads=8
)
GAT_optimizer = optim.Adam(GAT_model.parameters(), lr=args.lr)

def main():
    start_time = time.time()
    path='C:/Users/QAU_n/Desktop/杨凤东/data'
    path1 = 'C:/Users/QAU_n/Desktop/杨凤东/data42'
    graphs_dict = load_data(path)

    gat_outputs_list = GAT_train(graphs_dict, GAT_model, GAT_optimizer, args)


    print("GCN training completed, start GRU training ..")

    # 处理数据
    p_X = process_gcn_output(gat_outputs_list)
    print(f"Shape after X processing: {[x.shape for x in p_X]}")
    X_init, sample_counts = extract_samples_from_concentration(p_X)
    print(sample_counts)
    print(X_init.shape)

    # 生成标签Q
    y = []
    for i, count in enumerate(sample_counts):
        label = np.full((count,), i)
        y.append(label)
    y = np.concatenate(y)

    #Convert to one hot encoding
    Y = to_categorical(y, num_classes=len(sample_counts))

    print(f"y 的形状: {Y.shape}")

    X = X_init
    print(f"划分前的 X 形状: {X.shape}")
    print(f"划分前 Y 的形状: {Y.shape}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_labels = np.argmax(Y, axis=1)

    fold = 1
    classification_reports = []
    all_y_true = []
    all_y_pred = []
    histories = []

    for train_index, test_index in skf.split(X, y_labels):
        print(f"Start the {fold} fold training ..")

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        print(f"The shape of the training set X in the {fold} fold: {X_train.shape}, Shape of training set Y: {Y_train.shape}")
        print(f"The shape of test set X in the {fold} fold:{X_test.shape}, Shape of Test Set Y: {Y_test.shape}")

        #Build a model
        input_shape = X_train.shape[1:]
        num_classes = Y.shape[1]
        model = build_model(input_shape, num_classes)

        #Define callback
        checkpoint = ModelCheckpoint(f"gru_fold_{fold}.h5",
                                     monitor="val_loss",
                                     mode="min",
                                     save_best_only=True,
                                     verbose=1)
        #When the indicator does not show significant improvement within a period of patience epochs, terminate the training prematurely
        earlystop = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=10,
                                  verbose=1,
                                  restore_best_weights=True)
        #This indicator does not improve within the period of patience epochs and will automatically reduce the learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.80,
                                      patience=5,
                                      verbose=1)

        callbacks = [checkpoint,reduce_lr]

        #Training model
        history_fit = model.fit(x=X_train,
                                y=Y_train,
                                batch_size=32,
                                epochs=100,
                                verbose=1,
                                validation_data=(X_test, Y_test),
                                callbacks=callbacks)
        histories.append(history_fit.history)

        plot_performance(history=history_fit, fold=fold)

        predict_y_prob = model.predict(X_test)
        predict_y = np.argmax(predict_y_prob, axis=1)
        true_y = np.argmax(Y_test, axis=1)

        #Calculate classification report
        report = classification_report(true_y, predict_y, output_dict=True)
        classification_reports.append(report)
        print(f"Classification report for the {fold} fold:")
        print(classification_report(true_y, predict_y, digits=5))

        #Calculate the confusion matrix of the current fold
        cm = confusion_matrix(true_y, predict_y)
        print(f"The confusion matrix of the {fold} fold:")
        print(cm)

        all_y_true.extend(true_y)
        all_y_pred.extend(predict_y)

        fold += 1

    #Draw the average accuracy and loss of all folds
    min_epochs = min(len(history['accuracy']) for history in histories)
    truncated_histories = [{k: v[:min_epochs] for k, v in history.items()} for history in histories]

    avg_train_acc = np.mean([history['accuracy'] for history in truncated_histories], axis=0)
    avg_val_acc = np.mean([history['val_accuracy'] for history in truncated_histories], axis=0)
    avg_train_loss = np.mean([history['loss'] for history in truncated_histories], axis=0)
    avg_val_loss = np.mean([history['val_loss'] for history in truncated_histories], axis=0)

    font_path = 'C:/Windows/Fonts/simhei.ttf'
    font_prop = fm.FontProperties(fname=font_path)
    rcParams['font.family'] = font_prop.get_name()
    rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 1, 1)
    plt.plot(avg_train_acc, label="Average Training Accuracy")
    plt.plot(avg_val_acc, label="Average Validation Accuracy")
    plt.title("Average accuracy of five-fold cross-validation", fontsize=17)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.legend(loc="upper left")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(avg_train_loss, label="Average Training Loss")
    plt.plot(avg_val_loss, label="Average Validation Loss")
    plt.title("Average loss of five-fold cross-validation", fontsize=17)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.legend(loc="upper left")
    plt.grid()

    plt.tight_layout()
    plt.show()

    #Final comprehensive classification report
    print("Comprehensive classification report after six fold cross validation:")
    print(classification_report(all_y_true, all_y_pred, digits=5))

    cm = confusion_matrix(all_y_true, all_y_pred)
    print("The confusion matrix after six fold cross validation:")
    print(cm)

    report = classification_report(all_y_true, all_y_pred, output_dict=True)

    #Calculate macro average accuracy, recall, and F1 score
    macro_avg_report = classification_report(all_y_true, all_y_pred, output_dict=True)
    macro_avg_accuracy = accuracy_score(all_y_true, all_y_pred)
    macro_avg_recall = macro_avg_report['macro avg']['recall']
    macro_avg_f1 = macro_avg_report['macro avg']['f1-score']

    print(f"Macro average accuracy: {macro_avg_accuracy:.5f}")
    print(f"Macro average recall rate: {macro_avg_recall:.5f}")
    print(f"Macro average F1 score: {macro_avg_f1:.5f}")

    #Calculate overall accuracy, recall, and F1 score
    overall_accuracy = accuracy_score(all_y_true, all_y_pred)
    overall_recall = recall_score(all_y_true, all_y_pred, average='macro')
    overall_f1 = f1_score(all_y_true, all_y_pred, average='macro')

    print(f"Overall accuracy: {overall_accuracy:.5f}")
    print(f"Overall recall rate: {overall_recall:.5f}")
    print(f"Overall F1 score: {overall_f1:.5f}")

    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time
    print(f"Total runtime: {total_time:. 2f} seconds")  # 打印总运行时间，保留两位小数

if __name__ == "__main__":
    main()

