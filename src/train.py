import argparse
import pickle
import time
import os
import numpy as np
import torch
from rtpt import RTPT
from sklearn.metrics import accuracy_score, recall_score, roc_curve
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from logic_utils import get_lang
from neumann_utils import (generate_captions, get_data_loader, get_model,
                           get_prob, save_images_with_captions,
                           to_plot_images_clevr, to_plot_images_kandinsky)
from tensor_encoder import TensorEncoder
from visualize import plot_reasoning_graph

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int,
                        default=1, help="Batch size in beam search")
    parser.add_argument("--num-objects", type=int, default=3,
                        help="The maximum number of objects in one image")
    parser.add_argument("--dataset")  # , choices=["member"])
    parser.add_argument("--rtpt-name", default="HS")  # , choices=["member"])
    parser.add_argument(
        "--dataset-type", choices=['vilp', 'clevr-hans', 'kandinsky'], help="vilp or kandinsky or clevr")
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument("--no-train", action="store_true",
                        help="Perform prediction without training model")
    parser.add_argument("--small-data", action="store_true",
                        help="Use small training data.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--plot", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--plot-graph", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--t-beam", type=int, default=4,
                        help="Number of rule expantion of clause generation.")
    parser.add_argument("--n-beam", type=int, default=5,
                        help="The size of the beam.")
    parser.add_argument("--n-max", type=int, default=50,
                        help="The maximum number of clauses.")
    parser.add_argument("--program-size", "-m", type=int, default=1,
                        help="The size of the logic program.")
    #parser.add_argument("--n-obj", type=int, default=2, help="The number of objects to be focused.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="The number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="The learning rate.")
    parser.add_argument("--n-data", type=float, default=200,
                        help="The number of data to be used.")
    parser.add_argument("--pre-searched", action="store_true",
                        help="Using pre searched clauses.")
    parser.add_argument("-T", "--infer-step", type=int, default=10,
                        help="The number of steps of forward reasoning.")
    parser.add_argument("--term-depth", type=int, default=3,
                        help="The number of steps of forward reasoning.")
    args = parser.parse_args()
    return args


def predict(NEUMANN, I2F, loader, args, device,  th=None, split='train'):
    predicted_list = []
    target_list = []
    count = 0

    for i, sample in tqdm(enumerate(loader, start=0)):
        # to cuda
        imgs, target_set = map(lambda x: x.to(device), sample)
        target_set = target_set.float()

        # infer and predict the target probability
        V_0 = I2F(imgs)
        V_T = NEUMANN(V_0)
        predicted = get_prob(V_T, NEUMANN, args)
        predicted_list.append(predicted.detach())
        target_list.append(target_set.detach())
        if args.plot:
            if args.dataset_type == 'kandinsky':
                imgs = to_plot_images_kandinsky(imgs.squeeze(1))
            else:
                imgs = to_plot_images_clevr(imgs.squeeze(1))
            captions = generate_captions(
                V_T, NEUMANN.atoms, I2F.pm.e, th=0.3)
            save_images_with_captions(
                imgs, captions, folder='result/{}/'.format(args.dataset_type) + args.dataset + '/' + split + '/', img_id_start=count, dataset=args.dataset)
        count += V_T.size(0)  # batch size

    predicted = torch.cat(predicted_list, dim=0).detach().cpu().numpy()
    target_set = torch.cat(target_list, dim=0).to(
        torch.int64).detach().cpu().numpy()

    if th == None:
        fpr, tpr, thresholds = roc_curve(target_set, predicted, pos_label=1)
        accuracy_scores = []
        print('ths', thresholds)
        for thresh in thresholds:
            accuracy_scores.append(accuracy_score(
                target_set, [m > thresh for m in predicted]))

        accuracies = np.array(accuracy_scores)
        max_accuracy = accuracies.max()
        max_accuracy_threshold = thresholds[accuracies.argmax()]
        rec_score = recall_score(
            target_set,  [m > thresh for m in predicted], average=None)

        print('target_set: ', target_set, target_set.shape)
        print('predicted: ', predicted, predicted.shape)
        print('accuracy: ', max_accuracy)
        print('threshold: ', max_accuracy_threshold)
        print('recall: ', rec_score)

        return max_accuracy, rec_score, max_accuracy_threshold
    else:
        accuracy = accuracy_score(target_set, [m > th for m in predicted])
        rec_score = recall_score(
            target_set,  [m > th for m in predicted], average=None)
        return accuracy, rec_score, th


def train_neumann(args, NEUMANN, I2F, optimizer, train_loader, val_loader, test_loader, device, writer, rtpt):
    bce = torch.nn.BCELoss()
    time_list = []
    iteration = 0
    for epoch in range(args.epochs):
        loss_i = 0
        start_time = time.time()
        for i, sample in tqdm(enumerate(train_loader, start=0)):
            # to cuda
            imgs, target_set = map(lambda x: x.to(device), sample)
            target_set = target_set.float()

            # convert the images to probabilistic facts (facts converting)
            V_0 = I2F(imgs)
            # infer and predict the target probability
            V_T = NEUMANN(V_0)
            # get the probabilities of the target atoms
            predicted = get_prob(V_T, NEUMANN, args)
            loss = bce(predicted, target_set)
            loss_i += loss.item()
            # compute the gradients
            loss.backward()
            # update the weights of clauses
            optimizer.step()

            writer.add_scalar("metric/train_loss", loss, global_step=iteration)
            iteration += 1

        # record time
        epoch_time = time.time() - start_time
        time_list.append(epoch_time)
        writer.add_scalar("metric/epoch_time_mean", np.mean(time_list))
        writer.add_scalar("metric/epoch_time_std", np.std(time_list))
        writer.add_scalar("metric/train_loss_epoch", loss_i, global_step=epoch)

        rtpt.step()#subtitle=f"loss={loss_i:2.2f}")
        print("loss: ", loss_i)

        if epoch % 1 == 0 and epoch > 0:
            NEUMANN.print_valuation_batch(V_T)
            print("Epoch {}: ".format(epoch))
            NEUMANN.print_program()
        """
        if epoch % 20 == 0 and epoch > 0:
            NEUMANN.print_program()
            print("Predicting on validation data set...")
            acc_val, rec_val, th_val = predict(
                NEUMANN, I2F, val_loader, args, device, th=0.7, split='val')
            writer.add_scalar("metric/val_acc", acc_val, global_step=epoch)
            print("acc_val: ", acc_val)

            print("Predicting on training data set...")
            acc, rec, th = predict(
                NEUMANN, I2F, train_loader, args, device, th=th_val, split='train')
            writer.add_scalar("metric/train_acc", acc, global_step=epoch)
            print("acc_train: ", acc)

            print("Predicting on test data set...")
            acc, rec, th = predict(
                NEUMANN, I2F, test_loader, args, device, th=th_val, split='train')
            writer.add_scalar("metric/test_acc", acc, global_step=epoch)
            print("acc_test: ", acc)
        """
    return loss


def main(n):
    args = get_args()
    print('args ', args)
    if args.no_cuda:
        device = torch.device('cpu')
    elif len(args.device.split(',')) > 1:
        # multi gpu
        device = torch.device('cuda')
    else:
        device = torch.device('cuda:' + args.device)

    print('device: ', device)
    name = 'rgnn/' + args.dataset + '/' + str(n)
    writer = SummaryWriter(f"runs/{name}", purge_step=0)

    # Create RTPT object
    rtpt = RTPT(name_initials='HS', experiment_name="NEUMANN_{}".format(args.dataset),
                max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()

    # Get torch data loader
    train_loader, val_loader,  test_loader = get_data_loader(args, device)

    # Load logical representations
    lark_path = 'src/lark/exp.lark'
    lang_base_path = 'data/lang/'
    lang, clauses, bk, bk_clauses, terms, atoms = get_lang(
        lark_path, lang_base_path, args.dataset_type, args.dataset, args.term_depth)
    print("{} Atoms:".format(len(atoms)))

    # Load the NEUMANN model
    NEUMANN, I2F = get_model(lang=lang, clauses=clauses, atoms=atoms, terms=terms, bk=bk, bk_clauses=bk_clauses,
                             program_size=args.program_size, device=device, dataset=args.dataset, dataset_type=args.dataset_type,
                             num_objects=args.num_objects, infer_step=args.infer_step, train=not(args.no_train))

    writer.add_scalar("graph/num_atom_nodes", len(NEUMANN.rgm.atom_node_idxs))
    writer.add_scalar("graph/num_conj_nodes", len(NEUMANN.rgm.conj_node_idxs))
    num_nodes = len(NEUMANN.rgm.atom_node_idxs) + \
        len(NEUMANN.rgm.conj_node_idxs)
    writer.add_scalar("graph/num_nodes", num_nodes)

    num_edges = NEUMANN.rgm.edge_index.size(1)
    writer.add_scalar("graph/num_edges", num_edges)

    writer.add_scalar("graph/memory_total", num_nodes + num_edges)

    print("NUM NODES: ", num_nodes)
    print("NUM EDGES: ", num_edges)
    print("MEMORY TOTAL: ", num_nodes + num_edges)
    # save the reasoning graph

    if args.plot_graph:
        print("Plotting reasoning graph...")
        base_path = 'plot/reasoning_graph/'
        os.makedirs(base_path, exist_ok=True)
        path = base_path + "{}_{}_rg.png".format(args.dataset_type, args.dataset)
        plot_reasoning_graph(path, NEUMANN.rgm)

    # CHECK memory of tensors
    """Check memoru of tensors.
    print('check tensors.. ')
    te = TensorEncoder(lang, atoms, clauses+bk_clauses, device, NEUMANN.rgm)
    I = te.encode()
    print(I)
    print(I.size())
    print("TENSOR MEMORY: ", I.size(0) * I.size(1) * I.size(2) * I.size(3))
    I = I.expand(args.batch_size, -1, -1, -1, -1)
    print("TENSOR MEMORY BATCH: ", I.size(0) * I.size(1) * I.size(2) * I.size(3) * I.size(4))
    """
    params = list(NEUMANN.parameters())
    print('parameters: ', list(params))

    if not args.no_train:
        optimizer = torch.optim.RMSprop(params, lr=args.lr)
        loss_list = train_neumann(args, NEUMANN, I2F, optimizer, train_loader,
                                  val_loader, test_loader, device, writer, rtpt)

    # validation split
    print("Predicting on validation data set...")
    acc_val, rec_val, th_val = predict(
        NEUMANN, I2F, val_loader, args, device, th=0.7, split='val')

    print("Predicting on training data set...")
    # training split
    acc, rec, th = predict(
        NEUMANN, I2F, train_loader, args, device, th=th_val, split='train')

    print("Predicting on test data set...")
    # test split
    acc_test, rec_test, th_test = predict(
        NEUMANN, I2F, test_loader, args, device, th=th_val, split='test')

    print("training acc: ", acc, "threashold: ", th, "recall: ", rec)
    print("val acc: ", acc_val, "threashold: ", th_val, "recall: ", rec_val)
    print("test acc: ", acc_test, "threashold: ", th_test, "recall: ", rec_test)


if __name__ == "__main__":
    for i in range(5):
        main(n=i)
