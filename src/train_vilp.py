import argparse
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from rtpt import RTPT
from sklearn.metrics import (accuracy_score, average_precision_score,
                             recall_score)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from fact_enumerator import FactEnumerator
# from lib.nsfr.src.nsfr_utils import denormalize_kandinsky, get_data_loader, get_prob, get_nsfr_model
from logic_utils import get_index_by_predname, get_lang
# from nsfr_utils import get_data_loader
from neumann_utils import get_data_loader, get_model
from neural_utils import MLP
# import sys
# sys.path.append('/Users/shindo/Workspace/rgnn/src/lib/nsfr/src')
# from nsfr_utils import get_nsfr_model, get_img2facts
from percept import YOLOPerceptionModule
from visualize import plot_atoms, plot_infer_embeddings

torch.autograd.set_detect_anomaly(True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size to train with"
    )
    parser.add_argument(
        "--num-images", type=int, default=1, help="Number of images per instance."
    )
    parser.add_argument(
        "--dataset",
        choices=["member", "delete", "append", "reverse", "sort"]
    )
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of threads for data loader")
    parser.add_argument("--num-objects", type=int, default=3,
                        help="The maximum number of objects in one image")
    parser.add_argument(
        "-m", "--program_size", type=int, default=3, help="The size of the logic program"
    )
    parser.add_argument("--dataset-type", default="vilp")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument(
        "-td", "--term-depth", type=int, default=3, help="Max term depth.")
    args = parser.parse_args()
    return args


def predict(v, atoms, predname):
    target_index = get_index_by_predname(
        pred_str=predname, atoms=atoms)
    return v[:, target_index]


def predict_multi(v, atoms, prednames):
    """Extracting values from the valuation tensor using given predicates.

    prednames = ['kp1', 'kp2', 'kp3']
    """
    # v: batch * |atoms|
    target_indices = []
    for predname in prednames:
        target_index = get_index_by_predname(
            pred_str=predname, atoms=atoms)
        target_indices.append(target_index)
    prob = torch.cat([v[:, i].unsqueeze(-1)
                      for i in target_indices], dim=1)
    B = v.size(0)
    N = len(prednames)
    assert prob.size(0) == B and prob.size(
        1) == N, 'Invalid shape in the prediction.'
    return prob


def compute_acc(outputs, targets):
    predicts = np.argmax(outputs, axis=1)
    return accuracy_score(targets, predicts)


def run(GNNR, IMG2FACT, atoms, loader, optimizer, criterion, writer, args, device, train=False, epoch=0, rtpt=None,
        max_obj_num=4):
    iters_per_epoch = len(loader)

    be = torch.nn.BCELoss()
    # be = torch.nn.BCEWithLogitsLoss()

    loss_list = []
    predicted_list = []
    target_list = []

    acc_list = []
    ap_list = []

    for i, (imgs, labels) in tqdm(enumerate(loader, start=epoch * iters_per_epoch)):
        ##print(imgs)
        ##assert imgs.size(0) == 0, "Invalid size of tensors." + str(imgs.size())
        # to cuda
        ##imgs, labels = samples
        imgs.to(device)
        labels.to(device)
        # reset grad
        if train:
            optimizer.zero_grad()
        # infer and predict the target probability
        # yolo net to predict each object
        facts = IMG2FACT(imgs)
        # y_nsfr = NSFR(facts)[:, 2:] # remove true/false
        y_gnnr = GNNR(facts)

        ##y_nsfr = predict_multi(y_nsfr, prednames=['kp1','kp2','kp3'], atoms=atoms)
        # y_gnnr = F.softmax(100*predict_multi(y_gnnr, prednames=['kp1','kp2','kp3'], atoms=atoms), dim=1)

        print("--")
        # print('y_nsfr: ', y_nsfr.detach())
        print('y_gnnr: ', y_gnnr.detach())

        # binary cross-entropy loss computation
        loss = be(y_gnnr, labels)
        loss_list.append(loss.item())
        if train:
            loss.backward()
            if optimizer != None:
                optimizer.step()
        # update parameters for the step
        # if optimizer != None and epoch > 0:

        # print('const embeddings: ', GNNR.nfm.const_embeddings)

        if train:
            writer.add_scalar("metric/train_loss", loss.item(), global_step=i)
            print('train loss: ', loss.item())

    # plot_infer_embeddings(GNNR.gnn.x_atom_list, atoms[2:])

    # print('y_gnnr: ', np.round(y_gnnr.detach().cpu().numpy(), 2))
    # print('y_nsfr: ', np.round(y_nsfr.detach().cpu().numpy(), 2))
    binary_predicted_list = np.where(np.array(predicted_list) > 0.5, 1, 0)
    # acc = accuracy_score(target_list, binary_predicted_list)
    # ap = average_precision_score(target_list, binary_predicted_list)
    # ap = 0

    if rtpt != None:
        rtpt.step(subtitle=f"loss={loss.item():2.2f}")

    return loss_list  # , acc, ap


def main():
    args = get_args()
    # device = select_device(args.device, args.batch_size)
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + args.device)

    start_epoch = 0
    # load logical representations
    lark_path = 'src/lark/exp.lark'
    lang_base_path = 'data/lang/'
    lang, clauses, bk, bk_clauses, terms, atoms = get_lang(
        lark_path, lang_base_path, args.dataset_type, args.dataset, args.term_depth)

    # FE = FactEnumerator()
    # atoms = FE.enumerate_facts(B=bk, C=clauses, infer_step=5)

    # print(atoms)
    # Neuro-Symbolic Forward Reasoner
    # NSFR = get_nsfr_model(
    #    args, lang, clauses, atoms, bk, device)
    # get torch data loader
    train_loader, val_loader, test_loader = get_data_loader(args, device)

    # GNN Reasoner
    neumann, img2facts = get_model(lang, clauses, atoms, terms, bk, bk_clauses, args.program_size, device, args.dataset, args.dataset_type, args.num_objects, infer_step=5, train=True)
    # setting optimizer
    params = list(neumann.parameters())
    # print(len(params), "parameters", params)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = torch.nn.SmoothL1Loss()

    # num of the runs
    N = 1

    # Create RTPT object
    rtpt = RTPT(name_initials='HS',
                experiment_name='NEUMANN_' + args.dataset, max_iterations=N * args.epochs)
    rtpt.start()

    # accs
    acc_list = []
    acc_val_list = []
    acc_test_list = []

    # aps
    ap_list = []
    ap_val_list = []
    ap_test_list = []
    for n in range(N):
        print('=== Trial ' + str(n) + ' ===')

        name = 'GNNReasoner' + args.dataset + '_' + str(n)
        writer = SummaryWriter(f"runs/{name}", purge_step=0)
        # loss
        loss_list_all = []
        loss_list_val_all = []
        loss_list_test_all = []

        for epoch in np.arange(start_epoch, args.epochs + start_epoch):
            # training step
            loss_list = run(
                neumann, img2facts, atoms, train_loader, optimizer, criterion, writer, args, device=device, train=False,
                epoch=epoch, rtpt=rtpt)
            # writer.add_scalar("metric/train_acc", acc, global_step=epoch)
            # writer.add_scalar("metric/train_ap", ap, global_step=epoch)

            cur_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar(
                "lr", cur_lr, global_step=epoch * len(train_loader))
            # scheduler.step()

            """
            # validation step
            loss_list_val, acc_val, ap_val = run(
                GNNR, NSFR, val_loader, None, criterion, writer, args, device=device, train=False, epoch=epoch, rtpt=rtpt)
            writer.add_scalar("metric/val_acc", acc_val, global_step=epoch)
            writer.add_scalar("metric/val_ap", ap_val, global_step=epoch)

            # test step
            loss_list_test, acc_test, ap_test = run(
                GNNR, NSFR, test_loader, None, criterion, writer, args, device=device, train=False, epoch=epoch, rtpt=rtpt)
            writer.add_scalar("metric/test_acc", acc_test, global_step=epoch)
            writer.add_scalar("metric/test_ap", ap_test, global_step=epoch)

            # extend loss history
            loss_list_all.extend(loss_list)
            loss_list_val_all.extend(loss_list_val)
            loss_list_test_all.extend(loss_list_test)
            """

        # NSFR.print_program()
        # append acc and ap history
        # acc_list.append(acc)
        # acc_val_list.append(acc_val)
        # acc_test_list.append(acc_test)
        # ap_list.append(ap)
        # ap_val_list.append(ap_val)
        # ap_test_list.append(ap_test)
    """
        # save loss
        picklename = "GNNReasoner_" + args.dataset + \
            "_train_loss" + str(n) + ".pickle"
        with open('runs/' + picklename, 'wb') as f:
            pickle.dump(loss_list_all, f)

        picklename = "GNNReasoner_" + args.dataset + \
            "_val_loss" + str(n) + ".pickle"
        with open('runs/' + picklename, 'wb') as f:
            pickle.dump(loss_list_val_all, f)

        picklename = "GNNReasoner_" + args.dataset + \
            "_test_loss" + str(n) + ".pickle"
        with open('runs/' + picklename, 'wb') as f:
            pickle.dump(loss_list_test_all, f)

    # save accs
    f = open("runs/GNNReasoner_" + args.dataset + "_acc.txt", "w")
    f.write("--train--\n")
    f.write(str(acc_list) + "\n")
    f.write("mean: " + str(np.mean(np.array(acc_list))))
    f.write("\n")
    f.write("std: " + str(np.std(np.array(acc_list))))
    f.write("\n")
    f.write("--val--\n")
    f.write(str(acc_val_list) + "\n")
    f.write("mean: " + str(np.mean(np.array(acc_val_list))))
    f.write("\n")
    f.write("std: " + str(np.std(np.array(acc_val_list))))
    f.write("\n")
    f.write("--test--\n")
    f.write(str(acc_test_list) + "\n")
    f.write("mean: " + str(np.mean(np.array(acc_test_list))))
    f.write("\n")
    f.write("std: " + str(np.std(np.array(acc_test_list))))
    f.close()

    # save APs
    f = open("runs/GNNReasoner_" + args.dataset + "_AP.txt", "w")
    f.write("--train--\n")
    f.write(str(ap_list) + "\n")
    f.write("mean: " + str(np.mean(np.array(ap_list))))
    f.write("\n")
    f.write("std: " + str(np.std(np.array(ap_list))))
    f.write("\n")
    f.write("--val--\n")
    f.write(str(ap_val_list) + "\n")
    f.write("mean: " + str(np.mean(np.array(ap_val_list))))
    f.write("\n")
    f.write("std: " + str(np.std(np.array(ap_val_list))))
    f.write("\n")
    f.write("--test--\n")
    f.write(str(ap_test_list) + "\n")
    f.write("mean: " + str(np.mean(np.array(ap_test_list))))
    f.write("\n")
    f.write("std: " + str(np.std(np.array(ap_test_list))))
    f.write("\n")
    f.close()  
    """


if __name__ == "__main__":
    main()
