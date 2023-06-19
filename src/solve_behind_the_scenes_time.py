import argparse
import time

import numpy as np
import torch
from rtpt import RTPT
from sklearn.metrics import accuracy_score, recall_score, roc_curve
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from logic_utils import get_lang
from neumann_utils import get_behind_the_scenes_loader, get_model, get_prob


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size to infer with")
    parser.add_argument("--num-objects", type=int, default=3,
                        help="The maximum number of objects in one image")
    parser.add_argument("--dataset", default="delete")  # , choices=["member"])
    parser.add_argument("--dataset-type", default="behind-the-scenes")
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--plot", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--n-data", type=int, default=10000,
                        help="The number of data to be used.")
    parser.add_argument("--infer-step", type=int, default=6,
                        help="The number of steps of forward reasoning.")
    parser.add_argument("--term-depth", type=int, default=3,
                        help="The number of steps of forward reasoning.")
    parser.add_argument("--question-json-path", default="data/behind-the-scenes/BehindTheScenes_questions.json")
    args = parser.parse_args()
    return args



def predict(NEUMANN, I2F, loader, args, device,  th=None, split='train'):
    predicted_list = []
    target_list = []
    count = 0

    start = time.time()
    for i, sample in tqdm(enumerate(loader, start=0)):
        imgs, query, target_set = map(lambda x: x.to(device), sample)
        
        # to cuda
        target_set = target_set.float()

        V_0 = I2F(imgs, query)
        V_T = NEUMANN(V_0)
        predicted = get_prob(V_T, NEUMANN, args)
        predicted = to_one_label(predicted, target_set)
        predicted = torch.softmax(predicted * 10, dim=1)
        predicted_list.append(predicted.detach())
        target_list.append(target_set.detach())
        """
        if args.plot:
            imgs = to_plot_images_clevr(imgs.squeeze(1))
            captions = generate_captions(
                V_T, NEUMANN.atoms, I2F.pm.e, th=0.3)
            save_images_with_captions(
                imgs, captions, folder='result/kandinsky/' + args.dataset + '/' + split + '/', img_id_start=count, dataset=args.dataset)
        """
        count += V_T.size(0)  # batch size
    reasoning_time = time.time() - start

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

        return max_accuracy, rec_score, max_accuracy_threshold, reasoning_time
    else:
        accuracy = accuracy_score(target_set, [m > th for m in predicted])
        rec_score = recall_score(
            target_set,  [m > th for m in predicted], average=None)
        return accuracy, rec_score, th, reasoning_time


def to_one_label(ys, labels, th=0.7):
    ys_new = []
    for i in range(len(ys)):
        y = ys[i]
        label = labels[i]
        # check in case answers are computed
        num_class = 0
        for p_j in y:
            if p_j > th:
                num_class += 1
        if num_class >= 2:
            # drop the value using label (the label is one-hot)
            drop_index = torch.argmin(label - y)
            y[drop_index] = y.min()
        ys_new.append(y)
    return torch.stack(ys_new)


def main(n):
    args = get_args()
    #name = 'VILP'
    print('args ', args)
    if args.no_cuda:
        device = torch.device('cpu')
    elif len(args.device.split(',')) > 1:
        # multi gpu
        device = torch.device('cuda')
    else:
        device = torch.device('cuda:' + args.device)

    print('device: ', device)
    name = 'neumann/behind-the-scenes/' + str(n)
    writer = SummaryWriter(f"runs/{name}", purge_step=0)

    # Create RTPT object
    rtpt = RTPT(name_initials='HS', experiment_name=name,
                max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()

    # load logical representations
    lark_path = 'src/lark/exp.lark'
    lang_base_path = 'data/lang/'
    lang, clauses, bk, bk_clauses, terms, atoms = get_lang(
        lark_path, lang_base_path, args.dataset_type, args.dataset, args.term_depth)

    print("{} Atoms:".format(len(atoms)))

    # get torch data loader
    question_json_path = 'data/behind-the-scenes/BehindTheScenes_questions_{}.json'.format(args.dataset)
    test_loader = get_behind_the_scenes_loader(question_json_path, args.batch_size, lang, args.n_data, device)

    NEUMANN, I2F = get_model(lang=lang, clauses=clauses, atoms=atoms, terms=terms, bk=bk, bk_clauses=bk_clauses,
                          program_size=args.program_size, device=device, dataset=args.dataset, dataset_type=args.dataset_type,
                          num_objects=args.num_objects, infer_step=args.infer_step, train=False)#train=not(args.no_train))

    writer.add_scalar("graph/num_atom_nodes", len(NEUMANN.rgm.atom_node_idxs))
    writer.add_scalar("graph/num_conj_nodes", len(NEUMANN.rgm.conj_node_idxs))
    num_nodes = len(NEUMANN.rgm.atom_node_idxs) + len(NEUMANN.rgm.conj_node_idxs)
    writer.add_scalar("graph/num_nodes", num_nodes)

    num_edges = NEUMANN.rgm.edge_index.size(1)
    writer.add_scalar("graph/num_edges", num_edges)

    writer.add_scalar("graph/memory_total", num_nodes + num_edges)

    print("=====================")
    print("NUM NODES: ", num_nodes)
    print("NUM EDGES: ", num_edges)
    print("MEMORY TOTAL: ", num_nodes + num_edges)
    print("=====================")

    params = list(NEUMANN.parameters())
    print('parameters: ', list(params))

    print("Predicting on test data set...")
    times = []
    # test split
    for j in range(n):
        acc_test, rec_test, th_test, time = predict(
            NEUMANN, I2F, test_loader, args, device, th=0.5, split='test')
        times.append(time)
    
    with open('out/time_{}_{}.txt'.format(args.dataset, args.n_data), 'w') as f:
        f.write("\n".join(str(item) for item in times))

    print("test acc: ", acc_test, "threashold: ", th_test, "recall: ", rec_test)

if __name__ == "__main__":
    main(n=1)

