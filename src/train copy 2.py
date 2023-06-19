import argparse
import math
import os
import pickle
import random
import time

import numpy as np
import torch
from rtpt import RTPT
from sklearn.metrics import accuracy_score, recall_score, roc_curve
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb
from clause_generator import ClauseGenerator
from logic_utils import get_lang
from mode_declaration import get_mode_declarations_vilp
from neumann_utils import (generate_captions, get_data_loader, get_model,
                           get_prob, save_images_with_captions,
                           to_plot_images_clevr, to_plot_images_kandinsky,
                           update_by_clauses, update_by_refinement)
from refinement import RefinementGenerator
from tensor_encoder import TensorEncoder
from visualize import plot_reasoning_graph


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch-size", type=int, default=4,
                        help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int,
                        default=1, help="Batch size in beam search")
    parser.add_argument("--num-objects", type=int, default=3,
                        help="The maximum number of objects in one image")
    parser.add_argument("-ds","--dataset")  # , choices=["member"])
    parser.add_argument("--rtpt-name", default="HS")  # , choices=["member"])
    parser.add_argument(
       "-dt", "--dataset-type", choices=['vilp', 'clevr-hans', 'kandinsky'], help="vilp or kandinsky or clevr")
    parser.add_argument("-d", "--device", default='cpu',
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
    parser.add_argument("-pg", "--plot-graph", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--t-beam", type=int, default=4,
                        help="Number of rule expantion of clause generation.")
    parser.add_argument("--n-beam", type=int, default=5,
                        help="The size of the beam.")
    parser.add_argument("--n-max", type=int, default=50,
                        help="The maximum number of clauses.")
    parser.add_argument("-ps", "--program-size", type=int, default=1,
                        help="The size of the logic program.")
    #parser.add_argument("--n-obj", type=int, default=2, help="The number of objects to be focused.")
    parser.add_argument("-e", "--epochs", type=int, default=100,
                        help="The number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="The learning rate.")
    parser.add_argument("--n-data", type=float, default=200,
                        help="The number of data to be used.")
    parser.add_argument("--pre-searched", action="store_true",
                        help="Using pre searched clauses.")
    parser.add_argument("-is", "--infer-step", type=int, default=8,
                        help="The number of steps of forward reasoning.")
    parser.add_argument("-td", "--term-depth", type=int, default=3,
                        help="The max depth of terms to be generated.")
    parser.add_argument("-pd", "--program-depth", type=int, default=3,
                        help="The max depth of terms to in the clauses to be generated.")
    parser.add_argument("-bl", "--body-len", type=int, default=2,
                        help="The len of body of clauses to be generated.")
    parser.add_argument("-tr","--trial", type=int, default=2,
                        help="The number of trials to generate clauses before the final training.")
    parser.add_argument("-thd", "--th-depth", type=int, default=2,
                        help="The depth to specify the clauses to be pruned after generation.")
    parser.add_argument("-ns", "--n-sample", type=int, default=5,
                        help="The number of samples on each step of clause generation..")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="Random seed.")
    #  max_depth=1, max_body_len=1, max_var_num=5
    parser.add_argument("-md", "--max-depth", type=int, default=1, help="Max depth of terms.")
    parser.add_argument("-ml", "--max-body-len", type=int, default=1, help="Max length of the body.")
    parser.add_argument("-mv", "--max-var", type=int, default=4, help="Max number of variables.")
    parser.add_argument("-te", "--trial-epochs", type=int, default=5, help="The number of epochs in trials.")
    parser.add_argument("-pr", "--pos-ratio", type=float, default=0.1, help="The ratio of the positive examples in the final training.")
    parser.add_argument("-nr", "--neg-ratio", type=float, default=1.0, help="The ratio of the negative examples in the final training.")
    parser.add_argument("--n-ratio", type=float, default=1.0,
                        help="The ratio of data to be used.")
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


def train_neumann(args, NEUMANN, I2F, optimizer, train_loader, val_loader, test_loader, device, writer, rtpt, epochs, trial):
    bce = torch.nn.BCELoss()
    time_list = []
    iteration = 0
    clause_scores = torch.zeros(len(NEUMANN.clauses, )).to(device)
    for epoch in range(epochs):
        loss_i = 0
        start_time = time.time()
        grad_sum = torch.zeros(args.program_size, (len(NEUMANN.clauses)), device=device)
        for i, sample in tqdm(enumerate(train_loader, start=0)):
            optimizer.zero_grad()
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
            #print(np.round(NEUMANN.clause_weights.grad.T.detach().cpu().numpy(), 2))
            #print(NEUMANN.clause_weights.grad.detach().shape)
            #print(NEUMANN.clause_weights.grad.detach().sum(dim=0).shape)
            grad_sum += NEUMANN.clause_weights.grad.detach()
            #print(grad_sum)
            # update the weights of clauses
            optimizer.step()

            #writer.add_scalar("metric/train_loss", loss, global_step=iteration)
            wandb.log({'metric/training_loss_trial_{}'.format(trial): loss.item()})
            iteration += 1

        clause_scores_grad, indices = grad_sum.min(dim=0)
        clause_scores_i = clause_scores_grad * (-1) / len(train_loader)
        clause_scores += clause_scores_i
        #selected_clause_indices = torch.stack([F.gumbel_softmax(clause_scores, tau=0.1, hard=True) for i in range(10)])
        #selected_clause_indices, _ = torch.max(selected_clause_indices, dim=0)


        #selected_clauses = []
        #for ci in selected_clause_indices.detach().cpu().numpy():
        #    if ci > 0:
        #        selected_clauses.append(NEUMANN.clauses[int(ci)])
        #print("SELECTED CLAUSES: ", list(set(selected_clauses)))

        #print(selected_clause_indices)

        #print("clause scores: ", clause_scores)
        # record time
        epoch_time = time.time() - start_time
        time_list.append(epoch_time)
        #writer.add_scalar("metric/epoch_time_mean", np.mean(time_list))
        #writer.add_scalar("metric/epoch_time_std", np.std(time_list))
        #writer.add_scalar("metric/train_loss_epoch", loss_i, global_step=epoch)
        wandb.log({'metric/training_loss_epoch': loss_i})

        rtpt.step()#subtitle=f"loss={loss_i:2.2f}")
        print("loss: ", loss_i)

        # NEUMANN.print_valuation_batch(V_T)
        #if epoch % 1 == 0 and epoch > 0:
        #    NEUMANN.print_valuation_batch(V_T)
        #    print("Epoch {}: ".format(epoch))
        #    NEUMANN.print_program()
        if (epoch > 0 and epoch % 5 == 0) or (trial > args.trial and epoch % 5 == 0):
            NEUMANN.print_program()
            print("Predicting on validation data set...")
            acc_val, rec_val, th_val = predict(
                NEUMANN, I2F, val_loader, args, device, th=0.5, split='val')
            wandb.log({'metric/validation_accuracy': acc_val})
            # writer.add_scalar("metric/val_acc", acc_val, global_step=epoch)
            #print("acc_val: ", acc_val)
            #if acc_val > 0.95:
            #    break
        """
        if epoch % 20 == 0 and epoch > 0:

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
    NEUMANN.print_program()
    return loss, NEUMANN.clause_weights.detach() #clause_scores / epochs


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

    #seed = n
    seed = args.seed
    seed_everything(seed)

    # start weight and biases
    wandb.init(project="NEUMANN", name="{}:seed_{}".format(args.dataset, args.seed))


    # Create RTPT object
    rtpt = RTPT(name_initials='HS', experiment_name="NEUMANN_{}".format(args.dataset),
                max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()


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
    # clause generator
    refinement_generator = RefinementGenerator(lang=lang, mode_declarations=get_mode_declarations_vilp(lang, args.dataset), max_depth=args.program_depth, max_body_len=args.max_body_len, max_var_num=args.max_var)
    clause_generator = ClauseGenerator(refinement_generator=refinement_generator, root_clauses=clauses, th_depth=args.th_depth, n_sample=args.n_sample)
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

    trial = 0
    stop_flag = False
    params = list(NEUMANN.parameters())
    optimizer = torch.optim.RMSprop(params, lr=args.lr)
    clause_scores = torch.ones((args.program_size, len(NEUMANN.clauses, ))).to(device)

    #too_simple_clauses = clauses
    times = []
    while trial < args.trial:
        print("=====TRIAL: {}====".format(trial))

        epochs = args.trial_epochs
        pos_ratio = 1.0
        neg_ratio =  min(0.2*(trial+1), 1.0)
        # neg_ratio = 0.1
        softmax_temp = 1.0
        lr = 1e-2
        print('lr={}'.format(lr))

        print(NEUMANN.clauses)
        # Get torch data loader
        train_loader, val_loader,  test_loader = get_data_loader(args, device, pos_ratio, neg_ratio)
        start = time.time()
        NEUMANN, new_gen_clauses = update_by_refinement(NEUMANN, clause_scores, clause_generator, softmax_temp=softmax_temp)
        # clause_generator.print_tree()
        params = list(NEUMANN.parameters())
        optimizer = torch.optim.RMSprop(params, lr=lr)
        # optimizer = torch.optim.SGD(params, lr=lr)
        optimizer.zero_grad()
        loss_list, clause_scores = train_neumann(args, NEUMANN, I2F, optimizer, train_loader,
                                  val_loader, test_loader, device, writer, rtpt, epochs=epochs, trial=trial)
        times.append(time.time() - start)
        trial += 1
        
    
    with open('out/learning_time_neumann_{}_{}.txt'.format(args.dataset, args.seed), 'w') as f:
        f.write("\n".join(str(item) for item in times))
    epochs = args.epochs
    # for delete:
    # pos_ratio = 0.1
    # for sort:
    pos_ratio = args.pos_ratio
    neg_ratio = args.neg_ratio
    softmax_temp = 1e-2
    lr =  1e-2#  * pow(0.1, min(trial, 3))
    print("==== Generated Refinement Tree ====")
    clause_generator.print_tree()
    generated_clauses = clause_generator.get_clauses_by_th_depth(args.th_depth)
    print("==== Extracted Clauses ====")
    for c in generated_clauses:
        print(c)
    NEUMANN = update_by_clauses(NEUMANN, generated_clauses)
    # Get torch data loader
    train_loader, val_loader,  test_loader = get_data_loader(args, device, pos_ratio, neg_ratio)
    params = list(NEUMANN.parameters())
    optimizer = torch.optim.RMSprop(params, lr=lr)
    # optimizer = torch.optim.SGD(params, lr=lr)
    optimizer.zero_grad()
    loss_list, clause_scores = train_neumann(args, NEUMANN, I2F, optimizer, train_loader,
                          val_loader, test_loader, device, writer, rtpt, epochs=epochs, trial=trial)
    
    wandb.finish()
    NEUMANN.print_program()
    # validation split
    print("Predicting on validation data set...")
    acc_val, rec_val, th_val = predict(
        NEUMANN, I2F, val_loader, args, device, th=0.5, split='val')
    print("Trial {}, acc_val: {}".format(trial, acc_val))
    if args.plot_graph:
        print("Plotting reasoning graph...")
        base_path = 'plot/reasoning_graph/'
        os.makedirs(base_path, exist_ok=True)
        path = base_path + "rg_{}_{}_trial_{}.png".format(args.dataset_type, args.dataset, trial)
        plot_reasoning_graph(path, NEUMANN.rgm)
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


def seed_everything(seed=42):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main(n=1)
