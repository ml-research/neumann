import argparse
import pickle
import sys
import time

import clingo
import numpy as np
import torch
from clingo.control import Control
from rtpt import RTPT
from sklearn.metrics import accuracy_score, recall_score, roc_curve
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ilasp_utils import (get_ilasp_background_knowledge,
                         get_ilasp_mode_declarations)
from logic_utils import get_lang
from mode_declaration import get_mode_declarations
from neumann_utils import (get_clause_evaluator, get_data_loader, get_model,
                           get_prob)
from tensor_encoder import TensorEncoder

sys.path.append('src/FFNSL')
sys.path.append('FFNSL/nsl')
sys.path.append('FFNSL/examples/follow_suit/')
from nsl.ilasp import ILASPSession, ILASPSystem
from nsl.utils import add_cmd_line_args, calc_example_penalty

# from nsfr_utils import save_images_with_captions, to_plot_images_clevr, generate_captions

torch.set_num_threads(10)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int,
                        default=1, help="Batch size in beam search")
    parser.add_argument("--num-objects", type=int, default=6,
                        help="The maximum number of objects in one image")
    parser.add_argument("--dataset", default="delete")  # , choices=["member"])
    parser.add_argument("--dataset-type", default="behind-the-scenes")
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument("--no-train", action="store_true",
                        help="Perform prediction without training model")
    parser.add_argument("--small-data", action="store_true",
                        help="Use small training data.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--plot", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--t-beam", type=int, default=4,
                        help="Number of rule expantion of clause generation.")
    parser.add_argument("--n-beam", type=int, default=5,
                        help="The size of the beam.")
    parser.add_argument("--n-max", type=int, default=50,
                        help="The maximum number of clauses.")
    parser.add_argument("--program-size", type=int, default=1,
                        help="The size of the logic program.")
    #parser.add_argument("--n-obj", type=int, default=2, help="The number of objects to be focused.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="The number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="The learning rate.")
    parser.add_argument("--n-ratio", type=float, default=1.0,
                        help="The ratio of data to be used.")
    parser.add_argument("--pre-searched", action="store_true",
                        help="Using pre searched clauses.")
    parser.add_argument("--infer-step", type=int, default=6,
                        help="The number of steps of forward reasoning.")
    parser.add_argument("--term-depth", type=int, default=3,
                        help="The number of steps of forward reasoning.")
    parser.add_argument("--question-json-path", default="data/behind-the-scenes/BehindTheScenes_questions.json")
    args = parser.parse_args()
    return args

# def get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device, train=False):


def discretise_NEUMANN(NEUMANN, args, device):
    lark_path = 'src/lark/exp.lark'
    lang_base_path = 'data/lang/'
    lang, clauses_, bk, terms, atoms = get_lang(
        lark_path, lang_base_path, args.dataset_type, args.dataset, args.term_depth)
    # Discretise NEUMANN rules
    clauses = NEUMANN.get_clauses()
    return get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device, train=False)

def valuation_to_ILP_example_batch(V_0, atoms, labels, example_ids, th=0.5):
    examples = []
    
    target_atom = [atom for atom in atoms if str(atom) == 'kp(img)']
    assert len(target_atom) == 1, "Too many wrong atoms!"
    target_atom = target_atom[0]

    for i, v in enumerate(V_0):
        label = labels[i]
        true_atoms = []
        for j in range(len(v)):
            if v[j] > th:
                true_atoms.append(atoms[j])
        if labels[i] == 1.0:
            true_atoms.append(target_atom)
        examples.append((true_atoms, label, example_ids[i]))


    examples = atoms_to_ILASP_text(examples)
    return examples

def atoms_to_ILASP_text(atoms_list):
    #whole_text = ""
    texts = []
    for atoms, label, id in atoms_list:
        # positive example
        if label == 1:
            #text = "#pos(" + "{}@1".format(id) +",{}, {},{\n" # penalty 1
            text = '#pos(eg(id{0})@{1}, {{ {2} }}, {{ {3} }}, {{\n'.format(id, 1, "kp(img)", "")
            #text = '#pos({{{0}}}, {{{1}}}, {{\n'.format("kp(img)", "")
            for atom in atoms:
                if not str(atom) == 'kp(img)':
                    text += str(atom)
                    text += ".\n" #\n"
            text += "})."#\n\n"
            texts.append(text.replace('.(__T__).', ''))
            # whole_text += "}).\n\n"
        # negative example
        elif label == 0:
            #text = '#neg({{{0}}}, {{{1}}}, {{\n'.format("", "")
            text = '#neg(eg(id{0})@{1}, {{ {2} }}, {{ {3} }}, {{\n'.format(id, 1, "kp(img)", "")
            #text = '#neg({{{0}}}, {{{1}}}, {{\n'.format("kp(img)", "")
            for atom in atoms:
                if not str(atom) == 'kp(img)':
                    text += str(atom)
                    text += ".\n" #\n"
            text += "})."#\n\n"
            texts.append(text.replace('.(__T__).', ''))
    return texts   
    #return whole_text.replace('.(__T__).\n', '')


def train_ilasp(NEUMANN, I2F, loader, atoms, args, device,  th=None, split='train'):
    predicted_list = []
    target_list = []

    background_knowledge = get_ilasp_background_knowledge(args.dataset)
    mode_declarations = get_ilasp_mode_declarations(args.dataset)
    ilp_examples = []
    #TODO: add CLINGO initialization here

   #print(ctl.solve(on_model=print))
    start = time.time()
    count = 0
    for i, sample in enumerate(tqdm(loader), start=0):
        imgs, target_set = map(lambda x: x.to(device), sample)
        batch_size = imgs.size(0)
        example_ids = list(range(count, count + batch_size))
        count += batch_size

        # to cuda
        target_set = target_set.float()

        V_0 = I2F(imgs)
        # V_T = NEUMANN(V_0)
        #  print('target set: ', target_set)
        # NEUMANN.print_valuation_batch(V_0)
        ilp_examples_batch = valuation_to_ILP_example_batch(V_0, atoms, target_set, example_ids)
        ilp_examples.extend(ilp_examples_batch)

    
    ilp_sess = ILASPSession(examples=ilp_examples, background_knowledge=background_knowledge, mode_declarations=mode_declarations)
    cached_lt_file = 'ilasp_cache/{}_ilpasp_cache.txt'.format(args.dataset)
    # cached_lt_file = cache_dir+'/learning_tasks/'+net_type+'/'+d+'/'+cached_lt_file_name
    with open(cached_lt_file, 'w') as lt_file:
        lt_file.write(ilp_sess.learning_task)
    ilp_sys = ILASPSystem(run_with_pylasp=False)
    learned_rules, output_info = ilp_sys.run(ilp_sess)
        
    reasoning_time = time.time() - start
    print('Reasoning Time: ', reasoning_time)
    print('Learned rules: ', learned_rules)
    """kp(img) :- shape(V1,sphere); color(V1,blue); color(V2,yellow).
       kp(img) :- color(V1,yellow); color(V2,blue); not shape(V2,cylinder). 
    """
    rule = learned_rules.replace(' ', '').replace(';', ',').replace('kp(img)', 'kp(X)') #.replace('\n','').replace('%','')
    return rule, reasoning_time

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


    ## train_pos_loader, val_pos_loader, test_pos_loader = get_vilp_pos_loader(args)
    #####train_pos_loader, val_pos_loader, test_pos_loader = get_data_loader(args)

    # load logical representations
    lark_path = 'src/lark/exp.lark'
    lang_base_path = 'data/lang/'
    lang, clauses, bk, bk_clauses, terms, atoms = get_lang(
        lark_path, lang_base_path, args.dataset_type, args.dataset, args.term_depth, use_learned_clauses=True)

    print("{} Atoms:".format(len(atoms)))

    # get torch data loader
    #question_json_path = 'data/behind-the-scenes/BehindTheScenes_questions_{}.json'.format(args.dataset)
    # test_loader = get_behind_the_scenes_loader(question_json_path, args.batch_size, lang, args.n_data, device)
    train_loader, val_loader, test_loader = get_data_loader(args, device, pos_ratio=args.n_ratio, neg_ratio=args.n_ratio)

    NEUMANN, I2F = get_model(lang=lang, clauses=clauses, atoms=atoms, terms=terms, bk=bk, bk_clauses=bk_clauses,
                          program_size=args.program_size, device=device, dataset=args.dataset, dataset_type=args.dataset_type,
                          num_objects=args.num_objects, infer_step=args.infer_step, train=False)#train=not(args.no_train))

    times = []
    val_accs = []
    test_accs = []
    # train split
    for j in range(n):
        train_loader, val_loader, test_loader = get_data_loader(args, device, pos_ratio=args.n_ratio, neg_ratio=args.n_ratio)
        learned_rule_str, time = train_ilasp(
            NEUMANN, I2F, train_loader, atoms, args, device, th=0.5, split='train')
        print("=== learned rule string ===")
        print(learned_rule_str)
        print("=====")
        from fol.data_utils import DataUtils
        du = DataUtils(lark_path, lang_base_path, args.dataset_type, args.dataset)
        learned_clauses = [du.parse_clause(learned_rule_str, lang)]
        times.append(time)
    
        NEUMANN, I2F = get_model(lang=lang, clauses=learned_clauses, atoms=atoms, terms=terms, bk=bk, bk_clauses=bk_clauses,
                          program_size=args.program_size, device=device, dataset=args.dataset, dataset_type=args.dataset_type,
                          num_objects=args.num_objects, infer_step=args.infer_step, train=False)
        
        train_loader, val_loader, test_loader = get_data_loader(args, device, pos_ratio=args.n_ratio, neg_ratio=args.n_ratio)
        
        acc_val, rec_val, th_val = predict(
            NEUMANN, I2F, val_loader, args, device, th=None, split='val')
        val_accs.append(acc_val)

        print("val acc: ", acc_val, "threashold: ", th_val, "recall: ", rec_val)
        acc_test, rec_test, th_test = predict(
            NEUMANN, I2F, test_loader, args, device, th=th_val, split='test')
        test_accs.append(acc_test)
        
        with open('out/learning_time/time_ilasp_{}_ratio_{}.txt'.format(args.dataset, args.n_ratio), 'w') as f:
            f.write("\n".join(str(item) for item in times))
        with open('out/learning_time/validation_accuracy_ilasp_{}_ratio_{}.txt'.format(args.dataset, args.n_ratio), 'w') as f:
            f.write("\n".join(str(item) for item in val_accs))
        with open('out/learning_time/test_accuracy_ilasp_{}_ratio_{}.txt'.format(args.dataset, args.n_ratio), 'w') as f:
            f.write("\n".join(str(item) for item in test_accs))


if __name__ == "__main__":
    main(n=5)

