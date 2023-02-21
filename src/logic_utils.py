from tqdm import tqdm

from fol.data_utils import DataUtils
from fol.language import DataType
from fol.logic import *

p_ = Predicate('.', 1, [DataType('spec')])
false = Atom(p_, [Const('__F__', dtype=DataType('spec'))])
true = Atom(p_, [Const('__T__', dtype=DataType('spec'))])


def get_lang(lark_path, lang_base_path, dataset_type, dataset, term_depth):
    """Load the language of first-order logic from files.

    Read the language, clauses, background knowledge from files.
    Atoms are generated from the language.
    """
    print("Loading FOL language.")
    du = DataUtils(lark_path=lark_path, lang_base_path=lang_base_path,
                   dataset_type=dataset_type, dataset=dataset)
    lang = du.load_language()
    clauses = add_true_atom(du.load_clauses(
        du.base_path + 'clauses.txt', lang))
    bk_clauses = add_true_atom(du.load_clauses(
        du.base_path + 'bk_clauses.txt', lang))
    bk = du.load_atoms(du.base_path + 'bk.txt', lang)
    terms = generate_terms(lang, max_depth=term_depth)
    atoms = generate_atoms(lang, terms, dataset_type)
    # atoms = du.get_facts(lang)
    return lang, clauses, bk, bk_clauses, terms, atoms

def get_lang_behind_the_scenes(lark_path, lang_base_path, term_depth):
    """Load the language of first-order logic from files.

    Read the language, clauses, background knowledge from files.
    Atoms are generated from the language.
    """
    print("Loading FOL language.")
    du = DataUtils(lark_path=lark_path, lang_base_path=lang_base_path,
                   dataset_type='behind-the-scenes')
    lang = du.load_language()
    clauses = add_true_atom(du.load_clauses(
        du.base_path + 'clauses.txt', lang))
    bk_clauses = add_true_atom(du.load_clauses(
        du.base_path + 'bk_clauses.txt', lang))
    bk = du.load_atoms(du.base_path + 'bk.txt', lang)
    print("Generating Temrs ...")
    terms = generate_terms(lang, max_depth=term_depth)
    print("Generating Atoms ...")
    atoms = generate_atoms(lang, terms, "")
    # atoms = du.get_facts(lang)
    return lang, clauses, bk, bk_clauses, terms, atoms


def add_true_atom(clauses):
    """Add true atom T to body: p(X,Y):-. => p(X,Y):-T.
    """
    cs = []
    for clause in clauses:
        if len(clause.body) == 0:
            clause.body.append(true)
            cs.append(clause)
        else:
            cs.append(clause)
    return cs


def to_list(func_term):
    if type(func_term) == FuncTerm:
        return [func_term.args[0]] + to_list(func_term.args[1])
    else:
        return [func_term]

def generate_terms(lang, max_depth):
    consts = lang.consts
    funcs = lang.funcs
    terms = consts
    for i in tqdm(range(max_depth)):
        new_terms = []
        for f in funcs:
            terms_list = []
            for in_dtype in f.in_dtypes:
                terms_dt = [term for term in terms if term.dtype == in_dtype]
                terms_list.append(terms_dt)
                args_list = list(set(itertools.product(*terms_list)))

            for args in args_list:
                assert len(args) == 2
                if not args[0] in to_list(args[1]):
                    # print(args[1], to_list(args[1]))
                    new_terms.append(FuncTerm(f, args))
                    new_terms = list(set(new_terms))
                    terms.extend(new_terms)

    for x in sorted(list(set(terms))):
        print(x)
    return sorted(list(set(terms)))

def __generate_terms(lang, max_depth):
    consts = lang.consts
    funcs = lang.funcs
    terms = consts
    for i in tqdm(range(max_depth)):
        new_terms = []
        for f in funcs:
            terms_list = []
            for in_dtype in f.in_dtypes:
                terms_dt = [term for term in terms if term.dtype == in_dtype]
                terms_list.append(terms_dt)
                args_list = list(set(itertools.product(*terms_list)))

            for args in args_list:
                new_terms.append(FuncTerm(f, args))
                new_terms = list(set(new_terms))
                terms.extend(new_terms)
    return sorted(list(set(terms)))

def generate_atoms(lang, terms, dataset_type, max_term_depth=2):
    # spec_atoms = [false, true]
    atoms = []
    # terms = generate_terms(lang, max_depth=max_term_depth)
    for pred in lang.preds:
        dtypes = pred.dtypes
        terms_list = [[term for term in terms if term.dtype == dtype]
                      for dtype in dtypes]
        # consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
        args_list = []
        for terms_ in tqdm(set(itertools.product(*terms_list))):
            if dataset_type in ['kandinsky', 'clevr-hans']:
                if len(list(set(terms_))) == len(terms_):
                    args_list.append(terms_)
            else:
                args_list.append(terms_)
        for args in args_list:
            atoms.append(Atom(pred, args))
    return [true] + sorted(list(set(atoms)))


def generate_bk(lang):
    atoms = []
    for pred in lang.preds:
        if pred.name in ['diff_color', 'diff_shape']:
            dtypes = pred.dtypes
            consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
            args_list = itertools.product(*consts_list)
            for args in args_list:
                if len(args) == 1 or (args[0] != args[1] and args[0].mode == args[1].mode):
                    atoms.append(Atom(pred, args))
    return atoms


def get_index_by_predname(pred_str, atoms):
    for i, atom in enumerate(atoms):
        if atom.pred.name == pred_str:
            return i
        assert 1, pred_str + ' not found.'


def parse_clauses(lang, clause_strs):
    du = DataUtils(lang)
    return [du.parse_clause(c) for c in clause_strs]
