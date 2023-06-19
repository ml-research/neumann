import os.path

from lark import Lark

from .exp_parser import ExpTree
from .language import DataType, Language
from .logic import Const, FuncSymbol, NeuralPredicate, Predicate


class DataUtils(object):
    """Utilities about I/O of logic.
    """

    def __init__(self, lark_path, lang_base_path, dataset_type='kandinsky', dataset=None):
        #if dataset == 'behind-the-scenes':
        # for behind the scenes
        #    self.base_path = lang_base_path + dataset_type + '/'
        #if: 
        self.base_path = lang_base_path + dataset_type + '/' + dataset + '/'
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_atom = Lark(grammar.read(), start="atom")
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_clause = Lark(grammar.read(), start="clause")

    def load_clauses(self, path, lang):
        """Read lines and parse to Atom objects.
        """
        clauses = []
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    if line[-1] == '\n':
                        line = line[:-1]
                    if len(line) == 0 or line[0] == '#':
                        continue
                    tree = self.lp_clause.parse(line)
                    clause = ExpTree(lang).transform(tree)
                    clauses.append(clause)
        return clauses

    def load_atoms(self, path, lang):
        """Read lines and parse to Atom objects.
        """
        atoms = []

        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    if line[-1] == '\n':
                        line = line[:-2]
                    else:
                        line = line[:-1]
                    tree = self.lp_atom.parse(line)
                    atom = ExpTree(lang).transform(tree)
                    atoms.append(atom)
        return atoms

    def load_preds(self, path):
        f = open(path)
        lines = f.readlines()
        preds = [self.parse_pred(line) for line in lines]
        return preds

    def load_neural_preds(self, path):
        f = open(path)
        lines = f.readlines()
        preds = [self.parse_neural_pred(line) for line in lines]
        return preds

    def load_consts(self, path):
        f = open(path)
        lines = f.readlines()
        consts = []
        for line in lines:
            consts.extend(self.parse_const(line))
        return consts

    def load_funcs(self, path):
        funcs = []
        if os.path.isfile(path):
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    funcs.append(self.parse_func(line))
        return funcs

    def load_terms(self, path, lang):
        f = open(path)
        lines = f.readlines()
        terms = []
        for line in lines:
            terms.extend(self.parse_term(line, lang))
        return
    def parse_pred(self, line):
        """Parse string to predicates.
        """
        line = line.replace('\n', '')
        pred, arity, dtype_names_str = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]
        assert int(arity) == len(
            dtypes), 'Invalid arity and dtypes in ' + pred + '.'
        return Predicate(pred, int(arity), dtypes)

    def parse_neural_pred(self, line):
        """Parse string to predicates.
        """
        line = line.replace('\n', '')
        pred, arity, dtype_names_str = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]
        assert int(arity) == len(
            dtypes), 'Invalid arity and dtypes in ' + pred + '.'
        return NeuralPredicate(pred, int(arity), dtypes)

    def parse_func(self, line):
        """Parse string to function symbols.
        (Format) name:arity:input_type:output_type
        """
        name, arity, in_dtypes, out_dtype = line.replace("\n", "").split(':')
        in_dtypes = in_dtypes.split(',')
        in_dtypes = [DataType(in_dtype) for in_dtype in in_dtypes]
        out_dtype = DataType(out_dtype)
        return FuncSymbol(name, int(arity), in_dtypes, out_dtype)
    
    

    def parse_const(self, line):
        """Parse string to constants.
        """
        line = line.replace('\n', '')
        dtype_name, const_names_str = line.split(':')
        dtype = DataType(dtype_name)
        const_names = const_names_str.split(',')
        return [Const(const_name, dtype) for const_name in const_names]

    def parse_term(self, line, lang):
        """Parse string to func_terms.
        """
        line = line.replace('\n', '')
        dtype_name, term_names_str = line.split(':')
        dtype = DataType(dtype_name)
        term_strs = term_names_str.split(',')
        terms = []
        for term_str in term_strs:
            if not term_str == '':
                print("term_str: ", term_str)
                tree = self.lp_term.parse(term_str)
                terms.append(ExpTree(lang).transform(tree))
        return terms
        #return [Const(const_name, dtype) for const_name in const_names]

    def parse_clause(self, clause_str, lang):
        tree = self.lp_clause.parse(clause_str)
        return ExpTree(lang).transform(tree)

    def get_clauses(self, lang):
        return self.load_clauses(self.base_path + 'clauses.txt', lang)

    def get_bk(self, lang):
        return self.load_atoms(self.base_path + 'bk.txt', lang)

    def get_facts(self, lang):
        return self.load_atoms(self.base_path + 'facts.txt', lang)

    def load_language(self):
        """Load language, background knowledge, and clauses from files.
        """
        preds = self.load_preds(self.base_path + 'preds.txt') + \
            self.load_neural_preds(self.base_path + 'neural_preds.txt')
        consts = self.load_consts(self.base_path + 'consts.txt')
        funcs = self.load_funcs(self.base_path + 'funcs.txt')
        #terms = self.load_terms(self.base_path + 'terms.txt')
        lang = Language(preds, funcs, consts)
        return lang
