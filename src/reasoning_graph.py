import itertools
import multiprocessing
import networkx as nx

from torch_geometric.utils import from_networkx
from fol.logic import Clause, Conjunction
from fol.logic_ops import subs_list, unify

import torch
from tqdm import tqdm


class ReasoningGraphModule(object):
    """Reasoning graph, which represents a forward-reasoning process as a bipartite graph.

    Args:
        clauses (list(clauses)): The set of clauses.
        facts (list(atom)): The set of ground atoms (facts).
        terms (list(term)): The set of ground terms.
        lang (language): A language of first-order logic.
        device (device): A device.
        max_term_depth (int): The maximum depth (height) of the term in terms of the functors.
        init (bool): The flag whether the initialization is performed or not.
        dataset_type (str): A dataset type.
    """

    def __init__(self, clauses, facts, terms, lang, device, max_term_depth=3, init=True, dataset_type=None):
        self.lang = lang
        self.clauses = clauses
        self.facts = facts
        self.terms = terms
        self.device = device
        self.max_term_depth = max_term_depth
        self.dataset_type = dataset_type
        if init:
            self.fact_index_dict = self._build_fact_index_dict(facts)
            self.grounded_clauses, self.clause_indices = self._ground_clauses(
                clauses, lang)
            # for i, clause in enumerate(self.grounded_clauses):
            #print(i, clause)
            self.atom_node_idxs = list(range(len(self.facts)))
            self.conj_node_idxs = list(range(len(self.facts), len(
                self.facts) + len(self.grounded_clauses) + 1))  # a dummy conj node
            #print('Building reasoning graph for {}'.format(str(clauses)))
            # build reasoning graph
            self.networkx_graph, self.node_labels, self.node_objects = self._build_rg()
            self.pyg_data = from_networkx(self.networkx_graph)
            self.edge_index = self.pyg_data.edge_index.to(device)
            self.edge_type = torch.tensor(self.pyg_data.etype).to(device)
            self.edge_clause_index = torch.tensor(
                self.pyg_data.clause_index).to(device)
            self.num_nodes = len(self.node_labels)

    def _build_fact_index_dict(self, facts):
        dic = {}
        for i, fact in enumerate(facts):
            dic[fact] = i
        return dic

    def __str__(self):
        N_atom_nodes = len(self.atom_node_idxs)
        N_conj_nodes = len(self.conj_node_idxs)
        return "Reasoning Graph(N_atom_nodes={}, N_conj_nodes={})".format(N_atom_nodes, N_conj_nodes)

    def __repr__(self):
        return self.__str__()

    def _get_fact_idx(self, fact):
        if not fact in self.fact_index_dict:
            return False, -1
        else:
            return True, self.fact_index_dict[fact]

    def _invalid_var_dtypes(self, var_dtypes):
        # check the contradiciton of the List[(var, dtype)]
        if len(var_dtypes) < 2:
            return False
        for i in range(len(var_dtypes)-1):
            for j in range(i, len(var_dtypes)):
                if var_dtypes[i][0] == var_dtypes[j][0] and var_dtypes[i][1] != var_dtypes[j][1]:
                    return True
        return False

    def _ground_clauses(self, clauses, lang):
        """
        Ground a clause using all of the constants in a language.

        Args:
            clause (Clause): A clause.
            lang (Language): A language.
        """
        grounded_clauses = []
        clause_indices = []
        # the body has existentially quantified variable!!
        # e.g. body atoms: [in(img,O1),shape(O1,square)]
        # theta_list: [(O1,obj1), (O1,obj2)]
        #theta_list = self.generate_subs(atoms)
        # print('theta_list:', theta_list)

        # TODO: Grounding
        print('Grounding clauses ...')
        for ci, clause in enumerate(clauses):
            print('Grounding Clause {}: '.format(ci), clause)
            # TODO: Do we need head unification????
            if len(clause.all_vars()) == 0:
                grounded_clauses.append(clause)
                clause_indices.append(ci)
            # some variables in the body
            # TODO: lookup facts!!
            else:
                theta_list = self.generate_subs([clause.head] + clause.body)
                for i, theta in enumerate(theta_list):
                    # print(i, theta)
                    grounded_head = subs_list(clause.head, theta)
                    # check the grounded clause can deduce at least one fact
                    if grounded_head in self.fact_index_dict:
                        grounded_body = [subs_list(bi, theta)
                                         for bi in clause.body]
                        grounded_clauses.append(
                            Clause(grounded_head, grounded_body))
                        clause_indices.append(ci)
        return grounded_clauses, clause_indices

    def get_terms_by_dtype(self, dtype):
        return [term for term in self.terms if term.dtype == dtype]

    def get_terms_by_dtype_and_depth(self, dtype, depth):
        return [term for term in self.terms if term.dtype == dtype and term.max_depth() + depth <= self.max_term_depth]

    def generate_subs(self, atoms):
        """Generate substitutions from given body atoms.

        Generate the possible substitutions from given list of atoms. If the body contains any variables,
        then generate the substitutions by enumerating constants that matches the data type.
        !!! ASSUMPTION: The body has variables that have the same data type
            e.g. variables O1(object) and Y(color) cannot appear in one clause !!!

        Args:
            body (list(atom)): The body atoms which may contain existentially quantified variables.

        Returns:
            theta_list (list(substitution)): The list of substitutions of the given body atoms.
        """

        # extract all variables and corresponding data types from given body atoms
        var_dtype_list = []
        for atom in atoms:
            vd_list = atom.all_vars_and_dtypes()
            for vd in vd_list:
                if not vd in var_dtype_list:
                    var_dtype_list.append(vd)

        var_depth_list = []
        for atom in atoms:
            vd_list = atom.all_vars_with_depth()
            for v, depth in vd_list:
                if not v in [vd[0] for vd in var_depth_list]:
                    var_depth_list.append((v, depth))
                else:
                    for i, (v_, depth_) in enumerate(var_depth_list):
                        if v == v_ and depth > depth_:
                            del var_depth_list[i]
                            var_depth_list.append((v, depth))

        #var_dtype_set = list(set(var_dtype_list))
        vars = [vd[0] for vd in var_dtype_list]
        dtypes = [vd[1] for vd in var_dtype_list]
        depths = [vd[1] for vd in var_depth_list]
        # in case there is no variables in the body
        if len(list(set(dtypes))) == 0:
            return []
        if self._invalid_var_dtypes(var_dtype_list):
            return []
        # check the data type consistency
        # assert len(list(set(dtypes))) == 1, "Invalid existentially quantified variables. " + \
        #    str(len(list(set(dtypes)))) + " data types in the body."
        n_vars = len(vars)
        # TODO: fetch consts for each variable by data type, cartesian product
        # TODO: func terms should be obtained
        ## consts_list = [self.lang.get_by_dtype(t) for t in dtypes]
        consts_list = []
        for i in range(len(dtypes)):
            consts_list.append(
                self.get_terms_by_dtype_and_depth(dtypes[i], depths[i]))

        subs_consts_list = []
        for consts in list(itertools.product(*consts_list)):
            # for consts in list(itertools.permutations(consts_list[0], 9)):
            if self.dataset_type in ['kandinsky', 'clevr-hans']:
                if len(list(set(consts))) == len(consts):
                    subs_consts_list.append(consts)
            else:
                subs_consts_list.append(consts)

        # TODO: Compute max depth of the variable in the atom

        # print(consts_list)
        # e.g. if the data type is shape, then subs_consts_list = [(red,), (yellow,), (blue,)]
        # subs_consts_list = itertools.permutations(consts, n_vars)

        theta_list = []
        # generate substitutions by combining variables to the head of subs_consts_list
        for subs_consts in tqdm(subs_consts_list):
            theta = []
            for i, const in enumerate(subs_consts):
                s = (vars[i], const)
                theta.append(s)
            # if theta not in theta_list:
            theta_list.append(theta)
        # e.g. theta_list: [[(Z, red)], [(Z, yellow)], [(Z, blue)]]
        return theta_list

    def _build_rg(self):
        """
        Build reasoning graph from clauses.

        TODO: get node logical objects? facts or conjunction.
        """

        # print('Building Reasoning Graph...')
        G, node_labels, node_objects = self._init_rg()
        edge_clause_index = []

        # add dummy edge T to T
        G.add_edge(0, len(self.facts)+len(self.grounded_clauses),
                   etype=0, color='r', clause_index=0)
        G.add_edge(len(self.facts)+len(self.grounded_clauses),
                   0, etype=1, color='r', clause_index=0)

        for i, gc in enumerate(tqdm(self.grounded_clauses)):
            head_flag, head_fact_idx = self._get_fact_idx(gc.head)
            body_fact_idxs = []
            body_flag = True
            for bi in gc.body:
                body_flag, body_fact_idx = self._get_fact_idx(bi)
                body_fact_idxs.append(body_fact_idx)
                if not body_flag:
                    # failed to find body fact in database
                    break
            if body_flag and head_flag:
                head_node_idx = head_fact_idx
                body_node_idxs = body_fact_idxs
                conj_node_idx = self.conj_node_idxs[i]
                for body_node_idx in body_node_idxs:
                    G.add_edge(conj_node_idx, head_node_idx, etype=1,
                               color='r', clause_index=self.clause_indices[i])
                    G.add_edge(body_node_idx, conj_node_idx, etype=0,
                               color='b', clause_index=self.clause_indices[i])
        return G, node_labels, node_objects

    def _to_edge_index(self, clauses):
        """
        Convert clauses into a edge index representing a reasoning graph.

        Args:
            clauses (list(Clauses)): A set of clauses.
        """
        gc_counter = 0
        edge_index = []
        edge_type = []
        edge_clause_index = []
        for i, clause in enumerate(clauses):
            grounded_clauses = self._ground_clause(clause)
            for gc in grounded_clauses:
                head_fact_idx = self.facts.index(gc.head)
                conj_idx = len(self.facts) + gc_counter
                body_fact_idxs = []
                for bi in gc.body:
                    body_fact_idx = self.facts.index(bi)
                    #G.add_edge(body_node_idx, conj_node_idx, etype=0, color='b')
                    new_edge = [body_fact_idx, conj_idx]
                    if not new_edge in edge_index:
                        edge_index.append([body_fact_idx, conj_idx])
                        edge_type.append(0)  # edge: atom_node -> conj_node
                        edge_clause_index.append(i)

                #G.add_edge(conj_node_idx, head_node_idx, etype=1, color='r')
                edge_index.append([conj_idx, head_fact_idx])
                edge_type.append(1)  # edge: conj_node -> atom_node
                edge_clause_index.append(i)
                gc_counter += 1
        edge_index = torch.tensor(edge_index).view((2, -1)).to(self.device)
        edge_clause_index = torch.tensor(edge_clause_index).to(self.device)
        edge_type = torch.tensor(edge_type).to(self.device)
        num_nodes = conj_idx

        # compute indicis for atom nodes and conjunction nodes
        self.atom_node_idxs = list(range(len(self.facts)))
        self.conj_node_idxs = list(
            range(len(self.facts), len(self.facts) + gc_counter))
        return edge_index, edge_clause_index, edge_type, num_nodes

    def _init_rg(self):
        G = nx.DiGraph()
        node_labels = {}
        node_objects = {}
        # atom_idxs = self.node_idx_dic['atom']
        # conj_idxs = self.node_idx_dic['conj']
        # print("Initializing the reasoning graph...")
        G.add_nodes_from(list(range(len(self.facts))))

        N_fact = len(self.facts)
        for i, fact in enumerate(self.facts):
            node_labels[i] = str(fact)
            node_objects[i] = fact
        # G.add_nodes_from(conj_idxs)
        G.add_nodes_from(
            list(range(N_fact, N_fact + len(self.grounded_clauses))))
        for i, conj in enumerate(self.grounded_clauses):
            node_labels[N_fact + i] = '∧'
            node_objects[N_fact + i] = Conjunction()

        # add dummy conj node and edge T -> dummy_con
        G.add_node(N_fact + len(self.grounded_clauses))
        node_labels[N_fact + len(self.grounded_clauses)] = '∧'
        node_objects[N_fact + len(self.grounded_clauses)] = Conjunction()

        return G, node_labels, node_objects
