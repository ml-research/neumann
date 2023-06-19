import numpy as np
from problog import get_evaluatable
from problog.core import ProbLog
from problog.program import PrologString


def to_ProbLog_program_batch(V_0, NEUMANN, th=0.5):

    atom_string_batch = get_atom_string(V_0, NEUMANN.atoms)
    clause_string = get_clause_string(NEUMANN)
    query_string = 'query(kp(img)).'
    
    programs = []
    for atom_string in atom_string_batch:
        program_str = atom_string + clause_string + query_string
        programs.append(PrologString(program_str))

    return programs


def get_atom_string(V_0, atoms):
    text_batch = []
    for i, v in enumerate(V_0):
        text = ''
        for j, v_j in enumerate(v):
            prob = str(np.round(v_j.detach().cpu().numpy(), 2))
            atom = str(atoms[j])
            if atom != '.(__T__)' and v_j > 0:
                atom_str = prob + '::' + atom + '.\n'
                text += atom_str
        text_batch.append(text)
    return text_batch

def get_clause_string(NEUMANN):
    clauses = NEUMANN.clauses + NEUMANN.bk_clauses
    # assume all clauses have weight of 1.0
    text = ''
    for clause in clauses:
        text += str(clause) + '\n'
    return text

    """
    examples = []
    
    target_atom = [atom for atom in atoms if str(atom) == 'kp(img)']
    assert len(target_atom) == 1, "Too many wrong atoms!"
    target_atom = target_atom[0]

    problog_strings = []

        label = labels[i]
        true_atoms = []
        for j in range(len(v)):
            if v[j] > th:
                true_atoms.append(atoms[j])
        if labels[i] == 1.0:
            true_atoms.append(target_atom)
        examples.append((true_atoms, label, example_ids[i]))
    """

def atoms_to_ProbLog_text(atoms_list):
    #whole_text = ""
    texts = []

    for atoms, label, id in atoms_list:
        # positive example
        if label == 1:
            # text = "#pos(" + "{}@1".format(id) +",{}, {},{\n" # penalty 1
            text = '#pos(eg(id{0})@{1}, {{ {2} }}, {{ {3} }}, {{\n'.format(id, 1, "", "")
            for atom in atoms:
                text += str(atom)
                text += "." #\n"
            text += "})."#\n\n"
            texts.append(text.replace('.(__T__).', ''))
            # whole_text += "}).\n\n"
        # negative example
        elif label == 0:
            0
            # text = "#neg(" + "{}@1".format(id) +",{}, {},{\n" # penalty 1
            """
            text = '#neg(eg(id{0})@{1}, {{ {2} }}, {{ {3} }}, {{\n'.format(id, 1, "", "")
            for atom in atoms:
                text += str(atom)
                text += "." # \n"
            text += "})." #.\n\n"
            texts.append(text.replace('.(__T__).', ''))
            """
            #whole_text += text
            #whole_text += "}).\n\n"
    return texts   

def get_ilasp_background_knowledge(dataset):
    if dataset == 'twopairs':
        background_knowledge = '''
            diff_color(red,blue).
            diff_color(blue,red).
            diff_color(red,yellow).
            diff_color(yellow,red).
            diff_color(blue,yellow).
            diff_color(yellow,blue).
            diff_shape(circle,square).
            diff_shape(square,circle).
            diff_shape(circle,triangle).
            diff_shape(triangle,circle).
            diff_shape(square,triangle).
            diff_shape(triangle,square).

            same_shape_pair(X,Y):-shape(X,Z),shape(Y,Z).
            same_color_pair(X,Y):-color(X,Z),color(Y,Z).
            diff_shape_pair(X,Y):-shape(X,Z),shape(Y,W),diff_shape(Z,W).
            diff_color_pair(X,Y):-color(X,Z),color(Y,W),diff_color(Z,W).
            '''
        #in4(O1,O2,O3,O4,X):-in(O1,X),in(O2,X),in(O3,X),in(O4,X).
        #P(A):-R1(O1,O2),R2(O1,O2),R3(O3,O4),R4(O3,O4).
        return background_knowledge


def get_ilasp_mode_declarations(dataset):
    if dataset == 'twopairs':
        mode_declarations = '''
            #constant(image,img).
            #constant(object,obj1).
            #constant(object,obj2).
            #constant(object,obj3).
            #constant(object,obj4).
            #constant(object,obj5).
            #constant(object,obj6).

            #constant(color,red).
            #constant(color,blue).
            #constant(color,yellow).
            #constant(shape,circle).
            #constant(shape,square).
            #constant(shape,triangle).

            #modeh(kp(var(image))).
            #modeb(1, color(var(object),const(color))).
            #modeb(1, shape(var(object),const(shape))).
            #modeb(2, same_color_pair(var(object),var(object))).
            #modeb(1, diff_color_pair(var(object),var(object))).
            #modeb(2, same_shape_pair(var(object),var(object))).
            #modeb(1, diff_shape_pair(var(object),var(object))).
            #modeb(1, closeby(var(object),var(object))).
            '''
            #modeb(1, online(var(object),var(object),var(object),var(object),var(object))).
            #modeb(1, in4(var(object),var(object),var(object),var(object),var(image))).
    return mode_declarations