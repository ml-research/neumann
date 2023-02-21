from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_atoms(x_atom, atoms, path):
    x_atom = x_atom.detach().cpu().numpy()
    labels = [str(atom) for atom in atoms]
    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(x_atom)
    fig, ax = plt.subplots(figsize=(30,30))
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1])
    for i, label in enumerate(labels):
        ax.annotate(label, (X_reduced[i,0], X_reduced[i,1]))
    plt.savefig(path)

def plot_infer_embeddings(x_atom_list, atoms):
    for i, x_atom in enumerate(x_atom_list):
        plot_atoms(x_atom, atoms, 'imgs/x_atom_' + str(i) + '.png')

