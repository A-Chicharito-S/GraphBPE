from data_utils import qm9, tu_mutag, tu_enzymes, tu_proteins, molecule_net
supported_dataset = { # 'qm9': qm9.AugmentedQM9(root='dataset/qm9'),
                     'mutag': tu_mutag.AugmentedMutag(root='dataset', name='MUTAG'),
                     'enzymes': tu_enzymes.AugmentedEnzymes(root='dataset', name='ENZYMES'),
                     'proteins': tu_proteins.AugmentedProteins(root='dataset', name='PROTEINS'),
                     'esol': molecule_net.AugmentedMoleculeNet(root='dataset', name='esol'),
                     'freesolv': molecule_net.AugmentedMoleculeNet(root='dataset', name='freesolv'),
                     'lipophilicity': molecule_net.AugmentedMoleculeNet(root='dataset', name='lipo')}


supported_model = {'GNN': ['GCN', 'GAT', 'GIN', 'GraphSAGE', 'GraphTransformer'],
                   'HGNN': ['HyperConv', 'HGNN', 'HGNNP', 'HNHN']}

