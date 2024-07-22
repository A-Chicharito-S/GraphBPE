# $Id$
#
# Copyright (C) 2002-2010 greg Landrum and Rational Discovery LLC
#
#   @@ All Rights Reserved @@
#  This file is part of the RDKit.
#  The contents are covered by the terms of the BSD license
#  which is included in the file license.txt, found at the root
#  of the RDKit source tree.
#
""" functions to match a bunch of fragment descriptors from a file

No user-servicable parts inside.  ;-)

"""

# https://github.com/idrugLab/FG-BERT/blob/main/utils.py

import os
from rdkit import RDConfig
from rdkit import Chem
import torch_geometric as pyg

defaultPatternFileName = os.path.join(RDConfig.RDDataDir, 'FragmentDescriptors.csv')


def _CountAndGetMatches(mol, patt, unique=True):
    substructures = mol.GetSubstructMatches(patt, uniquify=unique)
    return len(substructures), substructures

fg2op = {}
fg2sma = {}


def _LoadPatterns(fileName=None):
    if fileName is None:
        fileName = defaultPatternFileName
    try:
        with open(fileName, 'r') as inF:
            for line in inF.readlines():
                if len(line) and line[0] != '#':
                    splitL = line.split('\t')
                    if len(splitL) >= 3:
                        name = splitL[0]
                        descr = splitL[1]
                        sma = splitL[2]
                        descr = descr.replace('"', '')

                        patt = Chem.MolFromSmarts(sma)

                        if not patt or patt.GetNumAtoms() == 0:
                            raise ImportError('Smarts %s could not be parsed' % (repr(sma)))
                        fn = lambda mol, countUnique=True, pattern=patt: _CountAndGetMatches(mol, pattern,
                                                                                             unique=countUnique)
                        fn.__doc__ = descr
                        name = name.replace('=', '_')
                        name = name.replace('-', '_')
                        # 'fr_sulfone' is repeated twice!
                        if name not in fg2op.keys():
                            fg2op[name] = fn
                            fg2sma[name] = sma
                            print(name, sma, fn)
    except IOError:
        pass


_LoadPatterns()


def find_functional_groups(data: pyg.data.Data):
    # assume we have edge_index, smiles, node_names
    smiles = data.smiles
    print(smiles)
    mol = Chem.MolFromSmiles(smiles)
    assert data.x.shape[0] == mol.GetNumAtoms(), 'x: {}; num_atoms: {}'.format(data.x.shape, mol.GetNumAtoms())
    # check whether number of nodes matches
    node_split = []
    for fg in fg2op.keys():
        op = fg2op[fg]
        num_sub, sub_idx = op(mol)
        if num_sub == 0:
            continue
        for sub in sub_idx:
            sub_set = set(sub)
            print(fg2sma[fg], sub_set)
            already_existed = False
            for existing_split in node_split:
                if sub_set.issubset(existing_split):
                    break
            if not already_existed:
                node_split.append(sub_set)
                print(fg2sma[fg], sub_set)

    return [list(split) for split in node_split]






