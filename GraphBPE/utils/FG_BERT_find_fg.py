from rdkit import Chem
import os
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog


# https://github.com/idrugLab/FG-BERT/blob/main/utils.py

def get_common_fg():  # 47 FGs list
    fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    fparams = FragmentCatalog.FragCatParams(1, 6, fName)
    fg_list = []
    for i in range(fparams.GetNumFuncGroups()):
        fg_list.append(fparams.GetFuncGroup(i))
    fg_list.pop(27)

    x = [Chem.MolToSmiles(_) for _ in fg_list] + ['*C=C', '*F', '*Cl', '*Br', '*I', '[Na+]', '*P', '*P=O', '*[Se]',
                                                  '*[Si]']
    y = set(x)
    return list(y)


# def obsmitosmile(smi):
#     conv = ob.OBConversion()
#     conv.SetInAndOutFormats("smi", "can")
#     conv.SetOptions("K", conv.OUTOPTIONS)
#     mol = ob.OBMol()
#     conv.ReadString(mol, smi)
#     smile = conv.WriteString(mol)
#     smile = smile.replace('\t\n', '')
#     return smile


def split_node_w_fg(smiles, num_atoms, fg_structures):  # Getting functional groups (including rings) in molecules
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [[i] for i in range(num_atoms)]
    assert mol.GetNumAtoms() == num_atoms
    ssr = Chem.GetSymmSSSR(mol)
    num_ring = len(ssr)
    ring_dict = {}
    for i in range(num_ring):
        ring_dict[i + 1] = list(ssr[i])

    ring_list = []
    fg_list = []
    for ring_i in ring_dict.values():
        ring_list.append(ring_i)  # record rings

    for i in fg_structures:
        patt = Chem.MolFromSmarts(i)
        atomids = mol.GetSubstructMatches(patt)
        if len(atomids) == 0:  # no matching structure
            continue

        for atomid in atomids:
            already_existed = False
            set_new_fg = set(atomid)
            remove_fg_list = []
            for existing_fg in fg_list:
                set_existing_fg = set(existing_fg)
                if set_new_fg.issubset(set_existing_fg):  # new fg is contained by existing fg
                    already_existed = True
                    break
                if set_existing_fg.issubset(set_new_fg):  # existing fg is contained by new fg
                    remove_fg_list.append(existing_fg)
            # append the potential new fg
            if not already_existed:
                fg_list.append(list(atomid))
            # remove previous fgs that are contained by the new fg
            for fg_to_remove in remove_fg_list:
                fg_list.remove(fg_to_remove)

    fg_list = fg_list + ring_list
    fg_atoms = []
    for fg in fg_list:
        fg_atoms = fg_atoms + fg
    fg_atoms = set(fg_atoms)
    ori_atoms = set([i for i in range(num_atoms)])
    uni_ori_atoms = ori_atoms - fg_atoms
    for uni_ori_atom in uni_ori_atoms:
        fg_list.append([uni_ori_atom])
    return fg_list

# if __name__ == "__main__":
#     fg_structure = get_common_fg()
#     f_g_list = split_node_w_fg(smiles='C1=[N+](CCCCc2ccccc2)CCCC12CCCO2', num_atoms=20, fg_structures=fg_structure)
#     print(f_g_list)


