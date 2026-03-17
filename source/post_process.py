import os
from ase.db import connect
from ase.io import write
import matplotlib.pyplot as plt

def db_analyse_varC(energy_C, miu_C):
    energy_C = energy_C   # eV
    miu_C = miu_C

    idx_list = []
    corr_energy_list = []
    C_Fe_ratio_list = []
    db = connect('./FeC.db')

    for stru in db.select(calculator='dp'):
        idx_list.append(stru.id - 1)
        cur_energy = stru.energy
        atoms = stru.toatoms()
        cur_C_num = sum(1 for atom in atoms if atom.symbol == 'C')
        cur_Fe_num = sum(1 for atom in atoms if atom.symbol == 'Fe')
        C_Fe_ratio_list.append(cur_C_num / cur_Fe_num)
        cur_corr_energy = cur_energy - cur_C_num * (energy_C + miu_C)
        corr_energy_list.append(cur_corr_energy)

    lowest_energy = min(corr_energy_list)
    lowest_stru_id = corr_energy_list.index(lowest_energy) + 1
    lowest_stru = db.get_atoms(id = lowest_stru_id)

    rel_energy_list = [x - lowest_energy for x in corr_energy_list]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Accepted Structure Index')
    ax1.set_ylabel('DPA-2 Relative Energy (eV)')
    ax1.plot(idx_list, rel_energy_list, linewidth=1.0, color='Red', label='Energy')
    lines, labels = ax1.get_legend_handles_labels()

    if(len(set(C_Fe_ratio_list)) == 1): # fixed-composition case
        ax1.legend(lines, labels)
    else:
        ax2 = ax1.twinx()
        ax2.set_ylabel('C/Fe ratio', rotation=270, labelpad=15)
        ax2.plot(idx_list, C_Fe_ratio_list, linewidth=1.0, color='Blue', linestyle='--', label='C/Fe')
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2)

    fig.tight_layout()
    plt.savefig('MC_results', dpi=200)

    write('Lowest.cif', lowest_stru, format='cif')

def write_cifs(tag, traj):
    os.mkdir(f'{tag}_CIFs')
    os.chdir(f'{tag}_CIFs')
    for ind, image in enumerate(traj):
        write(f'{tag}-{ind:04d}.cif', image, format='cif')
    os.chdir('..')