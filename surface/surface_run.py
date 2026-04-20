# Zhe Deng in 2025-11-14
# Functions and parameters about running variable lattice Monte-Carlo (surface-only case)

import random
import numpy as np
from ase.db import connect
from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.neighborlist import NeighborList
from deepmd.calculator import DP
from dscribe.descriptors import SOAP

from read_params import read_params
from vlmc_surface import VLMC_surface
from post_process import db_analyse_varC, write_cifs

def SOAP_check(cur_stru, last_stru, threshold=0.20):
    natoms_cur_stru = len(cur_stru)
    natoms_last_stru = len(last_stru)
    # If natoms are different, two structures must be different
    if(natoms_cur_stru != natoms_last_stru):
        return False

    # SOAP-based fingerprint check to detect identical structure
    soap = SOAP(species=['Fe', 'C'], r_cut=4.0, n_max=3, l_max=3, periodic=True, sparse=False)
    cur_soap_vectors = soap.create(cur_stru)
    last_soap_vectors = soap.create(last_stru)
    cur_descriptors = cur_soap_vectors.mean(axis=1)
    last_descriptors = last_soap_vectors.mean(axis=1)

    # Check Fe
    cur_Fe_indices = [atom.index for atom in cur_stru if atom.symbol == 'Fe']
    cur_Fe_descriptors = np.sort([cur_descriptors[i] for i in cur_Fe_indices])
    last_Fe_indices = [atom.index for atom in last_stru if atom.symbol == 'Fe']
    last_Fe_descriptors = np.sort([last_descriptors[i] for i in last_Fe_indices])
    diff_Fe = np.abs(cur_Fe_descriptors - last_Fe_descriptors)
    print(f"Fe difference: {np.max(diff_Fe):.4f}")
    if(np.max(diff_Fe) > threshold):
        return False

    # Check C
    cur_C_indices = [atom.index for atom in cur_stru if atom.symbol == 'C']
    cur_C_descriptors = np.sort([cur_descriptors[i] for i in cur_C_indices])
    last_C_indices = [atom.index for atom in last_stru if atom.symbol == 'C']
    last_C_descriptors = np.sort([last_descriptors[i] for i in last_C_indices])
    diff_C = np.abs(cur_C_descriptors - last_C_descriptors)
    print(f"C difference: {np.max(diff_C):.4f}")
    if(np.max(diff_C) > threshold):
        return False
    
    return True

def subsurface_carbon_check(cur_stru, surface_fe_indices, threshold=2.4):
    # Carbon atoms are not allowed to enter subsurface here, for other situations, please remove this function manually
    temp_stru = cur_stru.copy()
    cutoffs_c = [threshold if atom.symbol == 'C' else 0.0 for atom in temp_stru]
    nl_c = NeighborList(cutoffs_c, skin=0, self_interaction=False, bothways=True)
    nl_c.update(temp_stru)

    fe_indices = [atom.index for atom in cur_stru if atom.symbol == 'Fe']
    bulk_fe_indices = [i for i in fe_indices if i not in surface_fe_indices]
    C_indices = [atom.index for atom in cur_stru if atom.symbol == 'C']
    for idx in C_indices:
        neighbors_c, _ = nl_c.get_neighbors(idx)
        neigh_bulk_fe_num = sum(1 for i in neighbors_c if i in bulk_fe_indices)
        if(neigh_bulk_fe_num > 2):
            return True

    return False

def run_MC(init_stru):
    dp_calc = DP(model=model)
    init_stru.calc = dp_calc
    init_stru_relax = BFGS(init_stru)
    init_stru_relax.run(fmax=0.05)
    print("DPA-2| Done relaxing  IS" + f", the energy is: {init_stru.get_potential_energy():.4f} eV")
    print(f"The given chemical potential of carbon is {miu_C:.2f} eV, with temperature = {temperature} K")
    print("======================================================================")

    db = connect("./FeC.db") # Only save structures that accepted by metropolis scheme
    db.write(init_stru, relaxed=True)
    db_all = connect("./FeC_all.db") # Save all calculated structures
    db_all.write(init_stru, relaxed=True)

    cur_stru = init_stru.copy()
    analyzer = VLMC_surface(cur_stru)

    prob_total = sum(prob for prob in prob_list)
    operations = [
        (prob_list[0] / prob_total, analyzer.migrate_carbon),
        (prob_list[1] / prob_total, analyzer.add_carbon),
        (prob_list[2] / prob_total, analyzer.remove_carbon),
        (prob_list[3] / prob_total, analyzer.random_rattle),
        (prob_list[4] / prob_total, analyzer.slide_atoms)
    ]

    # Run VLMC process for 'max_iterations' steps
    for step in range(max_iterations):
        vacancies = analyzer.vacancies
        vac_with_C = [v for v in vacancies if v.c_count > 0]
        vac_without_C = [v for v in vacancies if v.c_count == 0]

        if(len(vac_with_C) == 0):
            print("There is no vacancy with carbon atom, please check the current structure.")
            temp_stru = analyzer.add_carbon()
        elif(len(vac_without_C) == 0):
            print("There is no empty vacancy, please check the current structure.")
            temp_stru = analyzer.remove_carbon()
        # Random selection of mutation operators
        else:
            random_num = random.random()
            cumulative_prob = 0.0
            for prob, operator in operations:
                cumulative_prob += prob
                if random_num < cumulative_prob:
                    temp_stru = operator()
                    break
        cur_stru = temp_stru.copy()
        db_all.write(cur_stru, relaxed=False)
            
        cur_stru.calc = dp_calc
        local_relax = BFGS(cur_stru, maxstep=1.2)
        local_relax.run(fmax=0.05, steps=300)
        db_all.write(cur_stru, relaxed=True)

        cur_energy = cur_stru.get_potential_energy()
        cur_C_num = sum(1 for atom in cur_stru if atom.symbol == 'C')
        print(f"DPA-2| Done relaxing stru {step+1}, the energy is: {cur_energy:.4f} eV")

        rows = list(db.select(sort='id'))
        last_row = rows[-1]
        last_stru = db.get_atoms(id=last_row.id)
        last_energy = last_stru.get_potential_energy()
        last_C_num = sum(1 for atom in last_stru if atom.symbol == 'C')

        # E_corr = E_tot - N_C * miu_C
        cur_corr_energy = cur_energy - cur_C_num * (energy_C + miu_C)
        last_corr_energy = last_energy - last_C_num * (energy_C + miu_C)
        print(f"The corrected energy of cur_stru is {cur_corr_energy:.4f} eV, last_stru is {last_corr_energy:.4f} eV")
        
        # Discard identical structure or untargeted structure with subsurface carbon atoms
        is_identical = SOAP_check(cur_stru, last_stru)
        with_subsurface_c = subsurface_carbon_check(cur_stru, analyzer.surface_fe_indices)
        if(is_identical):
            print("This structure is identical to the last accepted one, discard it")
        elif(with_subsurface_c):
            print("This structure contains subsurface carbon atom(s), discard it")
        elif(random.random() < np.exp((last_corr_energy - cur_corr_energy) / (temperature * 8.617E-5))):
            print(f"This structure has been accepted with index {len(db)}")
            analyzer.update_stru(cur_stru)
            db.write(cur_stru, relaxed=True)
        else:
            print("This structure has been rejected")
            
        print("====================================================================")
    
if __name__ == "__main__":
    config = read_params()

    model = config['model']
    max_iterations = config['max_iterations']
    temperature = config['temperature']
    energy_C = config['energy_C']
    miu_C = config['miu_C']

    prob_mig_C = config['prob_mig_C']
    prob_add_C = config['prob_add_C']
    prob_rmv_C = config['prob_rmv_C']
    prob_rtt = config['prob_rtt']
    prob_sld = config['prob_sld']
    prob_list = [prob_mig_C, prob_add_C, prob_rmv_C, prob_rtt, prob_sld]

    init_stru = read(config['init_stru'])
    mask = init_stru.get_scaled_positions()[:, 2] <= 0.08
    fix = FixAtoms(mask=mask)
    init_stru.set_constraint(fix)
    run_MC(init_stru)
    
    write_cifs('FeC', read('./FeC.db', ':'))
    # write_cifs('FeC_all', read('./FeC_all.db', ':')) # debug only
    db_analyse_varC(energy_C, miu_C)
    print("Done! You can check the results now.")