# Zhe Deng in 2025-11-16
# 1. Generate all possible interstitial sites with Voronoi tessellation.
# 2. Change the current positions of C based on obtained vacancies, or perturbe the positions of Fe atoms.
# 3. Optimize the updated structure with DPA-2 potential (or other calculators, not included in this script, see run.py).

import random
import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList
from dscribe.descriptors import SOAP
from collections import namedtuple
from scipy.spatial import Voronoi
from scipy.cluster.hierarchy import linkage, fcluster

class VLMC_bulk:
    def __init__(self, stru):
        self.stru = stru.copy()
        self.fe_indices = [atom.index for atom in self.stru if atom.symbol == 'Fe']
        self.run_detection()

    def update_stru(self, new_stru):
        self.stru = new_stru.copy()
        self.run_detection()
    
    def run_detection(self):
        self.c_indices = [atom.index for atom in self.stru if atom.symbol == 'C']
        self.vacancies = self.get_vacancies()

    def apply_supercell(self, points):
        # Construct a 3x3x3 supercell of self.stru
        cell = self.stru.cell
        a1, a2, a3 = cell[0, :], cell[1, :], cell[2, :]
        offsets = np.array([i * a1 + j * a2 + k * a3 for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]])
        extended_points = np.vstack([points + offset for offset in offsets])

        return extended_points

    def filter_vacancies(self, vacancies):
        # Only accept vacancies inside the primitive cell
        inv_cell = np.linalg.inv(self.stru.cell)
        scaled_coords = np.dot(vacancies, inv_cell)
        is_in_range = (scaled_coords < 1) & (scaled_coords >= 0)
        mask = np.all(is_in_range, axis=1)

        return vacancies[mask]

    def get_vacancies_environment(self, vacancies_pos, Fe_threshold=2.4, C_threshold=1.2):
        """
        Get the local environment of bulk vancancies based on ase.neighborlist function
        :param vacancies_pos: positions of vacancies (N, 3)
        :param Fe_threshold: cutoff radius in determining neighboring Fe atoms of each vacancy, Ang
        :param C_threshold: cutoff radius in determining neighboring C atoms of each vacancy, Ang
        :return: coordination number list (N) carbon count list (N) carbon indice list (N, )
        """
        Fe_indices_set = set(self.fe_indices)
        C_indices_set = set(self.c_indices)
        temp_stru = self.stru.copy()

        # Add artificial atoms (He) to represent 'vacancies'
        for pos in vacancies_pos:
            temp_stru.append('He')
            temp_stru.positions[-1] = pos

        cutoffs_fe = [Fe_threshold if atom.symbol == 'He' else 0.0 for atom in temp_stru]
        nl_fe = NeighborList(cutoffs_fe, skin=0, self_interaction=False, bothways=True)
        nl_fe.update(temp_stru)

        cutoffs_c = [C_threshold if atom.symbol == 'He' else 0.0 for atom in temp_stru]
        nl_c = NeighborList(cutoffs_c, skin=0, self_interaction=False, bothways=True)
        nl_c.update(temp_stru)

        vacancy_indices = [i for i, atom in enumerate(temp_stru) if atom.symbol == 'He']
        coordination_numbers = []
        carbon_counts = []
        carbon_indices = []

        for vac_idx in vacancy_indices:
            neighbors_fe, _ = nl_fe.get_neighbors(vac_idx)
            cn = sum(1 for i in neighbors_fe if i in Fe_indices_set)
            coordination_numbers.append(cn)
        
            # Check the status of vacancies (with or without carbon atoms)
            neighbors_c, _ = nl_c.get_neighbors(vac_idx)
            c_indice = np.array([i for i in neighbors_c if i in C_indices_set])
            c_count = len(c_indice)
            carbon_counts.append(c_count)
            carbon_indices.append(c_indice)

        return coordination_numbers, carbon_counts, carbon_indices
    
    def remove_duplicates(self, vacancy_data, threshold=1.2):
        """
        Remove close vacancies, based on a distance-based criterion
        :param vacancy_data: namedtuple("Vacancy", ["position", "coord_num", "c_count", "c_indice"])
        :param threshold: cutoff distance in removing adjacent vacancies
        :return: filtered vacancy_data list
        """
        points = np.array([v.position for v in vacancy_data])
        dup_atoms = Atoms('He'*len(points), positions=points, cell=self.stru.cell, pbc=[True, True, True]) 

        cutoffs = [threshold/2] * len(dup_atoms)
        nl = NeighborList(cutoffs, skin=0, self_interaction=False, bothways=True)
        nl.update(dup_atoms)
        
        duplicates = set()
        unique_indices = []
        
        for i in range(len(dup_atoms)):
            if i not in duplicates:
                unique_indices.append(i)
                neighbors, _ = nl.get_neighbors(i)
                duplicates.update(neighbors)
        
        return [vacancy_data[i] for i in unique_indices]

    def get_vacancies(self):
        """
        Get bulk vacancies from self.stru
        :return: namedtuple("Vacancy", ["position", "coord_num", "c_count", "c_indice"])
        """
        Fe_positions = self.stru.positions[self.fe_indices]
        extended_points = self.apply_supercell(Fe_positions)
        vor = Voronoi(extended_points)
        vacancies_pos = vor.vertices
        
        filtered_vacancies_pos = self.filter_vacancies(vacancies_pos)
        coord_nums, c_counts, c_indices = self.get_vacancies_environment(filtered_vacancies_pos)
        Vacancy_tuple = namedtuple("Vacancy", ["position", "coord_num", "c_count", "c_indice"])
        vacancy_data = [Vacancy_tuple(pos, cn, cc, ci) for pos, cn, cc, ci in zip(filtered_vacancies_pos, coord_nums, c_counts, c_indices)]

        # Low cn sites are not suitable for new carbon atoms
        high_cn_vacancy_data = [v for v in vacancy_data if v.coord_num >= 5]
        final_vacancies = self.remove_duplicates(high_cn_vacancy_data)

        return final_vacancies
    
    def calculate_local_env(self):
        """
        Calculate the SOAP-based desriptors of self.stru (only consider Fe atoms here)
        :return: descriptor list of all atoms (N)
        """
        fe_atoms = [atom for atom in self.stru if atom.symbol == 'Fe']
        fe_stru = Atoms(symbols='Fe'*len(fe_atoms), positions=[atom.position for atom in fe_atoms], cell=self.stru.cell, pbc=self.stru.pbc)

        soap = SOAP(species=['Fe'], r_cut=4.0, n_max=3, l_max=3, periodic=True, sparse=False)
        soap_vectors = soap.create(fe_stru)
        soap_descriptors = soap_vectors.mean(axis=1)

        return soap_descriptors
    
    def hierarchical_clustering(self, val_list, threshold):
        """
        Cluster atoms based on given properties (val_list)
        :param val_list: to be clustered atomic property (SOAP-based quantities here)
        :param threshold: threshold when clustering
        :return: sorted clustered 2d list with clustered indices of given atoms 
        """
        data = np.array(val_list).reshape(-1, 1)

        Z = linkage(data, method='single', metric='euclidean')
        cluster_labels = fcluster(Z, t=threshold, criterion='distance')
        max_label = np.max(cluster_labels)

        cluster_result = [[] for _ in range(max_label)]
        for idx, label in enumerate(cluster_labels):
            cluster_result[label-1].append(idx)

        # Sort the result list based on the mean value of each cluster
        cluster_means = []
        for cluster in cluster_result:
            values = [val_list[i] for i in cluster]
            cluster_means.append(np.mean(values))
            
        sorted_indices = np.argsort(cluster_means)
        sorted_cluster_result = [cluster_result[i] for i in sorted_indices]
            
        return sorted_cluster_result
    
    """
    Below: Operators to update atomic positions of C (migrate, add, remove) and Fe (rattle or slide)
    """

    def migrate_carbon(self):
        print("Now trying to migrate a carbon atom randomly")
        temp_stru = self.stru.copy()
        C_indices = self.c_indices
        empty_vacancies = [v for v in self.vacancies if v.c_count == 0]
        src_indice = random.choice(C_indices)
        tgt_vacancy = random.choice(empty_vacancies)

        temp_stru.positions[src_indice] = tgt_vacancy[0]

        return temp_stru

    def add_carbon(self):
        print("Now trying to add a carbon atom randomly")
        temp_stru = self.stru.copy()
        avail_vacancies = [v for v in self.vacancies if v.c_count == 0]
        cur_vac = random.choice(avail_vacancies)
        cur_vac_pos = cur_vac[0]

        temp_stru.append('C')
        temp_stru.positions[-1] = cur_vac_pos
        
        return temp_stru

    def remove_carbon(self):
        print("Now trying to remove a carbon atom randomly")
        temp_stru = self.stru.copy()

        del_idx = random.choice(self.c_indices)
        del temp_stru[del_idx]
        
        return temp_stru
    
    def random_rattle(self, strength=0.8):
        print("Now trying to rattle all atoms randomly")
        temp_stru = self.stru.copy()
        for idx in range(len(temp_stru)):
            ith_pos = temp_stru.positions[idx]
            delta_pos = [random.uniform(-strength, strength) for _ in range(3)]
            temp_stru.positions[idx] = ith_pos + delta_pos
        
        return temp_stru
    
    def rattle_iron_with_groups(self, clustered_atoms, strength=0.8):
        print("Now trying to rattle all irons with groups, also move carbons")
        temp_stru = self.stru.copy()
        delta_pos_list = []
        for i in range(len(clustered_atoms)):
            delta_pos_list.append([random.uniform(-strength, strength) for _ in range(3)])
            for j in clustered_atoms[i]:
                temp_stru.positions[j] = temp_stru.positions[j] + delta_pos_list[i]
    
        # Move C based on the neighboring Fe atoms
        cutoffs_c = [2.4 if atom.symbol == 'C' else 0.0 for atom in self.stru]
        nl_c = NeighborList(cutoffs_c, skin=0, self_interaction=False, bothways=True)
        nl_c.update(self.stru)

        c_indices = self.c_indices
        for idx in c_indices:
            neighbors_c, _ = nl_c.get_neighbors(idx)
            delta_pos_idx = []
            for ngb_idx in neighbors_c:
                if(ngb_idx in c_indices):
                    continue
                else:
                    group_id = next(idx for idx, clu_list in enumerate(clustered_atoms) if ngb_idx in clu_list)
                    delta_pos_idx.append(delta_pos_list[group_id])
            delta_pos_idx_mean = np.mean(delta_pos_idx, axis=0)
            temp_stru.positions[idx] = temp_stru.positions[idx] + delta_pos_idx_mean
        
        return temp_stru
    
    def safe_group_rattle(self):
        soap_descriptors = self.calculate_local_env()
        clustered_atoms = self.hierarchical_clustering(soap_descriptors, 0.10)

        if(len(clustered_atoms) > 1):
            return self.rattle_iron_with_groups(clustered_atoms)
        else:
            print("Warning: all iron atoms are in the same group, switch to random_rattle()")
            return self.random_rattle()
    
    def slide_atoms(self, strength=1.6):
        print("Now trying to shift atoms in a selected axis-aligned slab")
        temp_stru = self.stru.copy()
        frac_positions = temp_stru.get_scaled_positions()
    
        direction = random.choice([0, 1, 2])
        generated = False
        while(not generated):
            pos_start = random.random()
            pos_end = random.random()
            if(pos_start > pos_end):
                pos_start, pos_end = pos_end, pos_start
            selected_atoms = [idx for idx in range(len(frac_positions)) if pos_start < frac_positions[idx, direction] < pos_end]
            if(len(selected_atoms) != 0):
                generated = True
        
        delta_pos = np.zeros(3)
        random_pos = random.uniform(strength/2, strength)
        theta = np.random.uniform(0, 2 * np.pi)
        move_pos_1 = random_pos * np.cos(theta)
        move_pos_2 = random_pos * np.sin(theta)

        move_direction = [i for i in [0, 1, 2] if i != direction]
        delta_pos[move_direction[0]] = move_pos_1
        delta_pos[move_direction[1]] = move_pos_2

        for i in selected_atoms:
            temp_stru.positions[i] = temp_stru.positions[i] + delta_pos
        
        return temp_stru