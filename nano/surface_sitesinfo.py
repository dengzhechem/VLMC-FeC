# Deng Zhe in 2026-1-19
# 1. Detect surface atoms based on surface normal vectors.
# 2. Generate all possible adsorption sites with Voronoi tessellation.
# 3. Filter surface sites with the descriptor-based site filtering algorithm.

import csv
import numpy as np
from ase import Atoms
from ase.db import connect
from ase.neighborlist import NeighborList
from collections import namedtuple
from scipy.spatial import Voronoi

from read_params import read_params

class SurfVacancyAnalyzer:
    def __init__(self, structure):
        self.stru = structure.copy()
        self.fe_indices = [atom.index for atom in self.stru if atom.symbol == 'Fe']
        self.run_detection()
    
    def update_stru(self, new_stru):
        self.stru = new_stru.copy()
        self.run_detection()
    
    def run_detection(self):
        self.c_indices = [atom.index for atom in self.stru if atom.symbol == 'C']
        self.neigh_sum_vecs = self.get_vec_summation()
        self.surface_fe_indices = self.get_surface_fe_indices()
        self.raw_surface_vacancies = self.get_surface_vacancies()

        filtered_vacancies, is_removed = self.remove_duplicates(self.raw_surface_vacancies)
        while(is_removed): # Run self.remove_duplicates iteratively
            filtered_vacancies, is_removed = self.remove_duplicates(filtered_vacancies)
        self.vacancies = self.merge_subset_vacancies(filtered_vacancies)
    
    def get_vec_summation(self, neigh_threshold=3.2):
        """
        Summation of displacement vectors (neighboring Fe atoms) of Fe atoms
        param neigh_threshold: cutoff distance in determining neighboring Fe atoms, Ang
        :return: list of result vectors (N, 3)
        """
        fe_positions = self.stru.positions[self.fe_indices]
        fe_stru = Atoms(symbols='Fe'*len(fe_positions), positions=fe_positions, cell=self.stru.cell, pbc=self.stru.pbc)

        cutoffs_fe = [neigh_threshold/2] * len(fe_positions)
        nl_fe = NeighborList(cutoffs_fe, skin=0, self_interaction=False, bothways=True)
        nl_fe.update(fe_stru)

        neigh_sum_vecs = []
        for idx in range(len(fe_stru)):
            sum_vec = np.array([0., 0., 0.])
            neighbors_fe, _ = nl_fe.get_neighbors(idx)
            for i in neighbors_fe:
                delta_vec = fe_stru.positions[idx] - fe_stru.positions[i]
                sum_vec += delta_vec / np.linalg.norm(delta_vec)
            mean_vec = sum_vec / len(neighbors_fe)
            neigh_sum_vecs.append(mean_vec)
        
        return neigh_sum_vecs

    def refine_surface_with_normalvec(self, surface_list_rough, displacement=3.2, neigh_threshold=3.2, probe_threshold=2.4):
        """
        Refine potential surface atoms to check if the direction along surface normal vector contains other Fe atom(s)
        :param surface_list_rough: index list of potential surface Fe atom
        :param displacement: distance along the surface normal vector, Ang
        :param neigh_threshold: cutoff distance in determining neighboring Fe atoms (for atoms not included in surface_rough_list), Ang
        :param probe_threshold: cutoff radius for refining surface atoms based on the probe point, Ang
        :return: index list of refined surface Fe atoms
        """
        fe_positions = self.stru.positions[self.fe_indices]
        fe_stru = Atoms('Fe'*len(fe_positions), positions=fe_positions, cell=self.stru.cell, pbc=self.stru.pbc) 

        cutoffs_refine = [neigh_threshold if i in surface_list_rough else 0.0 for i in range(len(fe_positions))]
        nl_refine = NeighborList(cutoffs_refine, skin=0, self_interaction=False, bothways=True)
        nl_refine.update(fe_stru)

        refine_idx_list = []
        for idx in surface_list_rough:
            neighbors_fe, _ = nl_refine.get_neighbors(idx)
            for i in neighbors_fe:
                if i not in refine_idx_list:
                    refine_idx_list.append(i)
        
        for idx in refine_idx_list:
            cur_pos = fe_stru.positions[idx]
            move_vec = displacement * self.neigh_sum_vecs[idx] / np.linalg.norm(self.neigh_sum_vecs[idx])
            fe_stru.append('He')
            fe_stru.positions[-1] = cur_pos + move_vec
        
        cutoffs_check = [probe_threshold if atom.symbol == 'He' else 0.0 for atom in fe_stru]
        nl_check = NeighborList(cutoffs_check, skin=0, self_interaction=False, bothways=True)
        nl_check.update(fe_stru)

        check_atom_indices = [atom.index for atom in fe_stru if atom.symbol == 'He']
        refined_surface_list = []
        for idx in range(len(check_atom_indices)):
            neighbors_he, _ = nl_check.get_neighbors(check_atom_indices[idx])
            cn = sum(1 for i in neighbors_he if i in self.fe_indices)
            if(cn == 0):
                refined_surface_list.append(refine_idx_list[idx])

        return refined_surface_list

    def get_surface_fe_indices(self):
        """
        Get surface Fe atoms with threshold 0.2 based on the magnitude of the net displacement vector, and then refine it with surface normal vectors
        :return: index list of surface Fe atoms
        """
        neigh_sum_vecs = self.neigh_sum_vecs

        surface_fe_idx_rough = []
        for i in range(len(neigh_sum_vecs)):
            if(np.linalg.norm(neigh_sum_vecs[i]) > 0.2):
                surface_fe_idx_rough.append(self.fe_indices[i])

        surface_fe_idx = self.refine_surface_with_normalvec(surface_fe_idx_rough)
        
        return surface_fe_idx

    def get_vacancies_environment(self, vacancies_pos, Fe_threshold=2.7, C_threshold=1.2):
        """
        Get the local environment of surface vancancies based on ase.neighborlist function
        To accommodate 5-fold sites, the value of Fe_threshold here is slightly larger than the default value (2.4)
        :param vacancies_pos: positions of vacancies (N, 3)
        :param Fe_threshold: cutoff radius in determing neighboring Fe atoms of each vacancy, Ang
        :param C_threshold: cutoff radius in determing neighboring C atoms of each vacancy, Ang
        :return: coordination number list (N), iron indice list (N, ), carbon count list (N), carbon indice list (N, )
        """
        surface_fe_indices_set = set(self.surface_fe_indices)
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

        vacancy_indices = [atom.index for atom in temp_stru if atom.symbol == 'He']
        coordination_numbers = []
        iron_indices = []
        carbon_counts = []
        carbon_indices = []

        for vac_idx in vacancy_indices:
            neighbors_fe, _ = nl_fe.get_neighbors(vac_idx)
            fe_indice = np.array([i for i in neighbors_fe if i in surface_fe_indices_set])
            cn = len(fe_indice)
            coordination_numbers.append(cn)
            iron_indices.append(fe_indice)
        
            # Check the status of vacancies (with or without carbon atoms)
            neighbors_c, _ = nl_c.get_neighbors(vac_idx)
            c_indice = np.array([i for i in neighbors_c if i in C_indices_set])
            c_count = len(c_indice)
            carbon_counts.append(c_count)
            carbon_indices.append(c_indice)

        return coordination_numbers, iron_indices, carbon_counts, carbon_indices

    def find_orth_basis(self, vec):
        vec_unit = vec / np.linalg.norm(vec)
        standard_bases = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        dot_products = [abs(np.dot(vec_unit, e)) for e in standard_bases]

        min_index = np.argmin(dot_products)
        e_selected = standard_bases[min_index]

        u = np.cross(vec_unit, e_selected)
        u_unit = u / np.linalg.norm(u)
        w_unit = np.cross(vec_unit, u_unit)
        basis_matrix = np.array([u_unit, w_unit, vec_unit]).T

        return basis_matrix

    def get_surface_vacancies(self, neigh_threshold=4.8, site_threshold=2.4):
        """
        Get surface vacancies from recognized surface Fe atoms 
        :param neigh_threshold: cutoff distance for identifying neighboring Fe atoms in local 2D Voronoi tessellation, Ang
        :param site_threshold: cutoff radius for validating sites based on the center of neighboring atoms, Ang
        :return: namedtuple("Vacancy", ["position", "coord_num", "fe_indice", "c_count", "c_indice"])
        """
        Fe_sur_indices = self.surface_fe_indices
        Fe_sur_positions = self.stru.positions[Fe_sur_indices]
        n_sur_featoms = len(Fe_sur_indices)

        temp_stru = Atoms('Fe'*n_sur_featoms, positions=Fe_sur_positions, cell=self.stru.cell, pbc=self.stru.pbc) 
        cutoffs = [neigh_threshold/2] * len(Fe_sur_indices)
        nl = NeighborList(cutoffs, skin=0, self_interaction=True, bothways=True)
        nl.update(temp_stru)

        vacancies_pos = []
        for atom_idx in range(n_sur_featoms):
            neighbors_idx, _ = nl.get_neighbors(atom_idx)
            neighbors_pos = Fe_sur_positions[neighbors_idx]
            center = np.mean(neighbors_pos, axis=0)
            relative_pos = neighbors_pos - center

            normal_vec = self.neigh_sum_vecs[Fe_sur_indices[atom_idx]]
            proj_matrix = self.find_orth_basis(normal_vec)
            projected_neighbors = np.dot(relative_pos, proj_matrix)
            xy_projected = projected_neighbors[:, :2]

            vor = Voronoi(xy_projected)
            for vertex in vor.vertices:
                dist_to_center = np.linalg.norm(vertex)
                if dist_to_center > site_threshold:
                    continue
                vertex_local = np.array([vertex[0], vertex[1], 0])
                vertex_global = center + np.dot(proj_matrix, vertex_local)
                adsorption_site = vertex_global + normal_vec / np.linalg.norm(normal_vec)
                vacancies_pos.append(adsorption_site)

        coord_nums, fe_indices, c_counts, c_indices = self.get_vacancies_environment(vacancies_pos)
        Vacancy_tuple = namedtuple("Vacancy", ["position", "coord_num", "fe_indice", "c_count", "c_indice"])
        vacancy_data = [Vacancy_tuple(pos, cn, fei, cc, ci) for pos, cn, fei, cc, ci in zip(vacancies_pos, coord_nums, fe_indices, c_counts, c_indices)]
        surface_vacancies = [v for v in vacancy_data if v.coord_num >= 3]

        return surface_vacancies

    def remove_duplicates(self, vacancy_data, threshold=1.2):
        """
        Remove adjacent vacancies based on 'r<1.2Angstrom' and leave vacancies that 'at the center of iron atoms'
        :param vacancy_data: namedtuple("Vacancy", ["position", "coord_num", "fe_indice", "c_count", "c_indice"])
        :param threshold: cutoff distance in removing adjacent vacancies, Ang
        :return: filtered vacancy_data list and a boolean varible is_removed
        """
        points = [v.position for v in vacancy_data]
        dup_atoms = Atoms('He'*len(points), positions=points, cell=self.stru.cell, pbc=self.stru.pbc) 

        cutoffs = [threshold/2] * len(dup_atoms)
        nl = NeighborList(cutoffs, skin=0, self_interaction=False, bothways=True)
        nl.update(dup_atoms)
        
        unique_indices = []
        for i in range(len(dup_atoms)):
            neighbors, _ = nl.get_neighbors(i)
            to_check_idx = [idx for idx in neighbors if idx in unique_indices]
            if(len(to_check_idx) == 0):
                unique_indices.append(i)
            else:
                i_neighbor_fe = vacancy_data[i].fe_indice
                i_dist_list = [np.linalg.norm(points[i] - self.stru.positions[idx]) for idx in i_neighbor_fe]
                i_dist_descriptor = max(i_dist_list) / min(i_dist_list) - 0.4 * (vacancy_data[i].coord_num // 5)
                for j in to_check_idx:
                    j_neighbor_fe = vacancy_data[j].fe_indice
                    j_dist_list = [np.linalg.norm(points[j] - self.stru.positions[idx]) for idx in j_neighbor_fe]
                    j_dist_descriptor = max(j_dist_list) / min(j_dist_list) - 0.4 * (vacancy_data[j].coord_num // 5)
                    if(i_dist_descriptor < j_dist_descriptor):
                        j_idx_in_list = unique_indices.index(j)
                        unique_indices[j_idx_in_list] = i
                        break
        
        is_removed = (len(vacancy_data) > len(unique_indices))
        
        return [vacancy_data[i] for i in unique_indices], is_removed
    
    def merge_subset_vacancies(self, vacancy_data):
        """
        Merge vacancies if a 3-fold or 4-fold site is in a 5-fold site
        :param vacancy_data: list(zip(vacancies_pos, coord_nums, fe_indices, c_counts, c_indices))
        :return: filtered vacancy_data list
        """
        vac_fe_indices = [vacancy.fe_indice for vacancy in vacancy_data]
        five_cn_vacancies = [set(arr) for arr in vac_fe_indices if len(arr) == 5]
        
        if(len(five_cn_vacancies) == 0):
            return vacancy_data
        
        subset_indices = []
        for i, arr in enumerate(vac_fe_indices):
            if(len(arr) == 3 or len(arr) == 4):
                arr_set = set(arr)
                is_subset = any(arr_set.issubset(five_fold_site) for five_fold_site in five_cn_vacancies)
                if is_subset:
                    subset_indices.append(i)

        filtered_vacancy_data = [vacancy_data[i] for i in range(len(vacancy_data)) if i not in subset_indices]
        return filtered_vacancy_data
    
if __name__ == "__main__":
    config = read_params()
    miu_C = config['miu_C']
    energy_C = config['energy_C']

    db = connect('./FeC.db')
    with open('surface_sites.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['stru_num', 'c_fe_ratio', 'corr_energy', '3fold', '4fold', '5fold'])
        for stru in db.select(calculator='dp'):
            cur_atoms = stru.toatoms()

            cur_energy = stru.energy
            cur_C_num = sum(1 for atom in cur_atoms if atom.symbol == 'C')
            cur_Fe_num = sum(1 for atom in cur_atoms if atom.symbol == 'Fe')
            cur_c_fe_ratio = cur_C_num / cur_Fe_num
            cur_corr_energy = cur_energy - cur_C_num * (energy_C + miu_C)

            cur_analyzer = SurfVacancyAnalyzer(cur_atoms)
            cur_vacancies = cur_analyzer.vacancies

            cn_list = [3, 4, 5]
            sites_percent = []
            for cn in cn_list:
                cn_sites_num = len([v for v in cur_vacancies if v.coord_num == cn])
                sites_percent.append(100*cn_sites_num/len(cur_vacancies))

            row_data = [stru.id-1, f"{cur_c_fe_ratio:.4f}", f"{cur_corr_energy:.4f}", f"{sites_percent[0]:.4f}", f"{sites_percent[1]:.4f}", f"{sites_percent[2]:.4f}"]
            writer.writerow(row_data)
            file.flush()