# Zhe Deng in 2025-11-16
# 1. Detect surface atoms based on surface normal vectors.
# 2. Generate all possible interstitial/adsorption sites with Voronoi tessellation.
# 3. Change the current positions of C based on obtained vacancies, or perturbe the positions of Fe atoms.
# 4. Optimize the updated structure with DPA-2 potential (or other calculators, not included in this script, see run.py).

import random
import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList
from collections import namedtuple
from scipy.spatial import Voronoi

class VLMC_nano:
    def __init__(self, stru):
        self.stru = stru.copy()
        self.fe_indices = [atom.index for atom in self.stru if atom.symbol == 'Fe']
        self.run_detection()

    def update_stru(self, new_stru):
        self.stru = new_stru.copy()
        self.run_detection()

    def run_detection(self):
        self.c_indices = [atom.index for atom in self.stru if atom.symbol == 'C']
        self.neigh_sum_vecs = self.get_vec_summation()
        self.surface_fe_indices = self.get_surface_fe_indices()
        
        self.raw_bulk_vacancies = self.get_bulk_vacancies()
        self.raw_surface_vacancies = self.get_surface_vacancies()
        self.vacancies = self.remove_duplicates(self.raw_surface_vacancies + self.raw_bulk_vacancies)
    
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

    def get_vacancies_environment(self, vacancies_pos, is_surface, Fe_threshold=2.4, C_threshold=1.2):
        """
        Get the local environment of surface or bulk vancancies based on ase.neighborlist function
        :param vacancies_pos: positions of vacancies (N, 3)
        :param is_surface: if True, only consider surface Fe atoms in calculation, boolean
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

        vacancy_indices = [atom.index for atom in temp_stru if atom.symbol == 'He']
        coordination_numbers = []
        carbon_counts = []
        carbon_indices = []
        if(is_surface):
            surface_fe_indices_set = set(self.surface_fe_indices)

        for vac_idx in vacancy_indices:
            neighbors_fe, _ = nl_fe.get_neighbors(vac_idx)
            if(is_surface):
                cn = sum(1 for i in neighbors_fe if i in surface_fe_indices_set)
            else:
                cn = sum(1 for i in neighbors_fe if i in Fe_indices_set)
            coordination_numbers.append(cn)
        
            # Check the status of vacancies (with or without carbon atoms)
            neighbors_c, _ = nl_c.get_neighbors(vac_idx)
            c_indice = np.array([i for i in neighbors_c if i in C_indices_set])
            c_count = len(c_indice)
            carbon_counts.append(c_count)
            carbon_indices.append(c_indice)

        return coordination_numbers, carbon_counts, carbon_indices

    def get_bulk_vacancies(self):
        """
        Get bulk vacancies from self.stru
        :return: namedtuple("Vacancy", ["position", "coord_num", "c_count", "c_indice"])
        """
        fe_indices = self.fe_indices
        fe_positions = self.stru.positions[fe_indices]
        vor = Voronoi(fe_positions)
        vacancies_pos = vor.vertices

        coord_nums, c_counts, c_indices = self.get_vacancies_environment(vacancies_pos, False)
        Vacancy_tuple = namedtuple("Vacancy", ["position", "coord_num", "c_count", "c_indice"])
        vacancy_data = [Vacancy_tuple(pos, cn, cc, ci) for pos, cn, cc, ci in zip(vacancies_pos, coord_nums, c_counts, c_indices)]
        bulk_vacancies = [v for v in vacancy_data if v.coord_num >= 5]

        return bulk_vacancies
    
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
    
    def get_surface_vacancies(self, neigh_threshold=4.8, site_threshold=2.4):
        """
        Get surface vacancies from recognized surface Fe atoms 
        :param neigh_threshold: cutoff distance for identifying neighboring Fe atoms in local 2D Voronoi tessellation, Ang
        :param site_threshold: cutoff radius for validating sites based on the center of neighboring atoms, Ang
        :return: namedtuple("Vacancy", ["position", "coord_num", "c_count", "c_indice"])
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

        coord_nums, c_counts, c_indices = self.get_vacancies_environment(vacancies_pos, True)
        Vacancy_tuple = namedtuple("Vacancy", ["position", "coord_num", "c_count", "c_indice"])
        vacancy_data = [Vacancy_tuple(pos, cn, cc, ci) for pos, cn, cc, ci in zip(vacancies_pos, coord_nums, c_counts, c_indices)]
        surface_vacancies = [v for v in vacancy_data if v.coord_num >= 3]

        return surface_vacancies
    
    """
    Below: Operators to update atomic positions of C (migrate, add, remove) and Fe (rattle)
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