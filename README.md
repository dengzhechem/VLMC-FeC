# Variable Lattice Monte Carlo
A Voronoi tessellation-guided structural exploration framework, currently specified for iron carbides, including three modules:
- Bulk crystals.
- 2D surfaces.
- Isolated nanoparticles.

Each module provides some examples in terms of iron carbides.

## Dependencies:
- [ASE](https://wiki.fysik.dtu.dk/ase/about.html): system initialization, local relaxation and neighboring atom detection.
- [SciPy](https://scipy.org/): implements Voronoi tessellation and clustering algorithms.
- [DScribe](https://singroup.github.io/dscribe): implements the SOAP descriptor.
- [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit): integrates the machine learning potential. Users may substitute their own methods by changing the ASE calculator. However, due to computational costs, DFT methods are not recommended except for small systems.

Other standard libraries: `json`, `random`, `numpy`, `collections` and `matplotlib`. Please ensure them are available in your environment.

## Usage:

### 1. Preparations
The workflow library files are located in the `./source` directory:
```bash
source
├── post_process.py
├── read_params.py
├── vlmc_bulk.py
├── vlmc_nano.py
└── vlmc_surface.py
```
Before running a VLMC simulation, please add the `source` directory to your PYTHONPATH:
```bash
export PYTHONPATH=/path/to/source:$PYTHONPATH
```

Users can use the machine learning model provided in this work (or the previous [FT<sup>2</sup>DP-v1](https://www.aissquare.com/models/detail?pageType=models&id=307) model).

### 2. Run Simulations
- Phase transformation of Fe<sub>3</sub>C and Fe<sub>5</sub>C<sub>2</sub>:
```bash
cd /your/VLMC/bulk/Fe3C (or /your/VLMC/bulk/Fe5C2)
python ./bulk_run.py
```
- Carbon-induced reconstruction on the fcc Fe(100) surface:
```bash
cd /your/VLMC/surface
python ./surface_run.py
```
- Morphology evolution of an iron carbide nanoparticle:
```bash
cd /your/VLMC/nano
python ./nano_run.py
```
Note that before running simulations, please check the parameters in `params.json` to ensure they meet your needs. The default `max_iterations` parameter is set to 500 for testing purposes.
### 3. Output & Visualization
Upon completion, the simulation generates the following output files:

- `MC_results.png`: A plot showing the relative grand potential of all accepted structures.
- `Lowest.cif`: The structure of obtained global minimum.
- `FeC_CIFs`: A directory containing all accepted structures extracted from the ase.db database.

## Example Trajectories
If you prefer not to run the VLMC simulations, you can also view the example trajectories in `./traj-examples`. This directory contains four representative trajectories in `.xyz` format with accepted structures from previous VLMC simulations.

## References:
If you find the VLMC scheme or FT<sup>2</sup>DP model helpful for your work, we would appreciate a citation to the following papers:

[1] Liu, Z.-Q.; Deng, Z.; Zhao, H.; Wang, H.; Chen, M.; Jiang, H. FT<sup>2</sup>DP: Large Atomic Model Fine-Tuned Machine Learning Potential for Accelerating Atomistic Simulation of Iron-Based Fischer-Tropsch Synthesis. Journal of Materials Informatics. 2025, 5 (2). https://doi.org/10.20517/jmi.2024.105.

[2] Deng, Z; Liu, Z.; Jiang, H. Variable Lattice Monte Carlo: A Voronoi Tessellation-Guided Structural Exploration Framework and Applications in Iron Carbides. The Journal of Physical Chemistry Letters. ASAP. https://doi.org/10.1021/acs.jpclett.6c00382.