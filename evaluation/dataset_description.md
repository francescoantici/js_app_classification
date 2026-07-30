# Dataset Composition

The evaluation dataset consists of 100 labeled entries, each representing a real-world high-performance computing (HPC) job script manually categorized by its workload type. The entries span approximately 15 distinct scientific and engineering domains and reference roughly 55–60 unique software packages.

## Domain Distribution

The table below shows the breakdown of entries by application domain. The largest domain — computational chemistry and materials science at 14% — still represents only a minority of the entries, indicating that the dataset is reasonably balanced rather than concentrated in a single field. Machine learning and deep learning workloads account for 13% of the corpus, reflecting the growing convergence of ML and traditional HPC. Climate, weather, and earth sciences (11%), CFD and fluid dynamics (9%), and astronomy and astrophysics (9%) are also well represented.

| Domain | Count | Percentage |
|---|---|---|
| Computational Chemistry & Materials Science (DFT/QM) | 14 | 14% |
| Machine Learning & Deep Learning | 13 | 13% |
| Climate, Weather & Earth Sciences | 11 | 11% |
| CFD & Fluid Dynamics | 9 | 9% |
| Astronomy & Astrophysics | 9 | 9% |
| Workflow Managers | 8 | 8% |
| Molecular Dynamics | 6 | 6% |
| Bioinformatics & Genomics | 7 | 7% |
| Data Processing & Analytics | 6 | 6% |
| Nuclear & Particle Physics | 4 | 4% |
| FEA & Structural Engineering | 3 | 3% |
| Scientific Computing & HPC Utilities | 4 | 4% |
| Other (Quantum Computing, Electromagnetics, Neuroscience) | 6 | 6% |
| **Total** | **100** | **100%** |

## Specific Software Packages

The table below enumerates the specific software packages or applications that appear in the dataset, together with their domains and frequency of occurrence. The most frequently occurring software is the workflow manager Nextflow (4 occurrences), followed by VASP (3), and a set of widely used HPC codes that each appear twice: GROMACS, LAMMPS, OpenFOAM, CP2K, WRF, SU2, Snakemake, and Cylc.

| Software / Application | Domain | Occurrences |
|---|---|---|
| VASP | DFT / Materials | 3 |
| GROMACS | Molecular Dynamics | 2 |
| LAMMPS | Molecular Dynamics | 2 |
| OpenFOAM | CFD | 2 |
| CP2K | DFT | 2 |
| WRF | Weather / Climate | 2 |
| SU2 | CFD | 2 |
| Nextflow | Workflow Manager | 4 |
| Snakemake | Workflow Manager | 2 |
| Cylc | Workflow Manager | 2 |
| Quantum ESPRESSO | DFT | 1 |
| Gaussian | Quantum Chemistry | 1 |
| ORCA | Quantum Chemistry | 1 |
| NWChem | Quantum Chemistry | 1 |
| AMBER | Molecular Dynamics | 1 |
| NAMD | Molecular Dynamics | 1 |
| ABAQUS | FEA | 1 |
| OpenMC | Nuclear Engineering | 1 |
| GEANT4 | Particle Physics | 1 |
| Gadget | Astrophysics | 1 |
| FLASH | Astrophysics | 1 |
| Nyx | Astrophysics | 1 |
| CACTUS | Astrophysics | 1 |
| CESM | Climate | 1 |
| MOM6 | Ocean Modeling | 1 |
| PISM | Glaciology | 1 |
| GeoClaw | Geophysical Flows | 1 |
| SPECFEM3D | Seismology | 1 |
| PyCBC | Gravitational Waves | 1 |
| AlphaFold | ML + Structural Biology | 1 |
| BLAST+ | Bioinformatics | 1 |
| ABySS | Genomics | 1 |
| BUSCO | Genomics | 1 |
| Octave | Scientific Computing | 1 |
| FFmpeg | Media Processing | 1 |
| Rsync | File Transfer Utility | 1 |
| NEST | Computational Neuroscience | 1 |

## Label Granularity

The labels in the dataset span a wide range of granularity. Some entries carry generic, domain-level labels such as `cfd simulation`, `climate`, or `genomics`, while others are highly specific, naming both the code and the methodology, for example, `dedispersed radio astronomy`, `combustion dns`, `thermal structural model`, or `cp2k ab-initio dft`. Several entries also capture the pipeline nature of the workload (e.g., `bioinformatics setup`, `cleanup cfd`) rather than the simulation code itself. This heterogeneity in annotation granularity reflects the diversity of real-world job scripts, where descriptive comments and scheduler directives vary considerably in informativeness.
