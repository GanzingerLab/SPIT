# SPIT
Single Particle Interaction Tracking (SPIT) is a Python library for single-molecule experiments. The modules in this library can be used to localize particles from a raw movie, link their movements across frames into trajectories, and run analyses on trajectories to detect local colocalization and interaction.

# Installation
1. `git clone https://github.com/jungmannlab/picasso` and `git checkout 491ae28`
3. `git clone https://github.com/GanzingerLab/SPIT`
4. cd to the local version of the repository create environment `conda env create -f=environment.yml`
5. activate environment with `activate SPIT`
6. build package: move to picasso repository and `python setup.py install`
7. build package: move to SPIT repository and `python setup.py install`


# Usage
SPIT is executable from the command line. The following modules can be either executed individually, or consecutively:

**Localize** - localizes particles in frames </br>
**Link** - links localizations into trajectories </br>
**Colocalize** - checks for interactions between trajectories </br>
**FRAP** - performs a FRAP analysis of a SLB </br>

![Diagram explanation of SPIT](/spit_diagram.png?raw=true "Diagram explanation of SPIT")

## Contributions
Christian Niederauer</br>
Miles Wang-Henderson</br>
Elia Escoffier

## Citing SPIT

