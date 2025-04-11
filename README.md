# SPIT
Single Particle Interaction Tracking (SPIT) is a Python library for single-molecule experiments. The modules in this library can be used to localize particles from a raw movie, link their movements across frames into trajectories, and run analyses on trajectories to detect local colocalization and interaction.

# Installation
1. install Picasso Desktop
2. install git and Anaconda (and GitHub Desktop)
3. using GitHub Desktop, clone picasso `https://github.com/jungmannlab/picasso`
4. open the Windows command line, move into the just created picasso folder (cd C:\...\Documents\GitHub\picasso) and type `git checkout 491ae28` to revert picasso to the version that works with SPIT
5. using GitHub Desktop, clone naclib `https://github.com/edovanveen/naclib` and SPIT `https://github.com/GanzingerLab/SPIT`
6. in Anaconda prompt, move into SPIT folder (using cd ..) and create environment using 

`conda env create -f=environment.yml`
7. activate the environment using activate SPIT
8. then build packages:

	8.1. move to picasso folder: cd ...Documents/Github/picasso and `pip install .`
	
	8.2. move to naclib folder: cd ...Documents/Github/naclib and `pip install .`
	
	8.3. move to SPIT and `pip install -e .`
9. install Spyder in the newly created SPIT environment: `conda install spyder`
10. finally, install lir in the SPIT environment. `pip install largestinteriorrectangle`



# Usage
Coming soon

## Contributions
Gerard Castro-Linares</br> 
Christian Niederauer</br>
Miles Wang-Henderson</br>
Elia Escoffier

## Citing SPIT

