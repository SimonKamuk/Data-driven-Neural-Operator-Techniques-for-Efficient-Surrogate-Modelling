#!/bin/bash

##Kø navn, fx hpc som er den generelle cpu. Der er også gpu køer
#BSUB -q hpc
##Antal gpuer vi vil bruge. Kommenter ud hvis cpu.
##BSUB -gpu "num=1:mode=exclusive_process"

##Navn på job. Hvis man vil lave mange jobs kan man skrive my_job_name[1-100] så får man 100 jobs.
#BSUB -J deeponet_param_search[1-729]
##Output log fil. Folderen skal eksistere før jobbet submittes. Job nummer indsættes automatisk ved %J i filnavnet.
#BSUB -o output/deeponet_param_search_%J_%I.out
##Antal cpu kerner
#BSUB -n 1
##Om kernerne må være på forskellige computere
#BSUB -R "span[hosts=1]"
##Ram pr kerne
#BSUB -R "rusage[mem=8GB]"
##Hvor lang tid må den køre hh:mm
#BSUB -W 10:00
##Modtag email på studiemail når jobbet starter
#BSUB -B
##mail når jobbet stopper
#BSUB -N

## Fjern alle pakker og load så dem man skal bruge. Brug "module avail" på dtu server for at se hvilke pakker der findes.
module purge
module load julia/1.8.2

## Konstant argument til programmet

## Hvis man skal lave et loop hvor programmet modtager forskellige argumenter.
## for x in 1 2 3
## do
julia convection_diffusion_var_time.jl $LSB_JOBINDEX
## done
## Har man oprettet jobbet som en liste af jobs my_job_name[1-100] så kan man bruge dette indeks fra 1 til 100 som argument til sit program med argumentet $LSB_JOBINDEX
