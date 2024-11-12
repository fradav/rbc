For building an image:

```bash
apptainer build rbc-micromamba-muse.sif micromamba-muse.def 
```


For muse:

```bash
srun --partition=imag --account=defimag -N 1 -n 1 -c 1 singularity run -c -H/lustre/yales2bio/HEMO_POUR_FRANCOIS-DAVID2/RBC_treatmet_folder3 -B/etc/slurm -B/usr/lib64/slurm -B/usr/bin/scancel -B/usr/bin/sbatch -B/usr/bin/srun -B/usr/lib64/libmunge.so.2 -B/var/run/munge ~/work/rbc-micromamba-muse.sif python dask-example.py
```

```python
from dask_jobqueue import SLURMCluster

container_image = "/storage/simple/users/collinf/work/rbc-micromamba-muse.sif"
working_dir = "/lustre/yales2bio/HEMO_POUR_FRANCOIS-DAVID2/RBC_treatmet_folder3"
containered_python_exe = f"singularity run -c -H{working_dir} {container_image} python"
# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

cluster = SLURMCluster(
    cores=1,
    processes=1,
    n_workers=1,
    interface='ib0',
    memory="2 GB",
    walltime='00:05:00',
    job_script_prologue=['module load singularity',],
    job_extra_directives=['--partition=imag',
               '--account=defimag',
               '-o myjob.%j.%N.out',
               '-e myjob.%j.%N.error'],
    python=containered_python_exe)

cluster.scale(10)
```

