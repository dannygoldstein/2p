import os
import shutil
import yaml
import scipy 
import simple as s
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

outdir = os.path.abspath('models')

if rank == 0:
    try:
        os.mkdir(outdir)
    except:
        shutil.rmtree(outdir)
        os.mkdir(outdir)
comm.Barrier()

# partition an iterable i into n parts
_split = lambda i,n: [i[:len(i)/n]]+_split(i[len(i)/n:],n-1) if n != 0 else []

with open('grid.yaml', 'r') as f:
    config = yaml.load(f.read())

grid = []
for n in config['nickel']:
    for e in config['energy']:
        for h in config['hhe']:
            for he in config['he']:
                for o in config['ox']:
                    grid.append((e, n, o, he, h))

my_jobs = _split(grid, size)[rank]
layers = s.heger_s15_layers()[1:]
delta = -config['alpha']
n = -config['beta']
zeta_v = np.sqrt(2 * (5 - delta) * (n - 5) / ((3 - delta) * (n - 3)))
mixer = s.DiffusionMixer(config['mixing'])

for job in my_jobs:
    masses = job[1:]
    vt = 6e3 * zeta_v * np.sqrt(job[0] * 2 * 1.38 / sum(masses)) # km/s
    profile = s.BrokenPowerLaw(config['alpha'], config['beta'], vt)
    atm = s.StratifiedAtmosphere(layers, masses, profile,
                                 thermal_energy=job[0] * 1e51,
                                 thermal_profile=profile,
                                 v_outer=config['v_outer'],
                                 nzones=config['nzones'])
    ma = mixer(atm)
    fig, axarr = ma.plot(show=False, thermal=True)
    name = os.path.join(outdir, '_'.join(['%0.2e'] * 5) % job)

    ma.write(name + '.mod')
    fig.savefig(name + '.pdf')
    plt.close(fig)
