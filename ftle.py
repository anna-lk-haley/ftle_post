import sys
import os
from pathlib import Path
import numpy as np
#import scipy.linalg as la
import math
import h5py
import pyvista as pv
from mpi4py import MPI
import vtk
from scipy.spatial import cKDTree as kdtree
import DMD
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#mpiexec -np 40 python ftle.py ../mesh_rez/cases/case_A/case_028_low/results/art_*

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def lyapunov(gradient):
    CG=gradient.T@gradient
    vals, _ = np.linalg.eig(CG)
    return np.max(vals)

if __name__=="__main__":
    results=sys.argv[1] #eg. case_043_low/results/art_*
    dd = DMD.Dataset(Path(results))
    dd = dd.assemble_mesh()
    outfolder='FTLE_files'
    if not Path(outfolder).exists():
        Path(outfolder).mkdir(parents=True, exist_ok=True)

    surf = pv.read('spectro_sigmoid.vtp')
    times = len(dd.up_files)
    time_chunk = 300 #tsteps all have to be the same length
    #the time chunks have to overlap
    remainder = times - time_chunk
    #divide up the remainder to get the start time for each chunk
    shift = math.floor(remainder/size) #this may not end up being the entire cycle
    p_shift = rank*shift
    tsteps = range(p_shift,p_shift+time_chunk)
    dt = dd._get_time(dd.up_files[1])-dd._get_time(dd.up_files[0])
    T = dt*time_chunk
    
    surf.points=surf.points*1e-3
    initial = dd.mesh.select_enclosed_points(surf)
    inside = initial.threshold(0.5, scalars='SelectedPoints')
    X = inside
    x = X.copy() #this is our flow map
    if rank ==0:
        print('Beginning Integration!')
    for idx in tsteps:    
        # get velocity at tstep and integrate
        dd.mesh.point_data['v'] = dd(idx)
        x0 = x.sample(dd.mesh)
        v = x0.point_data['v']
        #check if no longer inside the field
        v[x0.point_data['vtkValidPointMask']==0] = 0 #this point is no longer in the big domain so set to zero
        x.points += v*dt #new position
        if (rank ==0) and (idx%10==0):
            print('{}% done integrating!'.format(round(100*idx/time_chunk)))
    #place these values back onto the initial grid and take derivative
    X.point_data['flow_map']=x.points
    deriv = X.compute_derivative('flow_map') #compute flow map derivative
    #dd.mesh.point_data['flow_map'][ids]=gradient.point_data['gradient'] #put it into big mesh again
    
    #get Cauchy-Green tensor
    if rank==0:
        print('Computing FTLE field')
    gradient = deriv['gradient'].reshape((-1,3,3))
    lam=np.zeros((len(X.points)))
    for ndx in range(len(X.points)):
        lam = lyapunov(gradient[ndx])
        X.point_data['ftle_field']=1/T*np.log(np.sqrt(lam))
    with h5py.File(outfolder+'/ftle_{}.h5'.format(rank),'w') as f:
        f.create_dataset('ftle_field', data = X.point_data['ftle_field'])

    


        

        
