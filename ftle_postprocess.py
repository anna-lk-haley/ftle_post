import sys
import os
from pathlib import Path
import numpy as np
import math
import h5py
import pyvista as pv
import DMD
import imageio
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#~/xvfb-run-safe python ftle_postprocess.py ../mesh_rez/cases/case_A/case_028_low/results/art_*

def images_to_movie(imgs, outpath, fps=30):
    """ Write images to movie.
    Format inferred from outpath extension, 
    see imageio docs.
    """
    writer = imageio.get_writer(outpath, format='FFMPEG',fps=fps)

    for im in imgs:
        writer.append_data(imageio.imread(im))
    writer.close()

results=sys.argv[1] #eg. case_043_low/results/art_*
dd = DMD.Dataset(Path(results))
dd = dd.assemble_mesh()
outfolder='FTLE_files'
output_folder = outfolder+'/animation'

surf = pv.read('spectro_sigmoid.vtp')
surf.points=surf.points*1e-3
initial = dd.mesh.select_enclosed_points(surf)
X = initial.threshold(0.5, scalars='SelectedPoints') #the enclosed mesh
p=pv.Plotter(off_screen=True, window_size=[768, 768])
p.camera_position = [(-36.81804943201743e-3, -48.491128854376726e-3, 0.3239806443238584e-3),
 (13.994269117949933e-3, -20.242262123498968e-3, -0.06966475995984878e-3),
 (-0.050441468179376045e-3, 0.10457167339154855e-3, 0.993237344954367e-3)]
#p.add_mesh(X)
#p.show(screenshot=output_folder + '/mesh.png', auto_close=False)
#p.close()
#X.save('mesh_ftle.vtu')

silhouette = dict(color='black', line_width=3.0, decimate=None)

if not Path(output_folder).exists():    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for idx in range(80):
        actors = [x for x in p.renderer.actors.keys()]
        for a in actors:
            p.remove_actor(actors)
        with h5py.File(outfolder+'/ftle_{}.h5'.format(idx), 'r') as f:
            X.point_data['ftle_field']=np.array(f['ftle_field'])
        
        p.add_mesh(dd.mesh,
        color='w',
        opacity=1,
        show_scalar_bar=False,
        lighting=True,
        smooth_shading=True,
        specular=0.00,
        diffuse=0.9,
        ambient=0.5,
        culling='front',
        silhouette=silhouette,
        name='surf',
        )
        
        p.add_mesh(X.slice(origin=(0.046355046148334106,-0.01818768590686787,-0.005222244831179413),normal=(0.5247939564414609,0.8510598713689062,0.016982303379083272)), scalars='ftle_field')
        p.show(screenshot=output_folder + '/ftle_{:04d}.png'.format(idx), auto_close=False)
    p.close()
imgs = sorted(Path(output_folder).glob('*.png'))
images_to_movie(imgs, output_folder + '/ftle.mp4', fps=5)

