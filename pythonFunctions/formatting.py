from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/master/py-lddmm/')
sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/master/py-lddmm/base')

from base.meshes import Mesh

##############################################################
def writePostDeform(origRescaledVTK, geomFeatureVTK):
    
    fo = Mesh(mesh=origRescaledVTK)
    fn = Mesh(mesh=geomFeatureVTK)
    
    print("Sizes of vertices should be same")
    print(fo.vertices.shape)
    print(fn.vertices.shape)
    
    fn.updateJacobianFactor(fo.volumes)
    fn.save(geomFeatureVTK.replace('.vtk','_updated.vtk'))
    return

def scale(origVTK, s=1e-3):
    
    fo = Mesh(mesh=origVTK)
    fo.updateVertices(fo.vertices*s)
    fo.save(origVTK.replace('.vtk','_scaled.vtk'))
    return

def updateRigid(rigidFeatsVTK,featsVTK):
    fr = Mesh(mesh=rigidFeatsVTK)
    ff = Mesh(mesh=featsVTK)
    
    ff.updateVertices(fr.vertices)
    ff.save(featsVTK.replace('.vtk','_rigidTransform.vtk'))
    return