import numpy as np
import h5py
from . import surfaces
from . import curves as crvs
from . import surfaceSection

def readTagData(filename, contoursOnly=False, select_sections_tag = None):
    f = h5py.File(filename, 'r')
    fkeys = list(f.keys())
    if not contoursOnly:
        if 'Template' in fkeys:
            vert = np.array(f['Template']['Vertices']).T
            faces = np.array(f['Template']['Faces'], dtype=int).T -1
            Template = surfaces.Surface(surf=(faces, vert))
        else:
            Template = None

        if 'Tagplane' not in fkeys or 'Curves' not in fkeys:
            print('Missing information in ' + filename)
            return None, None, None

        tag_pl = np.array(f['Tagplane']['Corners']).T
        npl = tag_pl.shape[0]
        if select_sections_tag is not None:
            saxal = False
            select_plane = np.zeros(npl, dtype=bool)
            for a in select_sections_tag:
                if 'SAX' in a:
                    saxal = True
                    break
            k = 0
            for i in range(npl):
                labf = f['Tagplane']['Label'][i]
                if type(labf) in (bytes, np.bytes_):
                    labf = labf.decode()
                if ( labf == 'SAXAL' and saxal) or labf in select_sections_tag:
                    tag_pl[k, :] = tag_pl[i, :]
                    select_plane[i] = True
                    k += 1
            tag_pl = tag_pl[:k, :]
        else:
            select_plane = np.ones(npl, dtype=bool)



        g = f['Curves']
        phases = list(g.keys())
        curves = {}
        select_img = np.ones(len(g[phases[0]].keys()), dtype=bool)
        if select_sections_tag is not None:
            for k,im in enumerate(g[phases[0]].keys()):
                if im not in select_sections_tag:
                    select_img[k] = False
        for ph in phases:
            if select_sections_tag is not None:
                images = list(set(g[ph].keys()) & set(select_sections_tag))
            else:
                images = list(g[ph].keys())
            nim = len(images)
            curve_in_phase = {s: None for s in images}
            for im in images:
                h = g[ph][im]
                curve_in_image = {}
                #curve_in_image = {'Intersections' + str(k): None for k in range(npl)}
                intersections = list(h.keys())
                # if len(intersections) != npl:
                #     print('A possibly empty intersection must be provided for each tag plane')
                #     return None, None, None
                for ins in intersections:
                    if len(h[ins]) > 0:
                        vert = np.array(h[ins]['Vertices']).T
                        faces = np.array(h[ins]['Edge'], dtype=int).T -1
                        curve_in_image[ins] = crvs.Curve(curve=(faces,vert))
                curve_in_phase[im] = curve_in_image
            curves[ph] = curve_in_phase

        activity = np.array(f['ActiveMatrix'])[select_plane, :].astype(bool)
        activity = activity[:, select_img]

    epi_contours = []
    endo_contours = []
    if 'Epi' in fkeys:
        if select_sections_tag is not None:
            images = list(set(f['Epi'].keys()) & set(select_sections_tag))
        else:
            images = list(f['Epi'].keys())
        if f['Epi'][images[0]].ndim == 2:
            nphases = 1
        else:
            nphases = f['Epi'][images[0]].shape[0]
        epi_contours = [[] for i in range(nphases)]
        for img in f['Epi']:
            allc = np.array(f['Epi'][img])
            if allc.ndim == 2:
                allc = allc[None, :, :]
            for i in range(allc.shape[0]):
                c = crvs.Curve(curve=allc[i,:,:].T)
                c = surfaceSection.SurfaceSection(curve=c)
                epi_contours[i].append(c)
    if 'Endo' in fkeys:
        if select_sections_tag is not None:
            images = list(set(f['Endo'].keys()) & set(select_sections_tag))
        else:
            images = list(f['Endo'].keys())
        if f['Endo'][images[0]].ndim == 2:
            nphases = 1
        else:
            nphases = f['Epi'][images[0]].shape[0]
        endo_contours = [[] for i in range(nphases)]
        for img in f['Endo']:
            allc = np.array(f['Endo'][img])
            if allc.ndim == 2:
                allc = allc[None, :, :]
            for i in range(allc.shape[0]):
                c = crvs.Curve(curve=allc[i,:,:].T)
                c = surfaceSection.SurfaceSection(curve=c)
                endo_contours[i].append(c)

    contours = {}
    contours['Epi'] = epi_contours
    contours['Endo'] = endo_contours

    if contoursOnly:
        return contours
    else:
        return Template, tag_pl, curves, activity, contours

def writeTagData(filename, tag_pl, curves, Template = None):
    f = h5py.File(filename, 'w')
    if Template is not None:
        f.create_group('Template')
        f['Template'].create_dataset('Vertices', data=Template.vertices)
        f['Template'].create_dataset('Faces', data=Template.faces)

    f.create_dataset('Tag_planes', data=tag_pl)

    f.create_group('Curves')
    g = f['curves']
    for k,ph in enumerate(curves):
        g.create_group(f'Phase{k+1}')
        h = g[f'Phase{k+1}']
        for l,inter in enumerate(ph):
            if inter is not None:
                h.create_dataset('Vertices', data=inter.vertices)
                h.create_dataset('Edges', data=inter.faces)


