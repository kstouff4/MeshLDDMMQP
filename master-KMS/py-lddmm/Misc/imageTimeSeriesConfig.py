import logging
import numpy
from . import diffeomorphisms, regularGrid
import sys
import os

compute_path = os.environ["PYLDDMM_COMPUTE_PATH"]
log_file_name = "imageTimeSeries.log"
compute_output_dir = compute_path + "/output/time_series/"
lung_image_dir = compute_path + "/input/time_series/lung/"
biocard_image_dir = compute_path + "/input/time_series/biocard/"
multiproc_pool_size = 16
multiproc_pool_timeout = 5000
file_write_iter = 20

def configure(sim, config_name):
    sim.config_name = config_name
    modname = globals()['__name__']
    module = sys.modules[modname]
    method = getattr(module, config_name)
    method(sim)

def lung(sim):
    sim.dim = 3
    sim.num_target_times = 4
    sim.num_times_disc = 10
    sim.num_times = sim.num_times_disc * sim.num_target_times + 1
    sim.times = numpy.linspace(0, 1, sim.num_times)
    sim.dt = 1. / (sim.num_times - 1)
    sim.sigma = 1.
    sim.alpha = 7. #5.
    sim.gamma = 1.
    sim.Lpower = 2.
    sim.num_points = numpy.array([256,190,160])
    sim.dx = numpy.array([1.,1.,1.])
    sim.domain_max = None
    sim.cg_init_eps = 1e-3
    sim.rg = regularGrid.RegularGrid(sim.dim, \
                                     num_points=sim.num_points, \
                                     domain_max=sim.domain_max, \
                                     dx=sim.dx, mesh_name="lddmm")

    sim.I = numpy.zeros((sim.rg.num_nodes, sim.num_times))
    sim.I_interp = numpy.zeros((sim.rg.num_nodes, sim.num_times))
    sim.p = numpy.zeros((sim.rg.num_nodes, sim.num_times))

    sim.sc = diffeomorphisms.gridScalars()
    sim.sc.loadAnalyze(lung_image_dir + "ic007.hdr")
    sim.I[:,0.*sim.num_times_disc] = sim.sc.data.reshape(sim.rg.num_nodes)
    sim.sc.loadAnalyze(lung_image_dir + "ic009.hdr")
    sim.I[:,1.*sim.num_times_disc] = sim.sc.data.reshape(sim.rg.num_nodes)
    sim.sc.loadAnalyze(lung_image_dir + "ic011.hdr")
    sim.I[:,2.*sim.num_times_disc] = sim.sc.data.reshape(sim.rg.num_nodes)
    sim.sc.loadAnalyze(lung_image_dir + "ic013.hdr")
    sim.I[:,3.*sim.num_times_disc] = sim.sc.data.reshape(sim.rg.num_nodes)
    sim.sc.loadAnalyze(lung_image_dir + "ic015.hdr")
    sim.I[:,4.*sim.num_times_disc] = sim.sc.data.reshape(sim.rg.num_nodes)

    for t in range(sim.num_target_times):
        sim.rg.create_vtk_sg()
        sim.rg.add_vtk_point_data(sim.I[:,t*sim.num_times_disc], "I")
        sim.rg.vtk_write(t, "targets", output_dir=sim.output_dir)

    sim.pool_size = multiproc_pool_size
    sim.pool_timeout =  multiproc_pool_timeout
    sim.write_iter = file_write_iter
    sim.verbose_file_output = False

    logging.info("lung data image parameters: ")
    logging.info("dimension: %d" % (sim.dim))
    logging.info("num_points: %s" % (str(sim.rg.num_points)))
    logging.info("domain_max: %s" % (str(sim.rg.domain_max)))
    logging.info("dx: %s" % (str(sim.rg.dx)))
    logging.info("sigma: %f" % (sim.sigma))
    logging.info("dt: %f" % (sim.dt))
    logging.info("kernel params- alpha: %f, gamma: %f, Lpower: %f" % \
                        (sim.alpha, sim.gamma, sim.Lpower))

def biocard(sim):
    sim.dim = 3
    sim.num_target_times = 1
    sim.num_times_disc = 10
    sim.num_times = sim.num_times_disc * sim.num_target_times + 1
    sim.times = numpy.linspace(0, 1, sim.num_times)
    sim.dt = 1. / (sim.num_times - 1)
    sim.sigma = (1./255.)
    sim.alpha = .01
    sim.gamma = 1.
    sim.Lpower = 2.
    sim.num_points = numpy.array([40, 32, 40])
    sim.dx = numpy.array([.9375, 2., .9375])
    sim.domain_max = None
    sim.cg_init_eps = 1e-3
    sim.rg = regularGrid.RegularGrid(sim.dim, \
                                     num_points=sim.num_points, \
                                     domain_max=sim.domain_max, \
                                     dx=sim.dx, mesh_name="lddmm")
    sim.I = numpy.zeros((sim.rg.num_nodes, sim.num_times))
    sim.I_interp = numpy.zeros((sim.rg.num_nodes, sim.num_times))
    sim.p = numpy.zeros((sim.rg.num_nodes, sim.num_times))

    sim.sc = diffeomorphisms.gridScalars()
    sim.sc.loadAnalyze(biocard_image_dir + "regR2_cut.hdr")
    sim.I[:,0.*sim.num_times_disc] = sim.sc.data.reshape(sim.rg.num_nodes).copy()
    sim.sc.loadAnalyze(biocard_image_dir + "regR3_cut.hdr")
    sim.I[:,1.*sim.num_times_disc] = sim.sc.data.reshape(sim.rg.num_nodes).copy()
    #sim.sc.loadAnalyze(biocard_image_dir + "regR4_cut.hdr")
    #sim.I[:,2.*sim.num_times_disc] = sim.sc.data.reshape(sim.rg.num_nodes)
    #sim.sc.loadAnalyze(biocard_image_dir + "regR5_cut.hdr")
    #sim.I[:,3.*sim.num_times_disc] = sim.sc.data.reshape(sim.rg.num_nodes)

    sim.pool_size = multiproc_pool_size
    sim.pool_timeout =  multiproc_pool_timeout
    sim.write_iter = file_write_iter
    sim.verbose_file_output = False

    logging.info("Biocard image parameters: ")
    logging.info("dimension: %d" % (sim.dim))
    logging.info("num_points: %s" % (str(sim.rg.num_points)))
    logging.info("domain_max: %s" % (str(sim.rg.domain_max)))
    logging.info("dx: %s" % (str(sim.rg.dx)))
    logging.info("sigma: %f" % (sim.sigma))
    logging.info("dt: %f" % (sim.dt))
    logging.info("kernel params- alpha: %f, gamma: %f, Lpower: %f" % \
                        (sim.alpha, sim.gamma, sim.Lpower))

def lung_downsample(sim):
    sim.dim = 3
    sim.num_target_times = 5
    sim.num_times_disc = 10
    sim.num_times = sim.num_times_disc * sim.num_target_times + 1
    sim.times = numpy.linspace(0, 1, sim.num_times)
    sim.dt = 1. / (sim.num_times - 1)
    sim.sigma = .1
    sim.num_points_data = numpy.array([256, 184, 160])
    sim.mults = numpy.array([2,2,2]).astype(int)
    sim.num_points = (sim.num_points_data/sim.mults).astype(int)
    sim.dx_data = (1., 1., 1.)
    sim.domain_max_data = None
    sim.dx = None # (1.,1.,1.)
    sim.domain_max = numpy.array([128., 92., 80.])
    sim.gradEps = 1e-8
    sim.rg = regularGrid.RegularGrid(sim.dim, \
                                     num_points=sim.num_points, \
                                     domain_max=sim.domain_max, \
                                     dx=sim.dx, mesh_name="lddmm")
    sim.rg_data = regularGrid.RegularGrid(sim.dim, \
                                          num_points=sim.num_points_data, \
                                          domain_max=sim.domain_max_data, \
                                          dx=sim.dx_data, mesh_name="lddmm_data")

    sim.I = numpy.zeros((sim.rg.num_nodes, sim.num_times))
    sim.I_interp = numpy.zeros((sim.rg.num_nodes, sim.num_times))
    sim.p = numpy.zeros((sim.rg.num_nodes, sim.num_times))

    sc = diffeomorphisms.gridScalars()
    sc.loadAnalyze(lung_image_dir + "test007.hdr")
    sim.I[:,0.*sim.num_times_disc] = sim.apply_sync_filter( \
                        sc.data.reshape(sim.rg_data.num_nodes),sim.mults)
    sc.loadAnalyze(lung_image_dir + "test009.hdr")
    sim.I[:,1.*sim.num_times_disc] = sim.apply_sync_filter( \
                        sc.data.reshape(sim.rg_data.num_nodes),sim.mults)
    sc.loadAnalyze(lung_image_dir + "test011.hdr")
    sim.I[:,2.*sim.num_times_disc] = sim.apply_sync_filter( \
                        sc.data.reshape(sim.rg_data.num_nodes),sim.mults)
    sc.loadAnalyze(lung_image_dir + "test013.hdr")
    sim.I[:,3.*sim.num_times_disc] = sim.apply_sync_filter( \
                        sc.data.reshape(sim.rg_data.num_nodes),sim.mults)
    sc.loadAnalyze(lung_image_dir + "test015.hdr")
    sim.I[:,4.*sim.num_times_disc] = sim.apply_sync_filter( \
                        sc.data.reshape(sim.rg_data.num_nodes),sim.mults)

    sim.I /= 255.

    for t in range(sim.num_target_times):
        sim.rg.create_vtk_sg()
        sim.rg.add_vtk_point_data(sim.I[:,t*sim.num_times_disc], "I")
        sim.rg.vtk_write(t, "targets", output_dir=sim.output_dir)

    sim.pool_size = multiproc_pool_size
    sim.pool_timeout =  multiproc_pool_timeout
    sim.write_iter = file_write_iter
    sim.verbose_file_output = False

    logging.info("lung data image parameters: ")
    logging.info("dimension: %d" % (sim.dim))
    logging.info("num_points: %s" % (str(sim.rg.num_points)))
    logging.info("domain_max: %s" % (str(sim.rg.domain_max)))
    logging.info("dx: %s" % (str(sim.rg.dx)))
    logging.info("sigma: %f" % (sim.sigma))
    logging.info("dt: %f" % (sim.dt))

def test3d(sim):
    sim.dim = 3
    sim.num_target_times = 1
    sim.num_times_disc = 10
    sim.num_times = sim.num_times_disc * sim.num_target_times + 1
    sim.times = numpy.linspace(0, 1, sim.num_times)
    sim.dt = 1. / (sim.num_times - 1)
    sim.sigma = 1
    sim.alpha = .01
    sim.gamma = 1.
    sim.Lpower = 2.
    sim.num_points = (40, 32, 40)
    sim.dx = (.9375, 2., .9375)
    sim.domain_max = None
    sim.gradEps = 1e-8
    sim.rg = regularGrid.RegularGrid(sim.dim, \
                                     num_points=sim.num_points, \
                                     domain_max=sim.domain_max, \
                                     dx=sim.dx, mesh_name="lddmm")
    sim.I = numpy.zeros((sim.rg.num_nodes, sim.num_times))
    sim.I_interp = numpy.zeros((sim.rg.num_nodes, sim.num_times))
    sim.p = numpy.zeros((sim.rg.num_nodes, sim.num_times))
    for t in range(sim.num_times):
        loc = -1. + 0 * sim.dt * (t -(t % sim.num_times_disc))
        x_sqr = numpy.power(sim.rg.nodes[:,0]-loc, 2)
        y_sqr = numpy.power(sim.rg.nodes[:,1], 2)
        z_sqr = numpy.power(sim.rg.nodes[:,2], 2)
        r = 13 - 8 * sim.dt * (t - (t % sim.num_times_disc))
        nodes = numpy.where(x_sqr + y_sqr + z_sqr < r**2)[0]
        sim.I[nodes, t] = 50.
    logging.info("Test 3d parameters: ")
    logging.info("dimension: %d" % (sim.dim))
    logging.info("num_points: %s" % (str(sim.rg.num_points)))
    logging.info("domain_max: %s" % (str(sim.rg.domain_max)))
    logging.info("dx: %s" % (str(sim.rg.dx)))
    logging.info("sigma: %f" % (sim.sigma))
    logging.info("dt: %f" % (sim.dt))

def monkey(sim):
    sim.dim = 2
    sim.num_target_times = 1
    sim.num_times_disc = 10
    sim.num_times = sim.num_times_disc * sim.num_target_times + 1
    sim.times = numpy.linspace(0, 1, sim.num_times)
    sim.dt = 1. / (sim.num_times - 1)
    sim.sigma = 1.
    sim.alpha = .002
    sim.gamma = 1.
    sim.Lpower = 2.
    sim.num_points = numpy.array([80,80])
    sim.domain_max = numpy.array([.5, .5])
    #sim.dx = numpy.array([1.,1.,1.])
    sim.dx = None
    sim.rg = regularGrid.RegularGrid(sim.dim, \
                                     num_points=sim.num_points, \
                                     domain_max=sim.domain_max, \
                                     dx=sim.dx, mesh_name="lddmm")

    sim.I = numpy.zeros((sim.rg.num_nodes, sim.num_times))
    sim.I_interp = numpy.zeros((sim.rg.num_nodes, sim.num_times))
    sim.p = numpy.zeros((sim.rg.num_nodes, sim.num_times))

    sim.sc = diffeomorphisms.gridScalars()
    sim.sc.loadAnalyze("/cis/home/clr/cawork/monkey/monkey1_80_80.hdr")
    sim.I[:,0.*sim.num_times_disc] = sim.sc.data.reshape(sim.rg.num_nodes)
    sim.sc.loadAnalyze("/cis/home/clr/cawork/monkey/monkey2_80_80.hdr")
    sim.I[:,1.*sim.num_times_disc] = sim.sc.data.reshape(sim.rg.num_nodes)

    for t in range(sim.num_target_times):
        sim.rg.create_vtk_sg()
        sim.rg.add_vtk_point_data(sim.I[:,t*sim.num_times_disc], "I")
        sim.rg.vtk_write(t, "targets", output_dir=sim.output_dir)

    logging.info("monkey data image parameters: ")
    logging.info("dimension: %d" % (sim.dim))
    logging.info("num_points: %s" % (str(sim.rg.num_points)))
    logging.info("domain_max: %s" % (str(sim.rg.domain_max)))
    logging.info("dx: %s" % (str(sim.rg.dx)))
    logging.info("sigma: %f" % (sim.sigma))
    logging.info("dt: %f" % (sim.dt))
