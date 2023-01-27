from base import loggingUtils
import numpy
import optparse
import os
import shutil
import logging
from base import imageTimeSeries, imageTimeSeriesConfig

def loadData_for_async(fbase, t):
    from tvtk.api import tvtk
    r = tvtk.XMLStructuredGridReader(file_name="%s%d.vts" % (fbase, t))
    r.update()
    v = numpy.array(r.output.point_data.get_array("v")).astype(float)
    I = numpy.array(r.output.point_data.get_array("I")).astype(float)
    I_interp = numpy.array(r.output.point_data.get_array("I_interp") \
                        ).astype(float)
    p = numpy.array(r.output.point_data.get_array("p")).astype(float)
    mu = numpy.array(r.output.point_data.get_array("mu")).astype(float)
    mu_state = numpy.array(r.output.point_data.get_array("mu")).astype(float)
    logging.info("reloaded time %d." % (t))
    return (v,I,I_interp,p,mu,mu_state)


if __name__ == "__main__":
    output_directory_base = imageTimeSeriesConfig.compute_output_dir
    # set options from command line
    parser = optparse.OptionParser()
    parser.add_option("-o", "--output_dir", dest="output_dir")
    parser.add_option("-c", "--config_name", dest="config_name")
    (options, args) = parser.parse_args()
    output_dir = output_directory_base + options.output_dir
    # remove any old results in the output directory
    if os.access(output_dir, os.F_OK):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    loggingUtils.setup_default_logging(output_dir, imageTimeSeriesConfig)
    logging.info(options)
    its = imageTimeSeries.ImageTimeSeries(output_dir, options.config_name)
    #its.reset()
    its.computeMatching()
    its.writeData("final")
