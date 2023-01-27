import imageTimeSeries, imageTimeSeriesConfig
import optparse
import os
import shutil
import logging
from base import loggingUtils

output_directory_base = imageTimeSeriesConfig.compute_output_dir
# set options from command line
parser = optparse.OptionParser()
parser.add_option("-o", "--output_dir", dest="output_dir")
parser.add_option("-c", "--config_name", dest="config_name")
parser.add_option("-f", "--file_base", dest="fbase")
(options, args) = parser.parse_args()
output_dir = output_directory_base + options.output_dir
# remove any old results in the output directory
if os.access(output_dir, os.F_OK):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)
loggingUtils.setup_default_logging(output_dir, imageTimeSeriesConfig)
logging.info(options)
its = imageTimeSeries.ImageTimeSeries(output_dir, options.config_name)
its.loadData(options.fbase)
logging.info("Begin computing maps.")
its.computeMaps()
logging.info("Done.")
