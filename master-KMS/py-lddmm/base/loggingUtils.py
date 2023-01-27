import logging
import os

def setup_default_logging(output_dir=None, config=None, fileName=None, stdOutput=True):
    logger = logging.getLogger()
    #print 'Before', logger.handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    #print 'After', logger.handlers
    logger.setLevel(logging.INFO)
    #formatter = logging.Formatter("%(asctime)s-%(levelname)s: %(message)s")
    formatter = logging.Formatter("%(message)s")
    if config == None:
        log_file_name = fileName
    else:
        log_file_name = config.log_file_name
    if output_dir == None:
        output_dir = ""
    if fileName != None:
        if not os.access(output_dir, os.W_OK):
            if os.access(output_dir, os.F_OK):
                logging.error('Cannot save in ' + output_dir)
                return
            else:
                os.makedirs(output_dir)
        fh = logging.FileHandler("%s/%s" % (output_dir, log_file_name), mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if stdOutput:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
