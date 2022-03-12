from logging import Formatter


LOG_DIR = '../logs/'
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')
