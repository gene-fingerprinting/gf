from os import mkdir
from os.path import exists, join, abspath, dirname, pardir

BASE_DIR = abspath(join(dirname(__file__), pardir))

SRC_DIR = join(BASE_DIR, 'src')

DATA_DIR = join(BASE_DIR, 'data')
if not exists(DATA_DIR):
    mkdir(DATA_DIR)
MON_PREFIX = 'MS'
MON_NUM = 100
ETH_SAMPLE_N = 10
TUN_SAMPLE_N = 10
UMON_PREFIX = 'UMS'
UMON_NUM = 1709

MODEL_DIR = join(BASE_DIR, 'model')
if not exists(MODEL_DIR):
    mkdir(MODEL_DIR)
MODE_CW = 'ClosedWorld'
MODE_OW = 'OpenWorld'

RESULT_DIR = join(BASE_DIR, 'result')
if not exists(RESULT_DIR):
    mkdir(RESULT_DIR)
    