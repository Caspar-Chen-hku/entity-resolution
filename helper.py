import logging, logging.config
import os, sys, pathlib
import argparse, json, codecs

from collections import defaultdict as ddict

### system level help functions ###
def check_file(filename):
    return pathlib.Path(filename).is_file()

def get_logger(name):
    config_dict = json.load(open('config/log_config.json'))

    if os.path.isdir('log') == False:
        os.system('mkdir {}'.format('log'))

    config_dict['handlers']['file_handler']['filename'] = 'log/' + name
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger

def invert_dict(my_map, struct='o2o'):
    inv_map = {}
    #one to one
    if struct == 'o2o':
        for k, v in my_map.items():
            inv_map[v] = k
    #many to one
    elif struct == 'm2o':
        for k, v in my_map.items():
            inv_map[v] = inv_map.get(v, [])
            inv_map[v].append(k)
    #many to one list
    elif struct == 'm2ol':
        for k, v in my_map.items():
            for ele in v:
                inv_map[ele] = inv_map.get(ele, [])
                inv_map[ele].append(k)
    #many to one set
    elif struct == 'm2os':
        for k, v in my_map.items():
            for ele in v:
                inv_map[ele] = inv_map.get(ele, set())
                inv_map[ele].add(k)
    return inv_map
