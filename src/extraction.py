import sys
import const
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join, exists

from utility import save_obj, load_obj
from features import decaps_packets
from features import gf_features
from features import cumul_features
from features import df_features
from features import ddtw_features


def extract_features(fname, tunnel, feature_function):
    packets = pd.read_csv(fname)
    packets = decaps_packets(packets, tunnel)
    features = feature_function(packets)

    return features

def extract_monitored(path, tunnel, feature_function):
    X = []
    y = []
    if tunnel == 'ETH':
        num_samples = const.ETH_SAMPLE_N
    else:
        num_samples = const.TUN_SAMPLE_N
    for site in tqdm(range(const.MON_NUM), desc='%s-%s-%s' % (const.MON_PREFIX, tunnel, feature_function.__name__)):
        for sample in range(num_samples):
            fname = join(path, 'site-%d-sample-%d.csv' % (site, sample))
            features = extract_features(fname, tunnel, feature_function)

            X.append(np.array(features, dtype=float))
            y.append(site)

    return X, y

def extract_unmonitored(path, tunnel, feature_function):
    X = []
    y = []
    for site in tqdm(range(const.MON_NUM, const.MON_NUM + const.UMON_NUM), desc='%s-%s-%s' % (const.UMON_PREFIX, tunnel, feature_function.__name__)):
        fname = join(path, 'site-%d.csv' % site)
        features = extract_features(fname, tunnel, feature_function)
        
        X.append(np.array(features, dtype=float))
        y.append(site)

    return X, y

def extract(dataset, attack):
    path = join(const.DATA_DIR, dataset)
    if not exists(path):
        raise Exception('Cannot find %s' % path)

    if 'GF' in attack:
        feature_function = gf_features
    elif 'CUMUL' in attack:
        feature_function = cumul_features
    elif 'DF' in attack:
        feature_function = df_features
    elif 'DDTW' in attack:
        feature_function = ddtw_features
    else:
        raise Exception('Unsupported attack %s' % attack)

    prefix, tunnel = dataset.split('-')
    if prefix == const.MON_PREFIX:
        X, y = extract_monitored(path, tunnel, feature_function)
    elif prefix == const.UMON_PREFIX:
        X, y = extract_unmonitored(path, tunnel, feature_function)
    else:
        raise Exception('Unknown prefix %s' % prefix)

    save_obj(X, join(const.DATA_DIR, 'X_%s_%s.pkl' % (dataset, attack)))
    save_obj(y, join(const.DATA_DIR, 'y_%s.pkl' % dataset))

def load_dataset(dataset, attack):
    X_file = join(const.DATA_DIR, 'X_%s_%s.pkl' % (dataset, attack))
    y_file = join(const.DATA_DIR, 'y_%s.pkl' % dataset)

    if not exists(X_file) or not exists(y_file):
        extract(dataset, attack)

    X = load_obj(X_file)
    y = load_obj(y_file)

    return X, y

def extract_attack(attack):
    extract('MS-ETH', attack)
    extract('MS-SSH', attack)
    extract('MS-SS', attack)
    extract('MS-VPN', attack)
    extract('UMS-SSH', attack)
    extract('UMS-SS', attack)
    extract('UMS-VPN', attack)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='extraction', description='Website Fingerprinting Feature Extraction')
    parser.add_argument('attack', type=str,
                        help='attack, can be one of GF, CUMUL, DF, DDTW.')
    parser.add_argument('-d', type=str, metavar='DATASET', default=None,
                        help='traffic dataset, can be one of MS-ETH, MS-SSH, MS-SS, MS-VPN, UMS-SSH, UMS-SS, UMS-VPN.')
    
    if len(sys.argv) == 1:
        extract_attack('GF')
        extract_attack('CUMUL')
        extract_attack('DF')
        extract_attack('DDTW')
        
    else:
        args = parser.parse_args()
        if args.d is None:
            extract_attack(args.attack)
        else:
            extract(args.d, args.attack)
