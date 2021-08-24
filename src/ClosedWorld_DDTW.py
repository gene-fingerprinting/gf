import const
import argparse

from extraction import load_dataset
from Model_DDTW import DDTW

mode = const.MODE_CW

# Hyperparameters
exemplar = 3        # [1 ... 10]
K = 1               # [1, 3, 5]
window = 0.2        # [0.1 0.2 ... 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ClosedWorld_DDTW', description='Evaluation program of Time DDTW Attack in Closed-World')
    parser.add_argument('-e', type=int, metavar='[1 2 ... 10]', default=None,
                        help='number of exemplars.')
    parser.add_argument('-k', type=int, metavar='[1 3 5]', default=None,
                        help='k-Nearest Neighbor.')
    parser.add_argument('-w', type=float, metavar='[0.1 0.2 ... 1]', default=None,
                        help='window scale for warping path.')
    args = parser.parse_args()
    exemplar = exemplar if args.e is None else args.e
    K = K if args.k is None else args.k
    window = window if args.w is None else args.w

    # Train on MS-ETH data
    model = DDTW(exemplar, K, window, mode, n_jobs=2)
    print('Training ETH')
    X_eth, y_eth = load_dataset('MS-ETH', 'DDTW')
    model.train(X_eth, y_eth)

    # Test on tunnel data
    print('Testing SSH')
    X_ssh, y_ssh = load_dataset('MS-SSH', 'DDTW')
    model.test(X_ssh, y_ssh)

    print('Testing SS')
    X_ss, y_ss = load_dataset('MS-SS', 'DDTW')
    model.test(X_ss, y_ss)

    print('Testing VPN')
    X_vpn, y_vpn = load_dataset('MS-VPN', 'DDTW')
    model.test(X_vpn, y_vpn)
    