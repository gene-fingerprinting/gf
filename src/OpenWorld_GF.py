import const
import argparse

from extraction import load_dataset
from Model_GF import GF

mode = const.MODE_OW

# Hyperparameters
exemplar = 6        # [1 ... 10]
K = 1               # [1, 3, 5]
window = 0.6        # [0.1 0.2 ... 1]
quantile = 0.9      # [0 0.05 ... 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='OpenWorld_GF', description='Evaluation program of Gene Fingerprinting Attack in Open-World')
    parser.add_argument('-e', type=int, metavar='[1 2 ... 10]', default=None,
                        help='number of exemplars.')
    parser.add_argument('-w', type=float, metavar='[0.1 0.2 ... 1]', default=None,
                        help='window scale for warping path.')
    parser.add_argument('-q', type=float, metavar='[0 0.05 ... 1]', default=None,
                        help='quantile for threshold.')
    args = parser.parse_args()
    exemplar = exemplar if args.e is None else args.e
    window = window if args.w is None else args.w
    quantile = quantile if args.q is None else args.q

    # Train on MS-ETH data
    model = GF(exemplar, K, window, mode, quantile, n_jobs=2)
    print('Training ETH')
    X_eth, y_eth = load_dataset('MS-ETH', 'GF')
    model.train(X_eth, y_eth)

    # Test on tunnel data
    print('Testing SSH')
    X_ssh, y_ssh = load_dataset('MS-SSH', 'GF')
    X_ssh_u, y_ssh_u = load_dataset('UMS-SSH', 'GF')
    model.test(X_ssh + X_ssh_u, y_ssh + y_ssh_u)

    print('Testing SS')
    X_ss, y_ss = load_dataset('MS-SS', 'GF')
    X_ss_u, y_ss_u = load_dataset('UMS-SS', 'GF')
    model.test(X_ss + X_ss_u, y_ss + y_ss_u)

    print('Testing VPN')
    X_vpn, y_vpn = load_dataset('MS-VPN', 'GF')
    X_vpn_u, y_vpn_u = load_dataset('UMS-VPN', 'GF')
    model.test(X_vpn + X_vpn_u, y_vpn + y_vpn_u)
