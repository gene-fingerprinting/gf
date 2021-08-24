import const
import argparse

from extraction import load_dataset
from Model_CUMUL import CUMUL

mode = const.MODE_OW

# Hyperparameters
C = 2**17           # [2**11 2**13 ... 2**17]
gamma = 2**3        # [2**-3 2**-1 ... 2**3]
quantile = 0.9      # [0 0.05 ... 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='OpenWorld_CUMUL', description='Evaluation program of CUMUL Attack in Open-World')
    parser.add_argument('-c', type=int, metavar='2^[11 13 ... 17]', default=None,
                        help='parameter C for SVC.')
    parser.add_argument('-g', type=float, metavar='2^[-3 -1 ... 3]', default=None,
                        help='parameter gamma for SVC.')
    parser.add_argument('-q', type=float, metavar='[0 0.05 ... 1]', default=None,
                        help='quantile for threshold.')
    args = parser.parse_args()
    C = C if args.c is None else args.c
    gamma = gamma if args.g is None else args.g
    quantile = quantile if args.q is None else args.q

    # Train on MS-ETH data
    model = CUMUL(C, gamma, mode, quantile)
    print('Training ETH')
    X_eth, y_eth = load_dataset('MS-ETH', 'CUMUL')
    model.train(X_eth, y_eth)
    
    # Test on tunnel data
    print('Testing SSH')
    X_ssh, y_ssh = load_dataset('MS-SSH', 'CUMUL')
    X_ssh_u, y_ssh_u = load_dataset('UMS-SSH', 'CUMUL')
    model.test(X_ssh + X_ssh_u, y_ssh + y_ssh_u)

    print('Testing SS')
    X_ss, y_ss = load_dataset('MS-SS', 'CUMUL')
    X_ss_u, y_ss_u = load_dataset('UMS-SS', 'CUMUL')
    model.test(X_ss + X_ss_u, y_ss + y_ss_u)

    print('Testing VPN')
    X_vpn, y_vpn = load_dataset('MS-VPN', 'CUMUL')
    X_vpn_u, y_vpn_u = load_dataset('UMS-VPN', 'CUMUL')
    model.test(X_vpn + X_vpn_u, y_vpn + y_vpn_u)
