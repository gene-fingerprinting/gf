import const
import argparse

from extraction import load_dataset
from Model_DF import DF

mode = const.MODE_OW

# Hyperparameters
input_size = 5000           # [500 ... 7000]
quantile = 0.9              # [0 0.05 ... 1]
num_classes = const.MON_NUM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='OpenWorld_DF', description='Evaluation program of Deep Fingerprinting Attack in Open-World')
    parser.add_argument('-s', type=int, metavar='[500 ... 7000]', default=None,
                        help='input feature size.')
    parser.add_argument('-q', type=float, metavar='[0 0.05 ... 1]', default=None,
                        help='quantile for threshold.')
    args = parser.parse_args()
    input_size = input_size if args.s is None else args.s
    quantile = quantile if args.q is None else args.q

    # Train on MS-ETH data
    model = DF(input_size, num_classes, mode, quantile)
    print('Training ETH')
    X_eth, y_eth = load_dataset('MS-ETH', 'DF')
    model.train(X_eth, y_eth, verbose=True)

    # Test on tunnel data
    print('Testing SSH')
    X_ssh, y_ssh = load_dataset('MS-SSH', 'DF')
    X_ssh_u, y_ssh_u = load_dataset('UMS-SSH', 'DF')
    model.test(X_ssh + X_ssh_u, y_ssh + y_ssh_u)

    print('Testing SS')
    X_ss, y_ss = load_dataset('MS-SS', 'DF')
    X_ss_u, y_ss_u = load_dataset('UMS-SS', 'DF')
    model.test(X_ss + X_ss_u, y_ss + y_ss_u)

    print('Testing VPN')
    X_vpn, y_vpn = load_dataset('MS-VPN', 'DF')
    X_vpn_u, y_vpn_u = load_dataset('UMS-VPN', 'DF')
    model.test(X_vpn + X_vpn_u, y_vpn + y_vpn_u)
