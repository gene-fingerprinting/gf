import const
import argparse

from extraction import load_dataset
from Model_DF import DF

mode = const.MODE_CW

# Hyperparameters
input_size = 5000           # [500 ... 7000]
num_classes = const.MON_NUM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ClosedWorld_DF', description='Evaluation program of Deep Fingerprinting Attack in Closed-World')
    parser.add_argument('-s', type=int, metavar='[500 ... 7000]', default=None,
                        help='input feature size.')
    args = parser.parse_args()
    input_size = input_size if args.s is None else args.s

    # Train on MS-ETH data
    model = DF(input_size, num_classes, mode)
    print('Training ETH')
    X_eth, y_eth = load_dataset('MS-ETH', 'DF')
    model.train(X_eth, y_eth, verbose=True)

    # Test on tunnel data
    print('Testing SSH')
    X_ssh, y_ssh = load_dataset('MS-SSH', 'DF')
    model.test(X_ssh, y_ssh)

    print('Testing SS')
    X_ss, y_ss = load_dataset('MS-SS', 'DF')
    model.test(X_ss, y_ss)

    print('Testing VPN')
    X_vpn, y_vpn = load_dataset('MS-VPN', 'DF')
    model.test(X_vpn, y_vpn)
