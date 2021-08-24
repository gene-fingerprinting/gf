import const
import argparse

from extraction import load_dataset
from Model_CUMUL import CUMUL, optimizer

mode = const.MODE_CW

# Hyperparameters
C = 2**17           # [2**11 ... 2**17]
gamma = 2**3        # [2**-3 ... 2**3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ClosedWorld_CUMUL', description='Evaluation program of CUMUL Attack in Closed-World')
    parser.add_argument('-c', type=int, metavar='2^[11 13 ... 17]', default=None,
                        help='parameter C for SVC.')
    parser.add_argument('-g', type=float, metavar='2^[-3 -1 ... 3]', default=None,
                        help='parameter gamma for SVC.')
    parser.add_argument('-o', action='store_true',
						help='optimize paraneters.')
    args = parser.parse_args()
    C = C if args.c is None else args.c
    gamma = gamma if args.g is None else args.g
    optimize = args.o

    if optimize:
        X_eth, y_eth = load_dataset('MS-ETH', 'CUMUL')
        C, gamma = optimizer(X_eth, y_eth)
        print('Best parameters: C=%d gamma=%f' % (C, gamma))

    # Train on MS-ETH data
    model = CUMUL(C, gamma, mode)
    print('Training ETH')
    X_eth, y_eth = load_dataset('MS-ETH', 'CUMUL')
    model.train(X_eth, y_eth)

    # Test on tunnel data
    print('Testing SSH')
    X_ssh, y_ssh = load_dataset('MS-SSH', 'CUMUL')
    model.test(X_ssh, y_ssh)

    print('Testing SS')
    X_ss, y_ss = load_dataset('MS-SS', 'CUMUL')
    model.test(X_ss, y_ss)

    print('Testing VPN')
    X_vpn, y_vpn = load_dataset('MS-VPN', 'CUMUL')
    model.test(X_vpn, y_vpn)
