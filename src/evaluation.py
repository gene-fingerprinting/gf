import sys
import const
import argparse
import subprocess
from os.path import join

def eval_ClosedWorld(attack):
	if 'GF' in attack:
		cmd = 'python %s' % join(const.SRC_DIR, 'ClosedWorld_GF.py')
	elif 'CUMUL' in attack:
		cmd = 'python %s' % join(const.SRC_DIR, 'ClosedWorld_CUMUL.py')
	elif 'DF' in attack:
		cmd = 'python %s' % join(const.SRC_DIR, 'ClosedWorld_DF.py')
	elif 'DDTW' in attack:
		cmd = 'python %s' % join(const.SRC_DIR, 'ClosedWorld_DDTW.py')
	else:
		raise Exception('Unsupported attack %s' % attack)

	print('Closed-World Evaluation on %s' % attack)
	subprocess.call(cmd, shell=True)

def eval_OpenWorld(attack):
	if 'GF' in attack:
		cmd = 'python %s' % join(const.SRC_DIR, 'OpenWorld_GF.py')
	elif 'CUMUL' in attack:
		cmd = 'python %s' % join(const.SRC_DIR, 'OpenWorld_CUMUL.py')
	elif 'DF' in attack:
		cmd = 'python %s' % join(const.SRC_DIR, 'OpenWorld_DF.py')
	elif 'DDTW' in attack:
		cmd = 'python %s' % join(const.SRC_DIR, 'OpenWorld_DDTW.py')
	else:
		raise Exception('Unsupported attack %s' % attack)

	print('Open-World Evaluation on %s' % attack)
	subprocess.call(cmd, shell=True)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='evaluation', description='Website Fingerprinting Evaluation')
	parser.add_argument('attack', type=str,
						help='attack type, can be one of GF, CUMUL, DF, DDTW.')
	parser.add_argument('-o', action='store_true',
						help='Open-World evaluation.')
    
	if len(sys.argv) == 1:
		eval_ClosedWorld('GF')
		eval_ClosedWorld('CUMUL')
		eval_ClosedWorld('DF')
		eval_ClosedWorld('DDTW')
		eval_OpenWorld('GF')
		eval_OpenWorld('CUMUL')
		eval_OpenWorld('DF')
		eval_OpenWorld('DDTW')
        
	else:
		args = parser.parse_args()
		if args.o == True:
			eval_OpenWorld(args.attack)
		else:
			eval_ClosedWorld(args.attack)
