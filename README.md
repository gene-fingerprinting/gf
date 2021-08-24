# Gene Fingerprinting

Gene Fingerprinting (GF) is a Website Fingerprinting (WF) attack that makes use of the timing and size of packets to generate fingerprints that reflect the intrinsic profile of a website.
GF leverages Zero-shot Learning (ZSL) to reduce the effort to collect data from different tunnels and achieve a real open-world.

The repository contains the code for the GF attack, our implementations of known WF attacks, the the dataset for closed- and open-world evaluation.

:warning: ***Note:*** The code and dataset are for NON-COMMERCIAL RESEARCH USE ONLY!

## Dataset

We publish the dataset of web traffic traces used for attack evaluation.
The dataset is available here:

* [data](https://mega.nz/file/UZZl2CCZ#XDcTJHun8Pa9aig-CS0oU0hCjxZLMXF-DZ9EM3HQgtI) (SHA1: F134A8C1361022028A32E8DB3DB63D19C82B4414)

### Description

The dataset contains traffic samples used for closed- and open-world evaluation.
This includes the list of monitored and unmonitored sites, and the packet traces we collected while visiting those sites over different tunnels.

The dataset consists of the following seven parts:
| No. |   Name  |     Type    | Sites | Visits |    Tunnel   |  Purpose |       Scenario       |
|:---:|:-------:|:-----------:|:-----:|:------:|:-----------:|:--------:|:--------------------:|
|  1  |  MS-ETH |  Monitored  |  100  |   10   |   Ethernet  | Training | Closed- & Open-world |
|  2  |  MS-SSH |  Monitored  |  100  |   10   |   OpenSSH   |   Test   | Closed- & Open-world |
|  3  |  MS-SS  |  Monitored  |  100  |   10   | Shadowsocks |   Test   | Closed- & Open-world |
|  4  |  MS-VPN |  Monitored  |  100  |   10   |   OpenVPN   |   Test   | Closed- & Open-world |
|  5  | UMS-SSH | Unmonitored |  1709 |    1   |   OpenSSH   |   Test   |      Open-world      |
|  6  |  UMS-SS | Unmonitored |  1709 |    1   | Shadowsocks |   Test   |      Open-world      |
|  7  | UMS-VPN | Unmonitored |  1709 |    1   |   OpenVPN   |   Test   |      Open-world      |

The following table lists the tunnel settings for capturing web traffic.
|             |     Client    |    Proxy    |       Ciphersuite      |
|:-----------:|:-------------:|:-----------:|:----------------------:|
|   OpenSSH   | windows_8.1p1 |    8.2p1    |    chacha20-poly1305   |
| Shadowsocks | windows 4.4.0 | libev 3.3.4 | chacha20-ietf-poly1305 |
|   OpenVPN   | connect-3.2.3 |   as_2.8.7  |       AES-256-GCM      |

### Data Format

Each part of the dataset consists of traffic samples in CSV format, with the file name:
```
site-<y>-sample-<i>.csv -- for monitored sites
site-<y>.csv            -- for unmonitored sites

<y> : website number
<i> : sample number
```

Each file is a web traffic trace captured during a visit, and each line in the file contains a packet's timestamp, direction, and payload size. E.g.
```
timing,direction,size
0.0,out,0
0.00662994384765625,in,0
0.0066950321197509766,out,0
0.0073430538177490234,out,517
0.013636112213134766,in,0
0.03990316390991211,in,1460
0.04000711441040039,in,1460
0.04002714157104492,out,0
```

## Feature Extraction

Before feature extraction, please put the downloaded dataset in the `data` folder.

To extract features for all WF attacks, use the following command without arguments:
```
> python src\extraction.py
```

Or, extract features of a particular dataset for a certain attack.
Use the argument `-h` to get the help message:
```
> python src\extraction.py -h

usage: extraction [-h] [-d DATASET] attack

Website Fingerprinting Feature Extraction

positional arguments:
  attack      attack, can be one of GF, CUMUL, DF, DDTW.

optional arguments:
  -h, --help  show this help message and exit
  -d DATASET  traffic dataset, can be one of MS-ETH, MS-SSH, MS-SS, MS-VPN,
              UMS-SSH, UMS-SS, UMS-VPN.
```

### Ready-made

To save time on feature extraction, download the extracted features and corresponding labels from here:

* [pickles](https://mega.nz/file/lFQDSaAT#dJ6guM3la2TTxQBaLZiSj2DkPDKgKPmUIC_-4YrUiLM) (SHA1: 907C6188A8120AA6192D63C22E5A312398D9744F)

It contains two different files:
```   
X_<dataset>_<attack>.pkl -- feature vectors
y_<dataset>.pkl          -- labels

<dataset> : One of the seven different datasets: MS-ETH/SSH/SS/VPN and UMS-SSH/SS/VPN.
<attack>  : One of the attacks: GF, CUMUL, DF, and DDTW.
```

Likewise, please put these files in the `data` folder.

## Closed-world Evaluation

### Dependency

To run the DDTW attack, please properly install the following Python library first:

* [dtaidistance](https://github.com/gene-fingerprinting/dtaidistance-2.0.6_F-distance): Implementation of F-distance based on [dtaidistance-2.0.6](https://github.com/wannesm/dtaidistance). Support C-Extensions for Python to accelerate calculation.

### Reproduce Results

To evaluate the GF attack in the closed-world, use the following command:
```
> python src\evaluation.py GF

Closed-World Evaluation on GF
Training ETH
Electing: 100%|█████████████████████████████████████████████████| 100/100 [00:00<00:00, 133.44it/s]
Testing SSH
Predict: 100%|█████████████████████████████████████████████████| 1000/1000 [00:15<00:00, 63.43it/s]
Accuracy: 97.10 %
Testing SS
Predict: 100%|█████████████████████████████████████████████████| 1000/1000 [00:20<00:00, 48.24it/s]
Accuracy: 94.10 %
Testing VPN
Predict: 100%|█████████████████████████████████████████████████| 1000/1000 [00:16<00:00, 59.32it/s]
Accuracy: 95.20 %
```

To evaluate all WF attacks in the closed-world, simply use the following command without arguments:
```
> python src\evaluation.py
```

Or, evaluate a particular attack using its own command. E.g., the help message for the command of GF closed-world evaluation:
```
> python src\ClosedWorld_GF.py -h

usage: ClosedWorld_GF [-h] [-e [1 2 ... 10]] [-k [1 3 5]] [-w [0.1 0.2 ... 1]]

Evaluation program of Gene Fingerprinting Attack in Closed-World

optional arguments:
  -h, --help          show this help message and exit
  -e [1 2 ... 10]     number of exemplars.
  -k [1 3 5]          k-Nearest Neighbor.
  -w [0.1 0.2 ... 1]  window scale for warping path.
```

## Open-world Evaluation

### Reproduce Results

To evaluate the GF attack in the open-world, use the following command:
```
> python src\evaluation.py GF -o

Open-World Evaluation on GF
Training ETH
Distance: 100%|█████████████████████████████████████████████████| 1000/1000 [00:15<00:00, 66.55it/s]
Testing SSH
Distance: 100%|█████████████████████████████████████████████████| 2709/2709 [00:52<00:00, 51.14it/s]
TPR: 88.70 %    FNR: 11.30 %    FPR: 9.25 %     Precision: 84.88 %
Testing SS
Distance: 100%|█████████████████████████████████████████████████| 2709/2709 [01:05<00:00, 41.47it/s]
TPR: 80.90 %    FNR: 19.10 %    FPR: 7.67 %     Precision: 86.06 %
Testing VPN
Distance: 100%|█████████████████████████████████████████████████| 2709/2709 [00:52<00:00, 51.61it/s]
TPR: 89.50 %    FNR: 10.50 %    FPR: 10.53 %    Precision: 83.26 %
```

To evaluate all WF attacks in the open-world, simply use the following command:
```
> python src\evaluation.py -o
```

Or, evaluate a particular attack using its own command. E.g., the help message for the command of GF open-world evaluation:
```
> python src\OpenWorld_GF.py -h

usage: OpenWorld_GF [-h] [-e [1 2 ... 10]] [-w [0.1 0.2 ... 1]]
                    [-q [0 0.05 ... 1]]

Evaluation program of Gene Fingerprinting Attack in Open-World

optional arguments:
  -h, --help          show this help message and exit
  -e [1 2 ... 10]     number of exemplars.
  -w [0.1 0.2 ... 1]  window scale for warping path.
  -q [0 0.05 ... 1]   quantile for threshold.
```
