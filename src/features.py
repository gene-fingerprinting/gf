
def decaps_packets(packets, tunnel):
    '''
    Dncapsulate packets according to packet-oriented encapsulation.
    '''
    # SSH : 16B TAG | 0~7B padding | payload
    if tunnel == 'SSH':
        shrink = 20
    # Shadowsocks : 16B TAG | payload
    elif tunnel == 'SS':
        shrink = 16
    # OpenVPN : 8B header | 16B TAG | 20B IP header | 20B TCP header | payload
    elif tunnel == 'VPN':
        shrink = 64
    else:
        shrink = 0
    packets['size'] = packets['size'].apply(lambda x: x - shrink if x >= shrink else 0)

    return packets

# Hyperparameters
SAMPLE_INTERVAL = 20

def gf_features(packets):
    valid = (packets['direction'] == 'in') & (packets['size'] > 1000)
    if sum(valid) == 0:
        raise Exception('No valid packets')
    start = packets[valid].iloc[0]['timing'] * 1000 - SAMPLE_INTERVAL
    # Packet timing in millisecond
    packets['ms_time'] = packets['timing'] * 1000 - start
    # Cumulative size of incoming packets
    packets['cumul_size'] = packets.apply(lambda x: 0 if x['direction'] == 'out' else x['size'], axis=1).cumsum()

    # Time-sampling cumulative size
    duration = packets['ms_time'].iloc[-1]
    cumul_size = 0
    features = [cumul_size]
    for period in range(int(duration / SAMPLE_INTERVAL) + 1):
        cur_time = (period + 1) * SAMPLE_INTERVAL
        period_packets = packets[(packets['ms_time'] >= cur_time) & (packets['ms_time'] < cur_time + SAMPLE_INTERVAL)]
        if len(period_packets) == 0:
            features.append(cumul_size)
            cumul_size = features[-1]
        else:
            features.append(int(period_packets['cumul_size'].iloc[0]))
            cumul_size = int(period_packets['cumul_size'].iloc[-1])

    return features

def cumul_features(packets):
    # Packets with payload
    packets = packets[packets['size'] != 0]

    # First 4 features
    inpacket = 0
    outpacket = 0
    insize = 0
    outsize = 0
    for _, packet in packets.iterrows():
        if packet['direction'] == 'out':
            outsize += packet['size']
            outpacket += 1
        else:
            insize += packet['size']
            inpacket += 1
    features = [inpacket, outpacket, insize, outsize]

    # 100 interpolant features
    a = 0
    c = 0
    CT = []
    for _, packet in packets.iterrows():
        a += packet['size']
        if packet['direction'] == 'out':
            c -= packet['size']
        else:
            c += packet['size']
        CT.append([a, c])

    gap = float(CT[-1][0] ) / 100
    cur_a = 0
    CT_ptr = 0
    for _ in range(100):
        next_a = cur_a + gap
        while (CT[CT_ptr][0] < next_a):
            CT_ptr += 1
            if (CT_ptr >= len(CT) - 1):
                CT_ptr = len(CT) - 1
                break
        next_pt_c = CT[CT_ptr][1]
        next_pt_a = CT[CT_ptr][0]
        cur_pt_c = CT[CT_ptr - 1][1]
        cur_pt_a = CT[CT_ptr - 1][0]

        if (next_pt_a - cur_pt_a != 0):
            slope = (next_pt_c - cur_pt_c) / (next_pt_a - cur_pt_a)
        else:
            slope = 1000
        next_c = slope * (next_a - cur_pt_a) + cur_pt_c
        features.append(next_c)
        cur_a = next_a

    return features

def df_features(packets):
    # Packets with payload
    packets = packets[packets['size'] != 0].copy()
    packets['sign'] = packets['direction'].apply(lambda x: 1 if x == 'out' else -1)
    features = packets['sign'].tolist()

    return features

def ddtw_features(packets):
    # Uplink packet timings (including TCP ACK packets)
    packets = packets[packets['direction'] == 'out']
    features = packets['timing'].tolist()

    return features
    