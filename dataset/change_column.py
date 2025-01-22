import pandas as pd

# Baca file CSV
input_file = 'ftpbf_final.csv'  # Ganti dengan path file CSV yang sesuai
df = pd.read_csv(input_file)

# Mapping kolom lama ke kolom baru
column_mapping = {
    'dst_port': 'Dst Port',
    'protocol': 'Protocol',
    'flow_duration': 'Flow Duration',
    'tot_fwd_pkts': 'Tot Fwd Pkts',
    'tot_bwd_pkts': 'Tot Bwd Pkts',
    'totlen_fwd_pkts': 'TotLen Fwd Pkts',
    'totlen_bwd_pkts': 'TotLen Bwd Pkts',
    'fwd_pkt_len_max': 'Fwd Pkt Len Max',
    'fwd_pkt_len_min': 'Fwd Pkt Len Min',
    'fwd_pkt_len_mean': 'Fwd Pkt Len Mean',
    'fwd_pkt_len_std': 'Fwd Pkt Len Std',
    'bwd_pkt_len_max': 'Bwd Pkt Len Max',
    'bwd_pkt_len_min': 'Bwd Pkt Len Min',
    'bwd_pkt_len_mean': 'Bwd Pkt Len Mean',
    'bwd_pkt_len_std': 'Bwd Pkt Len Std',
    'flow_byts_s': 'Flow Byts/s',
    'flow_pkts_s': 'Flow Pkts/s',
    'flow_iat_mean': 'Flow IAT Mean',
    'flow_iat_std': 'Flow IAT Std',
    'flow_iat_max': 'Flow IAT Max',
    'flow_iat_min': 'Flow IAT Min',
    'fwd_iat_tot': 'Fwd IAT Tot',
    'fwd_iat_mean': 'Fwd IAT Mean',
    'fwd_iat_std': 'Fwd IAT Std',
    'fwd_iat_max': 'Fwd IAT Max',
    'fwd_iat_min': 'Fwd IAT Min',
    'bwd_iat_tot': 'Bwd IAT Tot',
    'bwd_iat_mean': 'Bwd IAT Mean',
    'bwd_iat_std': 'Bwd IAT Std',
    'bwd_iat_max': 'Bwd IAT Max',
    'bwd_iat_min': 'Bwd IAT Min',
    'fwd_psh_flags': 'Fwd PSH Flags',
    'bwd_psh_flags': 'Bwd PSH Flags',
    'fwd_urg_flags': 'Fwd URG Flags',
    'bwd_urg_flags': 'Bwd URG Flags',
    'fwd_header_len': 'Fwd Header Len',
    'bwd_header_len': 'Bwd Header Len',
    'fwd_pkts_s': 'Fwd Pkts/s',
    'bwd_pkts_s': 'Bwd Pkts/s',
    'pkt_len_min': 'Pkt Len Min',
    'pkt_len_max': 'Pkt Len Max',
    'down_up_ratio': 'Down/Up Ratio',
    'pkt_len_mean': 'Pkt Len Mean',
    'pkt_len_std': 'Pkt Len Std',
    'pkt_len_var': 'Pkt Len Var',
    'fin_flag_cnt': 'FIN Flag Cnt',
    'syn_flag_cnt': 'SYN Flag Cnt',
    'rst_flag_cnt': 'RST Flag Cnt',
    'psh_flag_cnt': 'PSH Flag Cnt',
    'ack_flag_cnt': 'ACK Flag Cnt',
    'urg_flag_cnt': 'URG Flag Cnt',
    'cwe_flag_count': 'CWE Flag Count',
    'ece_flag_cnt': 'ECE Flag Cnt',
    'pkt_size_avg': 'Pkt Size Avg',
    'fwd_seg_size_avg': 'Fwd Seg Size Avg',
    'bwd_seg_size_avg': 'Bwd Seg Size Avg',
    'fwd_byts_b_avg': 'Fwd Byts/b Avg',
    'fwd_pkts_b_avg': 'Fwd Pkts/b Avg',
    'fwd_blk_rate_avg': 'Fwd Blk Rate Avg',
    'bwd_byts_b_avg': 'Bwd Byts/b Avg',
    'bwd_pkts_b_avg': 'Bwd Pkts/b Avg',
    'bwd_blk_rate_avg': 'Bwd Blk Rate Avg',
    'subflow_fwd_pkts': 'Subflow Fwd Pkts',
    'subflow_fwd_byts': 'Subflow Fwd Byts',
    'subflow_bwd_pkts': 'Subflow Bwd Pkts',
    'subflow_bwd_byts': 'Subflow Bwd Byts',
    'init_fwd_win_byts': 'Init Fwd Win Byts',
    'init_bwd_win_byts': 'Init Bwd Win Byts',
    'fwd_act_data_pkts': 'Fwd Act Data Pkts',
    'fwd_seg_size_min': 'Fwd Seg Size Min',
    'active_mean': 'Active Mean',
    'active_std': 'Active Std',
    'active_max': 'Active Max',
    'active_min': 'Active Min',
    'idle_mean': 'Idle Mean',
    'idle_std': 'Idle Std',
    'idle_max': 'Idle Max',
    'idle_min': 'Idle Min'
}

# Pilih kolom yang ada dalam mapping
columns_to_keep = list(column_mapping.keys())

# Hapus kolom yang tidak ada dalam kolom yang baru
df_filtered = df[columns_to_keep]

# Ganti nama kolom sesuai dengan mapping baru
df_filtered = df_filtered.rename(columns=column_mapping)

# Simpan ke file CSV baru
output_file = 'ftpbf_sorted.csv'  # Ganti dengan path output yang diinginkan
df_filtered.to_csv(output_file, index=False)

print("Kolom berhasil diubah dan file disimpan.")
