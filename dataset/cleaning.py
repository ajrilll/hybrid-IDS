import pandas as pd

file_path = 'ftpbf.csv'  # Ganti dengan nama atau path file CSV kamu
df = pd.read_csv(file_path)

filtered_df = df[(df['dst_ip'] == '10.3.1.254') & (df['src_ip'] == '10.1.1.254')]

filtered_df.to_csv('ftpbf_final.csv', index=False)
