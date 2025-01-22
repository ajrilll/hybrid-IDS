import pandas as pd
import pickle
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import Pipeline

# Load the data
df = pd.read_csv('./dataset_ajril_v9.csv')
df.columns = list(map(lambda x: x.replace(" ", "_").lower(), df.columns))
df['label'] = df['label'].replace({
    'Benign': 'Normal',
})

# Define the feature columns
list_columns = ['bwd_pkt_len_std', 'init_bwd_win_byts', 'pkt_len_min', 'fwd_pkt_len_min', 'idle_max', 
    'idle_mean', 'bwd_iat_tot', 'bwd_iat_max', 'bwd_iat_std', 'bwd_pkt_len_max', 'idle_min', 
    'bwd_iat_mean', 'protocol', 'fwd_iat_min', 'flow_iat_min', 'idle_std', 'fwd_pkts/s', 
    'flow_pkts/s', 'bwd_pkts/s', 'flow_byts/s', 'bwd_iat_min', 'active_max', 'active_mean',
    'active_min', 'active_std', 'subflow_bwd_byts', 'totlen_bwd_pkts', 'fin_flag_cnt', 
    'bwd_pkt_len_mean', 'bwd_seg_size_avg', 'tot_bwd_pkts', 'subflow_bwd_pkts', 
    'tot_fwd_pkts', 'subflow_fwd_pkts', 'bwd_header_len', 'fwd_header_len', 'totlen_fwd_pkts', 
    'subflow_fwd_byts', 'fwd_act_data_pkts', 'fwd_iat_tot', 'bwd_pkt_len_min', 'fwd_iat_mean', 
    'fwd_iat_max', 'fwd_iat_std', 'fwd_seg_size_avg', 'fwd_pkt_len_mean', 'flow_iat_mean', 
    'fwd_pkt_len_std', 'fwd_seg_size_min', 'pkt_size_avg', 'pkt_len_mean', 'flow_iat_std', 
    'pkt_len_var', 'flow_duration', 'flow_iat_max', 'pkt_len_std', 'dst_port', 'fwd_pkt_len_max', 
    'init_fwd_win_byts', 'pkt_len_max', 'fwd_psh_flags', 'syn_flag_cnt', 'rst_flag_cnt', 
    'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'ece_flag_cnt', 'down/up_ratio',
    'fwd_byts/b_avg', 'fwd_pkts/b_avg', 'fwd_blk_rate_avg', 'bwd_byts/b_avg', 
    'bwd_pkts/b_avg', 'bwd_blk_rate_avg']

# Encode labels
label_mapping = {
    'Normal': 0,
    'DoS attacks-SYN-Flood': 1,
    'DoS attacks-UDP-Flood': 2,
    'DDoS attacks-LOIC-HTTP': 3,
    'DoS attacks-GoldenEye': 4,
    'DoS attacks-Hulk': 5,
    'DoS attacks-Slowloris': 6,
    'SSH-Bruteforce': 7,
    'FTP-BruteForce': 8
}
df['label'] = df['label'].map(label_mapping)

# Split the data
X = df[list_columns]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Define a pipeline with SelectKBest and RandomForest using the best parameters
pipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif, k=20)),  # Select the top 20 features
    ('random_forest', RandomForestClassifier(
        n_estimators=10,             # 100 Number of trees in the forest
        max_depth=10,                 # 20 Maximum depth of the tree
        min_samples_split=2,          # 2 Minimum samples required to split a node
        min_samples_leaf=1,           # 1 Minimum samples required in a leaf node
        random_state=42,              # Random seed for reproducibility
        n_jobs=-1                     # Use all available cores
    ))
])

# Measure training time
start_time = time.time()
pipeline.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Training Time: {train_time:.4f} seconds")

# Measure prediction time
start_time = time.time()
y_pred = pipeline.predict(X_test)
pred_time = time.time() - start_time
print(f"Prediction Time: {pred_time:.4f} seconds")

# Generate classification report
target_names = list(label_mapping.keys())
report = classification_report(y_test, y_pred, target_names=target_names, digits=6)
print("\nClassification Report for Best Model:")
print(report)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Correlation Matrix
corr_matrix = X_train.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Cross Validation for the model
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean CV score: {np.mean(cv_scores):.4f}')

# Save the best model
filename = 'RandomForest_Model.pkl'
pickle.dump(pipeline, open(filename, 'wb'))
