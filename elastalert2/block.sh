#!/bin/bash

# Ambil argumen dari ElastAlert2
SRC_IP=$1
DST_IP=$2

# Log nilai SRC_IP untuk debugging
#echo "SRC_IP: $SRC_IP" >> /tmp/block.log

# User, password, dan IP MikroTik
USER="admin"
PASSWORD="azrielDAR24"
MIKROTIK_IP="10.2.1.1"

# Login SSH ke MikroTik menggunakan sshpass dan jalankan perintah
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$MIKROTIK_IP" "/ip firewall filter add chain=forward src-address=$SRC_IP dst-address=$DST_IP action=drop"

