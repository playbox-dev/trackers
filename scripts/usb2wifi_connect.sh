#!/bin/bash

# 1. USB接続中のデバイス一覧取得（emulatorなど除外）
devices=$(adb devices | grep -w "device" | grep -v "emulator" | awk '{print $1}' | grep -v ":")

for serial in $devices; do
    echo "Processing device: $serial"

    # 2. Wi-Fi IPアドレス取得（wlan0）
    ip=$(adb -s "$serial" shell ip -f inet addr show wlan0 | grep -w inet | awk '{print $2}' | cut -d/ -f1)
    if [ -z "$ip" ]; then
        echo "  Failed to get IP address for $serial"
        continue
    fi

    echo "  Found IP: $ip"

    # 3. TCPモードに切り替え
    adb -s "$serial" tcpip 5555

    # 4. Wi-Fi経由で再接続
    adb connect "$ip:5555"

	echo ""
done

echo "All devices have been connected via Wi-Fi."
