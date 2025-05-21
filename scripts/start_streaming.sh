#!/bin/bash

# 接続されている全デバイスを取得
devices=($(adb devices | grep -w "device" | awk '{print $1}'))

# デバイスが見つからない場合は終了
if [ ${#devices[@]} -eq 0 ]; then
    echo "接続されているデバイスがありません。"
    exit 1
fi

echo "以下のデバイスでストリーミングを開始します:"
for i in "${!devices[@]}"; do
    device=${devices[$i]}
    stream_key="fairy_$((i+1))"  # 各デバイスにユニークなストリームキーを付与

    echo "  - $device (Stream Key: $stream_key)"

    adb -s "$device" shell am start \
        -n ai.fd.thinklet.app.squid.run/.MainActivity \
        -a android.intent.action.MAIN \
        -e streamUrl "rtmp://192.168.0.106:1935" \
        -e streamKey "$stream_key" \
        --ei longSide 720 \
        --ei shortSide 480 \
        --ei videoBitrate 4096 \
        --ei audioSampleRate 44100 \
        --ei audioBitrate 128 \
        --ez preview true &

    sleep 3  # アプリが完全に起動するまで待機

    adb -s "$device" shell input keyevent KEYCODE_CAMERA  # ストリーミング開始
done

echo "全デバイスでストリーミングを開始しました。"
