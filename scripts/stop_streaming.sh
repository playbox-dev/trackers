#!/bin/bash

# 接続されている全デバイスを取得
devices=$(adb devices | grep -w "device" | awk '{print $1}')

# デバイスが見つからない場合は終了
if [ -z "$devices" ]; then
    echo "接続されているデバイスがありません。"
    exit 1
fi

echo "以下のデバイスでストリーミングを停止し、アプリを終了します:"
for device in $devices; do
    echo "  - $device"
done

# 各デバイスに対して処理を実行
for device in $devices; do
    echo "ストリーミング停止中: $device"

    # KEYCODE_CAMERA を送信してストリーミングを停止
    adb -s $device shell input keyevent KEYCODE_CAMERA

    # 少し待機（ストリーミングが完全に停止するまでの時間を確保）
    sleep 2

    # ホーム画面に戻る
    echo "ホーム画面に移動: $device"
    adb -s $device shell input keyevent KEYCODE_HOME

    # 少し待機（ホーム画面に完全に戻るための時間を確保）
    sleep 1

    # アプリを完全終了（強制停止）
    echo "アプリを終了: $device"
    adb -s $device shell am force-stop ai.fd.thinklet.app.squid.run
done

echo "全デバイスでストリーミングを停止し、アプリを終了しました。"
