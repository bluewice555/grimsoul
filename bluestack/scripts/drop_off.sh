#!/system/bin/sh

iptables -F
iptables -P INPUT ACCEPT
iptables -P OUTPUT ACCEPT
iptables -P FORWARD ACCEPT

echo "drop OFF"
command -v termux-toast >/dev/null 2>&1 && termux-toast "drop OFF"
