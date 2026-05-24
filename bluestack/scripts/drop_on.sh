#!/system/bin/sh

iptables -F
iptables -P INPUT DROP
iptables -P OUTPUT DROP
iptables -P FORWARD DROP
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

echo "drop ON"
command -v termux-toast >/dev/null 2>&1 && termux-toast "drop ON"
