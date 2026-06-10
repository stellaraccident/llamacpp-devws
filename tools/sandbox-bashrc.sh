# sandbox-bashrc.sh - sourced inside bwrap interactive shells.
[ -f /etc/bashrc ] && . /etc/bashrc
if [ -n "$LLAMACPP_DEVWS_SANDBOX" ]; then
    PS1="[BWRAP] ${PS1:-\u@\h:\w\$ }"
fi
