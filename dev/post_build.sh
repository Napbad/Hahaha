#!/bin/bash

mkdir -p /workspace/extern/externlibs

find /workspace/extern/externlibs -mindepth 1 -delete

mkdir -p /workspace/extern/externlibs

git clone --depth 1 https://github.com/ocornut/imgui.git /workspace/extern/externlibs/imgui


GREEN_BOLD="\033[1;32m"  
RESET_STYLE="\033[0m"    
TERMINAL_WIDTH=$(tput cols) 

echo -e "${GREEN_BOLD}$(printf '=%.0s' $(seq 1 $TERMINAL_WIDTH))${RESET_STYLE}"

SUCCESS_TEXT="Dev Container Build Success"
TEXT_LENGTH=${#SUCCESS_TEXT}
OFFSET=$(( (TERMINAL_WIDTH - TEXT_LENGTH) / 2 ))
echo -e "${GREEN_BOLD}$(printf ' %.0s' $(seq 1 $OFFSET))${SUCCESS_TEXT}${RESET_STYLE}"

echo -e "${GREEN_BOLD}$(printf '=%.0s' $(seq 1 $TERMINAL_WIDTH))${RESET_STYLE}"

