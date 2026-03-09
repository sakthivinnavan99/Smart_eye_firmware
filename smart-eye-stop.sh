#!/bin/bash
sudo systemctl stop smart-eye.service
echo "Smart Eye stopped"
sudo systemctl status smart-eye.service --no-pager
