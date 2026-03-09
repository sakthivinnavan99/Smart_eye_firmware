#!/bin/bash
sudo systemctl restart smart-eye.service
echo "Smart Eye restarted"
sudo systemctl status smart-eye.service --no-pager
