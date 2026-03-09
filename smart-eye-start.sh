#!/bin/bash
sudo systemctl start smart-eye.service
echo "Smart Eye started"
sudo systemctl status smart-eye.service --no-pager
