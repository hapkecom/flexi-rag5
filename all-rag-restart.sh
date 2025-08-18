#!/bin/bash

APP=rag-app
sudo docker-compose stop $APP && sudo docker-compose rm -f $APP && sudo sudo docker-compose up -d --force-recreate $APP && sudo ./all-logs.sh
