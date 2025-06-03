#!/bin/bash
SERVICES=$1
echo "Using 'unbuffer' tool - to install: 'apt install expect'"

#sudo unbuffer docker-compose logs -f ${SERVICES} | egrep -e 'INFO|WARN|ERROR'
#sudo unbuffer docker-compose logs -f ${SERVICES} | egrep -v -e 'DEBUG'
sudo unbuffer docker-compose logs -f ${SERVICES} | egrep -e 'INFO|WARN|ERROR|perplexica|weaviate'
