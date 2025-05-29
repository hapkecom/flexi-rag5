#!/bin/bash

#mkdir -p /data/aisbreaker-workspace/weaviate-ais-rag/weaviate-data

#sudo docker-compose up -d --remove-orphans && sudo docker-compose logs -f searxng
#sudo docker-compose up -d --remove-orphans && sudo docker-compose logs -f searxng
#sudo docker-compose up -d --remove-orphans && sudo docker-compose logs -f rag-app
#sudo docker-compose up -d --remove-orphans && sudo docker-compose logs -f perplexica
echo "#docker-compose logs -f weaviate"
echo "#sudo docker-compose logs -f perplexica"
echo "#sudo docker-compose logs -f rag-app"
sudo docker-compose up -d --remove-orphans && sudo docker-compose logs -f


