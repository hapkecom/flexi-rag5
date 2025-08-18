#!/bin/bash -x
./rag-build-image.sh
docker push aisbreaker/rag-app
