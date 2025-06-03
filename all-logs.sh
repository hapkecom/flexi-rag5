#!/bin/bash
SERVICES=$1

sudo docker-compose logs -f ${SERVICES}
