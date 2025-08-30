#!/bin/bash

./build.sh

docker save autopet_baseline1 | gzip -c > nnunet_baseline_v3.tar.gz
