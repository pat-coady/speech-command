#!/bin/bash
DIR_NAME="kws"
ssh $1 "tar zcvf payload.tar.gz -C $DIR_NAME outputs"
scp $1:~/payload.tar.gz payload.tar.gz
tar --keep-newer-files -xvf payload.tar.gz
rm payload.tar.gz

