#!/bin/bash
DIR_NAME="kws"
/usr/local/opt/gnu-tar/bin/gtar czf payload.tar.gz $DIR_NAME/*.py $DIR_NAME/*.h5
scp payload.tar.gz $1:~/$DIR_NAME/.
ssh $1 "tar xvf $DIR_NAME/payload.tar.gz -C $DIR_NAME"
rm payload.tar.gz
ssh $1 "rm $DIR_NAME/payload.tar.gz"
