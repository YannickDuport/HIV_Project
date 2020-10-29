#!/bin/bash

for sam in "$3"/*sorted.sam
do
	echo $sam
	f="$(basename $sam .sorted.sam)".counts.tsv
	echo $f
	$1 -r $2 -s1 $sam -o $4/$f -d 1 -a 
done
