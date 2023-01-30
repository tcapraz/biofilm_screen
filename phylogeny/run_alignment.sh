#!/bin/bash


for filename in extracted_genes/*.fasta; do
       xbase=${filename##*/}
       xpref=${xbase%.*}
       echo alignment/"$xpref"_alignment.fasta
       mafft $filename > alignment/"$xpref"_alignment.fasta


done

