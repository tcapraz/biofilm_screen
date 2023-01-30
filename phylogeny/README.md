This directory contains scripts to construct a phylogenetic tree from a list of genes of interest.


Summary of the workflow:

* Run extract_goi_sequences.py to extract genes from whole genome fasta files and save each gene in a separate fasta in the extracted_genes directory.

* Run run_alignmnet.sh to run run mafft on each of the extracted_genes and save alignemnts in the alignement directory.

* run concatenate_fasta_files_by_entry.py to concatenate the individual alignments

* Run FastTree concat_alignment.fasta > tree_file to construct phylogenetic tree

The script calc_dist_mat.py can be used to extract a matrix of phylogenetic distances between all strains.
