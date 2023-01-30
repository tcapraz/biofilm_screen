import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import itertools
import os
from sys import exit

parser = argparse.ArgumentParser(description='')

parser.add_argument("--folder_with_fastas")
parser.add_argument("--out_file")
parser.add_argument("--check_same_length", default=True, action="store_true")
parser.set_defaults(roary=False)
args = parser.parse_args()

# Read all files in args.folder_with_fastas into memory
genes = os.listdir(str(args.folder_with_fastas))
print(genes)
fastas = [list(SeqIO.parse(str(args.folder_with_fastas) + "/" + x, "fasta")) for x in genes]

respective_header = []
if args.check_same_length:
	for fasta in fastas:
	    if not len(list(set([len(entry.seq) for entry in fasta]))) == 1:
        	exit("At least one of your fasta files doesn't have entries that are all the same length")

# make sure you have 'empty' (i.e. gapped sequences) in case a gene is missing
### Get ids of all entries in all fastas, get set
all_samples = []
for fasta in fastas:
	for f in fasta:
		all_samples.append(f.id)
all_samples = list(set(all_samples))
print("Seeing a total of {} samples".format(len(all_samples)))

### for each fasta, add gapped entries for missing sequences
for fasta in fastas:
	entryIDs = [x.id for x in fasta]
	sequences_to_add = [sampleID for sampleID in all_samples if sampleID not in entryIDs]
	l = len(fasta[0].seq)
	print("Number of samples before adding gaps: {}".format(len(fasta)))
	for s in sequences_to_add:
		t = SeqRecord("".join(["-"] * l))
		t.id = s
		fasta.append(t)
	print("Number of samples after adding gaps: {}".format(len(fasta)))
	### Reorder each fasta alphabetically
	fasta[:] = sorted(fasta, key = lambda x : x.id)




# Check that all entries now have the same orientation
all_headers = []
for fasta in fastas:
	all_headers.append("".join([x.id for x in fasta]))


if len(set(all_headers)) != 1:
	exit("Samples are not having the same orientation across fasta files or some samples are missing in some files.")

# Important: Since we reordered the order of genomeIDs within each fasta, we need to make sure that all_samples is concordant with that!!!
all_samples = [x.id for x in fastas[0]]

# Concatenate sequences sample-wise over all genes
seqs_out = []

for idx, sample in enumerate(all_samples):
	t = ""
	for fasta in fastas:
		t += str(fasta[idx].seq)
	t = [sample, t]
	seqs_out.append(t)

with open(str(args.out_file), "w") as f:
    for out_f in seqs_out:
        f.write(">" + str(out_f[0]) + "\n" + str(out_f[1]) + "\n")
