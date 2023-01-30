import pprint
from BCBio.GFF import GFFExaminer
from BCBio import GFF
from Bio import SeqIO
from pathlib import Path
import os

# provide list of tuples
# first elem: gene name
# second elem: locus_tag i.e. "b number"
# goi_list = [
#     ("rpoS","b2741"),
#     ("csgF","b1038"),
#     ("csgB",	"b1041"),
#     ("csgG"	,"b1037"),
#     ("bcsZ",	"b3531"),
#     ("csgE",	"b1039"),
#     ("bcsC",	"b3530"),
#     ("bcsA",	"b3533"),
#     ("bcsF",	"b3537"),
#     ("bcsG",	"b3538"),
#     ("bcsQ",	"b3534"),
#     ("bcsB",	"b3532"),
#     ("csgA"	,"b1042"),
#     ("csgC",	"b1043"),
#     ("bcsE",	"b3536" ),
#     ("csgD",	"b1040")
#     ]
goi_list = [("csgD",	"b1040")] 
in_dir = "gff/split"
fasta_in= []
gff_in = []
for file in os.listdir(in_dir):
    if file.endswith(".fasta"):
        fasta_in.append(os.path.join(in_dir, file))
for i in fasta_in:
    gff_in.append(i.split(".")[0]+".gff")


not_found = {}
for gene in goi_list:
    found = 0
    not_found[gene] = []
    all_sequences = {}
    for f,g in zip(fasta_in, gff_in):
        oldfound = found
        name = Path(f).stem
        seq =  {}
        csg_seq=""
        for seq_record in SeqIO.parse(f, "fasta"):
                seq[seq_record.id] = seq_record.seq
        
    
        #limit_info = dict(gff_source=["prokka"])
        with open(g) as in_handle:
                    
            
            for rec in GFF.parse(in_handle):
                for feature in rec.features:
                   
                    # print(feature.qualifiers)
                    if len( feature.sub_features) !=0:
                        qual = feature.sub_features[0].qualifiers
                        if "gene" in qual:
                            if qual["gene"][0] == gene[0]:
                                start = feature.location.start
                                end = feature.location.end
                                contig = rec.id
                                csg_seq = seq[contig][start:end]
                                found += 1
                                break
                        if "inference" in qual:
                            b_number = qual["inference"][-1].split(".faa:")[-1]
                            if b_number == gene[1]:
                                start = feature.location.start
                                end = feature.location.end
                                contig = rec.id
                                csg_seq = seq[contig][start:end]
                                found += 1
                                break
                    else:
                        qual = feature.qualifiers
                        if "locus_tag" in qual:
                            if qual["locus_tag"][0] == gene[1]:
                                start = feature.location.start
                                end = feature.location.end
                                contig = rec.id
                                csg_seq = seq[contig][start:end]
                                found += 1
                                break
                        if "inference" in qual:
                            b_number = qual["inference"][-1].split(".faa:")[-1]
                            if b_number == gene[1]:
                                start = feature.location.start
                                end = feature.location.end
                                contig = rec.id
                                csg_seq = seq[contig][start:end]
                                found += 1
                                break
                            
        if name in all_sequences:
            print("Stop non unique filenames!")
            break
        if csg_seq != "":
            all_sequences[name] = csg_seq
        print(name,str(found), " done!")
        if found <= oldfound:
            print(gene, "not found for", name)
            not_found[gene].append(name)

    outseq =[]
    
    for i in all_sequences:
        header = ">" + i  +"\n"
        outseq.append(header)
        outseq.append( str(all_sequences[i]) +"\n")
        
    with open(os.path.join("biofilm_tree","extracted_genes", gene[0]+".fasta"), "w") as f:
        for i in outseq:
            f.write(i)
        

        
