"""
Constants for primary analysis FASTQ parsing.

Contains constant sequences used to identify and parse targeted sequencing reads
for costimulatory domains, sgRNAs, and lineage tracing indices.
"""

# Costimulatory domain flanking sequences
# These are the constant regions flanking the variable costim sequence
COSTIM_LEFT = "tggaggatccgcc"   # Downstream constant (3' of costim)
COSTIM_RIGHT = "gccaccatggcc"   # Upstream constant (5' of costim)

# sgRNA flanking sequences
SGRNA_LEFT = "gttttagagctag"    # Downstream of sgRNA spacer
SGRNA_RIGHT = "cttgtggaaagg"    # Upstream of sgRNA spacer

# Lineage tracing index flanking sequences
LTI_LEFT = "tctagaggatcc"       # Downstream of LTI barcode
LTI_RIGHT = "gaattcgatatc"      # Upstream of LTI barcode

# FASTQ filename encoding schemes
# Maps position index in underscore-separated filename to metadata field
FASTQ_NAMING_SCHEME_PT_EC_RP_SS = {
    0: 'Patient',
    1: 'Exp Cond',
    2: 'Replicate',
    3: 'T subset',
}

FASTQ_NAMING_SCHEME_PT_EC_RP_EX = {
    0: 'Patient',
    1: 'Exp Cond',
    2: 'Replicate',
    3: 'Sort Cat',
}

# For MQ (Marquis) naming convention
FASTQ_NAMING_SCHEME_MQ_CT_T = {
    0: 'Patient',
    1: 'Cell Type',
    2: 'Timepoint',
}
