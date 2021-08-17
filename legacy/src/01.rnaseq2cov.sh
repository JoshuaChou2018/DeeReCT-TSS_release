module load bedtools
genomeCoverageBed -ibam $1 -dz -split -strand + > $1.+.depth

genomeCoverageBed -ibam $1 -dz -split -strand - > $1.-.depth
