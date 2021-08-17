# CAGE: https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/
# RNASeq: https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.RNAseq/

cell=NKtcell
mkdir -p ${cell}/rnaseq
mkdir -p ${cell}/cage

cd ${cell}/rnaseq
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.RNAseq/NK%2520T%2520cell%2520leukemia%2520cell%2520line%253aKHYG-1.RDhi10082.10777-110G3.hg19.nobarcode.bam
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.RNAseq/NK%2520T%2520cell%2520leukemia%2520cell%2520line%253aKHYG-1.RDhi10082.10777-110G3.hg19.nobarcode.bam.bai

cd ../cage
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/NK%2520T%2520cell%2520leukemia%2520cell%2520line%253aKHYG-1.CNhs11867.10777-110G3.hg19.nobarcode.bam
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/NK%2520T%2520cell%2520leukemia%2520cell%2520line%253aKHYG-1.CNhs11867.10777-110G3.hg19.nobarcode.bam.bai
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/NK%2520T%2520cell%2520leukemia%2520cell%2520line%253aKHYG-1.CNhs11867.10777-110G3.hg19.ctss.bed.gz

cd ../..


cell=leukemia
mkdir -p ${cell}/rnaseq
mkdir -p ${cell}/cage

cd ${cell}/rnaseq
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.RNAseq/acute%2520lymphoblastic%2520leukemia%2520%2528T-ALL%2529%2520cell%2520line%253aHPB-ALL.RDhi10067.10429-106D6.hg19.nobarcode.bam
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.RNAseq/acute%2520lymphoblastic%2520leukemia%2520%2528T-ALL%2529%2520cell%2520line%253aHPB-ALL.RDhi10067.10429-106D6.hg19.nobarcode.bam.bai

cd ../cage
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/acute%2520lymphoblastic%2520leukemia%2520%2528T-ALL%2529%2520cell%2520line%253aHPB-ALL.CNhs10746.10429-106D6.hg19.nobarcode.bam
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/acute%2520lymphoblastic%2520leukemia%2520%2528T-ALL%2529%2520cell%2520line%253aHPB-ALL.CNhs10746.10429-106D6.hg19.nobarcode.bam.bai
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/acute%2520lymphoblastic%2520leukemia%2520%2528T-ALL%2529%2520cell%2520line%253aHPB-ALL.CNhs10746.10429-106D6.hg19.ctss.bed.gz

cd ../..


cell=testicular
mkdir -p ${cell}/rnaseq
mkdir -p ${cell}/cage

cd ${cell}/rnaseq
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.RNAseq/testicular%2520germ%2520cell%2520embryonal%2520carcinoma%2520cell%2520line%253aNEC15.RDhi10074.10593-108D8.hg19.nobarcode.bam
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.RNAseq/testicular%2520germ%2520cell%2520embryonal%2520carcinoma%2520cell%2520line%253aNEC15.RDhi10074.10593-108D8.hg19.nobarcode.bam.bai

cd ../cage
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/testicular%2520germ%2520cell%2520embryonal%2520carcinoma%2520cell%2520line%253aNEC15.CNhs12362.10593-108D8.hg19.nobarcode.bam
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/testicular%2520germ%2520cell%2520embryonal%2520carcinoma%2520cell%2520line%253aNEC15.CNhs12362.10593-108D8.hg19.nobarcode.bam.bai
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/testicular%2520germ%2520cell%2520embryonal%2520carcinoma%2520cell%2520line%253aNEC15.CNhs12362.10593-108D8.hg19.ctss.bed.gz

cd ../..


cell=lung
mkdir -p ${cell}/rnaseq
mkdir -p ${cell}/cage

cd ${cell}/rnaseq
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.RNAseq/small%2520cell%2520lung%2520carcinoma%2520cell%2520line%253aNCI-H82.RDhi10091.10842-111E5.hg19.nobarcode.bam
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.RNAseq/small%2520cell%2520lung%2520carcinoma%2520cell%2520line%253aNCI-H82.RDhi10091.10842-111E5.hg19.nobarcode.bam.bai

cd ../cage
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/small%2520cell%2520lung%2520carcinoma%2520cell%2520line%253aNCI-H82.CNhs12809.10842-111E5.hg19.nobarcode.bam
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/small%2520cell%2520lung%2520carcinoma%2520cell%2520line%253aNCI-H82.CNhs12809.10842-111E5.hg19.nobarcode.bam.bai
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/small%2520cell%2520lung%2520carcinoma%2520cell%2520line%253aNCI-H82.CNhs12809.10842-111E5.hg19.ctss.bed.gz

cd ../..


cell=neuroblastoma
mkdir -p ${cell}/rnaseq
mkdir -p ${cell}/cage

cd ${cell}/rnaseq
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.RNAseq/neuroblastoma%2520cell%2520line%253aCHP-134.RDhi10071.10508-107D4.hg19.nobarcode.bam
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.RNAseq/neuroblastoma%2520cell%2520line%253aCHP-134.RDhi10071.10508-107D4.hg19.nobarcode.bam.bai

cd ../cage
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/neuroblastoma%2520cell%2520line%253aCHP-134.CNhs11276.10508-107D4.hg19.nobarcode.bam
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/neuroblastoma%2520cell%2520line%253aCHP-134.CNhs11276.10508-107D4.hg19.nobarcode.bam.bai
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/neuroblastoma%2520cell%2520line%253aCHP-134.CNhs11276.10508-107D4.hg19.ctss.bed.gz

cd ../..


cell=myeloma
mkdir -p ${cell}/rnaseq
mkdir -p ${cell}/cage

cd ${cell}/rnaseq
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.RNAseq/myeloma%2520cell%2520line%253aPCM6.RDhi10068.10474-106I6.hg19.nobarcode.bam
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.RNAseq/myeloma%2520cell%2520line%253aPCM6.RDhi10068.10474-106I6.hg19.nobarcode.bam.bai

cd ../cage
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/myeloma%2520cell%2520line%253aPCM6.CNhs11258.10474-106I6.hg19.nobarcode.bam
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/myeloma%2520cell%2520line%253aPCM6.CNhs11258.10474-106I6.hg19.nobarcode.bam.bai
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/myeloma%2520cell%2520line%253aPCM6.CNhs11258.10474-106I6.hg19.ctss.bed.gz

cd ../..


cell=gastrointestinal
mkdir -p ${cell}/rnaseq
mkdir -p ${cell}/cage

cd ${cell}/rnaseq
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.RNAseq/gastrointestinal%2520carcinoma%2520cell%2520line%253aECC12.RDhi10077.10615-108G3.hg19.nobarcode.bam
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.RNAseq/gastrointestinal%2520carcinoma%2520cell%2520line%253aECC12.RDhi10077.10615-108G3.hg19.nobarcode.bam.bai

cd ../cage
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/gastrointestinal%2520carcinoma%2520cell%2520line%253aECC12.CNhs11738.10615-108G3.hg19.nobarcode.bam
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/gastrointestinal%2520carcinoma%2520cell%2520line%253aECC12.CNhs11738.10615-108G3.hg19.nobarcode.bam.bai
wget https://fantom.gsc.riken.jp/5/datafiles/latest/basic/human.cell_line.hCAGE/gastrointestinal%2520carcinoma%2520cell%2520line%253aECC12.CNhs11738.10615-108G3.hg19.ctss.bed.gz

cd ../..

