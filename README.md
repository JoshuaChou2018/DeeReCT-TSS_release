# DeeReCT-TSS: A novel meta-learning-based method annotates TSS in multiple cell types based on DNA sequences and RNA-seq data



## Overview

![21001617642928_.pic_hd.jpg](README.assets/21001617642928_.pic_hd.jpg.png)

This repository contains the implementation of DeeReCT-TSS from 

Juexiao Zhou, et al. "DeeReCT-TSS: A novel meta-learning-based method annotates TSS in multiple cell types based on DNA sequences and RNA-seq data"

If you use our work in your research, please cite our paper:

Juexiao Zhou, Bin Zhang et al. DeeReCT-TSS: A novel meta-learning-based method annotates TSS in multiple cell types based on DNA sequences and RNA-seq data, 21 June 2021, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-640669/v1]



## Model Design

![PID.1.model](README.assets/PID.1.model.jpg)

## File Description

The input of our model has two components with both sequence information from the reference genome and coverage information from RNA-Seq data. Homogeneous hCAGE (CAGE sequencing on Heliscope single molecule sequencer) data and RNA-Seq data of same cell lines mapped into hg19 are downloaded from the Fantom5 project (https://fantom.gsc.riken.jp/5/). Example data from colon carcinoma are in folder `data/colon/`.

In the folder `data/colon/`, there are two subfolders `cage/` and `rnaseq/`, which contains the paired data downloaded from Fantom5 (Due to the space limitation of github, you need to run the `download.sh` inside each folder to get the complete data).

In the folder `cage/`:

- `colon carcinoma cell line:COLO-320.CNhs10737.10420-106C6.hg19.ctss.bed.gz`: is the ctss called by Fantom5
- `colon carcinoma cell line:COLO-320.CNhs10737.10420-106C6.hg19.nobarcode.bam`: the raw CAGE data from Fantom5
- `colon carcinoma cell line:COLO-320.CNhs10737.10420-106C6.hg19.nobarcode.bam.bai`: index of the corresponding CAGE data
- `colon carcinoma cell line:COLO-320.CNhs10737.10420-106C6.hg19.nobarcode.bam.depth`: coverage of each base called by genomeCoverageBed
- `colon.carcinoma.anno.final.data.txt`: TSSs in bed format
- `colon.carcinoma.anno.reads_5_rpm_0.5.bed`: TSSs after filtering in bed format

In the filder `rnaseq`:

- `colon carcinoma cell line:COLO-320.RDhi10063.10420-106C6.hg19.nobarcode.bam`: is the raw RNASeq data from Fantom5
- `colon carcinoma cell line:COLO-320.RDhi10063.10420-106C6.hg19.nobarcode.bam.bai`: index of the corresponding RNASeq data
- `colon carcinoma cell line:COLO-320.RDhi10063.10420-106C6.hg19.nobarcode.bam.+.depth`: coverage of each base on + strand called by genomeCoverageBed
- `colon carcinoma cell line:COLO-320.RDhi10063.10420-106C6.hg19.nobarcode.bam.-.depth`: coverage of each base on - strand called by genomeCoverageBed

We store the reference genome in the folder `ref/`. For example, we use `hg19`, which is stored in `ref/hg19/`, under which there is `hg19.fa` file. If you are working on other species, you may specify the reference genome by yourself.

To downlad hg19:

```
mkdir -p ref/hg19
cd ref/hg19
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/genes/hg19.refGene.gtf.gz
```

We store all trained model in the folder `model/`.

We store all codes in the folder `src/`.



## Prerequisites

The code is tested with the following dependencies:

- Python 3.6
- Biopython 1.78
- Numpy 1.19.2
- Scipy 1.5.2
- Scikit-learn 0.22.1
- Tensorflow-gpu 1.14.0
- Seaborn 0.11.1
- Matplotlib
- Samtools
- Bedtools

The code is not guaranteed to work if different versions are used.



## Pipeline

Currently, we only release codes of data preprocessing and genome-wide scanning. For any further inquries, please contact us. 

### Data Preprocessing

Calculate the covegra of each base for both CAGE data and RNASeq data

```
bash src/01.cage2cov.sh path/to/bam
bash src/01.rnaseq2cov.sh path/to/bam
```



### Genome-wide TSS Scanning

Calculate regions for scanning based on RNASeq data

```
bash src/05.getScanRegion.sh
```

To speed up, we can split scanning regions into multiple files and each file contains around 20,000,000 query positions, before running the script below, please modify the content according to your need.

```
python src/05.scan_region_split.py
```

Before continuing, make sure your file hierarchy goes like this:

```
-- root
   |-- scan_regions (contains files for scanning in bed format)
        |-- colon.RNAseq.bedgraph.merged.bed
        |-- colon.RNAseq.bedgraph.merged.0.bed
        |-- colon.RNAseq.bedgraph.merged.1.bed
        |-- ...
   |-- model
        |-- colon_model
             |-- model_best.npz
   |-- data (where you store your data)
        |-- colon (the cell type your are going to work with)
             |-- cage
             |-- rnaseq
                 |-- Aligned.sortedByCoord.out.bam.+.depth
                 |-- Aligned.sortedByCoord.out.bam.-.depth
```

Then we can scan with well trained model, eg:

```
python src/05.scan.app.py 0 data/colon \
     model/colon_model/model_best.npz \
     /scan_output \
     scan_regions/colon.RNAseq.bedgraph.merged \
     data/colon/rnaseq/Aligned.sortedByCoord.out.bam.+.depth \
     data/colon/rnaseq/Aligned.sortedByCoord.out.bam.-.depth \
     ref/hg19/hg19.fa \
     step_size (optional)
```

Command above will start scanning all regions specified in the file `scan_regions/colon.RNAseq.bedgraph.merged.0.bed` and store all scanning result under `data/colon/scan_output`

To merge all individual scanning results into a complete bedgraph, run:

```
python src/06.p2bed.py data/colon/scan_output \
     data/colon/scan_output.bedgraph \
     scan_regions/colon.RNAseq.bedgraph.merged.bed \
     step_size (optional)
```

Command above will output the merged complete bedgraph of all scanned positions, eg: `data/colon/scan_output.bedgraph`

To post-process prediction, call TSS peaks and get the final prediction result:

```
bash src/06.clusterTSS.sh
```



## Comparison

We compare our method with:

- PromID, source code provided under `/compare`
- TransPrise, source code provided under `/compare`
- PseDNC-DL, web server provided @ [http://nsclbio.jbnu.ac.kr/tools/iPSW/](http://nsclbio.jbnu.ac.kr/tools/iPSW/)



## Complete Data

To download data of all extra cell types used in this paper, just run the following code 

```
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

```



## Acknowledgement

This project is supported by KAUST and SUSTech. 

