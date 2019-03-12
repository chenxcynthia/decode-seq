
## try http:// if https:// URLs are not supported
        source("https://bioconductor.org/biocLite.R")
        biocLite("motifStack")

library(motifStack)
protein<-read.table("/Users/cynthiachen/Documents/Research/Internship-2018/code/DeepBCR/src/pwm_dict9.txt")
protein<-t(protein[,1:20])
motif<-pcm2pfm(protein)
motif<-new("pfm", mat=motif, name="CDR3_10000", 
            color=colorset(alphabet="AA",colorScheme="chemistry"))
motifStack(motif, layout="stack", ncex=1.0)


