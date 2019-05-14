
set.seed(1)
rstr <- function(n,k){   # vector of n random char(k) strings
  sapply(1:n,function(i){do.call(paste0,as.list(sample(letters,k,replace=T)))})
}

#str<- c(paste0("aa",rstr(10,3)),paste0("bb",rstr(10,3)),paste0("cc",rstr(10,3)))
str<- c('LKPKCC', 'PFPLCI', 'CVLHVL', 'PDENCL', 'MHFNVC', 'FHLIMT',
       'SFMNSC', 'NILKCL', 'QSCLCP', 'FFNILL', 'KHECII', 'MHKLCL',
       'FLKNKL', 'LIKLKC', 'PEANCI', 'EVKSPL', 'FVLHQL', 'CPPNVK',
       'PCLCSP', 'LDCCRC', 'IKCKKC', 'LVLDCS', 'FMLIMA', 'GVSDCE',
       'LHLGLM', 'QQTIRC', 'WVKCCI', 'LIESRC', 'TMSLCQ', 'FASDCL',
       'QLSDLN', 'DPVDPC', 'IMKSPS', 'VFPICK', 'MWSDCC', 'PIDKCM',
       'CHPICS', 'AMNPCE', 'PPPQTI', 'PQADKL', 'HVLLKP', 'DIVSEC')
#        'PFMCVK', 'CIECCA', 'GDFCCI', 'PDESPL', 'MIGNLC', 'NQNSCQ',
#        'MVPQAN', 'PQKNLK', 'FQEDAL', 'IFRHGM', 'FQFNQC', 'MVSNEL',
#        'LHMSLA', 'KFESCK', 'NFQHIL', 'SMLDAR', 'IIQSQP', 'MHTILK',
#        'FFTNVT', 'QHWNAL', 'KMKLVE', 'IICKAK', 'FHTNLR', 'HQFRLL',
#        'PHIACP', 'PEPATR', 'ILKCGK', 'PCVINN', 'HHLCNR', 'LQADCN',
#        'SLKIEK', 'CCLDCM', 'QHFRML', 'QQIDVC', 'WHQLKQ', 'LVMIPM',
#        'KIPHGN', 'QICSLM', 'LQHSCT', 'DMEKIT', 'DMKCCE', 'QFFSNM',
#        'CHLKLK', 'IDSICA', 'IHLRMR', 'IECCCP', 'DMLINV', 'EQKNRC',
#        'KQLGKC', 'PIDLNT', 'CHKICA', 'CPDKSA', 'QVVNLK', 'QMEHGI',
#        'HWQICK', 'PFEEKM', 'PMFRCI', 'QVEDPL'
# Levenshtein Distance
d  <- adist(str)
rownames(d) <- str
hc <- hclust(as.dist(d))
options(repr.plot.width=12, repr.plot.height=7)
plot(hc)
rect.hclust(hc,k=4)

# distancemodels <- stringdistmatrix(str,str,method = "jw")
# rownames(distancemodels) <- str
# hc <- hclust(as.dist(distancemodels))
# dfClust <- data.frame(str, cutree(hc, k=200))
df <- data.frame(str,cutree(hc,k=20))
names(df) <- c('kmer','cluster')
plot(table(df$cluster))

stringdistmatrix
