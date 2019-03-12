
suppressPackageStartupMessages(library(motifStack))
#####Input#####
# protein<-read.table(file.path(find.package("motifStack"),"extdata","cap.txt"))
# protein<-t(protein[,1:20])
# motif1<-pcm2pfm(protein)
# motif2<-pcm2pfm(protein)
# motif3<-pcm2pfm(protein)
# pfm1<-new("pfm", mat=motif1, name="CAP1", color=colorset(alphabet="AA",colorScheme="chemistry"))
# pfm2<-new("pfm", mat=motif2, name="CAP2", color=colorset(alphabet="AA",colorScheme="chemistry"))
# pfm3<-new("pfm", mat=motif3, name="CAP3", color=colorset(alphabet="AA",colorScheme="chemistry"))
# motifs<-list(pfm1, pfm2, pfm3)

pfms<-list()
tumors <- c("BRCA","Normal","READ/COAD","KIRC","UCEC","HNSC","LUAD","LUSC","PRAD","OV","THCA","STAD","SKCM")
num_motifs <- 5
for(i in 1:num_motifs){
    filename <- paste("/Users/cynthiachen/Documents/Research/Internship-2018/code/DeepBCR/src/pwm_top", i,
                      ".txt", sep = "")
    protein<-read.table(filename)
    protein<-t(protein[,1:20])
    motif = pcm2pfm(protein)
    name <- paste("Cancer type: ", tumors[i], sep = "")
    pfm <- new("pfm", mat=motif, name=name, color=colorset(alphabet="AA",colorScheme="chemistry"))
    pfms[[i]] <- pfm
}

#plot(motif)
#motifs<-importMatrix(dir(file.path(find.package("motifStack"), "extdata"),"newdata.meme$", full.names = TRUE))

###############################################################################
####### newMotifStack
newMotifStack <-function(pfms, 
                         layout=c("stack", "treeview", "phylog", "radialPhylog"), 
                         ...){
    if(!is.list(pfms)){
      if(is(pfms, "pcm")) pfms <- pcm2pfm(pfms)
        plot(pfms)
        return(invisible())
    }
    layout <- match.arg(layout)
    if(all(sapply(pfms, class)=="pcm")) pfms <- lapply(pfms, pcm2pfm)
    if (any(unlist(lapply(pfms, function(.ele) !inherits(.ele, c("pfm", "psam")))))) 
        stop("pfms must be a list of pfm, pcm or psam objects")
    if(all(sapply(pfms, class)=="psam")){
      psam <- TRUE
    }else{
      psam <- FALSE
    }
    if (length(pfms)<2)
        stop("length of pfms less than 2")
    if (length(pfms)==2){
        pfms <- DNAmotifAlignment(pfms)
        plotMotifLogoStack(pfms)
        return(invisible(list(phylog=NULL, pfms=pfms)))
    }
    pfmList2matrixList <- function(pfms){
        m <- lapply(pfms, pfm2pwm)
        names(m) <- unlist(lapply(pfms, function(.ele) .ele@name))
        m
    }
    
    ##calculate the distances
#   dots <- list(...)
#   if(!"phylog" %in% names(dots)){
#     if(!psam){
#       jaspar.scores <- readDBScores(file.path(find.package("MotIV"), "extdata", "jaspar2010_PCC_SWU.scores"))
#       d <- motifDistances(pfmList2matrixList(pfms), DBscores=jaspar.scores)
#       hc <- motifHclust(d, method="average")
#       pfms <- pfms[hc$order]
#       pfms <- DNAmotifAlignment(pfms)
#       phylog <- hclust2phylog(hc)
#     }
#   }
    if(layout=="treeview" && !exists("hc")){
      if(!psam){
        jaspar.scores <- readDBScores(file.path(find.package("MotIV"), "extdata", "jaspar2010_PCC_SWU.scores"))
        d <- motifDistances(pfmList2matrixList(pfms), DBscores=jaspar.scores)
        hc <- motifHclust(d, mothod="average")
      }
    }
    switch(layout,
           stack = {
               plotMotifLogoStack(pfms, ...)
           },
           treeview = {
               if(!exists("hc")){
                 stop("hc is required for plotMotifLogoStackWithTree.")
               }
              plotMotifLogoStackWithTree(pfms, hc=hc, ...)
           },
           phylog = {
             if(!exists("phylog")){
               stop("phylog is required for plotMotifStackWithPhylog.")
             }
               plotMotifStackWithPhylog(phylog=phylog, pfms=pfms, ...)
           },
           radialPhylog = {
             if(!exists("phylog")){
               stop("phylog is required for plotMotifStackWithRadialPhylog.")
             }
               args <- list(phylog=phylog, pfms=pfms, ...)
               for(i in names(args)){
                   if(i %in% c("col.leaves", "col.leaves.bg", "col.bg", "col.inner.label.circle", 
                               "col.outer.label.circle")){
                       args[[i]] <- args[[i]][hc$order]
                   }
               }
               do.call(plotMotifStackWithRadialPhylog, args)
           },
           plotMotifLogoStack(pfms, ...)
    )
    
    return(invisible(list(phylog=phylog, pfms=pfms)))
}
                                  
##############################################################################

# debugonce(newMotifStack)
# debugonce(motifDistances)
## plot stacks
newMotifStack(pfms, layout="stack", ncex=1.0)          


