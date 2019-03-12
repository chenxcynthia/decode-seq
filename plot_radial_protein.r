
# Change these variables
myLabels = c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6", "Cluster 7",
              "Cluster 8", "Cluster 9", "Cluster 10", "Cluster 11", "Cluster 12", "Cluster 13", "Cluster 14",
              "Cluster 15", "Cluster 16", "Cluster 17", "Cluster 18", "Cluster 19", "Cluster 20", "Cluster 21")
num_clusters <- 5

suppressPackageStartupMessages(library(motifStack))

# a scope where we can put mutable global state
.globals <- new.env(parent = emptyenv())

#####Input#####
#protein<-read.table(file.path(find.package("motifStack"),"extdata","cap.txt"))
# protein<-read.table("cynthia1.txt")
# protein<-t(protein[,1:20])
pfms<-list()
# myLabels = c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6", "Cluster 7",
#               "Cluster 8", "Cluster 9", "Cluster 10", "Cluster 11", "Cluster 12", "Cluster 13", "Cluster 14",
#               "Cluster 15", "Cluster 16", "Cluster 17", "Cluster 18", "Cluster 19", "Cluster 20", "Cluster 21")

for(i in 1:num_clusters){
    filename <- paste("/Users/cynthiachen/Documents/Research/Internship-2018/code/DeepBCR/synth_motifStackR/cynthia", i, ".txt", sep = "")
    protein<-read.table(filename)
    protein<-t(protein[,1:20])
  assign(paste("motif", i, sep = ""), pcm2pfm(protein))
  assign(paste("pfm", i, sep = ""), new("pfm", mat=get(paste("motif", i, sep = "")), name=myLabels[i], color=colorset(alphabet="AA",colorScheme="chemistry")))
  pfms[[paste("pfm", i, sep = "")]]<-get(paste("pfm", i, sep = ""))
}
# pfms<-list(pfm1, pfm2, pfm3, pfm4, pfm5, pfm6, pfm7, pfm8, pfm9, pfm10, pfm11, pfm12, pfm13, pfm14, pfm15, pfm16, pfm17, pfm18, pfm19, pfm20, pfm21, pfm22, pfm23, pfm24, pfm25, pfm26, pfm27, pfm28, pfm29, pfm30, pfm31, pfm32, pfm33, pfm34, pfm35, pfm36, pfm37, pfm38, pfm39, pfm40, pfm41, pfm42, pfm43, pfm44, pfm45, pfm46, pfm47, pfm48, pfm49, pfm50)

## When the number of motifs is too much to be shown in a vertical stack, 
## motifStack can draw them in a radial style.
## random sample from MotifDb
library("MotifDb")
## See system.file("LICENSE", package="MotifDb") for use restrictions.

matrix.fly <- query(MotifDb, "Dmelanogaster")
motifs2 <- as.list(matrix.fly)
# use data from FlyFactorSurvey
motifs2 <- motifs2[grepl("Dmelanogaster\\-FlyFactorSurvey\\-",
                         names(motifs2))]
# format the names
names(motifs2) <- gsub("Dmelanogaster_FlyFactorSurvey_", "",
                       gsub("_FBgn\\d+$", "",
                            gsub("[^a-zA-Z0-9]","_",
                                 gsub("(_\\d+)+$", "", names(motifs2)))))
motifs2 <- motifs2[unique(names(motifs2))]
pfmsOld <- sample(motifs2, num_clusters)
# creat a list of object of pfm 
motifs2 <- lapply(names(pfmsOld), 
                  function(.ele, pfmsOld){new("pfm",mat=pfmsOld[[.ele]], name=.ele)}
                  ,pfmsOld)
## trim the motifs
motifs2 <- lapply(motifs2, trimMotif, t=0.4)
## setting colors
library(RColorBrewer)
color <- brewer.pal(12, "Set3")

###############################################################################
# Private utils
###############################################################################
checkInteger <- function(N){
    !length(grep("[^[:digit:]]", as.character(N)))
}

hex2psrgb<-function(col){
    col<-col2rgb(col)
    col<-col/255
    col<-paste(col,collapse=" ")
    col
}

motifStack_private_fontsize <- 18
coloredSymbols <- function(ncha, font, color, rname, fontsize=motifStack_private_fontsize){
    symbols<-list()
    for(i in 1:ncha){
        ps<-paste("%!PS\n/",font," findfont\n",fontsize," scalefont\n",
                  hex2psrgb(color[i])," setrgbcolor\nsetfont\nnewpath\n0 0 moveto\n(",
                  rname[i],") show",sep="")
        psfilename<-tempfile()
        psfilename <- gsub("\\", "/", psfilename, fixed=TRUE)
        # step1 create a ps file
        cat(ps,file=paste(psfilename,".ps",sep=""))
        # step2 convert it by grImport::PostScriptTrace
        PostScriptTrace(paste(psfilename,".ps",sep=""), paste(psfilename,".xml",sep=""))
        # step3 read by grImport::readPicture
        symbols[[i]]<-readPicture(paste(psfilename,".xml",sep=""))
        unlink(c(paste(psfilename,".ps",sep=""), 
                 paste("capture",basename(psfilename),".ps",sep=""), 
                 paste(psfilename,".xml",sep="")))
    }
    symbols
}

addPseudolog2<-function(x){
    x <- round(x, digits=6)
    ifelse(x < .000001, -20, log2(x)) ## !important -10.
}

## get Information Entropy from matrix
getIE<-function(x){
    addPseudolog2(nrow(x))
}

UngappedAlignment<-function(pfms, i, threshold, minimalConsensus=0, rcpostfix="(RC)", revcomp=TRUE){
    if(class(pfms[[i]])!="pfm"){
        pcms <- pfms
        pfms <- lapply(pfms, pcm2pfm)
    }else{
        pcms <- NULL
    }
    res<-getAlignedICWithoutGap(pfms[[i-1]], pfms[[i]], threshold, revcomp)
    if(!is.null(pcms)) pfms <- pcms
    if(res$max>=minimalConsensus){
        if(res$rev){
            pfms[[i]]<-matrixReverseComplement(pfms[[i]])
            pfms[[i]]@name<-paste(pfms[[i]]@name, rcpostfix, sep="")
        }
        if(res$offset>0){
            pfms[[i]]<-addBlank(pfms[[i]], res$offset, FALSE)
        }else{
            if(res[1]<0){
                pfms[1:(i-1)]<-lapply(pfms[1:(i-1)],function(.ele) addBlank(.ele, -1*res$offset, FALSE))
            }
        }
    }
    pfms
}
getAlignedICWithoutGap<-function(pfm1, pfm2, threshold, revcomp=TRUE){
    if(class(pfm1)!="pfm" | class(pfm2)!="pfm") stop("class of pfm1 and pfm2 must be pfm")
    offset1<-getoffsetPosByIC(pfm1, pfm2, threshold)
    if(revcomp){
        pfm3<-matrixReverseComplement(pfm2)
        offset2<-getoffsetPosByIC(pfm1, pfm3, threshold)
    }else{
        offset2<-list(k=0,max=0)
    }
    offset<-0
    rev<-FALSE
    if((offset1$max < offset2$max)){
        rev<-TRUE
        max<-offset2$max
        offset<-offset2$k
    }else{
        max<-offset1$max
        offset<-offset1$k
    }
    list(offset=offset,rev=rev,max=max)    
}
getICbyBase<-function(p, pfmj){
    re<-rep(0, 5)
    re[1:4]<-pfmj * (addPseudolog2(pfmj) - addPseudolog2(p))
    re[5]<-sum(re[1:4])
    re
}
getALLRbyBase <- function(b, pfmi, pfmj){##return 2x ALLR
    sum(pfmj * (addPseudolog2(pfmi) - addPseudolog2(b))) + 
        sum(pfmi * (addPseudolog2(pfmj) - addPseudolog2(b)))
}
checkALLR <- function(...){
    getALLRbyBase(...)>0
}
SWU <- function(pattern, subject, b, 
                match=1, mismatch=-1, gap=-1000){## motif length will never longer than 1000
    if(class(pattern)!="matrix" || class(subject)!="matrix"){
        stop("pattern and subject must be numeric matrix")
    }
    m <- ncol(pattern)
    n <- ncol(subject)
    score <- matrix(0, nrow=m+1, ncol=n+1)
    point <- matrix("none", nrow=m+1, ncol=n+1)
    max_i <- 1
    max_j <- 1
    max_score <- 0
    for(i in 1:m){
        for(j in 1:n){
            diagonal_score <- 0
            left_score <- 0
            up_score <- 0
            #calculate match score
            col1 <- subject[, j]
            col2 <- pattern[, i]
            #score by KLLR
            if(checkALLR(b, col1, col2)){
                diagonal_score <- score[i, j] + match
            }else{
                diagonal_score <- score[i, j] + mismatch
            }
            
            #calculate gap scores
            up_score <- score[i, j+1] + gap
            left_score <- score[i+1, j] + gap
            
            if(diagonal_score <=0 && up_score <=0 && left_score <=0){
                score[i+1, j+1] <- 0
                point[i+1, j+1] <- "none"
                next # terminate this iteration of the loop
            }
            
            #choose best score
            if(diagonal_score >= up_score){
                if(diagonal_score >= left_score){
                    score[i+1, j+1] <- diagonal_score
                    point[i+1, j+1] <- "diagonal"
                }else{
                    score[i+1, j+1] <- left_score
                    point[i+1, j+1] <- "left"
                }
            }else{
                if(up_score > left_score){
                    score[i+1, j+1] <- up_score
                    point[i+1, j+1] <- "up"
                }else{
                    score[i+1, j+1] <- left_score
                    point[i+1, j+1] <- "left"
                }
            }
            
            #set maximum score
            if(score[i+1, j+1] > max_score){
                max_i <- i+1
                max_j <- j+1
                max_score <- score[i+1, j+1]
            }
        }
    }
    #trace-back
    align1 <- c()
    align2 <- c()
    
    i <- max_i
    j <- max_j
    while(1){
        if(point[i, j]=="none"){
            break
        }
        
        if(point[i, j]=="diagonal"){
            align1 <- c(align1, j-1)
            align2 <- c(align2, i-1)
            i <- i-1
            j <- j-1
        }else{
            if(point[i, j]=="left"){
                align1 <- c(align1, j-1)
                align2 <- c(align2, 0)
                j <- j-1
            }else{
                if(point[i, j]=="up"){
                    align1 <- c(align1, 0)
                    align2 <- c(align2, i-1)
                    i <- i-1
                }
            }
        }
    }
    align1 <- rev(align1)
    align2 <- rev(align2)
    if(length(align1)<1 | length(align2)<1){
        k <- 0
    }else{
        k=align1[1]-align2[1]
    }
    list(k=k, max=max_score)
}
ungappedScore <- function(query, subject, b, threshold){
    if(class(query)!="matrix" || class(subject)!="matrix"){
        stop("query and subject must be numeric matrix")
    }
    m <- ncol(query)
    n <- ncol(subject)
    score <- matrix(0, nrow=m+1, ncol=n+1)
    for(i in 1:m){
        for(j in 1:n){
            ic1 <- getICbyBase(b, query[, i])[5]
            ic2 <- getICbyBase(b, subject[, j])[5]
            if(ic1>=threshold && ic2>=threshold){
                value <- getICbyBase(b, c(query[, i]+subject[, j])/2)[5]
            }else{
                value <- 0
            }
            score[i+1, j+1] <- score[i, j] + value
        }
    }
    score[1:m, 1:n] <- 0
    max_score <- max(score)
    if(max_score==0){
        return(list(k=0, max=0))
    }
    idx <- which(score==max_score, arr.ind=TRUE)
    
    if(nrow(idx)>1){
        idxn <- apply(idx, 1, min)
        idx <- idx[which(idxn==min(idxn))[1], , drop=FALSE]
    }
    max_r <- idx[, "row"]
    max_c <- idx[, "col"]
    k <- max_r - max_c
    list(k=as.numeric(k), max=max_score)
}
getoffsetPosByIC<-function(pfm1, pfm2, threshold){
    if(class(pfm1)!="pfm" | class(pfm2)!="pfm") stop("class of pfm1 and pfm2 must be pfm")
    res <- ungappedScore(pfm1$mat, pfm2$mat, b=pfm1$background, threshold)
    res
}
getoffsetPosByIC_old<-function(pfm1, pfm2, threshold){
    if(class(pfm1)!="pfm" | class(pfm2)!="pfm") stop("class of pfm1 and pfm2 must be pfm")
    score1<-rep(0, ncol(pfm1@mat))
    score2<-rep(0, ncol(pfm2@mat))
    value1<-rep(0, ncol(pfm1@mat))
    value2<-rep(0, ncol(pfm2@mat))
    for(i in 1:ncol(pfm1@mat)){
        J<-ifelse(ncol(pfm1@mat)+1-i>ncol(pfm2@mat),ncol(pfm2@mat),ncol(pfm1@mat)+1-i)
        for(j in 1:J){
            ic1<-getICbyBase(pfm1@background, pfm1@mat[,i+j-1])
            ic2<-getICbyBase(pfm2@background, pfm2@mat[,j])
            #ic3<-getALLRbyBase(pfm1@background, pfm1@mat[,i+j-1], pfm2@mat[,j])
            ic3 <- getICbyBase(pfm1@background, (pfm1@mat[,i+j-1]+pfm2@mat[,j])/2)
            if(sd(pfm1@mat[,i+j-1])==0 || sd(pfm2@mat[,j])==0){
                corr <- 0
            }else{
                corr <- cor(pfm1@mat[,i+j-1], pfm2@mat[,j], method="spearman")
            }
            if(ic1[5]>threshold & ic2[5]>threshold){
                #a<-ic1[1:4]>mean(ic1[1:4])
                #b<-ic2[1:4]>mean(ic2[1:4])
                #if(ic3>0) score1[i]<-score1[i]+1
                #if(ic3>0.25) {
                #if(any(a&b)){
                if(corr>0){
                    score1[i]<-score1[i]+1
                    value1[i] <- value1[i]+ic3[5]
                }
            }
        }
    }
    for(i in 1:ncol(pfm2@mat)){
        J<-ifelse(ncol(pfm2@mat)+1-i>ncol(pfm1@mat),ncol(pfm1@mat),ncol(pfm2@mat)+1-i)
        for(j in 1:J){
            ic2<-getICbyBase(pfm2@background, pfm2@mat[,i+j-1])
            ic1<-getICbyBase(pfm1@background, pfm1@mat[,j])
            #ic3<-getALLRbyBase(pfm1@background, pfm1@mat[,j], pfm2@mat[,i+j-1])
            ic3 <- getICbyBase(pfm1@background, (pfm2@mat[,i+j-1]+pfm1@mat[,j])/2)
            if(sd(pfm1@mat[,j])==0 || sd(pfm2@mat[,i+j-1])==0){
                corr <- 0
            }else{
                corr <- cor(pfm1@mat[,j], pfm2@mat[,i+j-1], method="spearman")
            }
            if(ic1[5]>threshold & ic2[5]>threshold){
                #a<-ic1[1:4]>mean(ic1[1:4])
                #b<-ic2[1:4]>mean(ic2[1:4])
                #if(ic3>0) score2[i]<-score2[i]+1
                #if(ic3>0.25) {
                #if(any(a&b)){
                if(corr>0){
                    score2[i]<-score2[i]+1
                    value2[i] <- value2[i]+ic3[5]
                }
            }
        }
    }
    k1<-match(max(score1),score1)
    k2<-match(max(score2),score2)
    sc <- (score1[k1] < score2[k2]) || ((score1[k1] == score2[k2]) && 
                                            (value1[k1] < value2[k2]))
    k <- ifelse(sc, -1*(k2-1), (k1-1))
    max <- ifelse(sc, score2[k2], score1[k1])
    val <- ifelse(sc, value2[k2], value1[k1])
    list(k=k,max=max,val=val)
}

setClass("Rect", 
         representation(x="numeric", y="numeric", width="numeric", height="numeric"), 
         prototype=prototype(x=0, y=0, width=0, height=0)
)

setGeneric("isContainedIn", function(a, b) standardGeneric("isContainedIn"))

setMethod("isContainedIn", signature(a="Rect", b="Rect"), function(a, b){
    a@x >= b@x && a@y >= b@y && a@x+a@width <= b@x+b@width && a@y+a@height <= b@y+b@height
})

setClass("pos",
         representation(box="Rect", beta="numeric", sig="pfm", freq="numeric", norm="numeric")
)

getPFMid <- function(pfms, nodename, rcpostfix="(RC)"){
    pfmNames <- as.character(unlist(lapply(pfms, function(.ele) .ele@name)))
    pfmNames <- gsub(rcpostfix,"",pfmNames, fixed=TRUE)
    which(pfmNames==nodename)
}

getParentNode <- function(nodelist, nodename){
    for(i in 1:length(nodelist)){
        currNode <- nodelist[[i]]
        if(currNode@left==nodename) return(c(currNode@parent, "left"))
        if(currNode@right==nodename) return(c(currNode@parent, "right"))
    }
    NULL
}

makeLeaveNames <- function(ch){
    gsub(".", "_", make.names(ch), fixed=TRUE)
}

###############################################################################
####### motifstackOld
motifStackOld <-function(pfmsOld, 
                         ...){
    if(all(sapply(pfmsOld, class)=="pcm")) pfmsOld <- lapply(pfmsOld, pcm2pfm)
    if(all(sapply(pfmsOld, class)=="psam")){
        psam <- TRUE
    }else{
        psam <- FALSE
    }
    pfmList2matrixList <- function(pfmsOld){
        m <- lapply(pfmsOld, pfm2pwm)
        names(m) <- unlist(lapply(pfmsOld, function(.ele) .ele@name))
        m
    }
  
    #calculate the distances
    dots <- list(...)
    jaspar.scores <- readDBScores(file.path(find.package("MotIV"), "extdata", "jaspar2010_PCC_SWU.scores"))
    d <- motifDistances(pfmList2matrixList(pfmsOld), DBscores=jaspar.scores)
    hc <- motifHclust(d, method="average")
    phylog <- hclust2phylog(hc)
    return(invisible(list(phylog=phylog, hc=hc)))
}

###############################################################################
######## radial style stack
######## 
###############################################################################
newPlotMotifStackWithRadialPhylog <- function (hc, phylog, pfms=NULL,
                                               circle=.75, circle.motif=NA, cleaves=1, cnodes=0,
                                               labels.leaves=names(phylog$leaves), clabel.leaves=1,
                                               labels.nodes=names(phylog$nodes), clabel.nodes=0,
                                               draw.box=FALSE,
                                               col.leaves=rep("black", length(labels.leaves)),
                                               col.leaves.bg=NULL, col.leaves.bg.alpha=1,
                                               col.bg=NULL, col.bg.alpha=1,
                                               col.inner.label.circle=NULL, inner.label.circle.width="default",
                                               col.outer.label.circle=NULL, outer.label.circle.width="default",
                                               clockwise =FALSE, init.angle=if(clockwise) 90 else 0,
                                               angle=360, pfmNameSpliter=";", rcpostfix="(RC)", 
                                               motifScale=c("linear","logarithmic"), ic.scale=TRUE,
                                               plotIndex=FALSE, IndexCol="black", IndexCex=.8,
                                               groupDistance=NA, groupDistanceLineCol="red", 
                                               plotAxis=FALSE, font="Helvetica-Bold", fontsize=12)
{
  if (!inherits(phylog, "phylog"))
    stop("Non convenient data")
  leaves.number <- length(phylog$leaves)
  checkLength <- function(tobechecked){
    !((length(tobechecked)>=leaves.number)||is.null(tobechecked))
  }
  checkNA <- function(tobechecked){
    if(is.null(tobechecked)) return(FALSE)
    return(any(is.na(tobechecked)))
  }
  for(tobechecked in c("col.leaves", "col.leaves.bg", "col.bg", "col.inner.label.circle", "col.outer.label.circle")){
    if(checkLength(eval(as.symbol(tobechecked)))) stop(paste("the length of", tobechecked, "should be same as the length of leaves"))
    if(checkNA(eval(as.symbol(tobechecked)))) stop(paste("contain NA in", tobechecked))
  }
  motifScale <- match.arg(motifScale)
  leaves.names <- names(phylog$leaves)
  nodes.number <- length(phylog$nodes)
  nodes.names <- names(phylog$nodes)
  if (length(labels.leaves) != leaves.number)
    labels.leaves <- names(phylog$leaves)
  if (length(labels.nodes) != nodes.number)
    labels.nodes <- names(phylog$nodes)
  if (circle < 0)
    stop("'circle': non convenient value")
  leaves.car <- gsub("[_]", " ", labels.leaves)
  nodes.car <- gsub("[_]", " ", labels.nodes)
  opar <- par(mar = par("mar"), srt = par("srt"))
  on.exit(par(opar))
  par(mar = c(0.1, 0.1, 0.1, 0.1), mfrow=c(1,1))
  dis <- phylog$droot
  max_Dis <- max(dis)
  axis_pos <- circle
  if(!is.na(groupDistance)){
    groupDistance <- (max_Dis - groupDistance) * circle / max_Dis
  }
  dis <- dis/max_Dis
  rayon <- circle
  dis <- dis * rayon
  dist.leaves <- dis[leaves.names]
  dist.nodes <- dis[nodes.names]
  asp <- c(1, 1)
  if(is.null(pfms)){
    plot.default(0, 0, type = "n", asp = 1, xlab = "", ylab = "",
                 xaxt = "n", yaxt = "n", xlim = c(-2, 2), ylim = c(-2, 2),
                 xaxs = "i", yaxs = "i", frame.plot = FALSE)
  }else{
    pin <- dev.size("in") #pin <- par("pin")
    if (pin[1L] > pin[2L]) asp <- c(pin[2L]/pin[1L], 1)
    else asp <- c(1, pin[1L]/pin[2L])
    plot.default(0, 0, type = "n", asp=1, xlab = "", ylab = "",
                 xaxt = "n", yaxt = "n", xlim = c(-2.5, 2.5), ylim = c(-2.5, 2.5),
                 xaxs = "i", yaxs = "i", frame.plot = FALSE)
  }
  d.rayon <- rayon/(nodes.number - 1)
  twopi <- if (clockwise) -2 * pi else 2 * pi
  alpha <- twopi * angle * (1:leaves.number)/leaves.number/360 + init.angle * pi/180
  names(alpha) <- leaves.names
  x <- dist.leaves * cos(alpha)
  y <- dist.leaves * sin(alpha)
  xcar <- (rayon + d.rayon) * cos(alpha)
  ycar <- (rayon + d.rayon) * sin(alpha)
  
  rayonWidth <- max(unlist(lapply(leaves.names, strwidth, units="user", cex=clabel.leaves)))
  circle.motif <- ifelse(is.na(circle.motif), rayon + d.rayon + rayonWidth, circle.motif)
  ##for logos position
  maxvpwidth <- 0
  if(!is.null(pfms)){
    beta <- alpha * 180 / pi
    vpheight <- 2 * pi * angle * circle.motif / 360 / leaves.number / 5
    vpheight <- vpheight * asp[2L]
    xm <- circle.motif * cos(alpha) * asp[1L] / 5 + 0.5
    ym <- circle.motif * sin(alpha) * asp[2L] / 5 + 0.5
    pfmNamesLen <- sapply(pfms, function(.ele) 
      length(strsplit(.ele@name, pfmNameSpliter)[[1]]))
    if(motifScale=="linear")
      vph <- 2.5*vpheight*pfmNamesLen
    else vph <- 2.5*vpheight*(1+log2(pfmNamesLen+0.0001))
    maxvpwidth <- max(mapply(function(.ele, f) ncol(.ele@mat)*f, pfms, vph))
  }
  if(inner.label.circle.width=="default") inner.label.circle.width <- rayonWidth/10
  if(outer.label.circle.width=="default") outer.label.circle.width <- rayonWidth/10
  
  ratio <- if(!is.null(pfms)) 2.5/(circle.motif+maxvpwidth+outer.label.circle.width) else 1
  if(ratio < 1){
    x <- x * ratio
    y <- y * ratio
    xcar <- xcar * ratio
    ycar <- ycar * ratio
    dis <- dis * ratio
    groupDistance <- groupDistance * ratio
    xm <- (xm-.5) * ratio + .5
    ym <- (ym-.5) * ratio + .5
    vpheight <- vpheight * ratio
    axis_pos <- axis_pos * ratio
  }
  ##for plot background
  if(!is.null(col.bg)) col.bg <- highlightCol(col.bg, col.bg.alpha)
  if(!is.null(col.leaves.bg)) col.leaves.bg <- highlightCol(col.leaves.bg, col.leaves.bg.alpha)
  gamma <- twopi * angle * ((1:(leaves.number+1))-0.5)/leaves.number/360 + init.angle * pi/180
  n <- max(2, floor(200*360/leaves.number))
  plotBgArc <- function(r,bgcol,inr){
    if(ratio < 1){
      r <- r*ratio
      inr <- inr*ratio
    }
    t2xy <- function(rx,t) list(x=rx*cos(t), y=rx*sin(t))
    oldcol <- bgcol[1]
    start <- 1
    icnt <- 1
    for(i in 1:leaves.number){
      oldcol <- bgcol[i]
      if(i==leaves.number || bgcol[i+1]!=oldcol){
        P <- t2xy(r, seq.int(gamma[start], gamma[start+icnt], length.out=n*icnt))
        polygon(c(P$x, 0), c(P$y, 0), border=bgcol[i], col=bgcol[i])
        start <- i+1
        icnt <- 1
      } else {
        icnt <- icnt + 1
      }
    }
    if(inr!=0){
      P <- t2xy(inr, seq.int(0, twopi, length.out=n*leaves.number))
      polygon(c(P$x, 0), c(P$y, 0), border="white", col="white")
    }
  }
  if (clabel.leaves > 0) {
    if(!is.null(col.outer.label.circle)) ##plot outer.label.circle
      plotBgArc(circle.motif+maxvpwidth+outer.label.circle.width, col.outer.label.circle, circle.motif+maxvpwidth)
    if(!is.null(col.inner.label.circle)) #plot inner.label.circle
      plotBgArc(circle.motif, col.inner.label.circle, circle.motif - inner.label.circle.width)
    if(!is.null(col.leaves.bg)) ##plot leaves bg
      plotBgArc(circle.motif - inner.label.circle.width, col.leaves.bg, rayon+d.rayon)
    if(!is.null(col.bg)) ##plot center bg
      plotBgArc(ifelse(mean(dist.leaves)/max(dis) > .9, mean(dist.leaves), max(dis)-d.rayon), col.bg, 0)##plotBgArc(mean(dist.leaves), col.bg, 0)
    for(i in 1:leaves.number) {
      par(srt = alpha[i] * 180/pi)
      text(xcar[i], ycar[i], leaves.car[i], adj = 0, col=col.leaves[i], cex = par("cex") *
             clabel.leaves)
      segments(xcar[i], ycar[i], x[i], y[i], col = grey(0.7))
    }
    
    assign("tmp_motifStack_symbolsCache", list(), envir=.globals)
    if(!is.null(pfms)){
      ##extract names
      for(metaChar in c("\\","$","*","+",".","?","[","]","^","{","}","|","(",")"))
      {
        rcpostfix <- gsub(metaChar,paste("\\",metaChar,sep=""),rcpostfix,fixed=TRUE)
      }
      pfmNames <- lapply(pfms, function(.ele) .ele@name)
      for(i in 1:length(pfmNames)){
        pfmname <- unlist(strsplit(pfmNames[[i]], pfmNameSpliter))
        pfmname <- gsub(paste(rcpostfix,"$",sep=""),"",pfmname)
        # pfmIdx <- which(makeLeaveNames(labels.leaves) %in% makeLeaveNames(pfmname))
        pfmIdx <- i
        if(length(pfmIdx)==0) 
          pfmIdx <- which(makeLeaveNames(names(phylog$leaves)) 
                          %in% makeLeaveNames(pfmname))
        if(length(pfmIdx)>0){
          vph <- ifelse(motifScale=="linear",
                        vpheight*length(pfmname),
                        vpheight*(1+log2(length(pfmname))))
          vpw <- vph * ncol(pfms[[i]]@mat) / 2
          vpd <- sqrt(vph*vph+vpw*vpw) / 2
          angle <- median(beta[pfmIdx])
          if(length(pfmIdx)%%2==1){
            this.pfmIdx <- which(beta[pfmIdx] == angle)[1]
            vpx <- xm[pfmIdx[this.pfmIdx]] + vpd * cos(alpha[pfmIdx[this.pfmIdx]]) * asp[1L]
            vpy <- ym[pfmIdx[this.pfmIdx]] + vpd * sin(alpha[pfmIdx[this.pfmIdx]]) * asp[2L]
            vpx1 <- xm[pfmIdx[this.pfmIdx]] - inner.label.circle.width * cos(alpha[pfmIdx[this.pfmIdx]]) * asp[1L] *1.1/5
            vpy1 <- ym[pfmIdx[this.pfmIdx]] - inner.label.circle.width * sin(alpha[pfmIdx[this.pfmIdx]]) * asp[2L] *1.1/5
          }else{
            this.pfmIdx <- order(abs(beta[pfmIdx] - angle))[1:2]
            vpx <- median(xm[pfmIdx[this.pfmIdx]]) + vpd * cos(median(alpha[pfmIdx[this.pfmIdx]])) * asp[1L]
            vpy <- median(ym[pfmIdx[this.pfmIdx]]) + vpd * sin(median(alpha[pfmIdx[this.pfmIdx]])) * asp[2L]
            vpx1 <- median(xm[pfmIdx[this.pfmIdx]]) - inner.label.circle.width * cos(median(alpha[pfmIdx[this.pfmIdx]])) * asp[1L] *1.1/5
            vpy1 <- median(ym[pfmIdx[this.pfmIdx]]) - inner.label.circle.width * sin(median(alpha[pfmIdx[this.pfmIdx]])) * asp[2L] *1.1/5
          }
          pushViewport(viewport(x=vpx, y=vpy, width=vpw, height=vph, angle=angle))
          if(class(pfms[[i]])!="psam"){
            plotMotifLogoA(pfms[[i]], font=font, ic.scale=ic.scale, fontsize=fontsize)
          }else{
            plotAffinityLogo(pfms[[i]], font=font, fontsize=fontsize, newpage=FALSE)
          }
          popViewport()
          if(plotIndex) {
            grid.text(label=i, x=vpx1, 
                      y=vpy1, 
                      gp=gpar(col=IndexCol, cex=IndexCex), rot=angle, just="right")
          }
        }else{
          # browser()
          warning(paste("No leave named as ", paste(pfmname, collapse=", ")), sep="")
        }
      }
    }
    rm(list="tmp_motifStack_symbolsCache", envir=.globals)
  }
  if (cleaves > 0) {
    for (i in 1:leaves.number) points(x[i], y[i], pch = 21, col=col.leaves[i],
                                      bg = col.leaves[i], cex = par("cex") * cleaves)
  }
  ang <- rep(0, length(dist.nodes))
  names(ang) <- names(dist.nodes)
  ang <- c(alpha, ang)
  for (i in 1:length(phylog$parts)) {
    w <- phylog$parts[[i]]
    but <- names(phylog$parts)[i]
    ang[but] <- mean(ang[w])
    b <- range(ang[w])
    a.seq <- c(seq(b[1], b[2], by = pi/180), b[2])
    lines(dis[but] * cos(a.seq), dis[but] * sin(a.seq), col="#222222", lwd=par("cex") * clabel.leaves)
    x1 <- dis[w] * cos(ang[w])
    y1 <- dis[w] * sin(ang[w])
    x2 <- dis[but] * cos(ang[w])
    y2 <- dis[but] * sin(ang[w])
    segments(x1, y1, x2, y2, col="#222222", lwd=par("cex") * clabel.leaves)
  } 
  if(!is.na(groupDistance)){
    if(length(groupDistanceLineCol)!=length(groupDistance)){
      groupDistanceLineCol <- rep(groupDistanceLineCol, ceiling(length(groupDistance)/length(groupDistanceLineCol)))[1:length(groupDistance)]
    }
    for(i in 1:length(groupDistance)){
      symbols(x=0, y=0, circles=groupDistance[i], fg=groupDistanceLineCol[i], lty=2, inches=FALSE, add=TRUE)
    }
  }
  if (cnodes > 0) {
    for (i in 1:length(phylog$parts)) {
      w <- phylog$parts[[i]]
      but <- names(phylog$parts)[i]
      ang[but] <- mean(ang[w])
      points(dis[but] * cos(ang[but]), dis[but] * sin(ang[but]),
             pch = 21, bg = "white", cex = par("cex") * cnodes)
    }
  }
  
  ## draw distance indix  
  if(plotAxis){
    wd <- if(!is.null(pfms)) 5 else 4
    if(clockwise){
      vp <- viewport(x=0.5, y=0.5, width=axis_pos/wd, height=.1, 
                     xscale=c(0, max_Dis), angle=init.angle, just=c(0, 0))
      pushViewport(vp)
      grid.xaxis(gp=gpar(cex = par("cex") * clabel.leaves, col="lightgray"),
                 main=TRUE)
      popViewport()
    }else{
      vp <- viewport(x=0.5, y=0.5, width=axis_pos/wd, height=.1, 
                     xscale=c(0, max_Dis), angle=init.angle, just=c(0, 1))
      pushViewport(vp)
      grid.xaxis(gp=gpar(cex = par("cex") * clabel.leaves, col="lightgray"),
                 main=FALSE)
      popViewport()
    }
  }
  
  points(0, 0, pch = 21, cex = par("cex") * 2 * clabel.leaves, bg = "red")
  if (clabel.nodes > 0) {
    delta <- strwidth(as.character(length(dist.nodes)), cex = par("cex") *
                        clabel.nodes)
    for (j in 1:length(dist.nodes)) {
      i <- names(dist.nodes)[j]
      par(srt = (ang[i] * 360/2/pi + 90))
      x1 <- dis[i] * cos(ang[i])
      y1 <- dis[i] * sin(ang[i])
      symbols(x1, y1, delta, bg = "white", add = TRUE,
              inches = FALSE)
      text(x1, y1, nodes.car[j], adj = 0.5, cex = par("cex") *
             clabel.nodes)
    }
  }
  
  if (draw.box)
    box()
  return(invisible())
}

###############################################################################
####### newMotifstack
newMotifStack <-function(hc,
                         phylog,
                         pfms, 
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
    dots <- list(...)
    if(!"phylog" %in% names(dots)){
      if(!psam){
    #   jaspar.scores <- readDBScores(file.path(find.package("MotIV"), "extdata", "jaspar2010_PCC_SWU.scores"))
    #   d <- motifDistances(pfmList2matrixList(pfms), DBscores=jaspar.scores)
    #   hc <- motifHclust(d, method="average")
    #   pfms <- pfms[hc$order]
    #    pfms <- DNAmotifAlignment(pfms)
    #    phylog <- hclust2phylog(hc)
      }
    }
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
               args <- list(hc=hc, phylog=phylog, pfms=pfms, ...)
               for(i in names(args)){
                   if(i %in% c("col.leaves", "col.leaves.bg", "col.bg", "col.inner.label.circle", "col.outer.label.circle")){
                       args[[i]] <- args[[i]][hc$order]
                   }
               }
               ## debugonce(newPlotMotifStackWithRadialPhylog)
               do.call(newPlotMotifStackWithRadialPhylog, args)
           },
           plotMotifLogoStack(pfms, ...)
    )
    
    return(invisible(list(phylog=phylog, pfms=pfms)))
}

############################################################################

result<-motifStackOld(motifs2)
#debugonce(newMotifStack)
## plot logo stack with radial style
newMotifStack(result[["hc"]], result[["phylog"]], pfms, layout="radialPhylog", 
              circle=0.9, cleaves = 0.2, 
              clabel.leaves = 0.5, 
              col.bg=rep(color, each=5), col.bg.alpha=0.3, 
              col.leaves=rep(color, each=5),
              col.inner.label.circle=rep(color, each=5), 
              inner.label.circle.width=0.05,
              col.outer.label.circle=rep(color, each=5), 
              outer.label.circle.width=0.2, 
              circle.motif=1.2,
              angle=360)


# myLabels = c("sv_SOLEXA_5", "CG3919_SANGER_5", "gsb_SOLEXA_5",
#              "gsb_n_SOLEXA_5", "CG12029_SOLEXA_5", "gl_SANGER_5",
#              "srp_FlyReg", "CG32830_SANGER_10", "l_1_sc_da_SANGER_5",
#              "Adf1_SANGER_5", "brk_FlyReg", "Hand_da_SANGER_5",
#              "Fer2_da_SANGER_5", "Espl_FlyReg", "tai_Clk_SANGER_5",
#              "h_SANGER_5", "HLHmgamma_SANGER_10", "lola_PY_SANGER_2_5",
#              "Lag1_SOLEXA", "dl_NBT", "shn_F1_2_SANGER_5",
#              "CG4136_SOLEXA", "Optix_SOLEXA", "Optix_Cell",
#              "z_FlyReg", "lola_PT_SANGER_5", "grh_FlyReg",
#              "br_Z1_FlyReg", "C15_SOLEXA", "lola_PK_SANGER_5",
#              "CG12768_SANGER_5", "CG6276_SANGER_5", "CG3407_SOLEXA_2_5",
#              "her_SANGER_10", "nub_FlyReg", "lola_PJ_SANGER_5",
#              "ken_SANGER_10", "pnr_SANGER_5", "ovo_SOLEXA_5",
#              "Odsh_SOLEXA", "br_SOLEXA_10", "Gsc_Cell",
#              "CG32830_SOLEXA_5", "Pph13_SOLEXA", "Tup_SOLEXA",
#              "CG34031_Cell", "CG9876_Cell", "Unpg_Cell",
#              "sna_SOLEXA_5", "CG17181_SOLEXA_5")

# Documentation: https://www.bioconductor.org/packages/devel/bioc/manuals/motifStack/man/motifStack.pdf


# plotMotifStackWithRadialPhylog: merge motifs by distance threshold
