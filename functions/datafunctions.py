'''
module data:
Preparing csv data for machine learning
'''
from matplotlib.pyplot import style,rc,figure,plot,legend,savefig,\
xticks,yticks,xlabel,ylabel,xscale,yscale,title,tight_layout
from numpy import save,load,array,min,max,std,newaxis,\
mean,vstack,where,dot,cumsum,sum,savetxt,split
from scipy.linalg import svd
from pandas import DataFrame,read_csv
from os.path import join,isfile
from os import listdir,remove
from inspect import currentframe

# Common Names

Features = "features.csv"

# Defined Dictionaries

fdict = {'X':'Points:0', 'Y':'Points:1', 'Z':'Points:2',
         'U':'U:0', 'V':'U:1', 'W':'U:2', 'P':'p', 'T':'T',
         'WX':'vorticity:0','WY':'vorticity:1','WZ':'vorticity:2'}

# Shared Functions

def fileRemove(name):
  if isfile(name): remove(name)

def Exit(msg,arg=None):
  if arg is None: exit(msg)
  else: exit(msg%tuple(arg))

def retrieve_filename(var):
  callers_local_vars = currentframe().f_back.f_back.f_locals.items()
  return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def Exit(msg,arg=None):
  if arg is None: exit(msg)
  else: exit(msg%tuple(arg))

def meanSub(T): 
  return T - T.mean(axis=1).reshape(T.shape[0],1)

def cumSum(S):  
  return cumsum(S)/sum(S)

def rsplit(r=None,p=0,s=1):
  return split(r,s,axis=0)[p]

# Data Functions
#---------------
# fld = folder, f = filed, nm = file name, flst = field list,
# nrm = normalization method: 
# mmx: min-max, pmo: +/-1, msd: mean-std  
# ff = sampling field (from file), ftr = feature list of sample

def takeFeatures(fld=None,pth=None,nm=None):
  ftr,ftv = [],[] 
  nsmp = 0 
  for f in listdir(fld):          
    fn = f.rsplit(".",1)[0].split("-")
    if nsmp == 0: ftr = fn[::2]
    ftv.append(list(map(float,fn[1::2]))) 
    DataFrame(columns=ftr,data=array(ftv)).to_csv(join(pth,nm+"-"+Features),index=False) 
    nsmp += 1
  print("Features are taken:",nsmp)
  return nsmp,ftr,ftv

def takeField(fld=None,f=None,nrm=None,wrt=False,nm=False,pth=None,frst=False):
  ftr,ftv = [],[] 
  data = [] 
  nsmp = 0 
  for fi in listdir(fld):
    fn = fi.rsplit(".",1)[0].split("-")
    print(fn)
    if nsmp == 0: ftr = fn[::2]
    ftv.append(list(map(float,fn[1::2]))) 
    DataFrame(columns=ftr,data=array(ftv)).to_csv(join(pth,nm+"-2-"+Features),index=False) 
    data.append(read_csv(join(fld,fi),usecols=[fdict[f]]).to_numpy().T.flatten())
    if frst is True: break
    nsmp += 1 
  data = array(data).T
  stat = [min(data,axis=0),max(data,axis=0),mean(data,axis=0),std(data,axis=0)]
  if nrm == "mmx": data = (data-stat[0])/(stat[1]-stat[0])
  if nrm == "pmo": data = -1.0 + 2.0*(data-stat[0])/(stat[1]-stat[0])
  if nrm == "msd": data = (data-stat[2])/stat[3]
  if wrt is True: save(join(pth,nm+"-2"+".npy"),data)
  print("Field %s is taken"%f)
  return data

def combineField(flst=None,wrt=False,nm=False,pth=None):
  data = flst[0]
  for i in range(1,len(flst)): 
    data = vstack((data,flst[i]))
  if wrt is True: save(join(pth,nm+".npy"),data)
  print("Fields are combined.")
  return data

def selectSample(ff=None,ftr=None,id=None,axis="col",pth=None):
    sfd = load(join(pth,ff+".npy"))
    if id is None: # select by features
      idf = where((read_csv(join(pth,Features)).to_numpy() == ftr).all(axis=1))[0]
    if id is not None: # select by id number
      idf = id
    print("Sample %d is selected from %s"%(idf,ff))
    if axis == "col": return sfd[:,idf].flatten()
    if axis == "row": return sfd[idf,:].flatten()

#=== UNDER PREPARATION

# def CaseSplit(X=None,rstate=None,spsize=None,dir=None,name=None):
#   Xt,Xs,yt,ys,idt,ids = train_test_split(X,id,id,random_state=0,test_size=0.2)
#   Ft = FT[idt]; Fs = FT[ids]
#   ff = open(join(dir,name+"-train-ftr.txt"),"w")
#   for i in range(len(idt)):
#       ff.writelines("Test Case = %4d, Ra = %9.0f, Re = %5.0f\n"
#       %(idt[i],Ft[i,0],Ft[i,1]))
#   ff.close()
#   ff = open(join(dir,name+"an2d-test-ftr.txt"),"w")
#   for i in range(len(ids)):
#       ff.writelines("Test Case = %4d, Ra = %9.0f, Re = %5.0f\n"
#       %(ids[i],Fs[i,0],Fs[i,1]))
#   ff.close()

# def findCase(Case=None,):
#   get_case = None
#   get_pred = None
#   get_trn  = None

#   try:
#     get_case = int(where((FT == CASE_CHECK).all(axis=1))[0])
#     print("Test Case in Case = ",get_case)
#   except:
#     print("-> Case Chech ",CASE_CHECK," is not in the original list")
#   try:
#     get_pred = int(where((Fs == CASE_CHECK).all(axis=1))[0])
#     print("Test Case in idx_tst = ",get_pred)
#   except:
#     print("-> Case Chech ",CASE_CHECK," is not in the test list")
#   try:
#     get_trn  = int(where((Ft == CASE_CHECK).all(axis=1))[0])
#     print("Test Case in idx_trn = ",get_trn)
#   except:
#     print("-> Case Chech ",CASE_CHECK," is not in the train list")

#   print("get_case = ",get_case)
#   print("get_pred = ",get_pred)
#   print("get_trn  = ",get_trn)

#===

def reduction(D=None,ord=None,nm=False,pth=None):
  #-----------------------------
  # R[ord,nsamp]: reduced space
  # Prev Version: r = np.dot(X.T,U) 
  # R = dot(D.T,U) 
  #-----------------------------
  U,S,VT = svd(D,full_matrices=False)
  CS = cumSum(S).reshape(-1,1)
  R  = (VT*S[:,newaxis]).T
  save   (join(pth,"modes-"+nm+".npy"),U[:,:ord])
  save   (join(pth,"reduc-"+nm+".npy"),R)
  savetxt(join(pth,"evals-"+nm+".txt"),S[:ord], fmt="%.20e")
  savetxt(join(pth,"cumsm-"+nm+".txt"),CS[:ord],fmt="%.20e")
  print('Field %s is reduced.'%nm)

def reconstruction(ff=None,R=None,ord=None,pth=None):
  U = load(join(pth,ff+".npy"))
  return sum(U[:,:ord]*R[:ord],axis=1)      

# Plot Functions and Settings

Palette = \
  {"lk" :['k', '-',2],"lb" :['b', '-',2],"lr" :['r', '-',2],"lg" :['g', '-',2],
   "dk" :['k','--',2],"db" :['b','--',2],"dr" :['r','--',2],"dg" :['g','--',2],
   "ddk":['k','-.',2],"ddb":['b','-.',2],"ddr":['r','-.',2],"ddg":['g','-.',2],
   "ck" :['k', 'o',1],"cb" :['b', 'o',1],"cr" :['r', 'o',1],"cg" :['g', 'o',1],
   "tk" :['k', '^',1],"tb" :['b', '^',1],"tr" :['r', '^',1],"tg" :['g', '^',1],
   "sk" :['k', 's',1],"sb" :['b', 's',1],"sr" :['r', 's',1],"sg" :['g', 's',1],
   "lck":['k','-o',2],"lcb":['b','-o',2],"lcr":['r','-o',2],"lcg":['g','-o',2],
   "ltk":['k','-^',2],"ltb":['b','-^',2],"ltr":['r','-^',2],"ltg":['g','-^',2],
   "lsk":['k','-s',2],"lsb":['b','-s',2],"lsr":['r','-s',2],"lsg":['g','-s',2]}

def graph(name="Test",X=None,y=None,pd=None,pth=None):
  #-------------------
  # Plot configuration
  #-------------------
  normValue = 10
  plotSize = len(X)
  if "dark" in pd and pd["dark"] is "yes": style.use("dark_background")
  if isinstance(X,list) is False: Exit("\n [Error] Plot: Input X/y is not list.\n")
  if "legend" not in pd: # Set Legend
    setLegend = False
  else: 
    setLegend = True; Legend = pd["legend"]  
    if "legsize" not in pd: legSize = normValue
    else: legSize = pd["legsize"]
  if "title" not in pd: # Set Tile
    setTitle = False
  else: 
    setTitle = True; Title = pd["title"]
    if "titlesize" not in pd: titleSize = normValue
    else: titleSize = pd["titlesize"]
  if "ticksize" not in pd: # Set Ticks 
    tickSize = normValue
  else: tickSize = pd["ticksize"]
  if "labelsize" not in pd: # Set Lable 
    labSize = normValue
  else: labSize = pd["labelsize"]
  if "figsize" not in pd: # Set Figure Size
    figSize = (6.4,4.8)
  else: figSize = pd["figsize"]
  if "xlog" not in pd: xlog = 0 # Set Log Scales
  else: xlog = pd["xlog"]
  if "ylog" not in pd: ylog = 0
  else: ylog = pd["ylog"]
  rc("axes",labelsize=labSize)
  xtck,ytck = pd["xticks"],pd["yticks"]
  xlab,ylab = pd["xlabel"],pd["ylabel"]
  fig = figure()
  fig.set_figwidth(figSize[0])
  fig.set_figheight(figSize[1])
  pt = pd["pallet"]
  #-----------------
  # Plot diagram set
  #-----------------
  for p in range(plotSize):
    xx = array(X[p])
    yy = array(y[p])
    if "sort" in pd and pd["sort"] is "yes": 
      ix = xx.argsort()
      xx = xx[ix]
      yy = yy[ix]
    plot(xx,yy,pt[p][0]+pt[p][1],linewidth=pt[p][2])
  xticks(xtck,fontsize=tickSize); yticks(ytck,fontsize=tickSize)
  xlabel(xlab); ylabel(ylab)
  if xlog == 1: xscale("log")
  if ylog == 1: yscale("log")
  if setTitle is True: title(Title,fontsize=titleSize)
  if setLegend is True:
    if pd["putleg"] in pd:
      legend(Legend,ncol=len(Legend),bbox_to_anchor=(pd["putleg"][0],pd["putleg"][1]),
             prop={'size':legSize})
    else: legend(Legend,prop={'size':legSize})
  if "tight" in pd and pd["tight"] is "yes": tight_layout()
  savefig(join(pth,name),dpi=300)

def make_csv(cols=None,data=None,nm=None,pth=None):
  clist = ["X","Y","Z","U","V","W","P"]
  for c in clist:
    if c in cols: cols[cols.index(c)] = fdict[c]
  DataFrame(columns=cols,data=array(data).T).to_csv(
    join(pth,nm+".csv"),index=False)