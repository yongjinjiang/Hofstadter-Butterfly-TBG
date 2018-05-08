import numpy as np
from math import sqrt,pi,cos
from mpmath import asec
import sys

def neighbours(syst0,SC1,SC2,sc1L,sc1R,sc2L,sc2R,Base_a1,Base_a2,a1p,a2p,N_ToT,a,b,a1,b1,f1,f2,delta_a,s1x,s1y,s2x,s2y,m1,m2):
    xm,ym,zm=SC1/2+SC2/2;
    #n1,n2=(np.array([[1/sqrt(3),-1],[1/sqrt(3),1]])@np.array([xm,ym])).astype(int)
    n1m,n2m=((np.array((np.matrix([Base_a1[:2],Base_a2[:2]]).T).I))@np.array([xm,ym])).astype(int)
    n1pm,n2pm=((np.array((np.matrix([a1p[:2],a2p[:2]]).T).I))@np.array([xm,ym])).astype(int)


    ri=a(n1m,n2m).pos;     ria_tag=a(n1m,n2m).tag;
    xi,yi,zi=ri; ria=np.array([xi*f1,yi*f1,zi*f2]);
    
    ri=b(n1m,n2m).pos;     rib_tag=b(n1m,n2m).tag;
    xi,yi,zi=ri; rib=np.array([xi*f1,yi*f1,zi*f2]);
    
    ri=a1(n1pm,n2pm).pos;  ria1_tag=a1(n1pm,n2pm).tag;
    xi,yi,zi=ri; ria1=np.array([xi*f1,yi*f1,zi*f2]);
    
    ri=b1(n1pm,n2pm).pos;  rib1_tag=b1(n1pm,n2pm).tag;
    xi,yi,zi=ri; rib1=np.array([xi*f1,yi*f1,zi*f2]);
    
  #  f1=sqrt(3)*delta_a;
  #  f2=d_0;
    
    nns_a_tag=[];
    nns_b_tag=[];
    nns_a1_tag=[];
    nns_b1_tag=[];
    
    for nj in range(N_ToT):
             rj0=syst0.sites[nj].pos;
             xj,yj,zj=rj0;
             rj_abs=np.array([xj*f1,yj*f1,zj*f2]);
             
             rj_tag=syst0.sites[nj].tag
             rjtag_1,rjtag_2=rj_tag;
             
             if sum(abs(a(rjtag_1,rjtag_2).pos-rj0))<10**-8: which_sublattice=0;
             if sum(abs(b(rjtag_1,rjtag_2).pos-rj0))<10**-8: which_sublattice=1;
             if sum(abs(a1(rjtag_1,rjtag_2).pos-rj0))<10**-8: which_sublattice=2;
             if sum(abs(b1(rjtag_1,rjtag_2).pos-rj0))<10**-8: which_sublattice=3;
             
             #print('which_sublattice=',which_sublattice)
             #for st in {'a','b','a1','b1'}:
             ss=ria-rj_abs;
             dis=sqrt(np.dot(ss,ss))
             if (dis<5*delta_a):
                ss1=rj_tag-ria_tag; ss1=list(ss1);ss1.append(which_sublattice);
                nns_a_tag.append(tuple(ss1))

             
             ss=rib-rj_abs;
             dis=sqrt(np.dot(ss,ss))
             if (dis<5*delta_a):
                 ss1=rj_tag-rib_tag;ss1=list(ss1);ss1.append(which_sublattice);
                 nns_b_tag.append(tuple(ss1))


             ss=ria1-rj_abs;
             dis=sqrt(np.dot(ss,ss))
             if (dis<5*delta_a):
                ss1=rj_tag-ria1_tag;ss1=list(ss1);ss1.append(which_sublattice);
                nns_a1_tag.append(tuple(ss1))


             ss=rib1-rj_abs;
             dis=sqrt(np.dot(ss,ss))
             if (dis<5*delta_a):
                ss1=rj_tag-rib1_tag;ss1=list(ss1);ss1.append(which_sublattice);
                nns_b1_tag.append(tuple(ss1))
                
    n1i,n2i=syst0.sites[0].tag
    n1f,n2f=syst0.sites[N_ToT-1].tag
    
    for i in range(N_ToT):
            n1,n2=syst0.sites[i].tag
            if n1<n1i:
               n1i=n1;
            if n1>n1f:
               n1f=n1;
            
            if n2<n2i:
               n2i=n2;
            if n2>n2f:
               n2f=n2;
        #print('n1i,n1f,n2i,n2f',n1i,n1f,n2i,n2f)
#
    nn1=n1f-n1i+1;nn2=n2f-n2i+1;
    tag2id=np.ones((nn1,nn2,4),dtype=int)*(-100);  #tags of B sublattice
    id2tag=[[] for i in range(N_ToT)];

    def xyz_for_sublattice_i(n1,n2,which_i):
         if which_i==0:
           return a(n1,n2).pos
         elif which_i==1:
           return b(n1,n2).pos
         elif which_i==2:
           return a1(n1,n2).pos
         else: #which_j==3:
           return b1(n1,n2).pos

    i= 0; #id2tag=np.zeros(N_ToT,dtype=int)
    for which_i in range(4):
       for n1 in range(n1i,n1f+1):
          for n2 in range(n2i,n2f+1):
             xR,yR,zR=xyz_for_sublattice_i(n1,n2,which_i)
             ss1=s1x*xR+s1y*yR;
             ss2=s2x*xR+s2y*yR;
             if (sc1L<=ss1<sc1R)&(sc2L<=ss2<sc2R):
                tag2id[n1-n1i,n2-n2i,which_i]=i  #array indices should be >=0
                id2tag[i]=(n1,n2,which_i)
                i+=1

#    for which_i in range(4):
#       for n1 in range(n1i,n1f+1):
#          for n2 in range(n2i,n2f+1):
#             xR,yR,zR=xyz_for_sublattice_i(n1,n2,which_i)
#             ss1=s1x*xR+s1y*yR;
#             ss2=s2x*xR+s2y*yR;
#             if (sc1L<=ss1<sc1R)&(sc2L<=ss2<sc2R):
#                  print('tag2id=',n1,n2,which_i,tag2id[n1-n1i,n2-n2i,which_i])
#
#    print('correct numbering0')
#    sys.exit()
#    for n1 in range(n1i,n1f+1):
#          for n2 in range(n2i,n2f+1):
#             xR,yR,zR=a(n1,n2).pos
#             ss1=s1x*xR+s1y*yR;
#             ss2=s2x*xR+s2y*yR;
#             if (sc1L<=ss1<sc1R)&(sc2L<=ss2<sc2R):
#                tag2id[n1-n1i,n2-n2i,0]=i  #array indices should be >=0
#                id2tag[i]=(n1,n2,0)
#                i+=1
#
#    for n1 in range(n1i,n1f+1):
#          for n2 in range(n2i,n2f+1):
#             xR,yR,zR=b(n1,n2).pos
#             ss1=s1x*xR+s1y*yR;
#             ss2=s2x*xR+s2y*yR;
#             if (sc1L<=ss1<sc1R)&(sc2L<=ss2<sc2R):
#                tag2id[n1-n1i,n2-n2i,1]=i  #array indices should be >=0
#                id2tag[i]=(n1,n2,1)
#                i+=1
#    for n1 in range(n1i,n1f+1):
#          for n2 in range(n2i,n2f+1):
#             xR,yR,zR=a1(n1,n2).pos
#             ss1=s1x*xR+s1y*yR;
#             ss2=s2x*xR+s2y*yR;
#             if (sc1L<=ss1<sc1R)&(sc2L<=ss2<sc2R):
#                tag2id[n1-n1i,n2-n2i,2]=i  #array indices should be >=0
#                id2tag[i]=(n1,n2,2)
#                i+=1
#
#    for n1 in range(n1i,n1f+1):
#          for n2 in range(n2i,n2f+1):
#             xR,yR,zR=b1(n1,n2).pos
#             ss1=s1x*xR+s1y*yR;
#             ss2=s2x*xR+s2y*yR;
#             if (sc1L<=ss1<sc1R)&(sc2L<=ss2<sc2R):
#                tag2id[n1-n1i,n2-n2i,3]=i  #array indices should be >=0
#                id2tag[i]=(n1,n2,3)
#                i+=1
#
    if (i!=N_ToT):
            print('total number not right');sys.exit()
    print('correct numbering')
    #sys.exit()
    def nn_pos_for_ni(n1,n2,nnjs,nj):
         n1R_j=n1+nnjs[nj][0];
         n2R_j=n2+nnjs[nj][1];
         which_j=nnjs[nj][2]
         
         if which_j==0:
           return a(n1R_j,n2R_j).pos,n1R_j,n2R_j,which_j
         elif which_j==1:
           return b(n1R_j,n2R_j).pos,n1R_j,n2R_j,which_j
         elif which_j==2:
           return a1(n1R_j,n2R_j).pos,n1R_j,n2R_j,which_j
         else: #which_j==3:
           return b1(n1R_j,n2R_j).pos,n1R_j,n2R_j,which_j




    boundary_sites=[];
    bulk_B=[[] for i in range(N_ToT)]

   # nn_length=[[] for i in range(4)];
    #nn_length[0]=len(nns_a_tag); nn_length[1]=len(nns_b_tag);
    #nn_length[2]=len(nns_a1_tag);nn_length[3]=len(nns_b1_tag);

    nns=[[] for i in range(N_ToT)];
    
    nnjs=[[] for i in range(4)];
    nnjs[0]=nns_a_tag; nnjs[1]=nns_b_tag; nnjs[2]=nns_a1_tag; nnjs[3]=nns_b1_tag;
    nn_length=[len(nnjs[i]) for i in range(4)]
    print('nn_length=',nn_length)
    num_nn=[0 for i in range(N_ToT)]
    
    

    a1p_a1=1/2 *(1 + np.sqrt((m1**2 - m2**2)**2/(m1**2 + m1*m2 + m2**2)**2)+ (3*m1*m2)/(m1**2 + m1*m2 + m2**2));
    a2p_a1=-((2 *cos(pi/6 + asec(2 - (6*m1*m2)/(m1**2 + 4*m1*m2 + m2**2))))/sqrt(3));
    a1p_a2= -a2p_a1;
    a2p_a2=np.sqrt((m1**2 - m2**2)**2/(m1**2 + m1*m2 + m2**2)**2)


    for n1 in range(n1i,n1f+1):
        for n2 in range(n2i,n2f+1):
          for which in range(4):
             xR,yR,zR=xyz_for_sublattice_i(n1,n2,which)
             ss1=s1x*xR+s1y*yR;
             ss2=s2x*xR+s2y*yR;
             #kk+=1;print(kk)
             if (sc1L<=ss1<sc1R)&(sc2L<=ss2<sc2R):
                nn_count=0
                for nj in range(nn_length[which]):
                   (xR_j,yR_j,zR_j),n1R_j,n2R_j,which_j=nn_pos_for_ni(n1,n2,nnjs[which],nj)
                   ss1_j=s1x*xR_j+s1y*yR_j;
                   ss2_j=s2x*xR_j+s2y*yR_j;
                   for n1s in range(-1,2):
                      for n2s in range(-1,2):
                        if (sc1L<=ss1_j-n1s*(sc1R-sc1L) <sc1R)&(sc2L<=ss2_j-n2s*(sc2R-sc2L) <sc2R):
                            if which_j in [0,1]:
                               n1R_js=n1R_j-n1s*(int(round(sc1R-sc1L)))*m1-n2s*  (int(round(sc2R-sc2L)))*(-m2);
                               n2R_js=n2R_j-n1s*(int(round(sc1R-sc1L)))*(m2)-n2s*(int(round(sc2R-sc2L)))*(m1+m2); #print('n1R_js=',n1R_js,'n2R_js=',n2R_js)
                               nns[tag2id[n1-n1i,n2-n2i,which]].append((tag2id[n1R_js-n1i,n2R_js-n2i,which_j],n1s,n2s))  #array indices should be >=0
                            
                            elif which_j in [2,3]:
                               mm11=int(round(m1*a1p_a1 +m2*a1p_a2));  mm21=int(round(-m2*a1p_a1 + (m1+m2)*a1p_a2));
                               mm12=int(round(m1*a2p_a1 +m2*a2p_a2));  mm22=int(round(-m2*a2p_a1 + (m1+m2)*a2p_a2));
                               #print(mm11,mm21,mm12,mm22)
                               #sys.exit()
                               n1R_js=n1R_j-n1s*(int(round(sc1R-sc1L)))*mm11-n2s*(int(round(sc2R-sc2L)))*mm21;
                               n2R_js=n2R_j-n1s*(int(round(sc1R-sc1L)))*mm12-n2s*(int(round(sc2R-sc2L)))*mm22; #print('n1R_js=',n1R_js,'n2R_js=',n2R_js)
                               nns[tag2id[n1-n1i,n2-n2i,which]].append((tag2id[n1R_js-n1i,n2R_js-n2i,which_j],n1s,n2s))  #array indices should be >=0
                            #bulk_B[tag2id[n1R_j-n1i,n2R_j-n2i,which_j].append(tag2id[n1-n1i,n2-n2i,which])
                            nn_count+=1;
             
                            
                num_nn[tag2id[n1-n1i,n2-n2i,which]]=nn_count;

    #print('num_nn=',num_nn)
    #sys.exit()
    boundary_sites=[];
    for i in range(N_ToT):
        n1,n2,which=id2tag[i];
        if ((num_nn[i])!=nn_length[which]):
            boundary_sites.append(i)
    print('boundary_sites=',len(boundary_sites))

#    for ni in boundary_sites:
#        ri0=syst0.sites[ni].pos;      xi,yi,zi=ri0;  ri=np.array([xi*f1,yi*f1,zi*f2]);
#        for nj in boundary_sites:
#             rj0=syst0.sites[nj].pos;
#             for n1 in range(-1,2):
#                for n2 in range(-1,2):
#                   if not((n1==0)&(n2==0)):
#                     rjn=rj0+(n1*SC1+n2*SC2); xj,yj,zj=rjn;
#                     rj_abs=np.array([xj*f1,yj*f1,zj*f2]);
#                     ss=ri-rj_abs;
#                     dis=sqrt(np.dot(ss,ss));
#                     if (dis<5*delta_a):
#                           nns[ni].append((nj,n1,n2))

    for ni in range(N_ToT):
        n1,n2,which=id2tag[ni];
        if (len(nns[ni])!=nn_length[which]):
           print(len(nns[ni]),nn_length[which])
           print('number of neighbors not equal to len[nns[ni]]:  wrong!')
           sys.exit()


    return nns_a_tag,nns_b_tag,nns_a1_tag,nns_b1_tag,tag2id,n1i,n1f,n2i,n2f,nns
    #return nns
