#construct the Twisted graphene Bilayer model without magnetic field
#(REf: PHYSICAL REVIEW B 88, 125426 (2013):Periodic Landau gauge and quantum Hall effect in twisted bilayer graphene)
#3D model(with hoppings and magnetic field, Finite system, without kx,ky,etc)
from math import pi, sqrt, tanh, exp,acos,cos,sin
import csv
import sys
import tinyarray
import time
import os
import kwant
import cmath
# For computing eigenvalues
import scipy.sparse.linalg as sla
from numpy import linalg as LA
import numpy as np
import matlab.engine

# For plotting
from matplotlib import pyplot as plt
from datetime import datetime
#lattice parameters:
t=2.7;      #eV
t12=-0.48;  #eV
delta_a=0.142;    #nm:  all intraplane distance are in unit of sqrt(3)*delta_a
d_0=0.335*1000;    #nm:  all interplane distance are in unit of (delta_0)
delta_12=0.0453;  #nm: interplane decay length
#delta_12=0.026128;
delta=delta_12;   #nm: inplane decay length

#lattice of the supercells
(m1,m2)=(1,0);


# Define the graphene lattice
sin_30, cos_30 = (1 / 2, sqrt(3) / 2)
Base_a1=(cos_30, -sin_30,0);   Base_a2=(cos_30,sin_30,0);
A_position=(0, 0,0)
B_position=(1 / sqrt(3),0,0)


L1=m1*np.matrix(Base_a1)+m2*np.matrix(Base_a2);
L1_prime=m2*np.matrix(Base_a1)+m1*np.matrix(Base_a2);

alpha=acos((m1**2+4*m1*m2+m2**2)/(2*(m1**2+m1*m2+m2**2)));

M_Rotation=[[cos(pi/3-alpha),-sin(pi/3-alpha),0],[sin(pi/3-alpha),cos(pi/3-alpha),0],[0,0,1]];

#M_Rotation=[[cos(-alpha),-sin(-alpha),0],[sin(-alpha),cos(-alpha),0],[0,0,1]];


M_Rotation3=[[cos(pi/3),-sin(pi/3),0],[sin(pi/3),cos(pi/3),0],[0,0,1]];

a1p=np.transpose(np.matrix(M_Rotation)*np.transpose(np.matrix(Base_a1)));
a1p=a1p.tolist();a1p[0][2]=0;
a1p=tuple(a1p[0]);
a2p=np.transpose(np.matrix(M_Rotation)*np.transpose(np.matrix(Base_a2)));
a2p=a2p.tolist();a2p[0][2]=0;
a2p=tuple(a2p[0]);

L2=np.transpose(np.matrix(M_Rotation3)*np.transpose(L1));
L2=L2.tolist();L2=tuple(L2[0]);

L2_prime=np.transpose(np.matrix(M_Rotation3)*np.transpose(L1_prime));
L2_prime=L2_prime.tolist();L2_prime=tuple(L2_prime[0]);

L1=L1.tolist();L1=tuple(L1[0]);
L1_prime=L1_prime.tolist();L1_prime=tuple(L1_prime[0]);



A_Prime_position=np.transpose(np.matrix(M_Rotation)*np.transpose(np.matrix(A_position)));
A_Prime_position=A_Prime_position.tolist();A_Prime_position[0][2]=1;
#print(A_Prime_position)
A_Prime_position=tuple(A_Prime_position[0]);

B_Prime_position=np.transpose(np.matrix(M_Rotation)*np.transpose(np.matrix(B_position)));
B_Prime_position=B_Prime_position.tolist();B_Prime_position[0][2]=1;
B_Prime_position=tuple(B_Prime_position[0]);


graphene = kwant.lattice.general([Base_a1, Base_a2],
                                 [A_position, B_position])
a, b = graphene.sublattices

graphene1 = kwant.lattice.general([a1p, a2p],
                                 [A_Prime_position, B_Prime_position])
a1, b1 = graphene1.sublattices

supercell=kwant.lattice.general([L1, L2],[A_position])

c=supercell.sublattices[0]

eta=10**-12;
s1x=(np.sqrt(3)*m1+2*np.sqrt(3)*m2)/(3*(m1**2 + m1* m2 + m2**2));
s1y=(- 3*m1)/(3*(m1**2 + m1* m2 + m2**2));
s2x=(np.sqrt(3)*m1- np.sqrt(3)* m2)/(3* (m1**2 + m1*m2 + m2**2));
s2y=(3*m1+3*m2)/(3* (m1**2 + m1*m2 + m2**2));

f1=sqrt(3)*delta_a;
f2=d_0;
SC1=1;
SC2=1;
def make_system_step1(q=3):
    #def pause():
    #  programPause = raw_input("Press the <ENTER> key to continue...")
    
    print(Base_a1)
    print(Base_a2)
    print(a1p)
    print(a2p)
    print(L1)
    print(L2)
    print(A_Prime_position)
    print(B_Prime_position)
    #sys.exit()
    
    #soa1=np.array([list((0,0)+Base_a1[0:2]),list((0,0)+Base_a2[0:2])])
    soa1=np.array([list((0,0)+a1p[0:2]),list((0,0)+a2p[0:2])])
    #soa1=np.array([list((0,0)+L1[0:2]),list((0,0)+L2[0:2])])
    #soa1=np.array([list((0,0)+L1[0:2]),list((0,0)+L2[0:2]),list((0,0)+L1_prime[0:2]),list((0,0)+L2_prime[0:2])])
    
    #X, Y, U, V = zip(*soa1)
    #plt.figure()
    #ax = plt.gca()
    #ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
    #ax.set_xlim([-2, 4])
    #ax.set_ylim([-2, 4])
    #plt.draw()
    #plt.show()
    #print(a)
    #print(c)

    syst = kwant.Builder()
    #eta=10**-12;
    sc1L=0+eta;sc1R=1+eta; #boundaries along L1 direction
    sc2L=0+eta;sc2R=q+eta; #boundaries along L2 direction
    SC1=(sc1R-eta)*tinyarray.array(L1);
    SC2=(sc2R-eta)*tinyarray.array(L2);  #the magnetic unite cell
 
    start=datetime.now()
    
    s1x=(np.sqrt(3)*m1+2*np.sqrt(3)*m2)/(3*(m1**2 + m1* m2 + m2**2));
    s1y=(- 3*m1)/(3*(m1**2 + m1* m2 + m2**2));
    s2x=(np.sqrt(3)*m1- np.sqrt(3)* m2)/(3* (m1**2 + m1*m2 + m2**2));
    s2y=(3*m1+3*m2)/(3* (m1**2 + m1*m2 + m2**2));

    for n1 in range(-200,200):
       for n2 in range(-200,200):
          
           xR,yR,zR=a(n1,n2).pos;
           ss1=s1x*xR+s1y*yR;
           ss2=s2x*xR+s2y*yR;
           if ((sc1L<=ss1<sc1R)&(sc2L<=ss2<sc2R)):
              syst[a(n1,n2)]=0
          
           xR,yR,zR=b(n1,n2).pos
           ss1=s1x*xR+s1y*yR;
           ss2=s2x*xR+s2y*yR;
           if ((sc1L<=ss1<sc1R)&(sc2L<=ss2<sc2R)):
              syst[b(n1,n2)]=0
           
           xR,yR,zR=a1(n1,n2).pos
           ss1=s1x*xR+s1y*yR;
           ss2=s2x*xR+s2y*yR;
           if ((sc1L<=ss1<sc1R)&(sc2L<=ss2<sc2R)):
              syst[a1(n1,n2)]=0
           
           xR,yR,zR=b1(n1,n2).pos
           ss1=s1x*xR+s1y*yR;
           ss2=s2x*xR+s2y*yR;
           if ((sc1L<=ss1<sc1R)&(sc2L<=ss2<sc2R)):
              syst[b1(n1,n2)]=0

           #xR,yR,zR=c(n1,n2).pos
           #if ((sc1L<=(sqrt(3)*m1*xR+2*sqrt(3)*m2*xR - 3*m1*yR)/(3*(m1**2 + m1* m2 + m2**2))<sc1R)&(sc2L<=(sqrt(3)*m1*xR - sqrt(3)* m2*xR + 3* m1* yR +3* m2* yR)/(3* (m1**2 + m1*m2 + m2**2))<sc2R)):
           #    syst[c(n1,n2)]=0

    print('onsite construction finish:')
    start=datetime.now()
   # print (datetime.now()-start)

    syst0=syst.finalized()
    N_ToT=syst0.graph.num_nodes;
    print(N_ToT);
    
    
    def neighbors():
        nns=[[] for i in range(N_ToT)];
        for ni in range(N_ToT):
          print(ni)
          ri=syst0.sites[ni].pos;
          xi,yi,zi=ri; ri=np.array([xi*f1,yi*f1,zi*f2]);
          for nj in range(N_ToT):
             rj=syst0.sites[nj].pos;
             xj,yj,zj=rj; rj=np.array([xj*f1,yj*f1,zj*f2]);
             for n1 in range(-1,2):
                for n2 in range(-1,2):
                  rjn=rj+(n1*SC1+n2*SC2)*f1;
                  ss=ri-rjn;
                  dis=sqrt(np.dot(ss,ss))
                  if (dis<5*delta_a):
                     if not((n1==0)&(n2==0)&(nj==ni)):
                        nns[ni].append((nj,n1,n2))
        return nns
    nns=neighbors();
    return nns,syst,syst0,SC1,SC2
    
    
    
def make_system_step2(p,k1,k2,syst,syst0,nns,SC1,SC2,q):
    N_ToT=syst0.graph.num_nodes
    start=datetime.now()
    
    
   
    def amp(ri,rj):
      xi,yi,zi=ri; ri=[xi*f1,yi*f1,zi*f2];
      xj,yj,zj=rj; rj=[xj*f1,yj*f1,zj*f2];
      ss=np.array(ri)-np.array(rj);
      dis=sqrt(np.dot(ss,ss))
      #print('dis=',dis,'delta_a=',delta_a,'delta=',delta)
      if abs(zi-zj)<10**-8:
         dis1=dis-delta_a;
         dis1=dis1/delta;
         hop_t=t*np.exp(-dis1);
      else:
           dis2=dis-d_0;
           dis2=dis2/delta;
           hop_t=t12*np.exp(-dis2);
      #print('hop_t=',hop_t)
      return -hop_t
     
    def phase(ri,rj):
       xR,yR,zi=ri;
       #ss1_i=s1x*xR+s1y*yR+2*eta;
       #ss2_i=s2x*xR+s2y*yR;
       ss1_i=(sqrt(3)*m1*xR+2*sqrt(3)*m2*xR - 3*m1*yR)/(3*(m1**2 + m1* m2 + m2**2))  +2*eta;
       ss2_i=(sqrt(3)*m1*xR - sqrt(3)* m2*xR + 3* m1* yR +3* m2* yR)/(3* (m1**2 + m1*m2 + m2**2));
       xR,yR,zj=rj;
       ss1_j=(sqrt(3)*m1*xR+2*sqrt(3)*m2*xR - 3*m1*yR)/(3*(m1**2 + m1* m2 + m2**2))  +2*eta;
       ss2_j=(sqrt(3)*m1*xR - sqrt(3)* m2*xR + 3* m1* yR +3* m2* yR)/(3* (m1**2 + m1*m2 + m2**2));
       #ss1_j=s1x*xR+s1y*yR+2*eta;
       #ss2_j=s2x*xR+s2y*yR;
       ss=(ss1_i+ ss1_j - 4*eta)/2;
       phase1=(ss- np.floor(ss))*(ss2_j - ss2_i);
       phase2=0;
       #print('ssi=',ss1_i,ss2_i)
       #print('ssi=',ss1_j,ss2_j)
       
       #print('np.floor(ss1_i)=',np.floor(ss1_i))
       #print('np.floor(ss1_j)=',np.floor(ss1_j))
       
       if abs(np.floor(ss1_i)-np.floor(ss1_j))==1:
          mn1=max(np.floor(ss1_i),np.floor(ss1_j))
          ss = (mn1-ss1_j)/(ss1_i - ss1_j);  #r(ss)=ss*r_i+(1-ss)*r_j;
          phase2 = -(ss*ss2_i + (1 - ss)*ss2_j)
          #print('mn1=',mn1,'ss=',ss,'ss2_i=',ss2_i,'ss2_j=',ss2_j,'phase2=',phase2)
       phase1=np.exp(2*pi*1j*phase1*p/q )
       phase2=np.exp(2*pi*1j*phase2*p/q )
       return phase2*phase1
       
#       
#    def Given_link_ijs(family_i,tag_i,family_j,tag_j):
#       print(family_i,family_j)
#       print(tag_i,tag_j)
#       for ni in range(N_ToT):
#          if (syst0.sites[ni].family==family_i)&(syst0.sites[ni].tag==tag_i):
#             n_i=ni;
#            
#             break
#       for nj in range(N_ToT):
#          if (syst0.sites[nj].family==family_j)&(syst0.sites[nj].tag==tag_j):
#             n_j=nj;
#             break
#       return n_i,n_j
#
#
#
#    n_i,n_j=Given_link_ijs(a,tinyarray.array([0,1]),b,tinyarray.array([-1,1]))
#    ri=syst0.sites[n_i].pos; rj=syst0.sites[n_j].pos;
#    
#    
#   # ri=tinyarray.array([0,0,1]); rj=tinyarray.array([3.1,0.1,1]);
#    p=1;
#    print('q=',q)
#    print(phase(ri,rj,p))
#    print(phase(ri+SC1,rj+SC1,p))
#    print(phase(ri+6*tinyarray.array(L2),rj+6*tinyarray.array(L2),p))
#    print(phase(ri+q*tinyarray.array(L2),rj+q*tinyarray.array(L2),p))
#    sys.exit()

    
    def set_hoppings(ri,rj,n1n2_is):
       # k1,k2,p=k_and_p
       # ri=site1.pos;
       # rj=site2.pos;
        #if (site1.pos==tinyarray.array([0,0]))&(site2.pos==tinyarray.array([0,0]))&(site1.family=='a')&(site2.family=='b'):
         #  print('k_and_p=',k1,k2,p)
        #print(site1.tag,site2.tag,syst0.sites[0].tag)
        #print(site1.family,site2.family,syst0.sites[0].family)
        non_diag_term=0;
        #print('Again, n1n2_is=',n1n2_is)
        #print('hopping k1=',k1)
        #if (ri==syst0.sites[0].pos):
         # print('insethoppings_n1n2_is=',n1n2_is)
        #time.sleep(10)
#        (for i in range(len(n1n2_is)):
#             n1,n2=n1n2_is[i]
#             rjn=rj+n1*SC1+n2*SC2
#             non_diag_term=non_diag_term+np.exp(1j*(n1*k1+n2*k2))*amp(ri,rjn) *phase(ri,rjn);)

             #if not((n1==0)&(n2==0)):
                #print('n1=',n1,'n2=',n2,'Hamiltonian_elements=',np.exp(1j*(n1*k1+n2*k2))*amp(ri,rjn))
                
        for n1 in range(-1,2):
            for n2 in range(-1,2):
                 rjn=rj+n1*SC1+n2*SC2;
                 non_diag_term=non_diag_term+np.exp(1j*(n1*k1+n2*k2))*amp(ri,rjn)*phase(ri,rjn);
        rj1=rj-SC1
        rj2=rj-SC2;
        #non_diag_term=t*(1*phase(ri,rj)+np.exp(-1j*k2)*phase(ri,rj1)+np.exp(-1j*k1)*phase(ri,rj2))
        if abs(ri[2]-rj[2])>10**-3:
          non_diag_term=0
        return non_diag_term


    def set_hoppings1(ri,rj):
       # k1,k2,p=k_and_p
       # ri=site1.pos;
       # rj=site2.pos;
        #if (site1.pos==tinyarray.array([0,0]))&(site2.pos==tinyarray.array([0,0]))&(site1.family=='a')&(site2.family=='b'):
         #  print('k_and_p=',k1,k2,p)
        #print(site1.tag,site2.tag,syst0.sites[0].tag)
        #print(site1.family,site2.family,syst0.sites[0].family)
        non_diag_term=0;
        #print('Again, n1n2_is=',n1n2_is)
        #print('hopping k1=',k1)
        #if (ri==syst0.sites[0].pos):
         # print('insethoppings_n1n2_is=',n1n2_is)
        #time.sleep(10)
        for n1 in range(-2,3):
            for n2 in range(-2,3):
              rjn=rj+n1*SC1+n2*SC2
              non_diag_term=non_diag_term+np.exp(1j*(n1*k1+n2*k2))*amp(ri,rjn) #*phase(ri,rjn);
             #if not((n1==0)&(n2==0)):
                #print('n1=',n1,'n2=',n2,'Hamiltonian_elements=',np.exp(1j*(n1*k1+n2*k2))*amp(ri,rjn))
       # for n1 in range(-5,6):
        #    for n2 in range(-5,6):
                # rjn=rj+n1*SC1+n2*SC2;
                # non_diag_term=non_diag_term+np.exp(1j*(n1*k1+n2*k2))*amp(ri,rjn)*phase(ri,rjn,p);
        return non_diag_term



    def on_site_term(ri):
        #  k1,k2,p=k_and_p
          #print('on_site k1=',k1)
          #ri=site.pos;
          diag_term=0;
          for n1 in range(-1,2):
              for n2 in range(-1,2):
                 rjn=ri+n1*SC1+n2*SC2;
                 if not((n1==0)and(n2==0)):
                   diag_term=diag_term+np.exp(1j*(n1*k1+n2*k2))*amp(ri,rjn) *phase(ri,rjn);
          #sys.exit()
          return diag_term
        
    def rearrange(nn_i0,n1n2_i0):
       n1n2_is=[[] for i in range(len(nn_i0))];
       nn_is=[[] for i in range(len(nn_i0))];
       ns=0;
       
       n1n2_is[ns]=[n1n2_i0[0]];
       nn_is[ns]=nn_i0[0];
       for i in range(1,len(nn_i0)):
          if nn_i0[i]==nn_is[ns]:
             n1n2_is[ns].append(n1n2_i0[i]);
          else:
             ns=ns+1;
             nn_is[ns]=nn_i0[i];
             n1n2_is[ns]=[n1n2_i0[i]]
       return n1n2_is, ns, nn_is

    for ni in range(0,N_ToT):
          #print('ni=',ni);
          ri=syst0.sites[ni].pos;
          nnn_sites=nns[ni];lennn_i=len(nnn_sites);
          nn_i=[nnn_sites[i][0] for i in range(lennn_i)];
          #print('nn_i=',nn_i)
          
          n1n2_i=[nnn_sites[i][1:3] for i in range(lennn_i)];
          #print('lennn_i=',lennn_i,'n1n2_i=',n1n2_i)
          
          
          n1n2_i, ns, nn_i=rearrange(nn_i,n1n2_i); #压缩近邻
          #print('After rearrangement:n1n2_i=',n1n2_i,'ns=,',ns+1)
          #print('ni=',ni)
          #print('nn_i=',nn_i)
          #sys.exit()
        #set hopping
          for nj in range(ns+1):
               n1n2_is=n1n2_i[nj];
               #print('n1n2_is=',n1n2_is)
               if (nn_i[nj]>ni):
                  rj=syst0.sites[nn_i[nj]].pos;
                  syst[syst0.sites[ni],syst0.sites[nn_i[nj]]]=set_hoppings(ri,rj,n1n2_is);
                  #syst[syst0.sites[ni],syst0.sites[nn_i[nj]]]=t*(1+np.exp(-1j*k2)+np.exp(-1j*k1))
          #sys.exit()



#
#    for ni in range(1,N_ToT):
#          #print('ni=',ni);
#          ri=syst0.sites[ni].pos;
#          for nj in range(ni):
#                  rj=syst0.sites[nj].pos;
#                  syst[syst0.sites[ni],syst0.sites[nj]]=set_hoppings(ri,rj);
#                  #syst[syst0.sites[ni],syst0.sites[nn_i[nj]]]=t*(1+np.exp(-1j*k2)+np.exp(-1j*k1))
#          #sys.exit()


               
    for ni in range(0,N_ToT):
           ri=syst0.sites[ni].pos;
           syst[syst0.sites[ni]]=on_site_term(ri);

             


    def amp1(ri,rj):
      xi,yi,zi=ri; ri=[xi*f1,yi*f1,zi*f2];
      xj,yj,zj=rj; rj=[xj*f1,yj*f1,zj*f2];
      ss=np.array(ri)-np.array(rj);
      dis=sqrt(np.dot(ss,ss))
      dis1=dis-delta_a;
      dis1=dis1/delta;
      Vpp_pi=t*np.exp(-dis1);
      
      dis2=dis-d_0;
      dis2=dis2/delta;
      Vpp_sigma=t12*np.exp(-dis2);
      
      hop_t=Vpp_pi*(1-(ss[2]/dis)**2)+Vpp_sigma*(ss[2]/dis)**2
      
      
#      hop_t=0;
#      if (abs(dis-delta_a)<10**-5):
#         hop_t=t;
#      if (abs(dis-d_0)<10**-5):
#          hop_t=t12
      return -hop_t


    #print('system finish::')
    #print (datetime.now()-start)

   # syst0=syst.finalized()

    #N_ToT=syst0.graph.num_nodes;

#    def Given_link_ijs(family_i,tag_i,family_j,tag_j):
#       print(family_i,family_j)
#       print(tag_i,tag_j)
#       for ni in range(N_ToT):
#          if (syst0.sites[ni].family==family_i)&(syst0.sites[ni].tag==tag_i):
#             n_i=ni;
#            
#             break
#       for nj in range(N_ToT):
#          if (syst0.sites[nj].family==family_j)&(syst0.sites[nj].tag==tag_j):
#             n_j=nj;
#             break
#       return n_i,n_j
#
#

    #n_i,n_j=Given_link_ijs(a,tinyarray.array([0,0]),b,tinyarray.array([0,0]))
   # print('k1=',k1,'k2=',k2)
    #print('n_i=,n_j=',n_i,n_j)
    #sys.exit()
    #sys.exit()
    #print('ddddd1111111')
    return syst




def main():
    start=datetime.now()
    q=89; #if no manetic field, set q=1;
    print(alpha/pi*180)
    #syst, hoppings = make_system(q)
    nns,syst1,syst0,SC1,SC2 = make_system_step1(q)
   
  
    
    # To highlight the two sublattices of graphene, we plot one with
    # a filled, and the other one with an open circle:
    def family_colors(site):
        return 'g' if site.family in (a,b) else 'm'


    
    k1=0.1;k2=0.1;p=0;
    Bfields=range(0,q*(m1*m2+m1**2+m2**2),2);
    Num_p=len(Bfields);
    
    
    Num_kpoints=50;
    #from Gamma to K to Gamma
    k1_list=list(np.linspace(0,  2/3*(2*pi), Num_kpoints))+list(np.linspace(2/3*(2*pi),(2*pi), Num_kpoints));
    k1_list=list(reversed(k1_list));
    k2_list=list(np.linspace(0,  1/3*(2*pi), Num_kpoints))+list(np.linspace(1/3*(2*pi),(2*pi), Num_kpoints));
    k2_list=list(reversed(k2_list));
    #from  K to Gamma to M to K':
    k1_list=list(np.linspace( -2/3*(2*pi),(2*pi), Num_kpoints))+list(np.linspace((2*pi),0, Num_kpoints))+list(np.linspace(0,  2/3*(2*pi), Num_kpoints));
    k1_list=list(reversed(k1_list));
    k2_list=list(np.linspace(-1/3*(2*pi),(2*pi), Num_kpoints))+list(np.linspace((2*pi),0, Num_kpoints))+list(np.linspace(0,  1/3*(2*pi), Num_kpoints));
    k2_list=list(reversed(k2_list));

    Num_kpoints=len(k1_list);
    
    energies = [];
    energies_kpm = [];
    
    for p in Bfields:
        print(p);
    #for nk in range(len(k2_list)):
        #print(nk);
        #p=10;
     #   k1=k1_list[nk];
      #  k2=k2_list[nk];
 
        print('k1=',k1,'k2=',k2,'p=',p)
        # Obtain the Hamiltonian as a dense matrix
        # ham_mat = syst.hamiltonian_submatrix(args=[p], sparse=True)
        k_and_p=(k1,k2,p)
        
        #syst = make_system(k1,k2,q)
        syst=make_system_step2(p,k1,k2,syst1,syst0,nns,SC1,SC2,q)
        
        #print('ddddddddddd')
        syst=syst.finalized()
   
        #print('eeeeeeeeeee')

        
        ham_mat= syst.hamiltonian_submatrix(args=[], sparse=False)
       
        #print(syst.hamiltonian_submatrix(args=[k_and_p],to_sites=(0,1,2),from_sites=(0,1,2)))
        #sys.exit()

        
        # we only calculate the 15 lowest eigenvalues
        #ev = sla.eigsh(ham_mat, k=4*q*(m1*m2+m1**2+m2**2)-10, which='SM', return_eigenvectors=False)
        #print(ham_mat.shape)

        ev=LA.eigvalsh(ham_mat)
        ev=sorted(ev)
        energies.append(ev)
        #print(ev)
       
#        rho = kwant.kpm.SpectralDensity(ham_mat,num_vectors=10,num_moments=800)
#        rho.add_moments(energy_resolution=0.0001)
#        energies_rho,densities = rho()
#        
#        d_max=max(densities);
#        def peak_position():
#          index=[]
#          for id in range(1,len(densities)-1):
#             if (densities[id]>densities[id+1])&(densities[id]>densities[id-1])&(densities[id]>d_max/50):
#               index.append(id)
#          return index
#
#        indexes=peak_position()
#        energies_kpm.append([energies_rho[i1] for i1 in indexes]);
#        print('nk_kpm=',nk)
    #print(np.array(energies).reshape(Num_kpoints,4*q*(m1*m2+m1**2+m2**2)))
    
    filename='Butterfly_Bilayer_with_B_q={0}-m1={1}-m2={2}.dat'.format(q,m1,m2);
    #res = [x, y, z, ....]
    csvfile = "/Users/jyj/Dropbox/A-Ningbo/graphene_Hafstder_butterfly/code/"+filename
    with open(csvfile, "w") as output:
       writer = csv.writer(output, lineterminator='\n')
       writer.writerows(energies)

#    filename1='Butterfly_Bilayer_no_B_q={0}-m1={1}-m2={2}_kpm.dat'.format(q,m1,m2);
#    csvfile1 = "/Users/jyj/Dropbox/A-Ningbo/graphene_Hafstder_butterfly/code/"+filename1
#    with open(csvfile1, "w") as output:
#       writer = csv.writer(output, lineterminator='\n')
#       writer.writerows(energies_kpm)

  

    plt.figure()
    plt.plot(Bfields, energies,'-')
   # plt.plot(k1_list, energies,'*')
    #plt.plot(range(len(k2_list)), np.array(energies),'-')
    #plt.plot(range(Num_p), np.array(energies),'-')

    plt.xlabel("magnetic field [arbitrary units]")
    plt.ylabel("energy ")



    plt.show()
    sys.exit()

# fsyst = syst.finalized()
    print (datetime.now()-start)
    start=datetime.now()
    rho = kwant.kpm.SpectralDensity(fsyst)
    energies, densities = rho()
   
    print (datetime.now()-start)
    energies, densities = rho.energies, rho.densities
    
    plt.figure()
    plt.plot(energies,densities )
    #pyplot.ylim(-1.6,1.6)
    
    plt.xlabel("energy")
    plt.ylabel("DOS[eV]")

#  pyplot.title('band structure along z with'+' Wx=' + str(Width));
    
    plt.show()
    


    sys.exit()

# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()




