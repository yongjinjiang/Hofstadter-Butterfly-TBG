# Tutorial 2.5. Beyond square lattices: graphene
# ==============================================
#
# Physics background
# ------------------
#  Transport through a graphene quantum dot with a pn-junction
#
# Kwant features highlighted
# --------------------------
#  - Application of all the aspects of tutorials 1-3 to a more complicated
#    lattice, namely graphene
import numpy as np
from math import pi, sqrt, tanh, exp, tan,acos,sin,cos
import csv
from kpmMethod import *
import numpy as np
from scipy.signal import find_peaks_cwt
from scipy.sparse import identity
from numpy import linalg as LA

import sys
import kwant
import tinyarray
import cmath
# For computing eigenvalues
import scipy.sparse.linalg as sla

# For plotting
from matplotlib import pyplot
from datetime import datetime

sq3=sqrt(3);


# Define the graphene lattice
sin_30, cos_30 = (1 / 2, sqrt(3) / 2)
graphene = kwant.lattice.general([(cos_30, -sin_30), (cos_30,sin_30)],
                                 [(0, 0), (1 / sqrt(3),0)])

Base_a1=(cos_30, -sin_30); Base_a2=(cos_30,sin_30);


a, b = graphene.sublattices

Nx=1;
Ny=1;

def make_system_step1(q):
    L1=tinyarray.array((3/sqrt(3),0));SC1=L1;
    L2=tinyarray.array((0,q));        SC2=L2;
   
    syst = kwant.Builder()
    
    ##primitive graphene,dispersion without B:
#    for i in range(1):
#    #for i in range(q):
#      for j in range(1):
#        # On-site Hamiltonian
#        syst[a(i, j)] =0
#        syst[b(i, j)] =0
#    syst[a(0,0),b(0,0)]=-1-np.exp(1j*k1)-np.exp(1j*k2)
#    
    def rectangle(pos):
        xR, yR = pos
        #print(pos)
        #sys.exit()
        # xR,yR=list(x*v1+y*v2)
        #return (xR <=Lx2)&(yR<=Ly2)&(xR >= Lx1)&(yR>=Ly1)
        return   (-cos_30/3-0.00001<=xR<3/sqrt(3)*Nx-1/sqrt(3))&(0-0.0001<=yR<=Ny*q-0.0001)
    syst[graphene.shape(rectangle, (0, 0))] = 0
    
    
    syst0=syst.finalized()
    N_ToT=syst0.graph.num_nodes;
    print(N_ToT);
    
    
    def neighbors():
        nns=[[] for i in range(N_ToT)];
        for ni in range(N_ToT):
          #print(ni)
          ri=syst0.sites[ni].pos;
        
          for nj in range(N_ToT):
             rj=syst0.sites[nj].pos;
             
             for n1 in range(-1,2):
                for n2 in range(-1,2):
                  rjn=rj+(n1*SC1+n2*SC2);
                  ss=ri-rjn;
                  dis=sqrt(np.dot(ss,ss))
                  if (abs(dis-1/sqrt(3))<10**-5):
                     if not((n1==0)&(n2==0)&(nj==ni)):
                        nns[ni].append((nj,n1,n2))
        return nns
    nns=neighbors();
    return nns,syst,syst0
    
    
    
def make_system_step2(k1,k2,syst,syst0,nns,p,q):
    L1=tinyarray.array((3/sqrt(3),0));SC1=L1;
    L2=tinyarray.array((0,q));        SC2=L2;
    
    
    N_ToT=syst0.graph.num_nodes
    start=datetime.now()
    
    def amp(ri,rj):
      hop_t=0;
      ss=ri-rj;
      dis=sqrt(np.dot(ss,ss))
      if (abs(dis-1/sqrt(3))<10**-5):
        hop_t=-1;
      return hop_t

    
    
    def hoppping(site1,site2):
      r1=site1.pos;
      r2=site2.pos;
      return -1*phase(r1,r2)
    
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
                 #print(rj,rjn);
                 #print(amp(ri,rjn));
                 non_diag_term=non_diag_term + np.exp(1j*(n1*k1+n2*k2)) *amp(ri,rjn)*phase(ri,rjn);
        
        #rj1=rj-SC1
        #rj2=rj-SC2;
        #non_diag_term=t*(1*phase(ri,rj)+np.exp(-1j*k2)*phase(ri,rj1)+np.exp(-1j*k1)*phase(ri,rj2))
        #if abs(ri[2]-rj[2])>10**-3:
        #  non_diag_term=0
        return non_diag_term

    
    def phase(rj,ri):
       xi,yi=ri;
       xj,yj=rj;
       phase=(yi+yj)/2* (xj-xi);
       phase=np.exp(2*pi*1j*phase*p/q*2*sqrt(3))  #p/q
       return phase
    
    
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

    
#    syst[graphene.neighbors()] =  hoppping;
#    
#    hopping = ((1*Nx, 1*Nx), a, b);
#    r1=(0,0);r2=(0,1/sqrt(3)); #from r2 to r1
#    syst[kwant.builder.HoppingKind(hopping[0],hopping[1],hopping[2])] =-phase(r1,r2)*np.exp(1j*k1) #kx shold be changed to kx
#    hopping = ((-q*Ny, q*Ny-1), b, a)
#    r1=(0,0);r2=(1/2/sqrt(3),1/2);
#    syst[kwant.builder.HoppingKind(hopping[0],hopping[1],hopping[2])] =-phase(r1,r2)* np.exp(1j*k2) #k2 shold be changed to ky
#    hopping = ((1-q*Ny, q*Ny), a, b)
#    r1=(0,0);r2=(-1/2/sqrt(3),1/2);
#    syst[kwant.builder.HoppingKind(hopping[0],hopping[1],hopping[2])] =-phase(r1,r2)* np.exp(1j*k2) #k2 shold be changed to ky

    def family_colors(site):
        return 1 if site.family == a else 0
    #kwant.plot(syst, site_color=family_colors, site_lw=0.1,colorbar=False)
    
    
#    n_i,n_j=Given_link_ijs(a,tinyarray.array([0,1]),b,tinyarray.array([-1,1]))
#    ri=syst0.sites[n_i].pos; rj=syst0.sites[n_j].pos;
#    
#    sys.exit()
#



    #print([syst0.sites[i].tag for i in  range(N_ToT)])
    
    #print(N_ToT);
    
    
    
   
    def phase_check():
          aa1=np.array(Base_a1);aa2=np.array(Base_a2);
          r1=(aa1+aa2)/3;
          r2=aa2;
          r3=(4*aa2-2*aa1)/3;
          r4=aa2-aa1;
          r5=(aa2-2*aa1)/3;
          r6=np.array([0,0]);
          r0=(2*aa2-aa1)/3;
          r=np.zeros((6,2))
          
#          print(phase(r3,r2))
#          print(phase(r3+SC1,r2+SC1))
#          print(phase(r5,r6))
#          print(phase(r5+SC1,r6+SC1))
#          print(phase(r6,r1))
#          print(phase(r6+SC1,r1+SC1))
#          sys.exit()

          r[0,:]=r1;r[1,:]=r2;r[2,:]=r3;r[3,:]=r4;
          r[4,:]=r5;r[5,:]=r6;
          rr=r;
          print(r1+r2+r3+r4+r5-6*r0)
          print('sum=',r.sum(0)-6*r0)
         
          print('total phase1=',phase(r1,r6)        *phase(r2,r1)        *phase(r3,r2)        *phase(r4,r3)        *phase(r5,r4)        *phase(r6,r5))
          for n1 in range(-5,5):
            for n2 in range(-5,5):
              r=rr+n1*aa1+n2*aa2*0.1
              print('total phase=',phase(r[0,:],r[5,:])*phase(r[1,:],r[0,:])*phase(r[2,:],r[1,:])*phase(r[3,:],r[2,:])*phase(r[4,:],r[3,:])*phase(r[5,:],r[4,:]))
#          
          print('the periocity of phase efactor:')
          for i in range(5):
            ri=r[i,:];rj=r[i+1,:];
          #the periocity of phase factor:
            print(str(i+1)+'---'+str(i+2)+':')
            print(phase(ri,rj))
            print(phase(ri+tinyarray.array(L1),rj+tinyarray.array(L1)))
            print(phase(ri+(q-1)*tinyarray.array(L2),rj+(q-1)*tinyarray.array(L2)))
            print(phase(ri+q*tinyarray.array(L2),rj+q*tinyarray.array(L2)))
          
          ri=r[5,:];rj=r[0,:];
          #the periocity of phase factor:
          print(str(6)+'---'+str(1)+':')
          print(phase(ri,rj))
          print(phase(ri+tinyarray.array(L1),rj+tinyarray.array(L1)))
          print(phase(ri+(q-1)*tinyarray.array(L2),rj+(q-1)*tinyarray.array(L2)))
          print(phase(ri+q*tinyarray.array(L2),rj+q*tinyarray.array(L2)))
          print('q=',q,'p=',p)
          print('total phase=',(phase(r[0,:],r[5,:])*phase(r[1,:],r[0,:])*phase(r[2,:],r[1,:])*phase(r[3,:],r[2,:])*phase(r[4,:],r[3,:])*phase(r[5,:],r[4,:]))**(2*q))
          return '1111here'
    #ss=phase_check()
    #print('herehere')
    #sys.exit()
    return syst


def main():
    start=datetime.now()
    q=89;
    nns,syst1,syst0 = make_system_step1(q)
    
    
    # To highlight the two sublattices of graphene, we plot one with
    # a filled, and the other one with an open circle:
    def family_colors(site):
        return 1 if site.family == a else 0

    def hopping_colors(site1,site2):
         return 1 if site1.tag-site2.tag==hoppings[1][0] else 0
    

    # Plot the closed system without leads.
    #kwant.plot(syst, site_color=family_colors, site_lw=0.1, hop_color=hopping_colors,colorbar=False)
    #kwant.plot(syst, site_color=family_colors, site_lw=0.1,colorbar=False)
   
    #fsyst = syst.finalized()
    
    
    
    k1=0.1;k2=0.1;p=1;
   
    #Num_p=len(Bfields);

    
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
    

    energies = []
    
    Bfields=range(q)
    
    for p in Bfields:
        print(p);
    #for nk in range(len(k2_list)):
     #   print(nk);k1=k1_list[nk];k2=k2_list[nk];
      #  print('k1=',k1,'k2=',k2,'p=',p,'q=',q)
        syst=make_system_step2(k1,k2,syst1,syst0,nns,p,q)
        fsyst = syst.finalized()
        ham_mat = fsyst.hamiltonian_submatrix(args=[], sparse=False)
       # energies_rho1,densities1 = kpm(ham_mat)  #我的程序
        
        # we only calculate the 15 lowest eigenvalues
        #ev = sla.eigsh(ham_mat, k=2*q-2, which='SM', return_eigenvectors=False)
        ev=LA.eigvalsh(ham_mat)
        ev=sorted(ev)
        energies.append(ev)
    
      #以下三行，算 DOS， 不是 LDOS：
        #rho = kwant.kpm.SpectralDensity(ham_mat,num_vectors=10,num_moments=800)
        #rho.add_moments(energy_resolution=0.0001)
        #energies_rho,densities = rho()
        #break

   
    m1=200;m2=100;
    
    #np.savetxt(f,energies)
    
    filename='Butterfly_q={0}-m1={1}-m2={2}.dat'.format(q,m1,m2);

    #res = [x, y, z, ....]
    csvfile = "/Users/jyj/Dropbox/A-Ningbo/graphene_Hafstder_butterfly/code/"+filename
    with open(csvfile, "w") as output:
       writer = csv.writer(output, lineterminator='\n')
       writer.writerows(energies)
    
    
#    pyplot.figure()
#    pyplot.plot(energies_rho, densities,'*');  pyplot.xlabel("DOS from KPM]")
#    pyplot.figure()
#    pyplot.plot(energies_rho1, densities1.real,'*');  pyplot.xlabel("DOS from my KPM]")
#    pyplot.show()
#    sys.exit()

   # file('data2','w').write('\n'.join(' '.join(repr(item)for item in
#row)for row in rows)+'\n')
    #np.save(f,energies)
    

    pyplot.figure()
    pyplot.plot(Bfields, energies,'*');  pyplot.xlabel("magnetic field [arbitrary units]");pyplot.ylabel("energy [t]")
#
#    pyplot.figure()
#    pyplot.plot(energies_rho, densities,'*');  pyplot.xlabel("DOS from KPM]")

 #   pyplot.plot(kregion, energies,'*')
 #   pyplot.plot(range(len(k2_list)), np.array(energies),'-');pyplot.xlabel("from  K to Gamma to M to K")


    pyplot.show()
    sys.exit()
    
    #sys.exit()

# fsyst = syst.finalized()
    print (datetime.now()-start)
    start=datetime.now()

    ham_mat = fsyst.hamiltonian_submatrix(args=[(k1,k2,p)], sparse=True)
    rho = kwant.kpm.SpectralDensity(ham_mat,num_vectors=15,num_moments=8000)
    #energies, densities = rho()
   
    print (datetime.now()-start)
   
    energies1, densities = rho.energies, rho.densities
    energies=  list(np.linspace(min(energies1),max(energies1),6*10**4))
    rho.add_moments(energy_resolution=0.0001)

    energies,densities = rho()

    print('len_energy=',len(energies))

    #indexes = detect_peaks(densities, mph=0.04, mpd=100)
    #indexes = find_peaks_cwt(densities, np.arange(1, 550))

    d_e=(energies[1]-energies[0])*10**4; d_max=max(densities);
    def peak_position():
       index=[]
       for id in range(1,len(densities)-1):
           if (densities[id]-densities[id+1]> d_e)&(densities[id]-densities[id-1]> d_e)&(densities[id]>d_max/50):
              index.append(id)
       return index

    indexes=peak_position()
    print(len(indexes),2*q)
    pyplot.figure()
    pyplot.plot(energies,densities)
    pyplot.plot([energies[i1] for i1 in indexes],[densities[i2] for i2 in indexes],'r*')


    #pyplot.ylim(-1.6,1.6)
    #pyplot.show()
    
   
    
    pyplot.xlabel("energy")
    pyplot.ylabel("DOS[eV]")

#  pyplot.title('band structure along z with'+' Wx=' + str(Width));
    
    pyplot.show()
    


    sys.exit()

# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()




