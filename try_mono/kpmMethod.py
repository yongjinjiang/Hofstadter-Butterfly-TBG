
import numpy as np
from math import pi, sqrt, tanh, exp, tan,acos,sin,cos
import scipy.sparse.linalg as sla
from scipy.sparse import identity
import sys


sq3=sqrt(3);

def kpm(H):  #H is sparce matrix of
#kpm的程序实现，算DOS，原理见书的279页：


  H=H.tocsr();


  
  R=10;
  M=2000; Ne=20000;
  eplison=M**-1;
  Ndim=H.shape[1]
  
  def cot(s):
    return tan(s)**-1
  
  i=0;Jac=np.zeros(M-1);
  Jac0=1/(M+1)*((M-i+1)*cos(pi*i/(M+1))+sin(pi*i/(M+1))*cot(pi/(M+1)));
  #for i=1:M-1
  for j in range(M-1):
    i=j+1;
    Jac[j]=1/(M+1)*((M-i+1)*cos(pi*i/(M+1))+sin(pi*i/(M+1))*cot(pi/(M+1)));
  
  #print(Jac0)
  #print(Jac)
  #sys.exit()
  


  v=np.zeros(2)
  #print('1111')

  ##v(1)=eigs(H,1, 'SR');  %,opts);
#  #v(2)=eigs(H,1, 'LR');  %,opts);
  v[0]= sla.eigsh(H, k=1, which='SA', return_eigenvectors=False)
  v[1]= sla.eigsh(H, k=1, which='LA', return_eigenvectors=False)
  v=sorted(v);
  rmax=v[1];rmin=v[0];
  ra=(rmax-rmin)/(2-eplison);
  rb=(rmax+rmin)/2;
  H=(H-rb*identity(Ndim))/ra;


#  #mu=0;mu(M-1)=0;
  mu=np.zeros(M-1).astype('complex128');

  mu0=0;
  for ir in range(R):
      
      #v0=(np.random.uniform(-0.5,0.5,Ndim))*sq3*2+1j*np.zeros(Ndim);
      v0=(np.random.uniform(-0.5,0.5,Ndim))*sq3*2+1j*np.zeros(Ndim);
      #v0=np.array([1,0,0])
      
      #print(mu.dtype,v0.dtype)
      #sys.exit()
      #v0=v0.tocsr()
      v00=v0;
      
#   v0=sparse(v0');
 #    v00=v0;
 
      mu0=np.dot(v0.conjugate(),v00)+mu0;
     
      v1=H.dot(v0)
      #print(type(v1),v1.shape,v1.dtype)
      mu[0]=np.dot(v1.conjugate(),v00)+mu[0]
     # print(type(np.matrix(v1)), np.matrix(v1).shape)
     # print(type(np.matrix(v00)), np.matrix(v00).shape)
      
      #mu[0]=(np.matrix(v1)*np.matrix(v00).getH())[0,0]
      #+mu[0]
     
      
      v2=2*H.dot(v1)-v00;
      mu[1]=np.dot(v2.conjugate(),v0)+mu[1];
      
      for im in range(2,M-1):
          v0=v1;v1=v2;
          v2=2*H.dot(v1)-v0;
          mu[im]=np.dot(v2.conjugate(),v00)+mu[im];

  mu0=mu0/R; mu=mu/R;
  mu0=mu0*Jac0;
  mu=mu*Jac;
  
  #x0=-10:20/Ne:10-20/Ne;
  ww=(rmax-rmin)/50;
  x0=np.linspace(rmin-ww,rmax+ww,Ne)
  x0=x0[(x0>rmin)&(x0<rmax)];
  x=(x0-rb)/ra;
  Mb=x.shape[0];
  arcx=np.arccos(x);
  #print('arcx=',arcx[0:5])
  gma=mu0*(np.ones(Mb).reshape(1,Mb))+2*np.matrix(mu)*np.cos(np.matrix(range(1,M)).T*arcx);
  #print(Mb,np.ones(Mb).shape,np.matrix(mu).shape,(np.matrix(range(M-1)).T*arcx).shape,'ttttt')
  
  #print('%%%',(np.matrix(mu)*np.cos(np.matrix(range(M-1)).T*arcx)).shape)
  #print(gma.shape)
  f=gma/np.sqrt(1-x**2);
  #print(gma.shape)
  energies_rho=x0.reshape(1,Mb);
  density=f/pi;
 # print(type(energies_rho),energies_rho.shape,type(np.array(density)),np.array(density).reshape((1922,)).shape )
  #print('3333')
  #return energies_rho,density
  return energies_rho,density