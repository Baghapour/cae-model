from CoolProp.CoolProp import PropsSI
from numpy import array,arange
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt

# Fluid Property and Problem Definition

class Fluid(object):
  def __init__(self,fluid=None,P=None,T=None):
    self.fluid = fluid
    self.Pref  = P
    self.Tref  = T
    self.rho   = PropsSI('D','T',T,'P',P,fluid)
    self.mu    = PropsSI('V','T',T,'P',P,fluid)
    self.Pr    = PropsSI('PRANDTL','T',T,'P',P,fluid)
    self.beta  = PropsSI('ISOBARIC_EXPANSION_COEFFICIENT','T',T,'P',P,fluid)
    self.M     = self.rho*1000/PropsSI('DMOLAR','T',T,'P',P,fluid)
    self.Cv    = PropsSI('CVMASS','T',T,'P',P,fluid)
    self.Cp    = PropsSI('CPMASS','T',T,'P',P,fluid)
    self.nu    = self.mu/self.rho

def freeConvSet(
  name=None,g=9.81,fluid=None,dT=None,L=None,P0=None,T0=None,Re=None,Rew=None,Rr=None):
  f = Fluid(fluid=fluid,P=P0,T=T0)
  beta = f.beta
  rho  = f.rho
  mu   = f.mu
  Pr   = f.Pr
  nu   = f.nu
  M    = f.M
  Cv   = f.Cv
  Cp   = f.Cp
  alf  = nu/Pr
  Gr   = g*beta*dT*L**3/nu**2
  Ra   = Gr*Pr
  Tc   = T0
  Th   = T0+dT
  file = open(name+".txt","w")
  ftr  = ["Ra","Pr","dT","P0","Tc","Th","rho","mu","nu","alf","M","Cv","Cp","beta","L","g"]
  val  = [  Ra,  Pr,  dT,  P0,  Tc,  Th,  rho,  mu,  nu,  alf,  M,  Cv,  Cp, beta,  L,  g]
  if Re is not None: 
    U = Re*nu/L
    ftr += ["Re","U"]
    val += [Re,U]
  if Rew is not None: 
    OMG = Rew*nu/(Rr*L) 
    UR  = OMG*Rr
    ftr += ["Rew", "Rr","omg","Ur"]
    val += [Rew,Rr,OMG,UR]
  for ft,va in zip(ftr,val):
    file.write("%5s\t%.4e\n"%(ft,va))
  file.close()

if __name__ == "__main__":

  annulus_mix = True
  rayben_les  = False

  # Check Pr = 0.710 +/- 0.003
  import numpy as np
  Prr = [
    7.0719e-01,0.713,7.0719e-01,0.713,
    7.0922e-01,0.7105,7.0922e-01,0.7105,
    7.0956e-01,0.7105,7.0956e-01,0.711,
    7.0956e-01,0.711,0.706,0.706,0.72,0.712,0.712,0.7065,0.7105,0.706]
  Prr = np.array(Prr)
  print(Prr.mean(),Prr.std())

'''
  if annulus_mix is True:
    # Free convection 
    #----------------------
    # Ra between 1e3 to 9e3
    #----------------------
    freeConvSet(name="annul-Ra-1e3-9e3n",fluid="Air",
                dT=5,
                L=0.02,
                P0=1.17156e5, #1e5,
                T0=300)
    #----------------------
    # Ra between 1e4 to 9e4
    #----------------------
    freeConvSet(name="annul-Ra-1e4-9e4n",fluid="Air",
                dT=1,
                L=0.02,
                P0=3.69190e5, #1e5,
                T0=300) 
    #----------------------
    # Ra between 1e5 to 1e6
    #----------------------
    freeConvSet(name="annul-Ra-1e5-1e6n",fluid="Air",
              dT=10,
              L=0.04,
              P0=4.12525e5, #1e5,
              T0=300) 
    
    # Mixed convection 
    #----------------------
    # Ra between 1e3 to 9e3
    #----------------------
    freeConvSet(name="annul-Ra-1e3-9e3",fluid="Air",
               dT=6.5,
               L=0.02,
               P0=1.17156e5,
               Rew=500,
               Rr=0.04,
               T0=300)
    #----------------------
    # Ra between 1e4 to 9e4
    #----------------------
    freeConvSet(name="annul-Ra-1e4-9e4",fluid="Air",
              dT=5.0,
              L=0.02,
              P0=3.69190e5,
              Rew=300,
              Rr=0.04,
              T0=300) 
    # #----------------------
    # # Ra between 1e5 to 1e6
    # #----------------------
    freeConvSet(name="annul-Ra-1e5-1e6",fluid="Air",
                 dT=1.25,
                L=0.08,
                P0=4.12525e5,
                Rew=600,
                Rr=0.08,
                T0=300) 
'''

