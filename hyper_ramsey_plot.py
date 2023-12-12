# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:14:56 2023

@author: Nakarin
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
# Ref
#  [1] Yudin, V. I., et al. "Hyper-Ramsey spectroscopy of optical clock transitions." Physical Review A 82.1 (2010): 011804.
#  [2] Hobson, R., et al. "Modified hyper-Ramsey methods for the elimination of probe shifts in optical clocks." Physical Review A 93.1 (2016): 010501.



##################################################
# Hyper Ramsey 

Omega0=1
tau=np.pi/(2*Omega0)
T=20/Omega0

# Pi/2 pulse
def W(tau,Omega0,delta_p,phi):
    Omega = np.sqrt(Omega0**2+delta_p**2)
    return np.array([[np.cos(Omega*tau/2)+1j*delta_p*np.sin(Omega*tau/2)/Omega,
                      -1j*np.exp(-1j*phi)*Omega0*np.sin(Omega*tau/2)/Omega],
                      [-1j*np.exp(1j*phi)*Omega0*np.sin(Omega*tau/2)/Omega, 
                      np.cos(Omega*tau/2)-1j*delta_p*np.sin(Omega*tau/2)/Omega]])
# free propagation
def V(T,delta):
    return np.array([[np.exp(1j*T*delta/2),
                      0],
                     [0, 
                      np.exp(-1j*T*delta/2)]])


Omega0=1
tau=np.pi/(2*Omega0)
T=20/Omega0
phi0 = np.array([[0], [1]])  # Initial state phi0 definition
delta_i=np.linspace(-.5,.5,1000)

shift_tot0 = []
error_i0 = []
pg=[]
pe=[]

delta = 0

for i in tqdm(range(len(delta_i))):
    Delta_sh = delta_i[i]
    Delta_step=0
    correction=0
    error_sig=1
    delta = 0
    

    # while (error_sig>1.5e-2):
    delta_p = delta-(Delta_sh-Delta_step)
    delta_dark = delta
    # Rotate last pulse by pi/2
    c0 = np.dot(W(tau,Omega0,delta_p,0), phi0) # Pi/2 pulse
    c1 = np.dot(V(T,delta_dark), c0) # Free evolution
    c2 = np.dot(W(tau*2,Omega0,delta_p,np.pi), c1) # -Pi pulse
    c3 = np.dot(W(tau,Omega0,delta_p,np.pi/2), c2) # Pi/2 pulse
    cg=c3[0]
    ce=c3[1]
    pe_p = (ce*np.conj(ce))
    
    c0=0; c1=0; c2=0; c3=0
    
    # Rotate last pulse by -pi/2
    c0 = np.dot(W(tau,Omega0,delta_p,0), phi0) # Pi/2 pulse
    c1 = np.dot(V(T,delta_dark), c0) # Free evolution
    c2 = np.dot(W(tau*2,Omega0,delta_p,np.pi), c1) # -Pi pulse
    c3 = np.dot(W(tau,Omega0,delta_p,-np.pi/2), c2) # Pi/2 pulse
    cg=c3[0]
    ce=c3[1]
    pe_m = (ce*np.conj(ce))
    error_sig = (np.real(pe_p)[0] - np.real(pe_m)[0])/T
    delta+= error_sig
        # if Delta_sh>0.3:
        #     print(error_sig)

    shift_tot0.append(Delta_sh-Delta_step)
    error_i0.append(delta)

shift_tot0 = np.array(shift_tot0)
error_i0 = np.array(error_i0)



##################################################
# Ramsey 
shift_tot1 = []
error_i1 = []
pg=[]
pe=[]

delta = 0

for i in tqdm(range(len(delta_i))):
    Delta_sh = delta_i[i]
    Delta_step=0
    correction=0
    error_sig=1
    delta = 0
    
    
    while (error_sig>2e-2):
        delta_p = delta-(Delta_sh-Delta_step)
        delta_dark = delta
        # Rotate last pulse by pi/2
        c0 = np.dot(W(tau,Omega0,delta_p,0), phi0) # Pi/2 pulse
        c1 = np.dot(V(T,0), c0) # Free evolution
        c3 = np.dot(W(tau,Omega0,delta_p,np.pi/2), c1) # Pi/2 pulse
        cg=c3[0]
        ce=c3[1]
        pe_p = (ce*np.conj(ce))
        
        c0=0
        c1=0
        c2=0
        c3=0
        
        # Rotate last pulse by -pi/2
        c0 = np.dot(W(tau,Omega0,delta_p,0), phi0) # Pi/2 pulse
        c1 = np.dot(V(T,0), c0) # Free evolution
        c3 = np.dot(W(tau,Omega0,delta_p,-np.pi/2), c1) # Pi/2 pulse
        cg=c3[0]
        ce=c3[1]
        pe_m = (ce*np.conj(ce))
        error_sig = (np.real(pe_p) - np.real(pe_m))/T
        delta-= error_sig[0]
        
    shift_tot1.append(Delta_sh-Delta_step)
    error_i1.append(delta)

shift_tot1 = np.array(shift_tot1)
error_i1 = np.array(error_i1)


fig, ax = plt.subplots(figsize=(12,8))
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
# Set the font size of the x-axis and y-axis tick labels
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.plot(shift_tot1,error_i1*T,'--',label='Ramsey ', linewidth=2.5)
plt.plot(shift_tot0,error_i0*T,label='Hyper-Ramsey', linewidth=2.5)
plt.ylim([-0.4,0.401])
plt.xlim([-0.5,0.5])
plt.yticks(np.arange(-0.4, 0.401,.2))
plt.legend(fontsize=22)
plt.xlabel(r'$(\Delta_{sh}-\Delta_{st})/\Omega_0$', fontsize=25) 
plt.ylabel(r'$(\omega_{L}-\omega_0)T$', fontsize=25) 
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.savefig('hyper-ramsey_compare.pdf', dpi=150)