import qiskit.circuit.library as clib
import qiskit as qk
import math
import numpy as np

EPSILON=1e-10

def CheckQPE(sv, nq_phase, msystem):
    N_phase=int(math.pow(2.0,nq_phase)+EPSILON)
    format_str_phase='{:0%db}'%(nq_phase)
    probs=np.zeros(N_phase)
    for p in range(N_phase):
        for j in range(msystem.N):
            for aj in range(2):
                for k in range(msystem.N):
                    for ak in range(2):
                        for a_hhl in range(2):
                            if((j==0) and (aj==0) and (k==0) and (ak==0) and (a_hhl==0)):
                                s=sv[p*msystem.N*2*msystem.N*2*2+j*2*msystem.N*2*2+aj*msystem.N*2*2+k*2*2+ak*2+a_hhl]
                                probs[p]=probs[p]+s.real*s.real+s.imag*s.imag
    probs=probs/sum(probs)
    print('-------------------------------------------------')
    print('|phase> register analysis:')
    for p in range(N_phase):
        p_bits=format_str_phase.format(p)
        phi=float(p)/float(N_phase)
        lam=math.sin(2.0*math.pi*phi)*msystem.X-msystem.d
        print('|%s,%9f>: %f;   %f'%(p_bits,lam,probs[p],phi))

def ExtractSolution(sv, nq_phase, msystem):
    N_phase=int(math.pow(2.0,nq_phase)+EPSILON)
    format_str_phase='{:0%db}'%(msystem.n)
    sol=np.zeros(msystem.N,dtype=np.complex_)
    for p in range(N_phase):
        for j in range(msystem.N):
            for aj in range(2):
                for k in range(msystem.N):
                    for ak in range(2):
                        for a_hhl in range(2):
                            if((p==0) and (aj==0) and (k==0) and (ak==0) and (a_hhl==0)):
                                s=sv[p*msystem.N*2*msystem.N*2*2+j*2*msystem.N*2*2+aj*msystem.N*2*2+k*2*2+ak*2+a_hhl]
                                sol[j]=s
    if(msystem.d/msystem.X>1.0):
        C = abs(msystem.X-msystem.d)-EPSILON
    else:
        k0 = float(N_phase)/(2.0*math.pi) * math.asin(msystem.d/msystem.X)
        if((k0%1)<EPSILON):
            C=min(abs(math.sin(2.0*math.pi*round(k0+1.0)/float(N_phase))*msystem.X-msystem.d)-EPSILON,
                  abs(math.sin(2.0*math.pi*round(k0-1.0)/float(N_phase))*msystem.X-msystem.d)-EPSILON)
        else:
            C=abs(math.sin(2.0*math.pi*round(k0)/float(N_phase))*msystem.X-msystem.d)-EPSILON
    print()
    print(C)
    print('-------------------------------------------------')
    print('Solution: ')
    ret_val=np.zeros(msystem.M,dtype=np.complex_)
    for j in range(msystem.M):
        if(msystem.expand):
            ret_val[j] = sol[j+msystem.M]/C
        else:
            ret_val[j] = sol[j]/C
        print('%f + i(%f) -> %f'%(ret_val[j].real,ret_val[j].imag,np.absolute(ret_val[j])))
    return ret_val

# basic circuit for a quantum Fourier transform
def QFT(nq):
    reg=qk.QuantumRegister(nq,name='q')
    circ=qk.QuantumCircuit(reg,name='QFT')
    for i in range(nq):
        circ.append(clib.HGate(),[reg[nq-i-1]])
        k=1
        for j in range(i+1,nq):
            fac=1/math.pow(2.0,k)
            circ.append(clib.CZGate().power(fac),[reg[nq-j-1],reg[nq-i-1]])
            k=k+1
    for i in range(int(nq/2.0)):
        circ.append(clib.SwapGate(),[reg[i],reg[nq-i-1]])
    return circ.to_gate()

# basic circuit for quantum phase estimation
def QPE(U, nq_phase,nq_vec):
    reg_phase=qk.QuantumRegister(nq_phase)
    reg_vec=qk.QuantumRegister(nq_vec)
    circ=qk.QuantumCircuit(reg_phase,reg_vec,name='QPE')
    Uc=U.control(1,None,'1')
    for i in range(nq_phase):
        circ.append(clib.HGate(),[reg_phase[nq_phase-i-1]])
    for i in range(nq_phase):
        p=int(math.pow(2.0,i)+EPSILON)
        circ.append(Uc.power(p), [reg_phase[i]]+reg_vec[:])
    circ.append(QFT(nq_phase).inverse(),reg_phase)
    return circ.to_gate()

# basic circuit for HHL rotation
def HHLRotation(nq_phase, msystem):
    reg_phase=qk.QuantumRegister(nq_phase)
    reg_a=qk.QuantumRegister(1)
    circ=qk.QuantumCircuit(reg_phase,reg_a,name='Rc')
    N_phase=int(math.pow(2.0,nq_phase)+EPSILON)
    # calculate C
    print()
    if(msystem.d/msystem.X>1.0):
        print("AA")
        C = abs(msystem.X-msystem.d)-EPSILON
    else:
        k0 = float(N_phase)/(2.0*math.pi) * math.asin(msystem.d/msystem.X)
        if((k0%1)<EPSILON):
            print("AB")
            C=min(abs(math.sin(2.0*math.pi*round(k0+1.0)/float(N_phase))*msystem.X-msystem.d)-EPSILON,
                  abs(math.sin(2.0*math.pi*round(k0-1.0)/float(N_phase))*msystem.X-msystem.d)-EPSILON)
        else:
            print("AC")
            C=abs(math.sin(2.0*math.pi*round(k0)/float(N_phase))*msystem.X-msystem.d)-EPSILON
    print(C)
    format_str_k='{:0%db}'%(nq_phase)
    for k in range(N_phase):
        k_bits=format_str_k.format(k)
        phi=float(k)/float(N_phase)
        lam=math.sin(2.0*math.pi*phi)*msystem.X-msystem.d
        if(abs(lam)>EPSILON):
            circ.append(clib.RYGate(2.0*math.acos(C/lam)).control(nq_phase,None,k_bits), reg_phase[:]+reg_a[:])
    return circ.to_gate()

