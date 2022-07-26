import qiskit.circuit.library as clib
import qiskit as qk
import math
import numpy as np

EPSILON=1e-10

def PrintStatevector(sv, nq_phase, msystem):
    N_phase=int(math.pow(2.0,nq_phase)+EPSILON)
    N_work=int(math.pow(2.0,max(1,msystem.n-1))+EPSILON)
    N_main=int(math.pow(2.0,msystem.n)+EPSILON)
    
    nq_work=max(1,msystem.n-1)
    
    f_a='{:0%db}'%(1)
    f_w='{:0%db}'%(nq_work)
    f_m='{:0%db}'%(msystem.n)
    f_p='{:0%db}'%(nq_phase)
    
    print("|%s>|%s>|%s>|%s>|%s>|%s>|%s>|%s>"%(
        "rp".ljust(max(2,nq_work)),
        "r1".ljust(max(2,msystem.n)),
        "r1w".ljust(max(3,nq_work)),
        "r1a",
        "r2".ljust(max(2,msystem.n)),
        "r2w".ljust(max(3,nq_work)),
        "r2a",
        "HHL"))
    
    c_tot=0
    for c_p in range(N_phase):
        s_p=f_p.format(c_p)
        for c_r1 in range(N_main):
            s_r1=f_m.format(c_r1)
            for c_r1w in range(N_work):
                s_r1w=f_w.format(c_r1w)
                for c_r1a in range(2):
                    s_r1a=f_a.format(c_r1a)
                    for c_r2 in range(N_main):
                        s_r2=f_m.format(c_r2)
                        for c_r2w in range(N_work):
                            s_r2w=f_w.format(c_r2w)
                            for c_r2a in range(2):
                                s_r2a=f_a.format(c_r2a)
                                for c_hhl in range(2):
                                    s_hhl=f_a.format(c_hhl)
                                    
                                    print("|%s>|%s>|%s>|%s>|%s>|%s>|%s>|%s>: %10.5f + (%10.5f)i"%( 
                                        s_p.ljust(max(2,nq_work)),
                                        s_r1.ljust(max(2,msystem.n)),
                                        s_r1w.ljust(max(3,nq_work)),
                                        s_r1a.ljust(3),
                                        s_r2.ljust(max(2,msystem.n)),
                                        s_r2w.ljust(max(3,nq_work)),
                                        s_r2a.ljust(3),
                                        s_hhl.ljust(3),
                                        sv[c_tot].real,
                                        sv[c_tot].imag
                                        ))
                                    c_tot=c_tot+1

def CheckQPE(sv, nq_phase, msystem):
    N_phase=int(math.pow(2.0,nq_phase)+EPSILON)
    N_work=int(math.pow(2.0,max(1,msystem.n-1))+EPSILON)
    format_str_phase='{:0%db}'%(nq_phase)
    probs=np.zeros(N_phase)
    for p in range(N_phase):
        for l in range(msystem.N*N_work*2*msystem.N*N_work*2*2):
            s=sv[p*msystem.N*N_work*2*msystem.N*N_work*2*2 + l]
            probs[p]=probs[p]+s.real*s.real+s.imag*s.imag
    probs=probs/sum(probs)
    print(sv)
    print('-------------------------------------------------')
    print('|phase> register analysis:')
    print(msystem.X,msystem.d)
    for p in range(N_phase):
        p_bits=format_str_phase.format(p)
        phi=float(p)/float(N_phase)
        lam=msystem.bnorm*(math.sin(2.0*math.pi*phi)*msystem.X-msystem.d)  # eigenvalues should correspond to those of A0, with two modes for each eigenvalue
        print('|%s,%9f>: %f;   %f'%(p_bits,lam.real,probs[p],phi))

def ExtractSolution(sv, nq_phase, msystem):
    N_phase=int(math.pow(2.0,nq_phase)+EPSILON)
    N_work=int(math.pow(2.0,max(1,msystem.n-1))+EPSILON)
    format_str_phase='{:0%db}'%(msystem.n)
    sol=np.zeros(msystem.N,dtype=np.complex_)
    for j in range(msystem.N):
        s=sv[j*N_work*2*msystem.N*N_work*2*2]
        sol[j]=s
    # calculate C
    if(msystem.d/msystem.X>1.0):
        C = abs(msystem.X-msystem.d)
    else:
        k0 = round(float(N_phase)/(2.0*math.pi) * math.asin(msystem.d/msystem.X))
        C1=abs(math.sin(2.0*math.pi*k0/float(N_phase))*msystem.X-msystem.d)
        C2=abs(math.sin(2.0*math.pi*(k0+1)/float(N_phase))*msystem.X-msystem.d)
        C3=abs(math.sin(2.0*math.pi*(k0-1)/float(N_phase))*msystem.X-msystem.d)
        if((k0-1)<0):
            C=min(C1,C2)
            if(C1<EPSILON):
                C=C2
        elif((k0+1)>=N_phase):
            C=min(C1,C3)
            if(C1<EPSILON):
                C=C3
        else:
            # *Note: only one can be zero
            if(C1<EPSILON):
                C=min(C2,C3)
            elif(C2<EPSILON):
                C=min(C1,C3)
            elif(C3<EPSILON):
                C=min(C1,C2)
            else:
                C=min(C1,C2,C3)
    C=C*(1.-EPSILON)
    print(C)
    print('-------------------------------------------------')
    print('Solution: ')
    ret_val=np.zeros(msystem.M,dtype=np.complex_)
    print(msystem.b)
    print(sol)
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

# basic circuit for quantum phase estimation (Fig. 1 on p. 3)
def QPE(U, nq_phase,nq_vec):
    reg_phase=qk.QuantumRegister(nq_phase)
    reg_vec=qk.QuantumRegister(nq_vec)
    circ=qk.QuantumCircuit(reg_phase,reg_vec,name='QPE')
    Uc=U.control(1,None,'1')
    for i in range(nq_phase):
        circ.append(clib.HGate(),[reg_phase[nq_phase-i-1]])
    for i in range(nq_phase):
        p=int(math.pow(2.0,i)+EPSILON)
        for j in range(p):
            circ.append(Uc, [reg_phase[i]]+reg_vec[:])
    circ.append(QFT(nq_phase).inverse(),reg_phase)
    return circ.to_gate()

# basic circuit for HHL rotation (Fig. 7 on p. 12)
def HHLRotation(nq_phase, msystem):
    reg_phase=qk.QuantumRegister(nq_phase)
    reg_a=qk.QuantumRegister(1)
    circ=qk.QuantumCircuit(reg_phase,reg_a,name='Rc')
    N_phase=int(math.pow(2.0,nq_phase)+EPSILON)
    # calculate C
    if(msystem.d/msystem.X>1.0):
        C = abs(msystem.X-msystem.d)
    else:
        k0 = round(float(N_phase)/(2.0*math.pi) * math.asin(msystem.d/msystem.X))
        C1=abs(math.sin(2.0*math.pi*k0/float(N_phase))*msystem.X-msystem.d)
        C2=abs(math.sin(2.0*math.pi*(k0+1)/float(N_phase))*msystem.X-msystem.d)
        C3=abs(math.sin(2.0*math.pi*(k0-1)/float(N_phase))*msystem.X-msystem.d)
        if((k0-1)<0):
            C=min(C1,C2)
            if(C1<EPSILON):
                C=C2
        elif((k0+1)>=N_phase):
            C=min(C1,C3)
            if(C1<EPSILON):
                C=C3
        else:
            # *Note: only one can be zero
            if(C1<EPSILON):
                C=min(C2,C3)
            elif(C2<EPSILON):
                C=min(C1,C3)
            elif(C3<EPSILON):
                C=min(C1,C2)
            else:
                C=min(C1,C2,C3)
    C=C*(1.-EPSILON)
    print(C)
    format_str_k='{:0%db}'%(nq_phase)
    for k in range(N_phase):
        k_bits=format_str_k.format(k)
        phi=float(k)/float(N_phase)
        lam=math.sin(2.0*math.pi*phi)*msystem.X-msystem.d
        if(abs(lam)>EPSILON):
            circ.append(clib.RYGate(2.0*math.acos(C/lam)).control(nq_phase,None,k_bits), reg_phase[:]+reg_a[:])
    return circ.to_gate()

