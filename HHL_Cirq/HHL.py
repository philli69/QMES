import sys
import math
import numpy as np
import cirq
import scipy.linalg as slin
from cirq import Simulator as sim

# simulation parameters
EPSILON=0.00000000001
trials=10000
nq_anc=1
nq_phase=16
nq_vec=6
N_anc=int(math.pow(2.0,nq_anc)+EPSILON)
N_phase=int(math.pow(2.0,nq_phase)+EPSILON)
N_vec=int(math.pow(2.0,nq_vec)+EPSILON)

# calculation parameters
ii=complex(0.0,1.0)
t=1.0
#C=t/(2.0*math.pi)
C=1.0/(N_phase*t)

vec_init_state=[]
for i in range(N_anc):
    for j in range(N_phase):
        for k in range(N_vec):
            vec_init_state.append(0.0)
vec_init_state[16]=1.0
vec_init_state[17]=1.0
vec_init_state[18]=1.0
vec_init_state[19]=1.0
vec_init_state[20]=1.0
vec_init_state[21]=1.0
vec_init_state[22]=1.0
vec_init_state[23]=1.0
norm_vec_init=math.sqrt(sum([abs(x)*abs(x) for x in vec_init_state]))
vec_init_state=np.array([x/norm_vec_init for x in vec_init_state])

H=[]
f=open('unitary.dat','r')
i=0
for line in f.readlines():
    H.append([])
    j=0
    for num in line.split(' '):
        z=num.split(',')
        x=float(z[0])
        y=float(z[1])
        #H[i].append(ii*t*complex(x,y))
        H[i].append(complex(x,y))
        j=j+1
    i=i+1

#H=np.array([[-3.0,2.0,2.0,1.0],
            #[2.0,3.0,2.0,2.0],
            #[2.0,2.0,3.0,2.0],
            #[1.0,2.0,2.0,3.0]])/6.0
#unitary = slin.expm(1j*H*t)

# initialize the environment
reg_anc = cirq.LineQubit.range(nq_anc)
reg_phase = cirq.LineQubit.range(nq_anc,nq_anc+nq_phase)
reg_vec = cirq.LineQubit.range(nq_anc+nq_phase,nq_anc+nq_phase+nq_vec)
circ = cirq.Circuit()

# auxilliary circuits
# main unitary
class HamSim(cirq.EigenGate,cirq.Gate):
    def __init__(self,H,t,nq,exponent=1.0):
        cirq.EigenGate.__init__(self,exponent=exponent)
        cirq.Gate.__init__(self)
        self.H=H
        self.t=t
        self.nq=nq
        ws,vs=np.linalg.eigh(H)
        self.eigen_components=[]
        #eigs=[]
        for w,v in zip(ws,vs.T):
            theta = w*t/math.pi
            #eigs.append(w)
            P=np.outer(v,np.conj(v))
            self.eigen_components.append((theta,P))
        #norm = math.sqrt(sum([abs(x)*abs(x) for x in eigs]))
        #eigs = [x/norm for x in eigs]
        #print(eigs)
        #sys.exit()
    def _num_qubits_(self):
        return self.nq
    def _with_exponent(self,exponent):
        return HamSim(self.H,self.t,self.nq,exponent)
    def _eigen_components(self):
        return self.eigen_components
# QFT
class QFT(cirq.Gate):
    def __init__(self,nq):
        super(QFT,self)
        self.nq=nq
    def _num_qubits_(self):
        return self.nq
    def _decompose_(self,reg):
        nq=len(reg)
        for i in range(nq):
            yield cirq.H(reg[i])
            k=1
            for j in range(i+1,nq):
                fac=1.0/math.pow(2.0,k)
                yield cirq.CZ(reg[j],reg[i])**(fac)
                k=k+1
        for i in range(int(nq/2.0+EPSILON)):
            yield cirq.SWAP(reg[i],reg[nq-i-1])
    def _circuit_diagram_info_(self,args):
        return ["QFT"] * self.nq
# QPE
class QPE(cirq.Gate):
    def __init__(self,nq_phase,nq_vec,U):
        super(QPE,self)
        self.nq_vec=nq_vec
        self.nq_phase=nq_phase
        self.U=U
    def _num_qubits_(self):
        return self.nq_phase+self.nq_vec
    def _decompose_(self,qubits):
        reg=[*qubits]
        reg_phase=reg[:self.nq_phase]
        reg_vec=reg[self.nq_phase:]
        for i in range(self.nq_phase):
            yield cirq.H(reg_phase[i])
        for i in range(self.nq_phase):
            p=int(math.pow(2.0,i)+EPSILON)
            yield (self.U.controlled_by(reg_phase[self.nq_phase-i-1]))**p
            #yield self.U(*reg_vec).controlled_by(reg_phase[self.nq_phase-i-1])**p
        yield QFT(self.nq_phase).on(*reg_phase)**-1
    def _circuit_diagram_info_(self,args):
        return ["QPE"] * self.num_qubits()

# main workspace------------------------------------------------------
U=HamSim(H,t,nq_vec,1.0).on(*reg_vec)
#U=cirq.MatrixGate(unitary)
qpe_gate=QPE(nq_phase,nq_vec,U).on(*(np.concatenate((reg_phase,reg_vec),axis=None)))
# run QPE
circ.append(qpe_gate)
# rotate conditioned on the eigenvalues
format_str_phase='{:0%db}'%(nq_phase)
for k in range(N_phase):
    ktemp=k
    if ktemp==0:
        ktemp=N_phase
    lam=2.0*math.pi*float(ktemp)/(t*N_phase)
    if lam>math.pi:
        lam=lam-2.0*math.pi
    if abs(lam)<EPSILON:
        theta=0.0  # if this has nontrivial contribution to inverse problem, your matrix is singular
    else:
        theta=math.asin(C/lam)
    k_bits=format_str_phase.format(k)
    k_list=[]
    for i in range(nq_phase):
        k_list.append(int(k_bits[i]))
    circ.append(cirq.ry(2.0*theta).on(*reg_anc).controlled_by(control_values=k_list,*reg_phase))
# uncompute QPE
circ.append(qpe_gate**-1)
# end main workspace--------------------------------------------------

# draw and run the circuit
#print(circ)
sim=cirq.Simulator()
result=sim.simulate(circ, initial_state=vec_init_state,qubit_order=reg_anc[:]+reg_phase[:]+reg_vec[:])

# process and report results
statevector=result.final_state_vector
#format_str_anc='{:0%db}'%(nq_anc)
#format_str_vec='{:0%db}'%(nq_vec)
#for i in range(N_anc):
    #bit_str_anc=format_str_anc.format(i)
    #for j in range(N_vec):
        #bit_str_vec=format_str_vec.format(j)
        #print('|%s>|%s>: %f + i(%f)' % (bit_str_anc,bit_str_vec,statevector[i*N_phase*N_vec+j].real,statevector[i*N_anc*N_vec+j].imag))

vec=[]
for i in range(N_vec):
    vec.append(statevector[N_phase*N_vec+i])
vec_norm=math.sqrt(sum([abs(x)*abs(x) for x in vec]))
vec = [x/vec_norm for x in vec]
print(math.sqrt(sum([abs(x)*abs(x) for x in vec])))
for x in vec:
    print(x)
print()
vec=[]
for i in range(N_vec):
    vec.append(statevector[i])
vec_norm=math.sqrt(sum([abs(x)*abs(x) for x in vec]))
vec = [x/vec_norm for x in vec]
print(math.sqrt(sum([abs(x)*abs(x) for x in vec])))
for x in vec:
    print(x)

format_str_phase='{:0%db}'%(nq_phase)
for i in range(N_phase):
    bit_str_phase=format_str_phase.format(i)
    prob=0.0
    for j in range(N_vec):
        for k in range(N_anc):
            prob=prob + abs(statevector[k*N_phase*N_vec + i*N_vec + j])
    lam=2.0*math.pi*float(i)/(t*N_phase)
    if lam>math.pi:
        lam=lam-2*math.pi
    if prob>0.1:
        print('|%s>(%f): %f'%(bit_str_phase,lam,prob))

#format_str_all='{:0%db}'%(nq_anc+nq_phase+nq_vec)
#for i in range(len(statevector)):
    #x=statevector[i]
    #bit_str=format_str_all.format(i)
    #print('|%s>: %f + i(%f)'%(bit_str,x.real,x.imag))








