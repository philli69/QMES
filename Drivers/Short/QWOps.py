import qiskit.circuit.library as clib
import qiskit as qk
import math
import numpy as np
import MatrixProcedures

EPSILON=1e-10

class Operators:
    
    
    def __init__(self, msystem):
        self.msystem=msystem
    
    # prepares a work register to implement a control (Fig. 3 on p. 8)
    # *Note: THIS NEEDS A CORRESPONDING UNDO OR THE BASE REGISTER BIT FLIPS WILL REMAIN
    def prepare_work(self,ctrl_bits):
        reg_base = qk.QuantumRegister(self.msystem.n)
        reg_work = qk.QuantumRegister(max(1,self.msystem.n-1))
        circ = qk.QuantumCircuit(reg_base,reg_work)
        
        ctrl_bits_flip=ctrl_bits[::-1]
        # switch desired 0's to 1's
        for p in range(self.msystem.n):
            if ctrl_bits_flip[p]=='0':
                circ.append(clib.XGate(),[reg_base[p]])
        # set work bits
        if(self.msystem.n>1):
            circ.append(clib.CCXGate(),[reg_base[0]]+[reg_base[1]]+[reg_work[0]])
            for p in range(1,self.msystem.n-1):
                circ.append(clib.CCXGate(),[reg_base[p+1]]+[reg_work[p-1]]+[reg_work[p]])
        else:
            circ.append(clib.CXGate(),[reg_base[0]]+[reg_work[0]])
        
        return circ.to_gate()
    
    # shorthand definition for negative identity operator
    def NI(self):
        reg=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg)
        circ.append(clib.ZGate(),reg)
        circ.append(clib.XGate(),reg)
        circ.append(clib.ZGate(),reg)
        circ.append(clib.XGate(),reg)
        return circ.to_gate()
    
    # shorthand definition for a factor of the imaginary unit
    def iI(self):
        reg=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg)
        circ.append(clib.SGate(),reg)
        circ.append(clib.XGate(),reg)
        circ.append(clib.SGate(),reg)
        circ.append(clib.XGate(),reg)
        return circ.to_gate()
    
    # shorthand definition for the gate which modifies (only) the phase of the |0> state
    def P1(self,phi):
        reg=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg)
        circ.append(clib.XGate(),reg)
        circ.append(clib.PhaseGate(phi),reg)
        circ.append(clib.XGate(),reg)
        return circ.to_gate()
    
    # shorthand definition for the gate which negates (only) the component of the |0> state
    def Z1(self):
        reg=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg)
        circ.append(clib.XGate(),reg)
        circ.append(clib.ZGate(),reg)
        circ.append(clib.XGate(),reg)
        return circ.to_gate()
    
    # prepares |phi_j> from the |0> state (Fig. 6 on p. 12)
    def Bj(self, j):
        reg_r2=qk.QuantumRegister(self.msystem.n)
        reg_r2w=qk.QuantumRegister(max(1,self.msystem.n-1))
        reg_r2a=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg_r2,reg_r2w,reg_r2a)
        format_str_k='{:0%db}'%(self.msystem.n)
        # start in uniform superposition
        for k in range(self.msystem.n):
            circ.append(clib.HGate(),[reg_r2[k]])
        # NOT the ancilla
        circ.append(clib.XGate(),reg_r2a[:])
        # rotate ancilla to produce desired state
        for c in range(len(self.msystem.A_indices[j])):
            k_bits=format_str_k.format(self.msystem.A_indices[j][c])
            r=np.absolute(self.msystem.A_elements[j][c])
            t=np.angle(self.msystem.A_elements[j][c])
            theta=np.arcsin(math.sqrt(r*self.msystem.N/self.msystem.X))
            omega=-0.5*t
            if((abs(abs(t)-math.pi)<EPSILON) and (j<self.msystem.A_indices[j][c])):
                omega=-omega
            circ.append(self.prepare_work(k_bits),reg_r2[:]+reg_r2w[:])
            circ.append(clib.RYGate(-2.0*theta).control(1,None,'1'),[reg_r2w[max(0,self.msystem.n-2)]]+reg_r2a[:])
            circ.append(self.P1(omega).control(1,None,'1'),[reg_r2w[max(0,self.msystem.n-2)]]+reg_r2a[:])
            circ.append(self.prepare_work(k_bits).inverse(),reg_r2[:]+reg_r2w[:])
        
        return circ.to_gate()
    
    # prepares |zeta_j> from the |0> state (no figure, quite trivial in our model. explained shortly after (57))
    def Bp(self):
        reg_r2=qk.QuantumRegister(self.msystem.n)
        reg_r2a=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg_r2,reg_r2a)
        # flip the ancilla bit to |1> (or NOT it, if the ancilla isn't prepared in |0>)
        circ.append(clib.XGate(),reg_r2a[:])
        return circ.to_gate()
    
    # conditional state preparation operator (Fig. 5 on p. 12)
    def T0(self):
        reg_r1=qk.QuantumRegister(self.msystem.n)
        reg_r1w=qk.QuantumRegister(max(1,self.msystem.n-1))
        reg_r1a=qk.QuantumRegister(1)
        reg_r2=qk.QuantumRegister(self.msystem.n)
        reg_r2w=qk.QuantumRegister(max(1,self.msystem.n-1))
        reg_r2a=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg_r1,reg_r1w,reg_r1a,reg_r2,reg_r2w,reg_r2a)
        format_str_j='{:0%db}'%(self.msystem.n)
        # for each state of |r1>, prepare |r2>
        for j in range(self.msystem.N):
            j_bits=format_str_j.format(j)
            circ.append(self.prepare_work(j_bits), reg_r1[:]+reg_r1w[:])
            circ.append(self.Bj(j).control(2,None,'01'), [reg_r1w[max(0,self.msystem.n-2)]]+reg_r1a[:]+reg_r2[:]+reg_r2w[:]+reg_r2a[:])
            circ.append(self.prepare_work(j_bits).inverse(), reg_r1[:]+reg_r1w[:])
        return circ.to_gate()
    
    # the swap operation (includes ancilla bits) (no figure, defined in (24) on p. 5)
    def S(self):
        reg_r1=qk.QuantumRegister(self.msystem.n)
        reg_r1a=qk.QuantumRegister(1)
        reg_r2=qk.QuantumRegister(self.msystem.n)
        reg_r2a=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg_r1,reg_r1a,reg_r2,reg_r2a)
        for jk in range(self.msystem.n):
            circ.append(clib.SwapGate(), [reg_r1[jk],reg_r2[jk]])
        circ.append(clib.SwapGate(), reg_r1a[:]+reg_r2a[:])
        return circ.to_gate()
    
    # reflection about the |0> state
    def R0(self):
        reg_r2=qk.QuantumRegister(self.msystem.n)
        reg_r2w=qk.QuantumRegister(max(1,self.msystem.n-1))
        reg_r2a=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg_r2,reg_r2w,reg_r2a)
        # negate all states
        circ.append(self.NI(), reg_r2a[:])
        cstr=''
        for i in range(self.msystem.n):
            cstr=cstr+'0'
        # negate the all-|0>'s state
        circ.append(self.prepare_work(cstr), reg_r2[:]+reg_r2w[:])
        circ.append(self.Z1().control(1,None,'1'), [reg_r2w[max(0,self.msystem.n-2)]]+reg_r2a[:])
        circ.append(self.prepare_work(cstr).inverse(), reg_r2[:]+reg_r2w[:])
        return circ.to_gate()
    
    # reflection about |phi_j> (see (48) on p. 7)
    def RBj(self,j):
        reg_r2=qk.QuantumRegister(self.msystem.n)
        reg_r2w=qk.QuantumRegister(max(1,self.msystem.n-1))
        reg_r2a=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg_r2,reg_r2w,reg_r2a)
        Bj=self.Bj(j)
        circ.append(Bj.inverse(), reg_r2[:]+reg_r2w[:]+reg_r2a[:])
        circ.append(self.R0(), reg_r2[:]+reg_r2w[:]+reg_r2a[:])
        circ.append(Bj, reg_r2[:]+reg_r2w[:]+reg_r2a[:])
        return circ.to_gate()
    
    # reflection about |zeta_j> (see (49) on p. 7)
    def RBp(self):
        reg_r2=qk.QuantumRegister(self.msystem.n)
        reg_r2w=qk.QuantumRegister(max(1,self.msystem.n-1))
        reg_r2a=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg_r2,reg_r2w,reg_r2a)
        Bp=self.Bp()
        circ.append(Bp.inverse(), reg_r2[:]+reg_r2a[:])
        circ.append(self.R0(), reg_r2[:]+reg_r2w[:]+reg_r2a[:])
        circ.append(Bp, reg_r2[:]+reg_r2a[:])
        return circ.to_gate()
    
    # the walk operator (Fig. 9 on p. 14)
    def W(self):
        reg_r1=qk.QuantumRegister(self.msystem.n)
        reg_r1w=qk.QuantumRegister(max(1,self.msystem.n-1))
        reg_r1a=qk.QuantumRegister(1)
        reg_r2=qk.QuantumRegister(self.msystem.n)
        reg_r2w=qk.QuantumRegister(max(1,self.msystem.n-1))
        reg_r2a=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg_r1,reg_r1w,reg_r1a,reg_r2,reg_r2w,reg_r2a)
        format_str_j='{:0%db}'%(self.msystem.n)
        for j in range(self.msystem.N):
            j_bits=format_str_j.format(j)
            circ.append(self.prepare_work(j_bits), reg_r1[:]+reg_r1w[:])
            circ.append(self.RBj(j).control(2,None,'01'), [reg_r1w[max(0,self.msystem.n-2)]]+reg_r1a[:]+reg_r2[:]+reg_r2w[:]+reg_r2a[:])
            circ.append(self.prepare_work(j_bits).inverse(), reg_r1[:]+reg_r1w[:])
        circ.append(self.RBp().control(1,None,'1'), reg_r1a[:]+reg_r2[:]+reg_r2w[:]+reg_r2a[:])
        circ.append(self.S(), reg_r1[:]+reg_r1a[:]+reg_r2[:]+reg_r2a[:])
        circ.append(self.iI(), [reg_r1[self.msystem.n-1]])
        
        return circ.to_gate()






