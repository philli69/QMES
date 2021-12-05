import qiskit.circuit.library as clib
import qiskit as qk
import math
import numpy as np
import MatrixProcedures

EPSILON=1e-10

class Operators:
    
    
    def __init__(self, msystem):
        self.msystem=msystem
    
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
    
    # prepares |phi_j> from the |0> state
    def Bj(self, j):
        reg_r2=qk.QuantumRegister(self.msystem.n)
        reg_r2a=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg_r2,reg_r2a)
        format_str_k='{:0%db}'%(self.msystem.n)
        # start in uniform superposition
        for k in range(self.msystem.n):
            circ.append(clib.HGate(),[reg_r2[k]])
        # rotate ancilla to produce desired state
        for k in range(self.msystem.N):
            k_bits=format_str_k.format(k)
            r=np.absolute(self.msystem.A[j][k])
            t=np.angle(self.msystem.A[j][k])
            theta=np.arccos(math.sqrt(r*self.msystem.N/self.msystem.X))
            omega=-0.5*t
            if((abs(abs(t)-math.pi)<EPSILON) and (j<k)):
                omega=-omega
            circ.append(clib.RYGate(2.0*theta).control(self.msystem.n,None,k_bits),reg_r2[:]+reg_r2a[:])
            circ.append(self.P1(omega).control(self.msystem.n,None,k_bits),reg_r2[:]+reg_r2a[:])
        return circ.to_gate()
    
    # prepares |zeta_j> from the |0> state
    def Bp(self):
        reg_r2=qk.QuantumRegister(self.msystem.n)
        reg_r2a=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg_r2,reg_r2a)
        # flip the ancilla bit to |1>
        circ.append(clib.XGate(),reg_r2a[:])
        return circ.to_gate()
    
    # conditional state preparation operator
    def T(self):
        reg_r1=qk.QuantumRegister(self.msystem.n)
        reg_r1a=qk.QuantumRegister(1)
        reg_r2=qk.QuantumRegister(self.msystem.n)
        reg_r2a=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg_r1,reg_r1a,reg_r2,reg_r2a)
        format_str_j='{:0%db}'%(self.msystem.n)
        # for each state of |r1>, prepare |r2>
        for j in range(self.msystem.N):
            j_bits=format_str_j.format(j)
            circ.append(self.Bj(j).control(self.msystem.n+1,None,'0'+j_bits),reg_r1[:]+reg_r1a[:]+reg_r2[:]+reg_r2a[:])
        return circ.to_gate()
    
    # the swap operation (includes ancilla bits)
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
        reg_r2a=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg_r2,reg_r2a)
        # negate all states
        circ.append(self.NI(), reg_r2a[:])
        cstr=''
        for i in range(self.msystem.n):
            cstr=cstr+'0'
        # negate the all-|0>'s state
        circ.append(self.Z1().control(self.msystem.n,None,cstr), reg_r2[:]+reg_r2a[:])
        return circ.to_gate()
    
    # reflection about |phi_j>
    def RBj(self,j):
        reg_r2=qk.QuantumRegister(self.msystem.n)
        reg_r2a=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg_r2,reg_r2a)
        Bj=self.Bj(j)
        circ.append(Bj.inverse(), reg_r2[:]+reg_r2a[:])
        circ.append(self.R0(), reg_r2[:]+reg_r2a[:])
        circ.append(Bj, reg_r2[:]+reg_r2a[:])
        return circ.to_gate()
    
    # reflection about |zeta_j>
    def RBp(self):
        reg_r2=qk.QuantumRegister(self.msystem.n)
        reg_r2a=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg_r2,reg_r2a)
        Bp=self.Bp()
        circ.append(Bp.inverse(), reg_r2[:]+reg_r2a[:])
        circ.append(self.R0(), reg_r2[:]+reg_r2a[:])
        circ.append(Bp, reg_r2[:]+reg_r2a[:])
        return circ.to_gate()
    
    # the walk operator
    def W(self):
        reg_r1=qk.QuantumRegister(self.msystem.n)
        reg_r1a=qk.QuantumRegister(1)
        reg_r2=qk.QuantumRegister(self.msystem.n)
        reg_r2a=qk.QuantumRegister(1)
        circ=qk.QuantumCircuit(reg_r1,reg_r1a,reg_r2,reg_r2a)
        format_str_j='{:0%db}'%(self.msystem.n)
        for j in range(self.msystem.N):
            j_bits=format_str_j.format(j)
            circ.append(self.RBj(j).control(self.msystem.n+1,None,'0'+j_bits),reg_r1[:]+reg_r1a[:]+reg_r2[:]+reg_r2a[:])
        circ.append(self.RBp().control(1,None,'1'),reg_r1a[:]+reg_r2[:]+reg_r2a[:])
        circ.append(self.S(),reg_r1[:]+reg_r1a[:]+reg_r2[:]+reg_r2a[:])
        circ.append(self.iI(),[reg_r1[self.msystem.n-1]])
        return circ.to_gate()

