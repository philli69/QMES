import sys
import math
import qiskit as qk
import MatrixProcedures as mp
import QAlgs as qa
import QWOps as qw
import qiskit.circuit.library as clib
from qiskit import *

EPSILON=1e-10
shots=4096
start_op=80
stop_op=130

IBMQ.enable_account('')
provider=IBMQ.get_provider(hub='ibm-q-research-2', group='uni-manitoba-1')

back1 = qk.BasicAer.get_backend('statevector_simulator')
back2 = provider.get_backend('ibmq_jakarta')
if(start_op==0):
    f = open('./job_data.dat', 'w')
else:
    f = open('./job_data.dat', 'a')

# the choice of nq_phase affects the accuracy of QPE
nq_phase=2

# initialize the matrix equation, and prepare it for the quantum procedure
print("Building system... ",end='')
sys.stdout.flush()
msystem = mp.MatrixSystem(M=2,expand=False)
msystem.RandInit()
msystem.PrepSystem()
print("Done.")

# calculate C
C=0.9999999998

# get the first initializer
init = qk.extensions.Initialize(msystem.b).copy(name='Init')

# initialize the quantum system itself
reg_phase=qk.QuantumRegister(nq_phase)
reg_r1=qk.QuantumRegister(msystem.n)
reg_r1a=qk.QuantumRegister(1)
reg_r2=qk.QuantumRegister(msystem.n)
reg_r2a=qk.QuantumRegister(1)
reg_a_hhl=qk.QuantumRegister(1)
reg_class=qk.ClassicalRegister(nq_phase+2*msystem.n+3)

reg_all=reg_phase[:]+reg_r1[:]+reg_r1a[:]+reg_r2[:]+reg_r2a[:]+reg_a_hhl[:]

l=0
def run_op(circ_init, op, reg_work, reg_all, reg_class):
    global l, start_op, stop_op, back1, back2
    
    circ = circ_init
    
    circ.append(op, reg_work)
    
    circ_transpiled=qk.transpile(circ,back1,optimization_level=2,basis_gates=['cx','id','rz','sx','x'])
    result1 = back1.run(circ_transpiled).result()
    statevector=result1.get_statevector()
    
    if((l>=start_op)and(l<stop_op)):
        circ.measure(reg_all,reg_class)
        circ_transpiled=qk.transpile(circ,back2,optimization_level=2,basis_gates=['cx','id','rz','sx','x'])
        job_id = back2.run(circ_transpiled,shots=shots).job_id()
        
        circ_temp=qk.QuantumCircuit(reg_work)
        circ_temp.append(op,reg_work)
        circ_transpiled=qk.transpile(circ_temp,back2,optimization_level=2,basis_gates=['cx','id','rz','sx','x'])
        n_ops=circ_transpiled.size()
        
        f.write('%d,%d,%s\n'%(l,n_ops,job_id))
        print('%d,%d,%s'%(l,n_ops,job_id))
    elif(l>=stop_op):
        f.close()
        sys.exit()
    else:
        print('Operator %d skipped.'%(l))
    
    l=l+1
    return qk.extensions.Initialize(statevector)

# initial T
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_r1)
init=run_op(circ_init, clib.HGate(), reg_r2, reg_all,reg_class)

# QPE
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate(),[reg_phase[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate(),[reg_phase[1]], reg_all,reg_class)

circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate().control(2,None,'01'),[reg_phase[0]]+[reg_r1a[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'11'),[reg_phase[0]]+[reg_r1a[0]]+[reg_r2a[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(1,None,'1'),[reg_phase[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(1,None,'1'),[reg_phase[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'01'),[reg_phase[0]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(2,None,'01'),[reg_phase[0]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'01'),[reg_phase[0]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate().control(2,None,'01'),[reg_phase[0]]+[reg_r1a[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'11'),[reg_phase[0]]+[reg_r1a[0]]+[reg_r2a[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SwapGate().control(1,None,'1'), [reg_phase[0]]+reg_r1[:]+reg_r2[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SwapGate().control(1,None,'1'), [reg_phase[0]]+reg_r1a[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SGate().control(1,None,'1'),[reg_phase[0]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[0]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SGate().control(1,None,'1'),[reg_phase[0]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[0]]+[reg_r1[0]], reg_all,reg_class)

circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate().control(2,None,'01'),[reg_phase[1]]+[reg_r1a[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'11'),[reg_phase[1]]+[reg_r1a[0]]+[reg_r2a[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(1,None,'1'),[reg_phase[1]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[1]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(1,None,'1'),[reg_phase[1]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[1]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'01'),[reg_phase[1]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(2,None,'01'),[reg_phase[1]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'01'),[reg_phase[1]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate().control(2,None,'01'),[reg_phase[1]]+[reg_r1a[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'11'),[reg_phase[1]]+[reg_r1a[0]]+[reg_r2a[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SwapGate().control(1,None,'1'), [reg_phase[1]]+reg_r1[:]+reg_r2[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SwapGate().control(1,None,'1'), [reg_phase[1]]+reg_r1a[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SGate().control(1,None,'1'),[reg_phase[1]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[1]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SGate().control(1,None,'1'),[reg_phase[1]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[1]]+[reg_r1[0]], reg_all,reg_class)

circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate().control(2,None,'01'),[reg_phase[1]]+[reg_r1a[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'11'),[reg_phase[1]]+[reg_r1a[0]]+[reg_r2a[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(1,None,'1'),[reg_phase[1]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[1]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(1,None,'1'),[reg_phase[1]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[1]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'01'),[reg_phase[1]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(2,None,'01'),[reg_phase[1]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'01'),[reg_phase[1]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate().control(2,None,'01'),[reg_phase[1]]+[reg_r1a[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'11'),[reg_phase[1]]+[reg_r1a[0]]+[reg_r2a[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SwapGate().control(1,None,'1'), [reg_phase[1]]+reg_r1[:]+reg_r2[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SwapGate().control(1,None,'1'), [reg_phase[1]]+reg_r1a[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SGate().control(1,None,'1'),[reg_phase[1]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[1]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SGate().control(1,None,'1'),[reg_phase[1]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[1]]+[reg_r1[0]], reg_all,reg_class)

circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SwapGate(),reg_phase, reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate(),[reg_phase[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.CZGate().power(0.5).inverse(),reg_phase, reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate(),[reg_phase[1]], reg_all,reg_class)

# HHL ancilla rotation
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
lam=-msystem.d
init=run_op(circ_init, clib.RYGate(2.0*math.acos(C/lam)).control(nq_phase,None,'00'), reg_phase[:]+reg_a_hhl[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
lam=msystem.X-msystem.d
init=run_op(circ_init, clib.RYGate(2.0*math.acos(C/lam)).control(nq_phase,None,'01'), reg_phase[:]+reg_a_hhl[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
lam=-msystem.d
init=run_op(circ_init, clib.RYGate(2.0*math.acos(C/lam)).control(nq_phase,None,'10'), reg_phase[:]+reg_a_hhl[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
lam=-msystem.X-msystem.d
init=run_op(circ_init, clib.RYGate(2.0*math.acos(C/lam)).control(nq_phase,None,'11'), reg_phase[:]+reg_a_hhl[:], reg_all,reg_class)

# inverse QPE
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate(),[reg_phase[1]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.CZGate().power(0.5),reg_phase, reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate(),[reg_phase[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SwapGate(),reg_phase, reg_all,reg_class)

circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[1]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SGate().inverse().control(1,None,'1'),[reg_phase[1]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[1]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SGate().inverse().control(1,None,'1'),[reg_phase[1]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SwapGate().control(1,None,'1'), [reg_phase[1]]+reg_r1a[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SwapGate().control(1,None,'1'), [reg_phase[1]]+reg_r1[:]+reg_r2[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'11'),[reg_phase[1]]+[reg_r1a[0]]+[reg_r2a[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate().control(2,None,'01'),[reg_phase[1]]+[reg_r1a[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'01'),[reg_phase[1]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(2,None,'01'),[reg_phase[1]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'01'),[reg_phase[1]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[1]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(1,None,'1'),[reg_phase[1]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[1]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(1,None,'1'),[reg_phase[1]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'11'),[reg_phase[1]]+[reg_r1a[0]]+[reg_r2a[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate().control(2,None,'01'),[reg_phase[1]]+[reg_r1a[0]]+[reg_r2[0]], reg_all,reg_class)


circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[1]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SGate().inverse().control(1,None,'1'),[reg_phase[1]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[1]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SGate().inverse().control(1,None,'1'),[reg_phase[1]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SwapGate().control(1,None,'1'), [reg_phase[1]]+reg_r1a[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SwapGate().control(1,None,'1'), [reg_phase[1]]+reg_r1[:]+reg_r2[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'11'),[reg_phase[1]]+[reg_r1a[0]]+[reg_r2a[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate().control(2,None,'01'),[reg_phase[1]]+[reg_r1a[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'01'),[reg_phase[1]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(2,None,'01'),[reg_phase[1]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'01'),[reg_phase[1]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[1]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(1,None,'1'),[reg_phase[1]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[1]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(1,None,'1'),[reg_phase[1]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'11'),[reg_phase[1]]+[reg_r1a[0]]+[reg_r2a[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate().control(2,None,'01'),[reg_phase[1]]+[reg_r1a[0]]+[reg_r2[0]], reg_all,reg_class)

circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[0]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SGate().inverse().control(1,None,'1'),[reg_phase[0]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[0]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SGate().inverse().control(1,None,'1'),[reg_phase[0]]+[reg_r1[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SwapGate().control(1,None,'1'), [reg_phase[0]]+reg_r1a[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.SwapGate().control(1,None,'1'), [reg_phase[0]]+reg_r1[:]+reg_r2[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'11'),[reg_phase[0]]+[reg_r1a[0]]+[reg_r2a[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate().control(2,None,'01'),[reg_phase[0]]+[reg_r1a[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'01'),[reg_phase[0]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(2,None,'01'),[reg_phase[0]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'01'),[reg_phase[0]]+reg_r2[:]+reg_r2a[:], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(1,None,'1'),[reg_phase[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(1,None,'1'),[reg_phase[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.ZGate().control(1,None,'1'),[reg_phase[0]]+[reg_r2[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.XGate().control(2,None,'11'),[reg_phase[0]]+[reg_r1a[0]]+[reg_r2a[0]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate().control(2,None,'01'),[reg_phase[0]]+[reg_r1a[0]]+[reg_r2[0]], reg_all,reg_class)

circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate(),[reg_phase[1]], reg_all,reg_class)
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate(),[reg_phase[0]], reg_all,reg_class)

# inverse T
circ_init=qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase,reg_class)
circ_init.append(init,reg_all)
init=run_op(circ_init, clib.HGate(),reg_r2, reg_all,reg_class)

f.close()

