import sys
import qiskit as qk
import MatrixProcedures as mp
import QAlgs as qa
import QWOps as qw

sim = qk.BasicAer.get_backend('statevector_simulator')

# the choice of nq_phase affects the accuracy of QPE
nq_phase=6

# initialize the matrix equation, and prepare it for the quantum procedure
print("Building system... ",end='')
sys.stdout.flush()
print()
msystem = mp.MatrixSystem(M=2,expand=False)
msystem.RandInit()
msystem.PrepSystem()
print("Done.")

# construct the top-level operators
print("Building operators... ",end='')
sys.stdout.flush()
qw_ops=qw.Operators(msystem)
init = qk.extensions.Initialize(msystem.b).copy(name='Init')
T=qw_ops.T().copy(name='T')
W=qw_ops.W().copy(name='W')
QPE=qa.QPE(W,nq_phase,2*msystem.n+2)
Rc=qa.HHLRotation(nq_phase, msystem)
print("Done.")

# initialize the quantum system itself
reg_phase=qk.QuantumRegister(nq_phase)
reg_r1=qk.QuantumRegister(msystem.n)
reg_r1a=qk.QuantumRegister(1)
reg_r2=qk.QuantumRegister(msystem.n)
reg_r2a=qk.QuantumRegister(1)
reg_a_hhl=qk.QuantumRegister(1)
circ = qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2,reg_r1a,reg_r1,reg_phase)

# append the computed operators to build the complete circuit
print("Building circuit... ",end='')
sys.stdout.flush()
circ.append(init, reg_r1)
circ.append(T, reg_r1[:]+reg_r1a[:]+reg_r2[:]+reg_r2a[:])
circ.append(QPE, reg_phase[:]+reg_r1[:]+reg_r1a[:]+reg_r2[:]+reg_r2a[:])
circ.append(Rc, reg_phase[:]+reg_a_hhl[:])
circ.append(QPE.inverse(), reg_phase[:]+reg_r1[:]+reg_r1a[:]+reg_r2[:]+reg_r2a[:])
circ.append(T.inverse(), reg_r1[:]+reg_r1a[:]+reg_r2[:]+reg_r2a[:])
print("Done.")

# transpile the circuit
print("Transpiling... ",end='')
sys.stdout.flush()
circ_transpiled = qk.transpile(circ,sim)
#circ_transpiled = qk.transpile(circ,sim,optimization_level=2,basis_gates=['cx','id','rz','sx','x'])
print("Done.")
print("Size of transpiled circuit: ", circ_transpiled.size())


# run the circuit and extract results
print("Running circuit... ",end='')
sys.stdout.flush()
result = sim.run(circ_transpiled).result()
statevector = result.get_statevector()
print("Done.")

# process the results: extract the solution and compare with a classical solution
# can also check QPE by removing Rc and QPE inverses in the main circuit, and uncommenting below
sol=qa.ExtractSolution(statevector,nq_phase,msystem)
msystem.CompareClassical(sol)
#qa.CheckQPE(statevector, nq_phase, msystem)

