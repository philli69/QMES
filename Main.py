# references are made to arXiv:2112.02600 (accurate as of v3)
import sys
import time
import qiskit as qk
import MatrixProcedures as mp
import QAlgs as qa
import QWOps as qw

tstart=time.time()
sim = qk.BasicAer.get_backend('statevector_simulator')

outfile = open('./output.txt','w')

# the choice of nq_phase affects the accuracy of QPE
nq_phase=6
MM=2

# initialize the matrix equation, and prepare it for the quantum procedure
print("Building system... ",end='')
sys.stdout.flush()
msystem=mp.MatrixSystem(M=MM,expand=False)
msystem.RandInit(D=2)
msystem.PrepSystem()
print("Done.")
msystem.PrintMatrix()

# construct the top-level operators
qw_ops=qw.Operators(msystem)
print(msystem.b)
init = qk.extensions.Initialize(msystem.b).copy(name='Init')
print()
print("Initializing...")
sys.stdout.flush()
T0=qw_ops.T0().copy(name='T0')
print("T0 built")
sys.stdout.flush()
W=qw_ops.W().copy(name='W')
print("W built")
sys.stdout.flush()
QPE=qa.QPE(W,nq_phase,max(4*msystem.n,6))
print("QPE built")
sys.stdout.flush()
Rc=qa.HHLRotation(nq_phase, msystem)
print("Rc built")
sys.stdout.flush()
print("Finished initialization.")

# initialize the quantum system itself
reg_phase=qk.QuantumRegister(nq_phase)
reg_r1=qk.QuantumRegister(msystem.n)
reg_r1w=qk.QuantumRegister(max(1,msystem.n-1))
reg_r1a=qk.QuantumRegister(1)
reg_r2=qk.QuantumRegister(msystem.n)
reg_r2w=qk.QuantumRegister(max(1,msystem.n-1))
reg_r2a=qk.QuantumRegister(1)
reg_a_hhl=qk.QuantumRegister(1)
circ = qk.QuantumCircuit(reg_a_hhl,reg_r2a,reg_r2w,reg_r2,reg_r1a,reg_r1w,reg_r1,reg_phase)

# append the computed operators to build the complete circuit (Fig. 4 on p. 11)
print()
print("Building circuit...")
sys.stdout.flush()
circ.append(init, reg_r1)
circ.append(T0, reg_r1[:]+reg_r1w[:]+reg_r1a[:]+reg_r2[:]+reg_r2w[:]+reg_r2a[:])
print("T0 added")
sys.stdout.flush()
circ.append(QPE, reg_phase[:]+reg_r1[:]+reg_r1w[:]+reg_r1a[:]+reg_r2[:]+reg_r2w[:]+reg_r2a[:])
print("QPE added")
sys.stdout.flush()
circ.append(Rc, reg_phase[:]+reg_a_hhl[:])
print("Rc added")
sys.stdout.flush()
circ.append(QPE.inverse(), reg_phase[:]+reg_r1[:]+reg_r1w[:]+reg_r1a[:]+reg_r2[:]+reg_r2w[:]+reg_r2a[:])
print("QPE* added")
sys.stdout.flush()
circ.append(T0.inverse(), reg_r1[:]+reg_r1w[:]+reg_r1a[:]+reg_r2[:]+reg_r2w[:]+reg_r2a[:])
print("T0* added")
sys.stdout.flush()
print("Finished building circuit.")

# transpile the circuit
print()
print("Transpiling... ",end='')
sys.stdout.flush()
#circ_transpiled = qk.transpile(circ,sim)
circ_transpiled = qk.transpile(circ,sim,optimization_level=0)
print("Done.")
sys.stdout.flush()
print("Size of transpiled circuit: ", circ_transpiled.size())


# run the circuit and extract results
print()
print("Running circuit... ",end='')
sys.stdout.flush()
result = sim.run(circ_transpiled).result()
statevector = result.get_statevector()
print("Done.")

# process the results: extract the solution and compare with a classical solution
# can also check QPE by removing Rc and QPE inverses in the main circuit, and uncommenting below
#qa.PrintStatevector(statevector,nq_phase,msystem)
#qa.CheckQPE(statevector, nq_phase, msystem)
sol=qa.ExtractSolution(statevector,nq_phase,msystem)
msystem.CompareClassical(sol)

print()
print("Miscellaneous items:")
print("%d %d %f"%(msystem.N,circ_transpiled.size(),time.time()-tstart))

outfile.write(str(msystem.N))
outfile.write('\n')
outfile.write(str(sum([len(x) for x in msystem.A_indices[:]])))
outfile.write('\n')
outfile.write(str(circ_transpiled.size()))
outfile.write('\n')
outfile.write(str(time.time()-tstart))
outfile.write('\n')

outfile.close()
