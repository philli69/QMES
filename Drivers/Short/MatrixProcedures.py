import math
import numpy as np
import random
import bisect

EPSILON=1e-10
i=complex(0.0,1.0)

class MatrixSystem:
    def __init__(self,M=1,expand=True):
        self.expand=expand
        self.M=M
        self.n=0
        self.N=0
        self.bnorm=0.0
        self.d=0.0
        self.ap=0.0
        self.X=0.0
        self.C=0.0
        self.A0_indices=[]
        self.A0_elements=[]
        self.b0=[]
        self.A_indices=[]
        self.A_elements=[]
        self.b=[]
    
    def FileInit(self):
        f=open('./MS.txt','r')
        lines=f.readlines()
        f.close()
        jp=0
        for c in range(len(lines)):
            l=lines[c].split(',')
            j=int(l[0])-1
            k=int(l[1])-1
            while(j>jp):
                jp=jp+1
                self.A0_indices.append([])
                self.A0_elements.append([])
            self.A0_indices[jp].append(k)
            self.A0_elements[jp].append(float(l[2]))
        f=open('./b0.txt','r')
        lines=f.readlines()
        f.close()
        for c in range(len(lines)):
            self.b0.append(float(l[0]))
    
    # parallel strip system
    def MoMInit(self):
        l=2.0/float(self.M)
        p=[]
        
        for j in range(self.M):
            if(j<self.M/2):
                self.b0.append(1.0)
                p.append([-0.5,l*j-0.5])
            else:
                p.append([0.5,l*(j-self.M/2)-0.5])
                self.b0.append(-1.0)
        
        for j in range(self.M):
            self.A0_indices.append([])
            self.A0_elements.append([])
            for k in range(self.M):
                self.A0_indices[j].append(k)
                if(j==k):
                    self.A0_elements[j].append(-l/(1.0)*(math.log(l)-1.5))
                else:
                    self.A0_elements[j].append(-l/(1.0)*math.log(math.sqrt((p[j][0]-p[k][0])**2.0+(p[j][1]-p[k][1])**2.0)))
    
    def RandInit(self,D=1):
        fill_prob = float(D-1)/float(self.M)  # remove one since we'll definitely put an element on the diagonal
        
        # initialize the rows of A0
        for j in range(self.M):
            # always put an element on the diagonal--avoids singular matrices
            self.A0_indices.append([j])
            self.A0_elements.append([])
        
        # generate D-1 random integers per row and store them as indices, ignoring recurrent indices (statistics on "D-1 nonzeros" could be better)
        for j in range(self.M):
            for c in range(D-1):
                k=random.randint(0,self.M-1)
                if(not (k in self.A0_indices[j])):
                    bisect.insort(self.A0_indices[j],k)
                    bisect.insort(self.A0_indices[k],j)
            
        # populate the matrix
        for j in range(self.M):
            for c in range(len(self.A0_indices[j])):
                k=self.A0_indices[j][c]
                if(k==j):
                    x=2.0*(random.random()-0.5)
                    self.A0_elements[j].append(complex(x,0.0))
                elif(k>j):
                    x=2.0*(random.random()-0.5)
                    y=2.0*(random.random()-0.5)
                    self.A0_elements[j].append(complex(x,y))
                    self.A0_elements[k].append(complex(x,-y))  # order works automatically
        
        # initialize b0 to a fully random vector
        for j in range(self.M):
            x=2.0*(random.random()-0.5)
            y=2.0*(random.random()-0.5)
            self.b0.append(np.complex(x,y))
    
    def PrepSystem(self):
        # determine the appropriate size of the system for quantum operation
        if(self.expand):
            self.n=math.ceil(math.log(self.M,2)-EPSILON)+1
            self.N=int(math.pow(2.0,self.n)+EPSILON)
        else:
            self.n=math.ceil(math.log(self.M,2)-EPSILON)
            self.N=self.M
        
        self.A_indices=[]
        self.A_elements=[]
        self.b=[]
        
        # set the elements of A and b
        if(self.expand):
            for j in range(self.N):
                self.A_indices.append([])
                self.A_elements.append([])
            
            for j in range(self.M):
                self.b.append(self.b0[j])
            for j in range(self.M,self.N):
                self.b.append(0.0)
            
            for j in range(self.M):
                for c in range(len(self.A0_indices[j])):
                    k=self.A0_indices[j][c]
                    self.A_indices[j].append(k+self.M)
                    self.A_indices[k+self.M].append(j)
                    self.A_elements[j].append(self.A0_elements[j][c])
                    self.A_elements[k+self.M].append(np.conjugate(self.A0_elements[j][c]))
            for j in range(2*self.M,self.N):
                self.A_indices[j].append(j)
                self.A_elements[j].append(1.0)
        else:
            for j in range(self.M):
                self.b.append(self.b0[j])
                self.A_indices.append([])
                self.A_elements.append([])
                for c in range(len(self.A0_indices[j])):
                    self.A_indices[j].append(self.A0_indices[j][c]);
                    self.A_elements[j].append(self.A0_elements[j][c]);
        
        # normalize b
        self.bnorm=math.sqrt(sum([np.absolute(x)*np.absolute(x) for x in self.b]))
        for j in range(self.N):
            self.b[j]=self.b[j]/self.bnorm
            for c in range(len(self.A_elements[j])):
                self.A_elements[j][c]=self.A_elements[j][c]/self.bnorm
        
        # get and apply the diagonal offset (if system is not expanded, use maximal in magnitude diagonal element of A. no modification needed if system is expanded)
        if(self.expand):
            self.d=0.0
        else:
            self.d=0.0
            for j in range(self.N):
                for c in range(len(self.A_indices[j])):
                    if(self.A_indices[j][c]==j):
                        if(self.d<np.absolute(self.A_elements[j][c])):
                            self.d=np.absolute(self.A_elements[j][c])
                    elif(self.A_indices[j][c]>j):
                        break
            for j in range(self.N):
                for c in range(len(self.A_indices[j])):
                    if(self.A_indices[j][c]==j):
                        self.A_elements[j][c]=self.A_elements[j][c]+self.d
                    elif(self.A_indices[j][c]>j):
                        break
        
        # calculate X (get upper bound on maximal in magnitude element of A first)
        self.ap=0.0
        for j in range(self.N):
            for elem in self.A_elements[j]:
                if(self.ap<np.absolute(elem)):
                    self.ap=np.absolute(elem)
        self.X=np.absolute(float(self.N)*self.ap+EPSILON)
    
    def CompareClassical(self,sol):
        A0=[]
        for j in range(self.M):
            A0.append([])
            for k in range(self.M):
                A0[j].append(complex(0.0,0.0))
        for j in range(self.M):
            for c in range(len(self.A0_indices[j])):
                k=self.A0_indices[j][c]
                A0[j][k]=self.A0_elements[j][c]
        A0inv=np.linalg.inv(A0)
        sol_class=A0inv.dot(self.b0)
        
        # get the relative global phase
        t1=np.angle(sol[0])
        t2=np.angle(sol_class[0])
        trel=t2-t1
        
        # compare
        print('----------------------------')
        print('Solution Comparison: ')
        print()
        print('%24s|%24s'%('Quantum','Classical'))
        print('-------------------------------------------------')
        for j in range(self.M):
            print('%10f+i(%10f)|%10f+i(%10f)'%((sol[j]*np.exp(i*trel)).real,(sol[j]*np.exp(i*trel)).imag,sol_class[j].real,sol_class[j].imag))
        print()
        print('Relative Errors:')
        summ=0.0
        for j in range(self.M):
            summ=summ+np.absolute(1.0-sol[j]/sol_class[j]*np.exp(i*trel))
            print(np.absolute(1.0-sol[j]/sol_class[j]*np.exp(i*trel)))
        print()
        print('Average Relative Error: ', summ/float(self.M))
    
    def PrintMatrix(self):
        for j in range(self.M):
            c=0
            for k in range(self.M):
                if(c<len(self.A0_indices[j])):
                    if(self.A0_indices[j][c]==k):
                        print(" %5.2f+(%5.2f) "%(self.A0_elements[j][c].real,self.A0_elements[j][c].imag),end='')
                        c=c+1
                    else:
                        print("  0.00+( 0.00) ",end='')
                else:
                    print("  0.00+( 0.00) ",end='')
            print("\n")
        print("\n")
        print("\n")
        for j in range(self.N):
            c=0
            for k in range(self.N):
                if(c<len(self.A_indices[j])):
                    if(self.A_indices[j][c]==k):
                        print(" %5.2f+(%5.2f) "%(self.A_elements[j][c].real,self.A_elements[j][c].imag),end='')
                        c=c+1
                    else:
                        print("  0.00+( 0.00) ",end='')
                else:
                    print("  0.00+( 0.00) ",end='')
            print("\n")

