import math
import numpy as np
import random

EPSILON=1e-10
i=complex(0.0,1.0)

class MatrixSystem:
    
    def __init__(self,M=1,expand=True):
        self.expand=expand
        self.M=M
        self.n=0
        self.N=0
        self.A0=None
        self.b0=None
        self.A=None
        self.b=None
        self.bnorm=0.0
        self.d=0.0
        self.ap=0.0
        self.X=0.0
        self.C=0.0
    
    def RandInit(self):
        self.A0 = np.zeros(shape=(self.M,self.M),dtype=np.complex_)
        self.b0 = np.zeros(shape=(self.M),dtype=np.complex_)
        
        # initialize A0 to a Hermitian matrix if expand is False, otherwise initialize A0 to a fully random matrix
        for j in range(self.M):
            for k in range(self.M):
                if(j<k or self.expand):
                    x=2.0*(random.random()-0.5)
                    y=2.0*(random.random()-0.5)
                    self.A0[j][k] = np.complex(x,y)
                elif(j==k):
                    x=2.0*(random.random()-0.5)
                    self.A0[j][k] = np.complex(x,0.0)
                else:
                    self.A0[j][k] = np.conjugate(self.A0[k][j])
        
        # initialize b0 to a fully random vector
        for j in range(self.M):
            x=2.0*(random.random()-0.5)
            y=2.0*(random.random()-0.5)
            self.b0[j] = np.complex(x,y)
    
    def PrepSystem(self):
        
        # determine the appropriate size of the system for quantum operation
        if(self.expand):
            self.n=math.ceil(math.log(self.M,2)-EPSILON)+1
            self.N=int(math.pow(2.0,self.n)+EPSILON)
        else:
            self.n=math.ceil(math.log(self.M,2)-EPSILON)
            self.N=self.M
        
        self.A = np.zeros(shape=(self.N,self.N),dtype=np.complex_)
        self.b = np.zeros(shape=(self.N),dtype=np.complex_)
        
        # set the elements of A and b
        for j in range(self.M):
            self.b[j]=self.b0[j]
            for k in range(self.M):
                if(self.expand):
                    self.A[j][k+self.M]=self.A0[j][k]
                    self.A[j+self.M][k]=np.conjugate(self.A0[k][j])
                else:
                    self.A[j][k]=self.A0[j][k]
        if(self.expand):
            for j in range(2*self.M,self.N):
                self.A[j][j]=np.complex(1.0,0.0)
        
        # normalize b
        self.bnorm=math.sqrt(sum([np.absolute(x)*np.absolute(x) for x in self.b]))
        for j in range(self.N):
            self.b[j]=self.b[j]/self.bnorm
            for k in range(self.N):
                self.A[j][k]=self.A[j][k]/self.bnorm
        
        # get and apply the diagonal offset (if system is not expanded, use maximal in magnitude diagonal element of A. no modification needed if system is expanded)
        if(self.expand):
            self.d=0.0
        else:
            self.d=np.absolute(self.A[0][0])
            for j in range(1,self.N):
                if(self.d<np.absolute(self.A[j][j])):
                    self.d=np.absolute(self.A[j][j])
            #self.d=3.0#HERE
            for j in range(self.N):
                self.A[j][j]=self.A[j][j]+self.d
        
        # calculate X (get upper bound on maximal in magnitude element of A first)
        self.ap=self.A[0][0]
        for j in range(self.N):
            for k in range(self.N):
                if(self.ap<np.absolute(self.A[j][k])):
                    self.ap=np.absolute(self.A[j][k])
        self.X=np.absolute(float(self.N)*self.ap+EPSILON)
    
    def CompareClassical(self,sol):
        A0inv=np.linalg.inv(self.A0)
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

