import numpy as np

class BaseRobot():

    def __init__(self):
        pass

    def mmatrix(self,*matrices):
        n=0
        for m in matrices:
            if (n==0):
                ma=m
                n=n+1
            elif (n==1):
                r=np.dot(ma,m)
                n=n+1
            else:
                r=np.dot(r,m)
        return r


    def sind(self,t):
        return np.sin(np.radians(t))

    def cosd(self,t):
        return np.cos(np.radians(t))    

    def rotax(self,t):
        Rx=np.array(([1,0,0,0],[0,self.cosd(t),-self.sind(t),0],[0,self.sind(t),self.cosd(t),0],[0,0,0,1]))
        return Rx

    def rotay(self,t):
        Ry=np.array(([self.cosd(t),0,self.sind(t),0],[0,1,0,0],[-self.sind(t),0,self.cosd(t),0],[0,0,0,1]))
        return Ry

    def rotaz(self,t):
        Rz=np.array(([self.cosd(t),-self.sind(t),0,0],[self.sind(t),self.cosd(t),0,0],[0,0,1,0],[0,0,0,1]))
        return Rz
    
    def trasx(self,Dx):
        Tx=np.array(([[1,0,0,Dx],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
        return Tx

    def trasy(self,Dy):
        Ty=np.array(([[1,0,0,0],[0,1,0,Dy],[0,0,1,0],[0,0,0,1]]))
        return Ty

    def trasz(self,Dz):
        Tz=np.array(([[1,0,0,0],[0,1,0,0],[0,0,1,Dz],[0,0,0,1]]))
        return Tz
    
    #Funciones vectorizadas para obtener multiples pocisiones en el brazo en una sola pasada
    def _create_stack(self, n):
        """Auxiliar para crear una pila de matrices identidad de tamaño (n, 4, 4)"""
        return np.eye(4)[np.newaxis, :, :].repeat(n, axis=0)

    def rotaxM(self, t):
        t = np.atleast_1d(t)
        n = len(t)
        c, s = self.cosd(t), self.sind(t)
        Rx = self._create_stack(n)
        Rx[:, 1, 1] = c
        Rx[:, 1, 2] = -s
        Rx[:, 2, 1] = s
        Rx[:, 2, 2] = c
        return Rx if n > 1 else Rx[0]

    def rotayM(self, t):
        t = np.atleast_1d(t)
        n = len(t)
        c, s = self.cosd(t), self.sind(t)
        Ry = self._create_stack(n)
        Ry[:, 0, 0] = c
        Ry[:, 0, 2] = s
        Ry[:, 2, 0] = -s
        Ry[:, 2, 2] = c
        return Ry if n > 1 else Ry[0]

    def rotazM(self, t):
        t = np.atleast_1d(t)
        n = len(t)
        c, s = self.cosd(t), self.sind(t)
        Rz = self._create_stack(n)
        Rz[:, 0, 0] = c
        Rz[:, 0, 1] = -s
        Rz[:, 1, 0] = s
        Rz[:, 1, 1] = c
        return Rz if n > 1 else Rz[0]

    def trasxM(self, Dx):
        Dx = np.atleast_1d(Dx)
        n = len(Dx)
        Tx = self._create_stack(n)
        Tx[:, 0, 3] = Dx
        return Tx if n > 1 else Tx[0]

    def trasyM(self, Dy):
        Dy = np.atleast_1d(Dy)
        n = len(Dy)
        Ty = self._create_stack(n)
        Ty[:, 1, 3] = Dy
        return Ty if n > 1 else Ty[0]

    def traszM(self, Dz):
        Dz = np.atleast_1d(Dz)
        n = len(Dz)
        Tz = self._create_stack(n)
        Tz[:, 2, 3] = Dz
        return Tz if n > 1 else Tz[0]

    
    def minv(self,R):
        return np.linalg.inv(R)
    
    def minvOld(self,R):
        r=np.zeros((4,4))
        a=np.zeros((3,3))
        p=np.zeros((3,1))
        a[0,0]=R[0,0]
        a[0,1]=R[0,1]
        a[0,2]=R[0,2]
        a[1,0]=R[1,0]
        a[1,1]=R[1,1]
        a[1,2]=R[1,2]
        a[2,0]=R[2,0]
        a[2,1]=R[2,1]
        a[2,2]=R[2,2]
        a=np.transpose(a)
        r[0,0]=a[0,0]
        r[0,1]=a[0,1]
        r[0,2]=a[0,2]
        r[1,0]=a[1,0]
        r[1,1]=a[1,1]
        r[1,2]=a[1,2]
        r[2,0]=a[2,0]
        r[2,1]=a[2,1]
        r[2,2]=a[2,2]
        a=-1*a
        p[0,0]=R[0,3]
        p[1,0]=R[1,3]
        p[2,0]=R[2,3]
        p1=np.dot(a,p)
        r[0,3]=p1[0,0]
        r[1,3]=p1[1,0]
        r[2,3]=p1[2,0]
        r[3,3]=1
        return r
    