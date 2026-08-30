import numpy as np

from gahomotopy.kinematics.base_robot import BaseRobot

class ROARM3DOF(BaseRobot):
    """3-DOF RoArm M2 robotic arm."""

    ga_ranges = {
        'radius': {'low': 0.05, 'high': 0.3},
        'obstacle_value': {'low': 100, 'high': 100000},
        'off_diagonal': 50,
        'diagonal': 50,
    }

    def __init__(self):
        super().__init__()          
        self.mb=np.eye(4)
        self.obstacles={}
        self.numSegments=200
        
    def XYToAngles(self,x,y):        
        pos=self.mb@self.trasx(x)@self.trasy(y)

        ang=self.M2newton(pos,[0,0,0])

        if(ang[0]<0):
            ang[0]=360+ang[0]

        if(ang[1]<0):
            ang[1]=360+ang[1]    
            
        return ang

    def invKinematics(self,x,y,z,sem):        
        pos=np.eye(4)@self.trasx(x)@self.trasy(y)@self.trasz(z)
        ang=self.inverseKinematics(pos,sem)

        #if(ang[0]<0):
        #    ang[0]=360+ang[0]

        #if(ang[1]<0):
        #    ang[1]=360+ang[1]    

        #if(ang[2]<0):
        #    ang[2]=360+ang[2]    
            
        return ang
    

    def CreateConfiSpaceVectorized(self):
        obstaclesCS = []
        
        # 1. Pre-process Obstacles
        # IMPORTANT: Ensure your obstacle 'center' now has 3 values [x, y, z]
        # Shape: (Num_Obstacles, 3)
        obs_centers = np.array([o['center'] for o in self.obstacles]) 
        
        # Shape: (Num_Obstacles,)
        obs_radii_sq = np.array([o['radius']**2 for o in self.obstacles]) 
        new_radii = np.array([o['radius']/8 for o in self.obstacles])

        # 2. Grid for w2 and w3
        w_range = np.arange(180) - 90
        w2_grid, w3_grid = np.meshgrid(w_range, w_range, indexing='ij')
        w2_flat = w2_grid.flatten()
        w3_flat = w3_grid.flatten()
        
        # 3. Outer Loop for w1
        for w1 in range(180):
            w1R = w1 - 180
            #w1R = w1 - 0
            
            # --- Vectorized Kinematics ---
            # Returns Shape: (32400, 100, 4, 4)
            all_poses = self.getMidPosBatch(w1R, w2_flat, w3_flat)
            
            # --- UPDATED: Extract X, Y, AND Z ---
            x_coords = all_poses[:, :, 0, 3] 
            y_coords = all_poses[:, :, 1, 3]
            z_coords = all_poses[:, :, 2, 3] # <--- Added Z extraction
            
            # --- UPDATED: 3D Vectorized Collision Check ---
            
            # Stack X, Y, Z: Shape (32400, 100, 3)
            robot_points = np.stack([x_coords, y_coords, z_coords], axis=-1)
            
            # Calculate squared 3D Euclidean distance
            # (P - O)**2
            # P: (32400, 100, 1, 3)
            # O: (1, 1, Num_Obs, 3)
            diff = robot_points[:, :, np.newaxis, :] - obs_centers[np.newaxis, np.newaxis, :, :]
            
            # Sum over the last axis (x, y, z)
            dist_sq = np.sum(diff**2, axis=-1)
            
            # Check collisions
            collisions = dist_sq < obs_radii_sq
            
            # Reduction: Did this configuration hit any obstacle with any segment?
            config_hits = np.any(collisions, axis=(1, 2))
            
            if np.any(config_hits):
                hit_indices = np.where(config_hits)[0]
                bad_w2 = w2_flat[hit_indices]
                bad_w3 = w3_flat[hit_indices]
                
                for idx, w2_val, w3_val in zip(hit_indices, bad_w2, bad_w3):
                    # Find which obstacle was hit to get the correct radius
                    # We take the first one hit if multiple are hit
                    specific_obs_idx = np.where(np.any(collisions[idx], axis=0))[0][0]
                    r_val = new_radii[specific_obs_idx]
                    
                    obstaclesCS.append({
                        'center': (w1R, w2_val, w3_val), 
                        'radius': r_val
                    })

        return obstaclesCS

    def getMidPosBatch(self, t1_scalar, t2_array, t3_array):
        """
        Calculates positions for a batch of angles.
        t1_scalar: single float
        t2_array: shape (N,)
        t3_array: shape (N,)
        Returns: (N, 100, 4, 4) matrix stack
        """
        num_configs = len(t2_array)
        
        # Generate K segments (Shape: 100)
        k = np.linspace(0, 1, self.numSegments)
        kn = k * 4
        
        # Segment limits (Shape: 100)
        k1 = np.clip(kn, 0, 1)
        k2 = np.clip(kn - 1, 0, 1)
        k3 = np.clip(kn - 2, 0, 1)
        k4 = np.clip(kn - 3, 0, 1)

        # --- Base Transformations (Broadcast Compatible) ---
        
        # T0: Base (1, 1, 4, 4)
        t0 = (np.eye(4) @ self.mb).reshape(1, 1, 4, 4)
        
        # T01: Rot Y (Scalar t1) -> (1, 1, 4, 4)
        t01 = t0 @ self.rotayM_batch(np.array([t1_scalar]))
        
        # T12: Trans Y (Depends on K segments) -> (1, 100, 4, 4)
        t12 = self.trasyM_batch_segments(5.196 * k1)
        
        # T23: Rot Z (Depends on t2 array) -> (N, 1, 4, 4)
        t23 = self.rotazM_batch(t2_array)
        
        # T34: Trans Y (Depends on K segments) -> (1, 100, 4, 4)
        t34 = self.trasyM_batch_segments(23.682 * k2)
        
        # T45: Trans X (Depends on K segments) -> (1, 100, 4, 4)
        t45 = self.trasxM_batch_segments(3.0 * k3)
        
        # T56: Rot Z (Depends on t3 array) -> (N, 1, 4, 4)
        t56 = self.rotazM_batch(t3_array)
        
        # T67: Trans X (Depends on K segments) -> (1, 100, 4, 4)
        t67 = self.trasxM_batch_segments(28.015 * k4)

        # --- Matrix Chain Multiplication with Broadcasting ---
        # NumPy matmul (@) broadcasts shapes like: (N, 1, 4, 4) @ (1, 100, 4, 4) -> (N, 100, 4, 4)
        
        m_02 = t01 @ t12    # (1, 100, 4, 4)
        m_03 = m_02 @ t23   # (1, 100) @ (N, 1) -> (N, 100, 4, 4)
        m_04 = m_03 @ t34   # (N, 100) @ (1, 100) -> (N, 100, 4, 4)
        m_05 = m_04 @ t45
        m_06 = m_05 @ t56
        res  = m_06 @ t67
        
        return res

    # --- Helper Matrix Functions ---
    # These helpers ensure outputs are (Batch, 1, 4, 4) or (1, Batch, 4, 4)
    # so they broadcast correctly against each other.
    
    def rotazM_batch(self, theta):
        # theta shape: (N,)
        # Returns: (N, 1, 4, 4)
        N = len(theta)
        rad = np.radians(theta)
        c = np.cos(rad)
        s = np.sin(rad)
        zs = np.zeros(N)
        os = np.ones(N)
        
        # Build matrices
        mat = np.array([
            [c, -s, zs, zs],
            [s,  c, zs, zs],
            [zs, zs, os, zs],
            [zs, zs, zs, os]
        ]).transpose(2, 0, 1) # Shape (N, 4, 4)
        
        return mat[:, np.newaxis, :, :] # Expand to (N, 1, 4, 4)

    def rotayM_batch(self, theta):
        N = len(theta)
        rad = np.radians(theta)
        c = np.cos(rad)
        s = np.sin(rad)
        zs = np.zeros(N)
        os = np.ones(N)
        
        mat = np.array([
            [ c, zs,  s, zs],
            [zs, os, zs, zs],
            [-s, zs,  c, zs],
            [zs, zs, zs, os]
        ]).transpose(2, 0, 1)
        return mat[:, np.newaxis, :, :]

    def trasyM_batch_segments(self, dists):
        # dists shape: (100,)
        # Returns: (1, 100, 4, 4) - Note the 1 at the start for broadcasting
        N = len(dists)
        zs = np.zeros(N)
        os = np.ones(N)
        
        mat = np.array([
            [os, zs, zs, zs],
            [zs, os, zs, dists],
            [zs, zs, os, zs],
            [zs, zs, zs, os]
        ]).transpose(2, 0, 1)
        return mat[np.newaxis, :, :, :]

    def trasxM_batch_segments(self, dists):
        N = len(dists)
        zs = np.zeros(N)
        os = np.ones(N)
        
        mat = np.array([
            [os, zs, zs, dists],
            [zs, os, zs, zs],
            [zs, zs, os, zs],
            [zs, zs, zs, os]
        ]).transpose(2, 0, 1)
        return mat[np.newaxis, :, :, :]
    
    def CreateConfiSpace(self):
        obstaclesCS=[]
        for w1 in range(180):
            for w2 in range(180):
                for w3 in range(180):
                    w1R=w1-90
                    w2R=w2-90
                    w3R=w3-90      
                    obs=False              
                    for i in range(100):
                        val=i/100                                                
                        pos=self.getMidPos(w1R,w2R,w3R,val)   
                        #pos=self.M2v(w1R,w2R,w3R)
                        x0=pos[0,3]
                        y0=pos[1,3]
                        z0=pos[2,3]
                        for obs in self.obstacles:
                            dis=(x0-obs['center'][0])**2+(y0-obs['center'][1])**2+(z0-obs['center'][2])**2
                            if(dis<obs['radius']**2):
                                newR=obs['radius']/8
                                obstaclesCS.append({'center': (w1R, w2R, w3R), 'radius': newR})
                                obs=True
                                break
                        if(obs):
                            break
        
        return obstaclesCS
    
    def getMidPos(self,t1,t2,t3,k,mb=np.eye(4)):
        #k = np.linspace(0, 1, self.numSegments)
        kn = k * 4
        
        # 2. Aplicar límites
        k1 = np.clip(kn, 0, 1)
        k2 = np.clip(kn - 1, 0, 1)
        k3 = np.clip(kn - 2, 0, 1)
        k4 = np.clip(kn - 3, 0, 1)                

        t0=np.eye(4)@self.mb
        t01=t0@self.rotay(t1)
        t12=self.trasy(5.196 * k1)
        t23=self.rotaz(t2)
        t34=self.trasy(23.682 * k2)
        t45=self.trasx(3.0 * k3)
        t56=self.rotaz(t3)
        t67=self.trasx(28.015 * k4)

        
        t02=t01@t12
        t03=t02@t23
        t04=t03@t34
        t05=t04@t45
        t06=t05@t56
        res=t06@t67
        
        return res 
    
    def CalculateDistanceToObstacles(self,w1,w2,w3): 
        numObs=len(self.obstacles)
        obsRet=np.zeros((self.numSegments,numObs))
        obs=False              
        for i in range(self.numSegments):
            val=i/self.numSegments                        
            pos=self.getMidPos(w1,w2,w3,val)   
            #pos=self.M2v(w1R,w2R,w3R)
            xk=pos[0,3]
            yk=pos[1,3]
            zk=pos[2,3]
            indexObs=0
            for obs in self.obstacles:
                circularEq=((xk-obs[0])**2)+((yk-obs[1])**2)+((zk-obs[2])**2)-(obs[3]**2)
                obsRet[i,indexObs]=circularEq
                indexObs+=1                        
        return obsRet    
    
    def drawArm(self,w1,w2,w3):
        pos=self.getMidPosFast(w1,w2,w3)
        #self.livePlot.getPointsAndGraph(pos)

    def CalculateDistanceToObstaclesFast(self,ang): 
        numObs=len(self.obstacles)
        obsRet=np.zeros((self.numSegments,numObs))
        obs=False               
        w1,w2,w3=ang
                                     
        pos=self.getMidPosFast(w1,w2,w3)

        xk = pos[:, 0, 3]  # Array de todas las X (tamaño numSegments)
        yk = pos[:, 1, 3]  # Array de todas las Y
        zk = pos[:, 2, 3]  # Array de todas las Z
        
        obs_array = np.array(self.obstacles) # Forma: (num_obs, 4)

        # Separamos los datos de los obstáculos
        obs_x = obs_array[:, 0] # X de cada obstáculo
        obs_y = obs_array[:, 1] # Y de cada obstáculo
        obs_z = obs_array[:, 2] # Z de cada obstáculo
        obs_r = obs_array[:, 3] # Radio de cada obstáculo

        # Usamos None (o np.newaxis) para forzar el broadcasting
        # Esto crea una matriz de (numSegments, num_obs) automáticamente
        dx = xk[:, np.newaxis] - obs_x
        dy = yk[:, np.newaxis] - obs_y
        dz = zk[:, np.newaxis] - obs_z

        dis=(dx**2 + dy**2 + dz**2) - (obs_r**2)

        crashM=dis<0

        crash = np.sum(crashM) > 0

        # Ecuación circular para todos los puntos y todos los obstáculos a la vez                                       
        return dis,crash
    
    def CalculateDistanceToObstaclesCS(self,w1,w2,w3): 
        numObs=len(self.obstacles)
        obsRet=np.zeros((self.numSegments,numObs))
        obs=False              
                                             
        
        obs_array = np.array(self.obstacles) # Forma: (num_obs, 4)

        # Separamos los datos de los obstáculos
        obs_x = obs_array[:, 0] # X de cada obstáculo
        obs_y = obs_array[:, 1] # Y de cada obstáculo
        obs_z = obs_array[:, 2] # Z de cada obstáculo
        obs_r = obs_array[:, 3] # Radio de cada obstáculo

        # Usamos None (o np.newaxis) para forzar el broadcasting
        # Esto crea una matriz de (numSegments, num_obs) automáticamente
        dx = w1 - obs_x
        dy = w2 - obs_y
        dz = w3 - obs_z

        # Ecuación circular para todos los puntos y todos los obstáculos a la vez                                       
        return (dx**2 + dy**2 + dz**2) - (obs_r**2)
    
    
    def getMidPosFast(self,t1,t2,t3,mb=np.eye(4)):
        #k = np.linspace(0, (self.numSegments - 1) / self.numSegments, self.numSegments)
        k = np.linspace(0, 1, self.numSegments)
        kn = k * 4
        
        # 2. Aplicar límites
        k1 = np.clip(kn, 0, 1)
        k2 = np.clip(kn - 1, 0, 1)
        k3 = np.clip(kn - 2, 0, 1)
        k4 = np.clip(kn - 3, 0, 1)                

        t0=np.eye(4)@self.mb
        t01=t0@self.rotayM(t1)
        t12=self.trasyM(5.196 * k1)
        t23=self.rotazM(t2)
        t34=self.trasyM(23.682 * k2)
        t45=self.trasxM(3.0 * k3)
        t56=self.rotazM(t3)
        t67=self.trasxM(28.015 * k4)

        
        t02=t01@t12
        t03=t02@t23
        t04=t03@t34
        t05=t04@t45
        t06=t05@t56
        res=t06@t67
        
        return res
                

    def setObstaclesMatrix(self,obstaclesx):
        self.obstacles=[]
        for obs in obstaclesx:
            obsx=self.mb[0,3]+obs['center'][0]
            obsy=self.mb[1,3]+obs['center'][1]
            obsz=self.mb[2,3]+obs['center'][2]
            self.obstacles.append({'center': (obsx, obsy, obsz), 'radius': obs['radius']})
            
    def setObstaclesPos(self,obstaclesx):
        self.obstacles=np.zeros((len(obstaclesx),4))
        index=0
        for obs in obstaclesx:
            obsx=obs['center'][0]
            obsy=obs['center'][1]
            obsz=obs['center'][2]
            radius=obs['radius']
            self.obstacles[index,0]=obsx
            self.obstacles[index,1]=obsy
            self.obstacles[index,2]=obsz
            self.obstacles[index,3]=radius            
            index+=1

        #self.livePlot.setObstacles(self.obstacles)

    def setObstaclesPosParams(self,obstaclesx,params):
        self.obstacles=np.zeros((len(obstaclesx),4))
        index=0
        for obs in obstaclesx:
            obsx=obs['center'][0]
            obsy=obs['center'][1]
            obsz=obs['center'][2]
            radius=obs['radius']
            self.obstacles[index,0]=obsx
            self.obstacles[index,1]=obsy
            self.obstacles[index,2]=obsz
            self.obstacles[index,3]=radius            
            index+=1

        #self.livePlot.setObstacles(self.obstacles)
        #self.livePlot.setParams(params)
    
    def setObstaclesDic(self,obstaclesx):
        self.obstacles=obstaclesx
            
    def setNumSegments(self,numSegments):
        self.numSegments=numSegments
        

    #Obtiene la cinematica directa para el brazo (dados los angulos la matriz espacial del efector final)

    def directKinematics(self, *args, mb=np.eye(4)):
        """Forward kinematics for the 3-DOF RoArm M2.

        Accepts either a single array-like of 3 joint angles or 3 separate
        scalars, so it is compatible with both the homotopy planner (which
        passes a numpy array, matching UR3E.directKinematics) and the legacy
        callers that unpack angles individually.

            directKinematics([t1, t2, t3])   # array form
            directKinematics(t1, t2, t3)     # scalar form
        """
        if len(args) == 1:
            ts = args[0]
            t1, t2, t3 = ts[0], ts[1], ts[2]
        elif len(args) == 3:
            t1, t2, t3 = args
        else:
            raise TypeError(
                f"directKinematics expects 1 array or 3 scalars, got {len(args)}"
            )

        t0=np.eye(4)@self.mb
        t01=t0@self.rotaz(t1)
        t12=self.trasz(51.96)
        t23=self.rotay(t2)
        t34=self.trasz(236.82)
        t45=self.trasx(30.)
        t56=self.rotay(t3)
        t67=self.trasx(280.15)
        t78=self.trasz(-14)

        
        t02=t01@t12
        t03=t02@t23
        t04=t03@t34
        t05=t04@t45
        t06=t05@t56
        t07=t06@t67
        t08=t07@t78
        
        return t08
    

    
    def compute_jacobian(self, angles, epsilon=1e-6):
        t1, t2, t3 = angles
        current_pos = self.directKinematics(t1, t2, t3)[:3, 3]
        J = np.zeros((3, 3))
        
        for i in range(3):
            perturbed_angles = list(angles)
            perturbed_angles[i] += epsilon
            new_pos = self.directKinematics(*perturbed_angles)[:3, 3]
            J[:, i] = (new_pos - current_pos) / epsilon
            
        return J

    def inverseKinematics(self, target, initial_guess=[0.0, 0.0, 0.0], max_iter=500, tol=1e-4):
        q = np.array(initial_guess, dtype=float)
        
        for i in range(max_iter):
            current_tf = self.directKinematics(q[0], q[1], q[2])
            current_pos = current_tf[:3, 3]

            target_pos = target[:3, 3]

            # Calculate error
            error = target_pos - current_pos
            
            if np.linalg.norm(error) < tol:
                return q # Success!
            
            # Newton-Raphson Step
            J = self.compute_jacobian(q)
            
            # Use pseudo-inverse to handle singularities (safer than np.linalg.inv)
            dq = np.linalg.pinv(J) @ error
            
            # Apply update
            q += dq
            
        return q # Return best effort
    
    def M2newton(self,vd,sem,mh=np.eye(4),mb=np.eye(4)):
        
        d=0.001
        calc=True
        failed=False
        j=np.zeros([12,3])
        b=np.ones([12,1])
        t1=sem[0]
        t2=sem[1]
        t3=sem[2]
        t1d=t1+d
        t2d=t2+d
        t3d=t3+d
        con=0
        error=0
        while calc:
            con=con+1
            t=self.directKinematics(t1,t2,t3)            
            tn=t-vd

            tn[0,0]=0
            tn[0,1]=0
            tn[0,2]=0

            tn[1,0]=0
            tn[1,1]=0
            tn[1,2]=0

            tn[2,0]=0
            tn[2,1]=0
            tn[2,2]=0
            #print("vd llegar")
            #print(vd)
            #print("error vd")
            #print(tn)
            b[0,0]=tn[0,0]
            b[1,0]=tn[0,1]
            b[2,0]=tn[0,2]
            b[3,0]=tn[0,3]
            b[4,0]=tn[1,0]
            b[5,0]=tn[1,1]
            b[6,0]=tn[1,2]
            b[7,0]=tn[1,3]
            b[8,0]=tn[2,0]
            b[9,0]=tn[2,1]
            b[10,0]=tn[2,2]
            b[11,0]=tn[2,3]
            tv1=[t1d,t1,t1]
            tv2=[t2,t2d,t2]
            tv3=[t3,t3,t3d]
            n=0
            while n<3:
                td=self.directKinematics(tv1[n],tv2[n],tv3[n])
                tj=(td-t)/d
                j[0,n]=tj[0,0]
                j[1,n]=tj[0,1]
                j[2,n]=tj[0,2]
                j[3,n]=tj[0,3]
                j[4,n]=tj[1,0]
                j[5,n]=tj[1,1]
                j[6,n]=tj[1,2]
                j[7,n]=tj[1,3]
                j[8,n]=tj[2,0]
                j[9,n]=tj[2,1]
                j[10,n]=tj[2,2]
                j[11,n]=tj[2,3]
                n=n+1
            R=np.linalg.pinv(j)@(-b)
            t1=t1+R[0,0]
            t2=t2+R[1,0]
            t3=t3+R[2,0]
            t1=t1%360
            t2=t2%360
            t3=t3%360
            
            if t1>180:
                t1=-(360-t1)
            if t1<-180:
                t1=360+t1

            if t2>180:
                t2=-(360-t2)
            if t2<-180:
                t2=360+t2
                
            if t3>180:
                t3=-(360-t2)
            if t3<-180:
                t3=360+t2

            t1d=t1+d
            t2d=t2+d
            t3d=t3+d
            tol=0.01
            for i in range(11):
                error+=b[i,0]

            if (abs(b[0,0])<tol and abs(b[1,0])<tol and abs(b[2,0])<tol and abs(b[3,0])<tol and abs(b[4,0])<tol and abs(b[5,0])<tol and abs(b[6,0])<tol and abs(b[7,0])<tol and abs(b[8,0])<tol and abs(b[9,0])<tol and abs(b[10,0])<tol and abs(b[11,0])<tol):
                calc=False
            if con>1000:
                calc=False
                failed=True
                #t1=sem[0]
                #t2=sem[1]
                #t3=sem[2]
        tetas=[t1,t2,t3]
        #print(con)
        #print(tetas)
        return tetas