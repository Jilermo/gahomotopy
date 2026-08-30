import numpy as np

from gahomotopy.kinematics.base_robot import BaseRobot


class UR3E(BaseRobot):
    """6-DOF UR3e robotic arm."""

    ga_ranges = {
        'radius': {'low': 0.05, 'high': 0.3},
        'obstacle_value': {'low': 100, 'high': 100000},
        'off_diagonal': 100,
        'diagonal': 100,
    }

    def __init__(self):
        super().__init__()          
        self.mb=np.eye(4)
        self.obstacles={}
        self.numSegments=200

    def normalizeAngles(self,ang):
        for i in range(len(ang)):
            if(ang[i]<0):
                ang[i]=360+ang[i]

        return ang
        
    def XYToAngles(self,x,y):        
        pos=self.mb@self.trasx(x)@self.trasy(y)

        ang=self.M2newton(pos,[0,0,0,0,0,0])

        ang=self.normalizeAngles(ang)
            
        return ang

    def invKinematics(self,x,y,z,sem):        
        pos=np.eye(4)@self.trasx(x)@self.trasy(y)@self.trasz(z)
        ang=self.inverseKinematics(pos,sem) 
            
        return ang    
    

    def CalculateDistanceToObstaclesFast(self,ang): 
        numObs=len(self.obstacles)
        obsRet=np.zeros((self.numSegments,numObs))
        obs=False              
        
                                     
        pos=self.getMidPosFast(ang)

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
    
    
    def CalculateSelfCollisionFast(self, ang, link_radius=50.0, min_seg_dist=2): 
        """
        Checks for self-collision by calculating the distance between all points 
        in the robotic arm, ignoring adjacent segments.
        
        link_radius: The physical radius/thickness of the robotic arm links.
        min_seg_dist: Minimum segment index difference to check. 
                      2 means segment 0 is checked vs 2, but not vs 1.
        """
        pos = self.getMidPosFast(ang)

        xk = pos[:, 0, 3]  # Array of all X coordinates
        yk = pos[:, 1, 3]  # Array of all Y coordinates
        zk = pos[:, 2, 3]  # Array of all Z coordinates
        
        # 1. Calculate ALL-TO-ALL distances using broadcasting
        # This creates an (numSegments, numSegments) matrix of distances
        dx = xk[:, np.newaxis] - xk[np.newaxis, :]
        dy = yk[:, np.newaxis] - yk[np.newaxis, :]
        dz = zk[:, np.newaxis] - zk[np.newaxis, :]

        dist_sq = dx**2 + dy**2 + dz**2

        # 2. Determine which kinematic segment (0 to 7) each point belongs to
        k = np.linspace(0, 1, self.numSegments)
        kn = k * 9
        # np.floor maps kn to integer segment IDs, clip ensures the last point at k=1 stays in segment 7
        segment_ids = np.clip(np.floor(kn), 0, 8).astype(int)

        # 3. Create an exclusion mask
        # We only check collision if the difference between segment IDs is >= min_seg_dist (e.g., 2).
        # We use (col - row) to only check the upper triangle of the matrix, avoiding double counting (i vs j and j vs i).
        valid_mask = (segment_ids[np.newaxis, :] - segment_ids[:, np.newaxis]) >= min_seg_dist

        # 4. Check for crashes
        # A collision happens if the distance squared is less than (2 * link_radius)^2
        collision_dist_sq = (2 * link_radius)**2

        # Apply the mask: set the distance of adjacent/same segments to infinity so they never trigger a crash
        dist_sq_masked = np.where(valid_mask, dist_sq, np.inf)

        dist_sq_masked_zero = np.where(valid_mask, dist_sq, 0)

        crashM = dist_sq_masked < collision_dist_sq
        crash = np.sum(crashM) > 0

        # Returning the actual distances (sqrt) and the crash boolean
        return np.sqrt(dist_sq_masked_zero), crash


    def getMidPosFast(self,ts,mb=np.eye(4)):
        #k = np.linspace(0, (self.numSegments - 1) / self.numSegments, self.numSegments)
        k = np.linspace(0, 1, self.numSegments)
        kn = k * 8
        
        # 2. Aplicar límites
        k1 = np.clip(kn, 0, 1)
        k2 = np.clip(kn - 1, 0, 1)
        k3 = np.clip(kn - 2, 0, 1)
        k4 = np.clip(kn - 3, 0, 1)
        k5 = np.clip(kn - 4, 0, 1)
        k6 = np.clip(kn - 5, 0, 1)
        k7 = np.clip(kn - 6, 0, 1)
        k8 = np.clip(kn - 7, 0, 1)
        #k9 = np.clip(kn - 8, 0, 1)


        t0=np.eye(4)@self.mb
        t01=t0@self.traszM(152*k1)@self.rotazM(ts[0])
        t12=self.trasyM(-131*k2)@self.rotayM(-ts[1])
        t23=self.trasxM(-244*k3)@self.rotayM(-ts[2])
        t34=self.trasyM(106*k4)
        t45=self.trasxM(-213*k5)
        t56=self.trasyM(-106*k6)@self.rotayM(-ts[3])
        t67=self.traszM(-85*k7)@self.rotazM(-ts[4])
        t78=self.trasyM(-92*k8)@self.rotayM(-ts[5])

        #On robot tool
        #t89=self.trasyM(-228.6*k9)


        
        t02=t01@t12
        t03=t02@t23
        t04=t03@t34
        t05=t04@t45
        t06=t05@t56
        t07=t06@t67
        t08=t07@t78
        #t09=t08@t89
        
        return t08
        
                

    #def setObstaclesMatrix(self,obstaclesx):
    #    self.obstacles=[]
    #    for obs in obstaclesx:
    #        obsx=self.mb[0,3]+obs['center'][0]
    #        obsy=self.mb[1,3]+obs['center'][1]
    #        obsz=self.mb[2,3]+obs['center'][2]
    #        self.obstacles.append({'center': (obsx, obsy, obsz), 'radius': obs['radius']})
            
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


    
    def setObstaclesDic(self,obstaclesx):
        self.obstacles=obstaclesx
            
    def setNumSegments(self,numSegments):
        self.numSegments=numSegments
        

    #Obtiene la cinematica directa para el brazo (dados los angulos la matriz espacial del efector final)

    def directKinematics(self,ts,mb=np.eye(4)):

        t0=np.eye(4)@self.mb
        t01=t0@self.trasz(152)@self.rotaz(ts[0])
        t12=self.trasy(-131)@self.rotay(-ts[1])
        t23=self.trasx(-244)@self.rotay(-ts[2])
        t34=self.trasy(106)
        t45=self.trasx(-213)
        t56=self.trasy(-106)@self.rotay(-ts[3])
        t67=self.trasz(-85)@self.rotaz(-ts[4])
        t78=self.trasy(-92)@self.rotay(-ts[5])

        #On robot tool
        #t89=self.trasy(-228.6)
        
        t02=t01@t12
        t03=t02@t23
        t04=t03@t34
        t05=t04@t45
        t06=t05@t56
        t07=t06@t67
        t08=t07@t78
        #t09=t08@t89
        
        return t08
    
    
    def compute_jacobian(self, angles, epsilon=1e-6):        
        current_pos = self.directKinematics(angles)
        J = np.zeros((6, 6))
        
        for i in range(6):
            perturbed_angles = list(angles)
            perturbed_angles[i] += epsilon
            new_pos = self.directKinematics(*perturbed_angles)
            J[:, i] = (new_pos - current_pos) / epsilon
            
        return J

    def inverseKinematics(self, target, initial_guess=[0.0, 0.0, 0.0,0.0, 0.0, 0.0], max_iter=500, tol=1e-4):
        q = np.array(initial_guess, dtype=float)
        
        for i in range(max_iter):
            current_tf = self.directKinematics(q)
            #current_pos = current_tf[:3, 3]

            
            #target_pos = target[:3, 3]

            # Calculate error
            error = target - current_tf
            
            if np.linalg.norm(error) < tol:
                return q # Success!
            
            # Newton-Raphson Step
            J = self.compute_jacobian(q)
            
            # Use pseudo-inverse to handle singularities (safer than np.linalg.inv)
            dq = np.linalg.pinv(J) @ error
            
            # Apply update
            q += dq
            
        return q # Return best effort
    

if __name__ == "__main__":
    arm=UR3E()

    ang=[-37.78, -119.13, -132.59, 18.03, 92.01,52.17]
    pos=arm.directKinematics(ang)
    print(pos)
    