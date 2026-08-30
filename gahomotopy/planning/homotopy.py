import numpy as np

from collections import deque
import statistics

class HomotopyPathPlanner:
    """
    Implementation of the Homotopy Path Planning Method using Spherical Algorithm
    Based on the paper example in Section VII.A - Three Circular Obstacles
    """

    def __init__(self,radiuses,obstacles,start,goal,obsVal,obsSign,a_matrix,arm):
        self.step=0
        # Start and goal points
        self.start = start  # Point A
        self.goal = np.array(goal)   # Point B

        self.arm=arm

        self.numVar=len(start)

        obs_list = []
        for obs in obstacles:
            x, y, z = obs['center']
            r = obs['radius']
            obs_list.append([x, y, z, r])


        self.a = np.array(a_matrix)

        self.a_0 = self._compute_a0()

        # Path storage
        self.path = []

        self.obstacles=obstacles

        self.centers = np.array([obs['center'] for obs in obstacles])
        self.radii_sq = np.array([obs['radius']**2 for obs in obstacles])

        self.radiuses = radiuses

        #Obstacle repulsion values
        self.obsRepSign=obsSign
        self.obsRepVal=obsVal

        #self.obsRep = np.full(len(obstacles), self.obsRepVal)

        #self.obsRep=np.array(obsRepVal)*np.array(obsRepSign)

        #self.obsRep=self.precomputeObsValues(center_clusters)

        self.obsRep=obsSign*obsVal

        self.maxWVal=20
        self.maxWValViolated=False
        self.maxWValViolatedVal=0

        self.Q=self.Q_value()

        self.cacheFnF0Val()

        self.sg=self.calculate_sg()

        self.lambdas=[]

        max_size = 8
        self.moving_queue = deque(maxlen=max_size)


    def precomputeObsValues(self,center_clusters):
        #multipliers_per_point = self.obsRepSign[center_clusters]
        mapped_values = [self.obsRepSign[label] for label in center_clusters]
        mapped_values_np=np.array(mapped_values)

        #multipliers_per_point = self.obsRepSign[center_clusters]

        #multipliers_reshaped = multipliers_per_point.reshape(-1, 1)

        #flattened = np.concatenate(multipliers_reshaped)
        #X_transformed_vectorized = self.obsRep * flattened
        X_transformed_vectorized = self.obsRep * mapped_values_np

        return X_transformed_vectorized

    def cacheFnF0Val(self):
        n=len(self.start)
        self.fStart = np.zeros(n)
        for eq_idx in range(n):
            # Compute partial derivative using finite differences
                if eq_idx < n-1:  # Linear equations (H1 to Hn-1)
                    self.fStart[eq_idx]=self.fn(self.start, eq_idx)
                else:  # Last equation with obstacles (Hn)
                    self.fStart[eq_idx]=self.ffinal(self.start, eq_idx)


    def calculate_sg(self):
        """
        Calculates the sign corrector 'sg' once at the start point.
        """
        # Get Jacobian at the start point (pos=start, lam=0)
        J_start = self.jacobian_nH(np.array(self.start), 0.0)

        # Extract nxn spatial part
        J_spatial_start = J_start[:, :self.numVar]

        # Calculate determinant
        det_start = np.linalg.det(J_spatial_start)

        # Calculate the natural sign term C
        C = ((-1)**(self.numVar + 1)) * det_start

        # sg is the sign of C
        sg = np.sign(C)

        if sg == 0:
            raise ValueError("Jacobian is singular at the start point. Cannot proceed.")

        return sg


    def _compute_a0(self):
        """
        Pre-compute a_0,k for all k equations
        a_0,k = -sum(a_j,k * w_goal,j)
        """
        a_0 = []
        for k in range(len(self.a)):
            a_k = np.array(self.a[k])
            a_0_k = -np.dot(a_k, self.goal)
            a_0.append(a_0_k)

        return np.array(a_0)



    def obstacle_function(self, pos, obs):
        """
        Calculate the obstacle function for a circular obstacle
        Equation (39): Obi(x,y) = (x - xi)^2 + (y - yi)^2 - ri^2
        """
        #cx, cy = obs['center']
        r = obs['radius']

        res=0
        for i in range(len(obs['center'])):
            res+=(pos[i] - obs['center'][i])**2
        res-=r**2
        return  res

    def obstacle_function_np(self, pos):
        """
        Calculate the obstacle function for a circular obstacle
        Equation (39): Obi(x,y) = (x - xi)^2 + (y - yi)^2 - ri^2
        """
        diffs = self.centers - pos  # Or pos_np - all_centers_np, squaring removes sign

        # 2. Square all differences
        #    sq_diffs is still (N, 3)
        sq_diffs = diffs**2

        # 3. Sum along the coordinate axis (axis=1)
        #    This performs (dx^2 + dy^2 + dz^2) for each obstacle.
        #    sq_distances is now (N,)
        sq_distances = np.sum(sq_diffs, axis=1)

        # 4. Subtract all squared radii
        #    Returns the final (N,) result array
        return sq_distances - self.radii_sq


    def W(self, pos):
        """
        Calculate W(x,y) - the combined repulsion field from all obstacles
        Equation (42): W(x,y) = sum(pi / Obi(x,y))
        """

        w_val = 0
        counter=0
        for obs in self.obstacles:
            obs_val = self.obstacle_function(pos,obs)
            if abs(obs_val) > 1e-10:  # Avoid division by zero
                #w_val += obs['p'] / obs_val
                #w_val += self.obsRepVal*self.obsRepSign[counter] / obs_val
                w_val += self.obsRepVal*self.obsRepSign[counter] / obs_val
            counter+=1
        return w_val

    def W_np(self, ang):
        """
        Calculate W(x,y) - the combined repulsion field from all obstacles
        Equation (42): W(x,y) = sum(pi / Obi(x,y))
        """

        #start_time = time.time()
        #obsValues=self.arm.CalculateDistanceToObstacles(pos[0],pos[1],pos[2])

        obsValues,crash=self.arm.CalculateDistanceToObstaclesFast(ang)

        #obsValuesSC,crashSC=self.arm.CalculateSelfCollisionFast(ang, link_radius=40.0, min_seg_dist=2)

        #end_time = time.time()

        # Calculate and print the duration
        #elapsed_time = end_time - start_time
        #print(f"Distance to obstacles calculated in {elapsed_time:.4f} seconds")


        #obsValues=1/obsValues

        #w_val=np.sum(obsValues*self.obsRep)

        #obsValues=obsValues/2

        obsValues=1/obsValues

        w_val=np.sum(obsValues*self.obsRep)

        #obsValuesSC=1/obsValuesSC
        #w_val=w_val+np.sum(obsValues*self.obsRep)



        # NOTE: The maxWValViolated check is intentionally commented out to match
        # the old (working) codebase. The old SphericalAlgorithmMulti.py had
        # these lines commented out, allowing the homotopy to continue tracking
        # even when the total repulsion field W exceeds maxWVal. Re-enabling this
        # check causes the GA to fail on most candidate paths because W easily
        # exceeds 20 when obstacle values range from 100 to 100000.
        #if w_val>self.maxWVal:
        #    self.maxWValViolated=True
        #    self.maxWValViolatedVal=w_val

        #if(crash or crashSC):
        if(crash):
            #self.maxWValViolated=True
            self.maxWValViolatedVal=w_val

        return w_val

    def Q_value(self):
        """
        Calculate Q = W(1,1) at the goal point
        Equation (43)
        """
        #res1=self.W_np(self.goal)
        #res2=self.W(self.goal)
        return self.W_np(self.goal)

    def ln(self, pos, k):
        """
        k-th linear equation in n-dimensional space
        """
        # Note: pos should be just spatial coordinates here

        a_k = np.array(self.a[k])
        return self.a_0[k] + np.dot(a_k, pos)

    def fn(self, pos, k):
        """
        k-th equation without obstacles
        """
        return self.ln(pos, k)

    def ffinal(self, pos, k):
        """
        Last equation with obstacle singularities
        Only used for the n-th equation
        """
        #return self.ln(pos, k) + self.W_np(pos) - self.Q_value()
        return self.ln(pos, k) + self.W_np(pos) - self.Q

    def Hn(self, pos, lam, k):
        """
        k-th homotopy equation (for linear equations)
        H_k = f_k(pos) - (1-λ)f_k(start_pos) = 0
        """
        return self.fn(pos, k) - (1 - lam) * self.fStart[k]

    def Hfinal(self, pos, lam, k):
        """
        Final homotopy equation (with obstacles)
        H_n = f_final(pos) - (1-λ)f_final(start_pos) = 0
        """
        return self.ffinal(pos, k) - (1 - lam) * self.fStart[k]


    def jacobian_nH(self, pos, lam):
        """
        Calculate the Jacobian matrix for n-dimensional homotopy system

        Structure for n spatial dimensions:
        - We have n equations (n-1 linear + 1 with obstacles)
        - Each equation has n+1 variables (n spatial + lambda)
        - Jacobian is n × (n+1) matrix:

        [∂H1/∂x1  ∂H1/∂x2  ...  ∂H1/∂xn  ∂H1/∂λ]
        [∂H2/∂x1  ∂H2/∂x2  ...  ∂H2/∂xn  ∂H2/∂λ]
        [  ...      ...    ...    ...      ...  ]
        [∂Hn/∂x1  ∂Hn/∂x2  ...  ∂Hn/∂xn  ∂Hn/∂λ]
        """
        eps = 1e-8
        n = len(pos)  # Number of spatial dimensions

        # Initialize Jacobian matrix (n equations × n+1 variables)
        J = np.zeros((n, n+1))

        # For each equation (row of Jacobian)
        for eq_idx in range(n):

            # Partial derivatives w.r.t. spatial variables (columns 0 to n-1)
            for var_idx in range(n):
                # Create perturbed copies
                pos_plus = pos.copy()
                pos_minus = pos.copy()
                pos_plus[var_idx] += eps
                pos_minus[var_idx] -= eps

                # Compute partial derivative using finite differences
                if eq_idx < n-1:  # Linear equations (H1 to Hn-1)
                    h_plus = self.Hn(pos_plus, lam, eq_idx)
                    h_minus = self.Hn(pos_minus, lam, eq_idx)
                else:  # Last equation with obstacles (Hn)
                    h_plus = self.Hfinal(pos_plus, lam, eq_idx)
                    h_minus = self.Hfinal(pos_minus, lam, eq_idx)

                J[eq_idx, var_idx] = (h_plus - h_minus) / (2 * eps)

            # Partial derivative w.r.t. lambda (last column)
            if eq_idx < n-1:  # Linear equations
                h_plus = self.Hn(pos, lam + eps, eq_idx)
                h_minus = self.Hn(pos, lam - eps, eq_idx)
            else:  # Last equation with obstacles
                h_plus = self.Hfinal(pos, lam + eps, eq_idx)
                h_minus = self.Hfinal(pos, lam - eps, eq_idx)

            J[eq_idx, n] = (h_plus - h_minus) / (2 * eps)

        return J

    def sphere_equation(self, pos, lam, center, radius):
        """
        Sphere equation for tracking
        S = (x-cx)^2 + (y-cy)^2 + (λ-cλ)^2 - r^2 = 0
        """
        #cx, cy, clam = center
        res=0
        for i in range(len(pos)):
            res+=(pos[i] - center[i])**2
        res+=(lam - center[-1])**2
        res-=radius**2
        #(x - cx)**2 + (y - cy)**2 + (lam - clam)**2 - radius**2
        return res

    def euler_predictor_n(self, current_point, sphere_center, radius):
        """
        Euler predictor step for n-dimensional case
        current_point: [x1, x2, ..., xn, lambda] - (n+1) dimensional
        """
        numVar = len(self.start)  # Number of spatial dimensions (n)
        pos = current_point[:-1]  # Spatial coordinates [x1, ..., xn]
        lam = current_point[-1]   # Lambda parameter

        # Calculate Jacobian - should be numEq x (numVar+1)
        J = self.jacobian_nH(pos, lam)

        # Extract nxn matrix for partial derivatives w.r.t spatial variables
        J_spatial = J[:, :numVar]  # This is correct - nxn matrix

        # Calculate determinant for sg parameter
        det = np.linalg.det(J_spatial)

        # Determine sign (you may need to adapt this logic for n-dim)
        #sg = -1 if self.m1 > self.m2 else 1


        # Calculate ∂λ/∂ρ - generalized for n dimensions
        # The exponent should be (n+1) not (2+1)
        dlam_drho = self.sg * (-1)**(numVar+1) * det

        # Calculate tangent vector components
        # The lambda column is at index numVar (or -1)
        b = -J[:, -1] * dlam_drho  # Changed from J[:, 2]

        try:
            # Solve for spatial components of tangent vector
            tangent_spatial = np.linalg.solve(J_spatial, b)
        except:
            # If singular, use small perturbation for ALL n dimensions
            tangent_spatial = np.ones(numVar) * 0.1

        # Construct full tangent vector [dx1/dρ, dx2/dρ, ..., dxn/dρ, dλ/dρ]
        tangent = np.append(tangent_spatial, dlam_drho)

        # Normalize tangent vector
        tangent = tangent / np.linalg.norm(tangent)

        # Predictor point
        predictor = sphere_center + radius * tangent


        return predictor

    def newton_raphson_corrector_n(self, predictor_point, sphere_center, radius, max_iter=40, tol=1e-6):
        """
        Newton-Raphson corrector to find actual point on homotopy curve
        Based on Section V.B and equation (25)
        Extended to n dimensions
        """
        numVar = len(self.start)
        pos = predictor_point[:numVar].copy()  # Make a copy to avoid modifying original
        lam = predictor_point[-1]

        for i in range(max_iter):
            # System of equations: H1=0, H2=0, ..., Hn=0, S=0
            F_list = []

            # Add each homotopy equation H_k
            for k in range(numVar):
                if k < numVar - 1:
                    H_n = self.Hn(pos, lam, k)
                else:
                    H_n = self.Hfinal(pos, lam, k)
                F_list.append(H_n)

            # Add sphere equation
            sphere_eq = self.sphere_equation(pos, lam, sphere_center, radius)
            F_list.append(sphere_eq)

            F = np.array(F_list)

            # Check convergence
            if np.linalg.norm(F) < tol:
                return np.array([*pos, lam]), True

            # Calculate full Jacobian including sphere equation
            # Jacobian size: (numVar + 1) x (numVar + 1)
            J_full = np.zeros((numVar + 1, numVar + 1))

            # Jacobian of H1, H2, ..., Hn
            J_H = self.jacobian_nH(pos, lam)
            J_full[:numVar, :] = J_H

            # Jacobian of sphere equation
            # Last row: derivatives with respect to all position variables and lambda
            for j in range(numVar):
                J_full[numVar, j] = 2 * (pos[j] - sphere_center[j])
            J_full[numVar, numVar] = 2 * (lam - sphere_center[-1])

            # Newton-Raphson update
            try:
                delta = np.linalg.solve(J_full, -F)
                pos += delta[:numVar]
                lam += delta[numVar]
            except np.linalg.LinAlgError:
                return np.array([*pos, lam]), False

        return np.array([*pos, lam]), False



    def track_path_multi(self, max_steps=1000):
        """
        Main tracking algorithm using spherical continuation
        """
        # Initialize at start point
        #current = np.array([self.start[0], self.start[1], 0.0])
        current = np.array([*self.start, 0.0])


        self.path = [current.copy()]

        failed=True

        for step in range(max_steps-1):
            #print("step "+str(step))
            self.step=step
            # Current sphere center
            sphere_center = current.copy()

            # Predictor step
            predictor = self.euler_predictor_n(current, sphere_center, self.radiuses)

            # Corrector step
            corrected, success = self.newton_raphson_corrector_n(predictor, sphere_center, self.radiuses)

            if self.maxWValViolated:
                print("max w violated at "+str(self.step))
                print(f"Max W Val Violated by "+ str(self.maxWValViolatedVal))
                print("Path Planning FAILED")
                failed=True
                break

            if not success:
                print(f"Newton-Raphson failed at step {step}")
                failed=True
                break

            # Update current point
            current = corrected
            self.path.append(current.copy())

            p1=self.path[-1]
            p1 = np.array(p1[:-1])
            p2 = np.array(self.goal)
            matr1=self.arm.directKinematics(p1)
            matr2=self.arm.directKinematics(p2)
            pos1=np.array([matr1[0,3],matr1[1,3],matr1[2,3]])
            pos2=np.array([matr2[0,3],matr2[1,3],matr2[2,3]])
            distance_to_goal = np.sum(np.abs(((pos1 - pos2)**2)))

            if step%500==0:

                distance_to_goal2=np.sqrt(distance_to_goal)
                print(f"Progress distance: {distance_to_goal2}")

                stdDevDistances=self.add_to_queue(distance_to_goal2)

                if stdDevDistances<0.2:
                    print(f"distances Standard Deviation {stdDevDistances}")
                    print("Algorithm stuck at "+str(self.step))
                    print("Path Planning FAILED")
                    failed=True
                    break


            # Check if reached goal (λ ≈ 1)

            if current[self.numVar] >= 0.99 and distance_to_goal<3:
            #if current[self.numVar] >= 0.99:
                failed=False
                print(f"Reached goal at step {step}")
                break

            # Keep constant radius
            self.lambdas.append(current[self.numVar-1])



        total_distance=10000
        if not failed:
            print("Algorithm SUCCEDED")
            print("Path completed on ", len(self.path), " readiuses")
            p1=self.path[-1]
            p1 = np.array(p1[:-1])
            p2 = np.array(self.goal)
            total_distance = np.sum((p1 - p2)**2)

            total_distance=self.getPathDistance(self.path)
            y1=1/100
            y2=1/100000
            total_distance=(total_distance+(self.step*y1))*y2
        else:
            print("Algorithm FAILED")
            p1=self.path[-1]
            _lambda=p1[-1]
            p1 = np.array(p1[:-1])
            p2 = np.array(self.goal)
            matr1=self.arm.directKinematics(p1)
            matr2=self.arm.directKinematics(p2)
            pos1=np.array([matr1[0,3],matr1[1,3],matr1[2,3]])
            pos2=np.array([matr2[0,3],matr2[1,3],matr2[2,3]])
            distance_to_goal = np.sum(np.abs(((pos1 - pos2)**2)))
            distance_to_goal=np.sqrt(distance_to_goal)
            #total_distance = 1000-(self.step/400)+distance_to_goal

            #total_distance = 1000000+(distance_to_goal/100)-(_lambda*100)
            x1=1/12000
            x2=1/100
            print(f"Distance to goal{distance_to_goal}")
            print(f"Final Lambda{_lambda}")
            total_distance = 10+(distance_to_goal*x2)

        print(f"Evaluation: {total_distance}")


        return np.array(self.path),current[2],failed,total_distance,self.lambdas

    def add_to_queue(self,value):
        self.moving_queue.append(value)

        # Standard deviation requires at least 2 data points
        if len(self.moving_queue) > 6:
            current_stdev = statistics.stdev(self.moving_queue)
            return current_stdev
        else:
            return 100

    def getPathDistance(self,path):
        pathCartesian=np.zeros([len(path),len(path[0])-1],dtype="float64")
        for i in range(len(path)):
            pos=self.arm.directKinematics(path[i])
            pathCartesian[i][0]=pos[0,3]
            pathCartesian[i][1]=pos[1,3]
            pathCartesian[i][2]=pos[2,3]

        diffs = np.diff(pathCartesian, axis=0)
        step_distances = np.sqrt(np.sum(diffs**2, axis=1))
        total_distance = np.sum(step_distances)
        return total_distance
