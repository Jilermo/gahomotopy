#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

namespace py = pybind11;
using namespace Eigen;

class FastUR3e {
private:
    Matrix4d mb;
    int numSegments;
    std::vector<Vector4d> obstacles;

    // Homotopy Matrices
    MatrixXd a;
    VectorXd a_0;
    VectorXd fStart;
    VectorXd obsRep;
    double Q;
    int numVar;

    // Kinematic Transformation Helpers
    Matrix4d rotaz(double t) {
        Matrix4d R = Matrix4d::Identity();
        double rad = t * M_PI / 180.0;
        R(0,0) = cos(rad); R(0,1) = -sin(rad);
        R(1,0) = sin(rad); R(1,1) = cos(rad);
        return R;
    }

    Matrix4d rotay(double t) {
        Matrix4d R = Matrix4d::Identity();
        double rad = t * M_PI / 180.0;
        R(0,0) = cos(rad); R(0,2) = sin(rad);
        R(2,0) = -sin(rad); R(2,2) = cos(rad);
        return R;
    }

    Matrix4d trasz(double d) { Matrix4d T = Matrix4d::Identity(); T(2,3) = d; return T; }
    Matrix4d trasy(double d) { Matrix4d T = Matrix4d::Identity(); T(1,3) = d; return T; }
    Matrix4d trasx(double d) { Matrix4d T = Matrix4d::Identity(); T(0,3) = d; return T; }

public:
    // Ensures heap allocations align with 16-byte memory vectorization limits
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FastUR3e(int segments = 200) : numSegments(segments), Q(0.0), numVar(0) {
        mb = Matrix4d::Identity();
    }

    void set_obstacles(py::array_t<double> obs_array) {
        auto buf = obs_array.request();
        double* ptr = (double*)buf.ptr;
        int num_obs = buf.shape[0];
        
        obstacles.clear();
        for (int i = 0; i < num_obs; i++) {
            obstacles.push_back(Vector4d(ptr[i*4 + 0], ptr[i*4 + 1], ptr[i*4 + 2], ptr[i*4 + 3] * ptr[i*4 + 3]));
        }
    }

    // Explicitly maps Row-Major NumPy matrix structures safely into Eigen configurations
    void configure_homotopy(py::array_t<double> a_matrix, 
                            py::array_t<double> a0_vector, 
                            py::array_t<double> fStart_vector, 
                            py::array_t<double> obsRep_vector, 
                            double q_val) {
        
        auto buf_a = a_matrix.request();
        int rows_a = buf_a.shape[0];
        int cols_a = buf_a.shape[1];
        // Read explicitly as RowMajor matching NumPy layout
        a = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>((double*)buf_a.ptr, rows_a, cols_a);

        auto buf_a0 = a0_vector.request();
        a_0 = Eigen::Map<Eigen::VectorXd>((double*)buf_a0.ptr, buf_a0.size);

        auto buf_fs = fStart_vector.request();
        fStart = Eigen::Map<Eigen::VectorXd>((double*)buf_fs.ptr, buf_fs.size);

        auto buf_or = obsRep_vector.request();
        obsRep = Eigen::Map<Eigen::VectorXd>((double*)buf_or.ptr, buf_or.size);

        Q = q_val;
        numVar = rows_a;
    }

    // Returns a tuple: (distances_matrix, crash_boolean)
    py::tuple calculate_distance_fast(std::vector<double> ts) {
        Matrix4d t0 = mb;
        Matrix4d t01 = t0 * trasz(152) * rotaz(ts[0]);
        Matrix4d t12 = trasy(-131) * rotay(-ts[1]);
        Matrix4d t23 = trasx(-244) * rotay(-ts[2]);
        Matrix4d t34 = trasy(106);
        Matrix4d t45 = trasx(-213);
        Matrix4d t56 = trasy(-106) * rotay(-ts[3]);
        Matrix4d t67 = trasz(-85) * rotaz(-ts[4]);
        Matrix4d t78 = trasy(-92) * rotay(-ts[5]);

        Matrix4d transforms[9];
        transforms[0] = t0;
        transforms[1] = t01;
        transforms[2] = transforms[1] * t12;
        transforms[3] = transforms[2] * t23;
        transforms[4] = transforms[3] * t34;
        transforms[5] = transforms[4] * t45;
        transforms[6] = transforms[5] * t56;
        transforms[7] = transforms[6] * t67;
        transforms[8] = transforms[7] * t78;

        // Allocate NumPy array for results to return to Python
        int num_obs = obstacles.size();
        py::array_t<double> dist_result({numSegments, num_obs});
        auto buf = dist_result.mutable_unchecked<2>();
        bool crash = false;

        for (int i = 0; i < numSegments; ++i) {
            double k = (double)i / (numSegments - 1);
            double kn = k * 8.0;
            
            int idx = std::min((int)floor(kn), 7);
            double local_k = kn - idx;

            // Interpolate position between joint frames
            Vector3d p1 = transforms[idx].block<3,1>(0,3);
            Vector3d p2 = transforms[idx+1].block<3,1>(0,3);
            Vector3d pos = p1 + local_k * (p2 - p1);

            for (int j = 0; j < num_obs; ++j) {
                double dx = pos(0) - obstacles[j](0);
                double dy = pos(1) - obstacles[j](1);
                double dz = pos(2) - obstacles[j](2);
                
                double dist_sq = (dx*dx + dy*dy + dz*dz) - obstacles[j](3); // (r^2)
                buf(i, j) = dist_sq;
                
                if (dist_sq < 0) crash = true;
            }
        }
        return py::make_tuple(dist_result, crash);
    }

    double compute_W_internal(const VectorXd& ts, bool& crash) {
        Matrix4d t0 = mb;
        Matrix4d t01 = t0 * trasz(152) * rotaz(ts(0));
        Matrix4d t12 = trasy(-131) * rotay(-ts(1));
        Matrix4d t23 = trasx(-244) * rotay(-ts(2));
        Matrix4d t34 = trasy(106);
        Matrix4d t45 = trasx(-213);
        Matrix4d t56 = trasy(-106) * rotay(-ts(3));
        Matrix4d t67 = trasz(-85) * rotaz(-ts(4));
        Matrix4d t78 = trasy(-92) * rotay(-ts(5));

        // Use an explicitly aligned vector container to avoid stack array allocation faults
        std::vector<Matrix4d, Eigen::aligned_allocator<Matrix4d>> transforms = {
            t0, t01, t01*t12, t01*t12*t23, t01*t12*t23*t34, 
            t01*t12*t23*t34*t45, t01*t12*t23*t34*t45*t56, 
            t01*t12*t23*t34*t45*t56*t67, t01*t12*t23*t34*t45*t56*t67*t78
        };

        double w_val = 0.0;
        int num_obs = obstacles.size();
        crash = false;

        for (int i = 0; i < numSegments; ++i) {
            double k = (double)i / (numSegments - 1);
            double kn = k * 8.0;
            int idx = std::min((int)floor(kn), 7);
            double local_k = kn - idx;

            Vector3d p1 = transforms[idx].block<3,1>(0,3);
            Vector3d p2 = transforms[idx+1].block<3,1>(0,3);
            Vector3d pos = p1 + local_k * (p2 - p1);

            for (int j = 0; j < num_obs; ++j) {
                double dx = pos(0) - obstacles[j](0);
                double dy = pos(1) - obstacles[j](1);
                double dz = pos(2) - obstacles[j](2);
                
                double dist_sq = (dx*dx + dy*dy + dz*dz) - obstacles[j](3);
                
                if (dist_sq < 0) crash = true;
                if (std::abs(dist_sq) > 1e-10) {
                    w_val += (1.0 / dist_sq) * obsRep(j);
                }
            }
        }
        return w_val;
    }

    VectorXd compute_H(const VectorXd& pos, double lam, bool& crash) {
        VectorXd H = a_0 + a * pos - (1.0 - lam) * fStart;
        double w_val = compute_W_internal(pos, crash);
        H(numVar - 1) += w_val - Q;
        return H;
    }

    MatrixXd jacobian_nH_cpp(const VectorXd& pos, double lam) {
        double eps = 1e-8;
        MatrixXd J = MatrixXd::Zero(numVar, numVar + 1);
        bool dummy_crash;
        
        for (int var_idx = 0; var_idx < numVar; ++var_idx) {
            VectorXd pos_plus = pos;
            VectorXd pos_minus = pos;
            pos_plus(var_idx) += eps;
            pos_minus(var_idx) -= eps;
            
            J.col(var_idx) = (compute_H(pos_plus, lam, dummy_crash) - compute_H(pos_minus, lam, dummy_crash)) / (2.0 * eps);
        }
        
        J.col(numVar) = (compute_H(pos, lam + eps, dummy_crash) - compute_H(pos, lam - eps, dummy_crash)) / (2.0 * eps);
        return J;
    }

    py::tuple newton_raphson_corrector_cpp(py::array_t<double> predictor_arr, py::array_t<double> sphere_arr, double radius, int max_iter, double tol) {
        auto buf_pred = predictor_arr.request();
        auto buf_sph = sphere_arr.request();
        
        VectorXd predictor_point = Eigen::Map<Eigen::VectorXd>((double*)buf_pred.ptr, buf_pred.size);
        VectorXd sphere_center = Eigen::Map<Eigen::VectorXd>((double*)buf_sph.ptr, buf_sph.size);
        
        VectorXd pos = predictor_point.head(numVar);
        double lam = predictor_point(numVar);
        bool success = false;
        bool violated_crash = false;
        
        for (int i = 0; i < max_iter; ++i) {
            bool current_crash = false;
            VectorXd H = compute_H(pos, lam, current_crash);
            if (current_crash) violated_crash = true;
            
            double sphere_eq = (pos - sphere_center.head(numVar)).squaredNorm() + std::pow(lam - sphere_center(numVar), 2) - std::pow(radius, 2);
            
            VectorXd F(numVar + 1);
            F.head(numVar) = H;
            F(numVar) = sphere_eq;
            
            if (F.norm() < tol) {
                success = true;
                break;
            }
            
            MatrixXd J_full = MatrixXd::Zero(numVar + 1, numVar + 1);
            J_full.topLeftCorner(numVar, numVar + 1) = jacobian_nH_cpp(pos, lam);
            
            for (int j = 0; j < numVar; ++j) {
                J_full(numVar, j) = 2.0 * (pos[j] - sphere_center(j));
            }
            J_full(numVar, numVar) = 2.0 * (lam - sphere_center(numVar));
            
            VectorXd delta = J_full.colPivHouseholderQr().solve(-F);
            
            pos += delta.head(numVar);
            lam += delta(numVar);
        }
        
        VectorXd result(numVar + 1);
        result.head(numVar) = pos;
        result(numVar) = lam;
        
        return py::make_tuple(result, success, violated_crash);
    }
};

PYBIND11_MODULE(ur3e_cpp, m) {
    py::class_<FastUR3e>(m, "FastUR3e")
        .def(py::init<int>(), py::arg("segments") = 200)
        .def("set_obstacles", &FastUR3e::set_obstacles)
        .def("calculate_distance_fast", &FastUR3e::calculate_distance_fast)
        .def("configure_homotopy", &FastUR3e::configure_homotopy)
        .def("newton_raphson_corrector_cpp", &FastUR3e::newton_raphson_corrector_cpp);
}