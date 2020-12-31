#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;
  n_aug_ = 7;
  n_sigma_ = 2*n_aug_ + 1;

  // initial state vector
  x_ = VectorXd(n_x_);
  x_.fill(0.0);

  Xsig_pred_ = MatrixXd(n_x_, n_sigma_);
  Xsig_pred_.fill(0.0);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_ <<      0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
            -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
             0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
            -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
            -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  is_initialized_ = false;
  lambda_ = 3 - n_x_;
  time_us_ = 0.0;

  // Init. weights
  weights_ = VectorXd(n_sigma_);
  double weight = 1/2*(lambda_+n_aug_);

  weights_(0) = lambda_/(lambda_+n_aug_);

  for (int i=1; i<2*n_aug_+1; ++i) {  
    weights_(i) = weight;
  }

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  int n_z_params;
  VectorXd z_;

  // Lidar input (px,py)
  if(meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    n_z_params = 2;
    z_ = VectorXd(n_z_params);
    z_ << meas_package.raw_measurements_[0],
         meas_package.raw_measurements_[1];
  }
  // Radar input (row, phi, row_dot)
  else
  {
    n_z_params = 3;
    z_ = VectorXd(n_z_params);
    z_ << meas_package.raw_measurements_[0],
         meas_package.raw_measurements_[1],
         meas_package.raw_measurements_[2];
  }


  if(!is_initialized_)
  {
    if(meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      x_ << z_(0),
            z_(1),
            2.2049,
            0.5015,
            0.3528; 
    }
    else
    {
      x_ << z_(0)*cos(z_(1)), // px = row * sin(phi)
            z_(0)*sin(z_(1)), // py = row * cos(phi)
            2.2049,
            0.5015,
            0.3528;       
    }
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
    return;
  }

  double delta_t = static_cast<double>((meas_package.timestamp_ - time_us_) * 1e-6);
  this->time_us_ = meas_package.timestamp_;


  Prediction(delta_t);

  // Updating step
  VectorXd z_out;
  MatrixXd S_out;

  if(meas_package.sensor_type_ == MeasurementPackage::LASER)
    PredictLidarMeasurement(z_out, S_out);
  
  else
    PredictRadarMeasurement(z_out, S_out);
  
  // Updating the state x & covariance P based on state prediction 
  // & current measurement
  UpdateState(meas_package,z_out, S_out, n_z_params);

  std::cout << "Step Done\n";
}

void UKF::Prediction(double delta_t){
  
  // Generating sigma points
  GenerateAugSigmaPoints(); 

  _Prediction(delta_t);

  // Predict state mean & covariance
  VectorXd xk_p;
  MatrixXd Pk_p;
  PredictMeanAndCovariance(xk_p, Pk_p);
  this->x_ = xk_p;
  this->P_ = Pk_p;

}

void UKF::GenerateAugSigmaPoints(){

  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_);

  // create augmented mean state
  x_aug.head(n_x_) = this->x_;
  x_aug(5) = this->std_a_;
  x_aug(6) = this->std_yawdd_;

  // create augmented covariance matrix
  MatrixXd Q = MatrixXd(2,2);
  Q << std_a_*std_a_, 0,
       0, std_yawdd_*std_yawdd_;

  P_aug.fill(0.0); // Filling the 7*7 matrix by zeros
  P_aug.topLeftCorner(n_x_, n_x_) = this->P_;
  P_aug.bottomRightCorner(n_aug_ - n_x_, n_aug_ - n_x_) = Q;

  // create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;

  double eq_const = sqrt(lambda_ + n_aug_);

  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i+1) = x_aug + (eq_const * A.col(i));
    Xsig_aug.col(i+n_aug_+1) = x_aug - (eq_const * A.col(i));
  }  
  
  generatedSigmaPoints = Xsig_aug;
}

void UKF::_Prediction(double delta_t) {
  
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  double delta_t_squared = delta_t * delta_t;  

  for (size_t i = 0; i < n_sigma_; i++)
  {
    VectorXd f(n_x_);
    VectorXd nu(n_x_);

    double p_x = generatedSigmaPoints(0,i);
    double p_y = generatedSigmaPoints(1,i);
    double v_k = generatedSigmaPoints(2,i);
    double yaw_k = generatedSigmaPoints(3,i);
    double yaw_k_dot = generatedSigmaPoints(4,i);

    double acc_noise = generatedSigmaPoints(5,i);
    double yaw_dot_dot_noise = generatedSigmaPoints(6,i);

    nu(0) = 0.5*delta_t_squared*cos(yaw_k)*acc_noise;
    nu(1) = 0.5*delta_t_squared*sin(yaw_k)*acc_noise;
    nu(2) = delta_t*acc_noise;
    nu(3) = 0.5*delta_t_squared*yaw_dot_dot_noise;
    nu(4) = delta_t*yaw_dot_dot_noise;

    // Checking if the yaw rate is equal to zero or not to avoid division by zero
    if (fabs(yaw_k_dot) > 0.001) 
    {
      f(0) = v_k/yaw_k_dot * ( sin (yaw_k + yaw_k_dot*delta_t) - sin(yaw_k));
      f(1) = v_k/yaw_k_dot * ( - cos(yaw_k + yaw_k_dot*delta_t) + cos(yaw_k));
    } 
    else 
    {
      f(0) = v_k*cos(yaw_k)*delta_t;
      f(1) = v_k*sin(yaw_k)*delta_t;
    }

    f(2) = 0.0;
    f(3) = yaw_k_dot*delta_t;
    f(4) = 0.0;

    Xsig_pred_.col(i) = generatedSigmaPoints.col(i).head(n_x_) + f + nu;
  }

}

void UKF::PredictMeanAndCovariance(VectorXd &x_out, MatrixXd &P_out){

  // create vector for predicted state
  VectorXd x = VectorXd(n_x_);
  x.fill(0.0);
  // create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);

  // predict state mean
  for (size_t i = 0; i < n_sigma_; i++)
    x += weights_(i) * Xsig_pred_.col(i);

  // predict state covariance matrix
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x;

    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P = P + weights_(i) * x_diff * x_diff.transpose() ;
  }

  x_out = x;
  P_out = P;
}

void UKF::PredictLidarMeasurement(VectorXd &z_out, MatrixXd &S_out) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
   int n_z = 2;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sigma_);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_*std_laspx_, 0,
       0, std_laspy_*std_laspy_;

  // transform each sigma point into measurement space
  for (size_t i = 0; i < n_sigma_; i++)
  {
    double p_x = this->Xsig_pred_(0,i);
    double p_y = this->Xsig_pred_(1,i);

    // Assuming w_k = zero
    Zsig(0,i) = p_x; 
    Zsig(1,i) = p_y;
  }
  
  this->Zsig_ = Zsig;

  // calculate mean predicted measurement
  for (size_t i = 0; i < n_sigma_; i++)
    z_pred += weights_(i) *  Zsig.col(i);
  
  // calculate innovation covariance matrix S
  for (int i = 0; i < n_sigma_; ++i) {  // iterate over sigma points

    // state difference
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S += weights_(i) * z_diff * z_diff.transpose() ;
  }
  S = S + R;
  
  z_out = z_pred;
  S_out = S;
}

void UKF::PredictRadarMeasurement(VectorXd &z_out, MatrixXd &S_out) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  int n_z = 3;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sigma_);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_*std_radr_, 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;
  /**
   * Student part begin
   */

  // transform each sigma point into measurement space
  for (size_t i = 0; i < n_sigma_; i++)
  {
    double p_x = this->Xsig_pred_(0,i);
    double p_y = this->Xsig_pred_(1,i);
    double v_k = this->Xsig_pred_(2,i);
    double yaw_k = this->Xsig_pred_(3,i);
    double yaw_k_dot = this->Xsig_pred_(4,i);

    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y) + 0.0; // Assuming w_k is zero
    Zsig(1,i) = atan2(p_y,p_x) + 0.0;
    Zsig(2,i) = ( ( (p_x*cos(yaw_k)*v_k) + (p_y*sin(yaw_k)*v_k) ) / (sqrt(p_x*p_x + p_y*p_y)) ) + 0.0;
  }

  this->Zsig_ = Zsig;

  // calculate mean predicted measurement
  for (size_t i = 0; i < n_sigma_; i++)
    z_pred += weights_(i) *  Zsig.col(i);
  
  // calculate innovation covariance matrix S
  for (int i = 0; i < n_sigma_; ++i) {  // iterate over sigma points

    // state difference
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S += weights_(i) * z_diff * z_diff.transpose() ;
  }

  S = S + R;
  
  z_out = z_pred;
  S_out = S;
}


void UKF::UpdateState(MeasurementPackage &meas_package, const  VectorXd &z_pred, 
                     const MatrixXd &S_pred, const int n_z){

  
  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  VectorXd z_meas = meas_package.raw_measurements_;

  /**
   * Student part begin
   */

  // calculate cross correlation matrix
  for (size_t i = 0; i < n_sigma_; i++)
  { 
    VectorXd x_diff = this->Xsig_pred_.col(i) - x_;

    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    VectorXd z_diff = this->Zsig_.col(i) - z_pred;
    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose() ;
  }
  
  // calculate Kalman gain K;
  MatrixXd K = Tc*S_pred.inverse();
  // update state mean and covariance matrix

  // residuals
  VectorXd z_diff = z_meas - z_pred;

  // angle normalization
  while (z_diff(1) > M_PI) z_diff(1) -= 2.0*M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2.0*M_PI;

  x_ = x_ + K*z_diff;
  P_ = P_ - K*S_pred*K.transpose();

}