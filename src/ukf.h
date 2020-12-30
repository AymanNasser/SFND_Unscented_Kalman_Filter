#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"

class UKF {
 
 private:
  /**
   * Generates the augmented sigma points
   * @param generated_augmented_sigma_points Are the generated points
   * */
  void GenerateAugSigmaPoints(MatrixXd &generated_augmented_sigma_points);

  /**
   * Predicts sigma points
   * @param delta_t Time between k and k+1 in s
   * @param generatedSigmaPoints The generated sigma points using unscented 
   * transformation with augmentation approach
   */
  void Prediction(MatrixXd &generatedSigmaPoints, double delta_t);

  /**
   * Predicts the state, and the state covariance
   * matrix
   * @param x_out Predicted state mean
   * @param P_out Predicted state covariance matrix
   */
  void PredictMeanAndCovariance(VectorXd &x_out, MatrixXd &P_out);

  /**
   * Transforms the predicted distribution into the measurment space of lidar 
   * @param meas_package The measurement at k+1
   */
  void PredictLidarMeasurement(MeasurementPackage meas_package);

  /**
   * Transforms the predicted distribution into the measurment space of radar
   * @param meas_package The measurement at k+1
   */
  void PredictRadarMeasurement(MeasurementPackage meas_package);
 
 public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // predicted sigma points matrix
  Eigen::MatrixXd Xsig_pred_;

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  // Weights of sigma points
  Eigen::VectorXd weights_;

  // State dimension
  const int n_x_ = 5;

  // Augmented state dimension
  const int n_aug_ = 7;

  const int n_sigma_ = 2*n_aug_ + 1;

  // Sigma point spreading parameter
  double lambda_;
};

#endif  // UKF_H