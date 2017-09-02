/**
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 *
 * Created by Frank Sanabri Macías <fsanm77@uo.edu.cu> on 05/12/2015.
 */

/****************************************************************/
/* File: kalman.cpp */
/* Sistemas de Vision Computacional Trabajo Final *
/* Fichero basico para filtro de Kalman */
/****************************************************************/
//...
 
//#include "stdafx.h"

#include "kalman.hpp"

KalmanFilter initKalman(const kfState8_t init)
{
    KalmanFilter kalf = KalmanFilter( STATE_SIZE, MEAS_SIZE, 0, 5);
	// Transition matrix A has main 4th-upper diagonal set to identity
	setIdentity(kalf.transitionMatrix, Scalar(1));
	//kalf.transitionMatrix.diag(-4) = Scalar(1);
	//if you need something more complex, you can use
    float A[] = {1,0,0,0,1,0,0,0,
	   0,1,0,0,0,1,0,0,
	   0,0,1,0,0,0,1,0,
	   0,0,0,1,0,0,0,1,
	   0,0,0,0,1,0,0,0,
	   0,0,0,0,0,1,0,0,
	   0,0,0,0,0,0,1,0,
	   0,0,0,0,0,0,0,1};

	kalf.transitionMatrix = Mat(STATE_SIZE, STATE_SIZE, CV_32F, A); /**/
	kalf.transitionMatrix = kalf.transitionMatrix.clone();

	//Measurement matrix H is the identity
	setIdentity(kalf.measurementMatrix, Scalar(1));

	//Process noise covariance matrix is diagonal with values SIGMA_Q
	setIdentity(kalf.processNoiseCov, Scalar(SIGMA_Q));

	//Measurement noise covariance matrix is diagonal, with values SIGMA_R
	setIdentity(kalf.measurementNoiseCov, Scalar(SIGMA_R));

	//Error covariance matrix a posteriori
	setIdentity(kalf.errorCovPost, Scalar(SIGMA_P));

	//Init the filter state. use copyTo()! If you use '=', both
	//variables reference the same data in memory.
	Mat st = (Mat_<float>(8,1) << init.x, init.y, init.w, init.h,
	init.vx, init.vy, init.vw, init.vh);
	st.copyTo(kalf.statePost);
	return kalf;
}

void updateKalman(KalmanFilter& kalf)
{
	Mat pred = kalf.predict();
	Mat measure(4,1,CV_32F);

	measure.at<float>(0,0) = pred.at<float>(0,0);
	measure.at<float>(1,0) = pred.at<float>(1,0);
	measure.at<float>(2,0) = pred.at<float>(2,0);
	measure.at<float>(3,0) = pred.at<float>(3,0);

	kalf.correct(measure);
	return;
}

void updateKalman(KalmanFilter& kalf, kfMeas4_t measurement)
{
	Mat meas = kalf.predict();//?????
	Mat measure(4,1,CV_32F);

	//update the values with our measurement
	measure.at<float>(0,0) = measurement.x;
	measure.at<float>(1,0) = measurement.y;
	measure.at<float>(2,0) = measurement.w;
	measure.at<float>(3,0) = measurement.h;
	kalf.correct(measure);
	return;
}

