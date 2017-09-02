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
/* File: kalman.hpp */
/* Sistemas de Vision Computacional Trabajo Final */
/* Fichero basico de cabecera para filtro de Kalman */
/****************************************************************/

#define SIGMA_R 0.5//0.5 // measurement noise cov
#define SIGMA_Q 0.1 //0.1// state noise cov
#define SIGMA_P 0.1 // error cov
#define STATE_SIZE 8
#define MEAS_SIZE 4

//#include "stdafx.h"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;




// State struct
typedef struct kfState8_t
{
	// 2D position
	float x;
	float y;
	// ROI size
	float w;
	float h;
	// 2D velocity
	float vx;
	float vy;
	// size change velocity
	float vw;
	float vh;
} kfState8_t;

// Measurement struct
typedef struct kfMeas4_t
{
	// 2D position
	float x;
	float y;
	// ROI size
	float w;
	float h;
} kfMeas4_t;

typedef struct blobs_t
{
	KalmanFilter kf;
	kfState8_t initState;
	kfMeas4_t measurement;
	unsigned int lifeTime;
	int lostN;
	int ID;
} blobs_t;

KalmanFilter initKalman(const kfState8_t init_state);

//update the KF with the measurement
void updateKalman(KalmanFilter& kalf, kfMeas4_t meas);

//update the KF with the prediction (when we have no measurement)
void updateKalman(KalmanFilter& kalf);

