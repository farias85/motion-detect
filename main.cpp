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

#include <QCoreApplication>

#include <iostream>
#include "kalman.hpp"
#include <fstream>

void printKalmanState(KalmanFilter kf);

#define SAVEOUTPUT 0
#define VISUALFLAG 1
#define NUMFRAMESMODEL 37
#define DIFF_TH 30.0/255
#define AREATH 80.0

#define MINDETECT 4
#define MAXLOST 10

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);


    /**********************************************************
        *  Variables Declarations
        ***********************************************************/
        //video variables
        VideoCapture captura;

        //image variables
        Mat frame,grayframe,bgModel,diffFrame,diffImage,bitImage;

        Mat bgModel2save,diffImage2save,bitImage2save;

        int c,frameCount = 0;

        //contours list & bounding box variables
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );
        RNG rng(12345);

        // kalman filter variables
        vector<KalmanFilter> kfList;
        vector<blobs_t> blobList;


        //kfState8_t initState;

        int numfilters=0,filterCount =0;
        vector<Mat> x_pred;
        vector<int> BB4filter;
        int numNewBBox;
        vector<blobs_t> blobsList;

        // run time evaluation functions
        time_t tstart,tend;

        // other variables
         CvFont font;
         char* filterIDstr="";
         Scalar colorsTrack[10]={Scalar(255,0,0),Scalar(0,255,0),Scalar(0,0,255),Scalar(127,0,0),Scalar(0,127,0),Scalar(0,0,127),Scalar(0,69,255),Scalar(211,0,148),Scalar(34,139,34),Scalar(127,0,127)};


        /**********************************************************
        *  reading  AVI file and initializing images
        ***********************************************************/
        //captura = VideoCapture("rtsp://10.30.72.197:554/ISAPI/streaming/channels/2?auth=YWRtaW46SHVhV2VpMTIz");
        //captura = VideoCapture("prueba.avi");
        captura = VideoCapture("Walk30000.avi");
        if ( !captura.isOpened() )
        {
            cerr << "No se ha podido abrir el fichero " << endl //argv[1]<<endl
                <<"El fichero no existe o el codec no es adecuado"<<endl;
            return -1;
        }
        captura.get(0);

        // Get the frame rate
        double rate= captura.get(CV_CAP_PROP_FPS);

        // visualizing windows
        namedWindow("SRC");
        //namedWindow("BGMODEL");
        //namedWindow("DIFF");
        //namedWindow("THRE");


         captura >> frame;
        //initializing images
        grayframe = cv::Mat::zeros(frame.rows,frame.cols,CV_32FC1);
        diffImage = cv::Mat::zeros(frame.rows,frame.cols,CV_32FC1);
        frame = cv::Mat::zeros(frame.rows,frame.cols,CV_8U);
        bgModel = Mat::zeros(frame.rows,frame.cols,CV_32FC1);
        Mat frame32 = Mat::zeros(frame.rows,frame.cols,CV_32FC1);
        Mat bImage(frame.size(), CV_MAKETYPE(frame.depth(), 1));

        //output flow inicialization
        ofstream outputResult("results.txt", ios::out);

        // font
        //font = fontQt("Times");
        cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);

        /**********************************************************
        *  frame cicle processing
        ***********************************************************/
        for (;;)
        {
            if ( frameCount == NUMFRAMESMODEL)
            {
                tstart = clock(); // poner como evento del sistema
            }

            // get actual frame
            captura >> frame;
            if (frame.empty()) // end of file
                break;


            /*
            string filename ="C:/temp/PeopleDetection2/PeopleDetection2/images/inputframes/frame_";
            char frameNumberstr[10];
            sprintf(frameNumberstr,"%05d.png",frameCount);
            filename = filename  + frameNumberstr; // frameCount
            imwrite(filename.c_str(), frame);*/

            // Processing here

            cvtColor(frame, grayframe, CV_BGR2GRAY); //Convert frame to gray and store in image
            grayframe.convertTo(frame32,5, 1.0/255.0, 1);//.0/255


            //cration of background Model
            if(frameCount <  NUMFRAMESMODEL)
            {
                accumulate(frame32, bgModel);
                if (frameCount ==  NUMFRAMESMODEL -1)
                {
                    bgModel = bgModel*(1.0/((double)NUMFRAMESMODEL));
                }
            }
            else
            {

               /* if (SAVEOUTPUT  )
                    outputResult << frameCount;

                if ( VISUALFLAG )
                    cout << frameCount  ;
*/

                //************************************************* *
                // apply background subtraction  and umbralization  *
                //************************************************* *
                absdiff(frame32, bgModel, diffImage);

                //threshold image
                threshold(diffImage,bitImage,DIFF_TH,255.0/255.0,CV_THRESH_BINARY);///255.0


                //find connected componet
                bitImage.convertTo(bImage,CV_8U,1,0);
                 //  cv::imshow("Prueba", bImage);
                //findContours(bImage,contours,hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
                findContours(bImage,contours,hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);


                // find bounding box of detected blogs with area grater than 50 pixels
                int numcontours = contours.size();
                int icountBB =0;
                boundRect.clear();

                for (int icontour  = 0; icontour < numcontours ; icontour  ++)
                {

                    double contArea = contourArea(Mat(contours[icontour]) ,0 );
                    if ( contArea > AREATH )
                    {
                        boundRect.push_back( boundingRect( Mat(contours[icontour] )) );

                        //drawing bounding box
                        Scalar color = Scalar( 0,  255 ,  255  );
                        Rect actrect = boundRect[icountBB];
                        rectangle( frame, actrect.tl(), actrect.br(), color, 2, 8, 0 );
                        icountBB++;
                    }
                }
                int numBBox  = icountBB;


                //**************************************************
                //      Asociation step                            *
                //**************************************************
                // prediction of new bounding boxes
                x_pred.clear();
                for (int ifilt = 0 ; ifilt < numfilters; ifilt ++)
                {
                    //printKalmanState(blobsList[ifilt].kf);
                    x_pred.push_back(blobsList[ifilt].kf.predict() );

                    //cout << "(" << x_pred[ifilt].at<float>(0,0) <<"," << x_pred[ifilt].at<float>(1,0)<<","<<x_pred[ifilt].at<float>(2,0)<<","<<x_pred[ifilt].at<float>(3,0)<<")"<<endl;
                }

                // create distance matrix
                //cout << endl;
                Mat dMat(numBBox,numfilters,CV_32F); // need to be cleared
              //  Mat dMat(numBBox,numfilters,CV_32FC1);
                if (( numfilters > 0 ) && ( numBBox > 0 ))
                {
                    for (int iBB = 0; iBB < numBBox ; iBB++)
                    {
                        for (int ifilt = 0 ; ifilt < numfilters; ifilt++)
                        {
                            // distance
                            float dist2  = pow( boundRect[iBB].x - x_pred[ifilt].at<float>(0,0), 2);
                            dist2 += pow( boundRect[iBB].y - x_pred[ifilt].at<float>(1,0), 2);
                            dist2 += pow( boundRect[iBB].width - x_pred[ifilt].at<float>(2,0), 2);
                            dist2 += pow( boundRect[iBB].height - x_pred[ifilt].at<float>(3,0), 2);
                            dist2 = sqrt(dist2);

                            if ( dist2 > 60 )
                                dMat.at<float>(iBB,ifilt) = 100001;
                            else
                                dMat.at<float>(iBB,ifilt) = dist2;
                            //cout<<" "<<dMat.at<float>(iBB,ifilt)<<" ";
                        }
                        //cout << endl;
                    }
                }



               if ( numBBox > numfilters  )
                {
                   //cout << " Start (numBBox > numfilters)"<< endl;
                    //asociate filters with closer bbox
                    for (int ifilt = 0 ; ifilt < numfilters ; ifilt++)
                    {
                        float mindist = 10000;
                        int minpos = -1;
                        for (int iBB  = 0; iBB < numBBox ; iBB++)
                        {
                            if ( dMat.at<float>(iBB,ifilt) < mindist)
                            {
                                mindist = dMat.at<float>(iBB,ifilt);
                                minpos = iBB;
                            }
                        }
                        BB4filter[ifilt] = minpos;
                    }

                    //create new filter for each new BB

                    //Mat newBBs= Mat::ones(numBBox,1,CV_8U);
                    int newBBs[numBBox];
                    for(int d = 0 ; d < numBBox; d++)
                    {
                        newBBs[d] = 1;
                    }

                    for (int ifilt= 0 ; ifilt < numfilters; ifilt++ )
                    {
                        newBBs[BB4filter[ifilt]] = 0;
                    }

                    for(int iBB1 = 0 ;  iBB1 < numBBox ; iBB1++)
                    {
                        int newbb =  newBBs[iBB1];
                        if ( newbb == 1)
                        {
                            kfState8_t initState;
                            initState.x = (float) boundRect[iBB1].x;
                            initState.y = (float) boundRect[iBB1].y;
                            initState.w = (float) boundRect[iBB1].width;
                            initState.h = (float) boundRect[iBB1].height;
                            initState.vx = 0;
                            initState.vy = 0;
                            initState.vh = 0;
                            initState.vw = 0;

                            blobs_t newblobs;
                            newblobs.initState =initState;
                            newblobs.kf = initKalman(initState);
                            newblobs.lifeTime = 1;
                            newblobs.lostN = 0;
                            newblobs.measurement.x = (float) boundRect[iBB1].x;
                            newblobs.measurement.y =
                            newblobs.measurement.w = (float) boundRect[iBB1].y;
                            newblobs.measurement.h = (float) boundRect[iBB1].height;
                            //filterCount = filterCount +1;
                            newblobs.ID = 0;//filterCount;
                            blobsList.push_back(newblobs);
                            //---

                            //printKalmanState(blobsList[numfilters].kf);

                            BB4filter.push_back(iBB1);
                            numfilters = numfilters + 1;

                        }
                    }

                 // cout << " End (numBBox > numfilters)"<< endl;
                  //newBBs.release();
                 // cout << " release"<< endl;
                }
                else
                {
                    if  ( numBBox ==  numfilters )
                    {
                       // cout << " Start (numBBox ==  numfilters )"<< endl;
                        //asociate filters with closer bbox
                        for (int ifilt = 0 ; ifilt < numfilters ; ifilt++)
                        {
                            float mindist = 10000;
                            int minpos = -1;
                            for (int iBB2  = 0; iBB2 < numBBox ; iBB2++)
                            {
                                if (dMat.at<float>(iBB2,ifilt) < mindist )
                                {
                                    mindist = dMat.at<float>(iBB2,ifilt);
                                    minpos = iBB2;
                                }

                            }
                            BB4filter[ifilt] = minpos;
                        }
                       // cout << " End (numBBox ==  numfilters )"<< endl;
                    }
                    else //if  numBBox < numfilters
                    {
                        //cout << " Start (numBBox < numfilters )"<< endl;
                        for (int ifilt = 0 ; ifilt < numfilters ; ifilt++)
                        {
                            float mindist = 10000;
                            int minpos = -1;
                            for (int iBB3  = 0; iBB3 < numBBox ; iBB3++)
                            {
                                if  ( dMat.at<float>(iBB3,ifilt)< mindist )
                                {
                                    mindist = dMat.at<float>(iBB3,ifilt);
                                    minpos = iBB3;
                                }
                            }
                            BB4filter[ifilt] = minpos;
                        }

                        //detect filters bbox duplicated asociation
                        //Mat dupFiltBB = Mat::zeros(1,numfilters,CV_32S);
                        int dupFiltBB[10];

                        for (int ifilt = 0; ifilt < 10; ifilt++)
                            dupFiltBB[ifilt]=0;

                        for(int iBB4  = 0; iBB4 < numBBox ; iBB4++)
                        {
                            int minpos = -1;
                            float mindist = 10000;
                            for (int ifilt = 0 ; ifilt < numfilters ; ifilt++)
                            {
                                if  ( dMat.at<float>(iBB4,ifilt)< mindist )
                                {
                                    mindist = dMat.at<float>(iBB4,ifilt);
                                    minpos = ifilt;
                                }
                            }
                            dupFiltBB[minpos] = 1;
                        }

                        for (int ifilt =0; ifilt< numfilters; ifilt++)
                        {
                            if ( dupFiltBB[ifilt] == 0)
                            {
                                BB4filter[ifilt] = -1;
                            }

                        }

                    // cout << " End (numBBox < numfilters )"<< endl;
                    }
                }

                //**************************************************
                //      kalman filter update                       *
                //**************************************************
                vector<int> del;
                for(int ifilt = 0; ifilt < numfilters; ifilt++)
                {
                    int ind = BB4filter[ifilt];
                    if (ind < 0)
                    {
                       // cout << " Start (ind < 0 )"<< endl;
                        blobsList[ifilt].lostN++;
                        if (blobsList[ifilt].lostN > MAXLOST || blobsList[ifilt].lifeTime < MINDETECT )
                        {
                            del.push_back(ifilt);
                            /*blobsList.erase(blobsList.begin()+ifilt);
                            BB4filter.erase(BB4filter.begin() + ifilt);
                            numfilters = numfilters - 1;*/
                        }
                        else
                        {
                           //  cout << " Start (ind < 0 ) else"<< endl;
                            updateKalman( blobsList[ifilt].kf );

                            //drawning tracked Bounding Boxes
                            Scalar colorRed = Scalar( 255 , 0, 0 );
                            Rect actrect ;

                            actrect.x      = (int) blobsList[ifilt].kf.statePost.at<float>(0);
                            actrect.y      = (int) blobsList[ifilt].kf.statePost.at<float>(1);
                            actrect.width  = (int) blobsList[ifilt].kf.statePost.at<float>(2);
                            actrect.height = (int) blobsList[ifilt].kf.statePost.at<float>(3);
                            int colorId =  (blobsList[ifilt].ID -1)% 10 ;
                            rectangle( frame, actrect.tl(), actrect.br(), colorsTrack[colorId], 2, 8, 0 );
                            //itoa(blobsList[ifilt].ID,filterIDstr,10);
                            //addText( frame, filterIDstr, Point((int)actrect.x,(int)actrect.y), font);

                            Mat resultpos;
                            blobsList[ifilt].kf.statePost.copyTo(resultpos );
                            if ( VISUALFLAG )
                            cout <<" "<<blobsList[ifilt].ID << " " <<resultpos.at<float>(0,0) <<" " << resultpos.at<float>(1,0) <<" " << resultpos.at<float>(2,0) <<" "<< resultpos.at<float>(3,0);
                            if ( SAVEOUTPUT )
                                outputResult << " "<< blobsList[ifilt].ID << " " <<resultpos.at<float>(0,0) <<" " << resultpos.at<float>(1,0) <<" " << resultpos.at<float>(2,0) <<" "<< resultpos.at<float>(3,0);

                           // cout << " End (ind < 0 ) else"<< endl;
                        }
                   //  cout << " End (ind < 0 )"<< endl;
                    }
                    else
                    {
                      //  cout << " Start (ind >= 0 )"<< endl;

                        kfMeas4_t measurement;
                        measurement.x = (float)  boundRect[ind].x;
                        measurement.y = (float)  boundRect[ind].y;
                        measurement.w = (float)  boundRect[ind].width;
                        measurement.h = (float)  boundRect[ind].height;

                        updateKalman( blobsList[ifilt].kf   ,measurement);

                        blobsList[ifilt].lifeTime++;
                        blobsList[ifilt].lostN=0;

                        if ( blobsList[ifilt].lifeTime > MINDETECT )
                        {

                            if ( blobsList[ifilt].ID == 0)   // assign ID for validate filters
                            {
                                filterCount = filterCount +1;
                                blobsList[ifilt].ID = filterCount;
                            }

                            //drawning tracked Bounding Boxes
                            Scalar colorRed = Scalar( 255 , 0, 0 );
                            Rect actrect ;

                            actrect.x      = (int) blobsList[ifilt].kf.statePost.at<float>(0);
                            actrect.y      = (int) blobsList[ifilt].kf.statePost.at<float>(1);
                            actrect.width  = (int) blobsList[ifilt].kf.statePost.at<float>(2);
                            actrect.height = (int) blobsList[ifilt].kf.statePost.at<float>(3);
                            int colorId =  (blobsList[ifilt].ID -1)% 10 ;
                            rectangle( frame, actrect.tl(), actrect.br(), colorsTrack[colorId], 2, 8, 0 );

                            //itoa(blobsList[ifilt].ID,filterIDstr,10);
                            //addText( frame, filterIDstr, Point((int)actrect.x,(int)actrect.y), font);
                            //cvPutText(frame.operator IplImage(),filterIDstr,Point((int)actrect.x,(int)actrect.y),&font,colorRed);
                            //addText(frame,string(filterIDstr),Point((int)actrect.x,(int)actrect.y),font);

                            Mat resultpos;
                            blobsList[ifilt].kf.statePost.copyTo(resultpos );
                            if ( VISUALFLAG )
                            cout <<" "<<blobsList[ifilt].ID << " " <<resultpos.at<float>(0,0) <<" " << resultpos.at<float>(1,0) <<" " << resultpos.at<float>(2,0) <<" "<< resultpos.at<float>(3,0);
                            if ( SAVEOUTPUT )
                                outputResult << " "<< blobsList[ifilt].ID << " " <<resultpos.at<float>(0,0) <<" " << resultpos.at<float>(1,0) <<" " << resultpos.at<float>(2,0) <<" "<< resultpos.at<float>(3,0);


                        }

                       // cout << " End (ind >= 0 )"<< endl;
                      }
                }

                for(int d = del.size() -1 ; d >=0;  d--)
                {
                    blobsList.erase(blobsList.begin() + del[d]);
                    BB4filter.erase(BB4filter.begin() + del[d]);
                    numfilters--;
                }


              /*  if ( VISUALFLAG )
                    cout << endl;
                if ( SAVEOUTPUT )
                    outputResult <<endl;*/

            }

            if ( VISUALFLAG )
            {
                imshow ("SRC",frame);

                if (frameCount > NUMFRAMESMODEL-1)
                {
                    //imshow ("BGMODEL",bImage);
                   // imshow ("DIFF",diffImage);
                    //imshow ("THRE",bitImage);
                }

                c = waitKey(25);
                if ( c == 'q' )
                    break;
                else if ( c == 'p' )
                    waitKey(0);
            }




            if ( SAVEOUTPUT )
            {
                string filename ="C:/Users/frank/Documents/Visual Studio 2008/Projects/PeopleDetection2/PeopleDetection2/images/frames/frame_";
                char frameNumberstr[10];
                sprintf(frameNumberstr,"%05d.png",frameCount);
                filename = filename  + frameNumberstr; // frameCount
                imwrite(filename.c_str(), frame);

                if (frameCount == NUMFRAMESMODEL )
                {
                    Mat bgModel2save,diffImage2save, bitImage2save;
                    filename ="C:/Users/frank/Documents/Visual Studio 2008/Projects/PeopleDetection2/PeopleDetection2/images/bgmodel/frame_";
                    filename = filename  + frameNumberstr; // frameCount
                    bgModel.convertTo(bgModel2save,0,255,0);
                    imwrite(filename.c_str(), bgModel2save);
                }

                if (frameCount > NUMFRAMESMODEL-1)
                {
                    filename ="C:/Users/frank/Documents/Visual Studio 2008/Projects/PeopleDetection2/PeopleDetection2/images/diffIm/frame_";
                    filename = filename  + frameNumberstr; // frameCount
                    diffImage.convertTo(diffImage2save,0,255,0);
                    imwrite(filename.c_str(), diffImage2save);

                    filename ="C:/Users/frank/Documents/Visual Studio 2008/Projects/PeopleDetection2/PeopleDetection2/images/detetIm/frame_";
                    filename = filename  + frameNumberstr; // frameCount
                    bitImage.convertTo(bitImage2save,0,255,0);
                    imwrite(filename.c_str(), bitImage2save);
                }
            }

            frameCount++;

        }

        tend = clock();
        //cout << "runing time: " << (((double)difftime(tend,tstart))*1000)/CLOCKS_PER_SEC<< "ms"<< endl;
        waitKey(0);
        captura.release();
         return a.exec();
    }

    void printKalmanState(KalmanFilter kf)
    {
        Mat Tm = kf.transitionMatrix;
        cout << " Transcition Matrix"<< endl;
        cout << Tm.at<float>(0,0)<<" "<<Tm.at<float>(0,1)<<" "<<Tm.at<float>(0,2)<<" "<<Tm.at<float>(0,3)<<" "<<Tm.at<float>(0,4)<<" "<<Tm.at<float>(0,5)<<" "<<Tm.at<float>(0,6)<<" "<<Tm.at<float>(0,7)<<endl;
        cout << Tm.at<float>(1,0)<<" "<<Tm.at<float>(1,1)<<" "<<Tm.at<float>(1,2)<<" "<<Tm.at<float>(1,3)<<" "<<Tm.at<float>(1,4)<<" "<<Tm.at<float>(1,5)<<" "<<Tm.at<float>(1,6)<<" "<<Tm.at<float>(1,7)<<endl;
        cout << Tm.at<float>(2,0)<<" "<<Tm.at<float>(2,1)<<" "<<Tm.at<float>(2,2)<<" "<<Tm.at<float>(2,3)<<" "<<Tm.at<float>(2,4)<<" "<<Tm.at<float>(2,5)<<" "<<Tm.at<float>(2,6)<<" "<<Tm.at<float>(2,7)<<endl;
        cout << Tm.at<float>(3,0)<<" "<<Tm.at<float>(3,1)<<" "<<Tm.at<float>(3,2)<<" "<<Tm.at<float>(3,3)<<" "<<Tm.at<float>(3,4)<<" "<<Tm.at<float>(3,5)<<" "<<Tm.at<float>(3,6)<<" "<<Tm.at<float>(3,7)<<endl;
        cout << Tm.at<float>(4,0)<<" "<<Tm.at<float>(4,1)<<" "<<Tm.at<float>(4,2)<<" "<<Tm.at<float>(4,3)<<" "<<Tm.at<float>(4,4)<<" "<<Tm.at<float>(4,5)<<" "<<Tm.at<float>(4,6)<<" "<<Tm.at<float>(4,7)<<endl;
        cout << Tm.at<float>(5,0)<<" "<<Tm.at<float>(5,1)<<" "<<Tm.at<float>(5,2)<<" "<<Tm.at<float>(5,3)<<" "<<Tm.at<float>(5,4)<<" "<<Tm.at<float>(5,5)<<" "<<Tm.at<float>(5,6)<<" "<<Tm.at<float>(5,7)<<endl;
        cout << Tm.at<float>(6,0)<<" "<<Tm.at<float>(6,1)<<" "<<Tm.at<float>(6,2)<<" "<<Tm.at<float>(6,3)<<" "<<Tm.at<float>(6,4)<<" "<<Tm.at<float>(6,5)<<" "<<Tm.at<float>(6,6)<<" "<<Tm.at<float>(6,7)<<endl;
        cout << Tm.at<float>(7,0)<<" "<<Tm.at<float>(7,1)<<" "<<Tm.at<float>(7,2)<<" "<<Tm.at<float>(7,3)<<" "<<Tm.at<float>(7,4)<<" "<<Tm.at<float>(7,5)<<" "<<Tm.at<float>(7,6)<<" "<<Tm.at<float>(7,7)<<endl;

        cout << " filter statePre "<< endl;
        cout << kf.statePre.at<float>(0,0)<< endl;
        cout << kf.statePre.at<float>(1,0)<< endl;
        cout << kf.statePre.at<float>(2,0)<< endl;
        cout << kf.statePre.at<float>(3,0)<< endl;
        cout << kf.statePre.at<float>(4,0)<< endl;
        cout << kf.statePre.at<float>(5,0)<< endl;
        cout << kf.statePre.at<float>(6,0)<< endl;
        cout << kf.statePre.at<float>(7,0)<< endl;
        cout << endl;

        cout << " filter statePost "<< endl;
        cout << kf.statePost.at<float>(0,0)<< endl;
        cout << kf.statePost.at<float>(1,0)<< endl;
        cout << kf.statePost.at<float>(2,0)<< endl;
        cout << kf.statePost.at<float>(3,0)<< endl;
        cout << kf.statePost.at<float>(4,0)<< endl;
        cout << kf.statePost.at<float>(5,0)<< endl;
        cout << kf.statePost.at<float>(6,0)<< endl;
        cout << kf.statePost.at<float>(7,0)<< endl;
        cout << endl;


}
