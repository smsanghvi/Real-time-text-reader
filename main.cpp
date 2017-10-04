/*****************************************************************************
​ * ​ ​ Copyright​ ​ (C)​ ​ 2017​ ​ by​ ​ Snehal Sanghvi
​ *
​ * ​ ​  Users​ ​ are  ​ permitted​ ​ to​ ​ modify​ ​ this​ ​ and​ ​ use​ ​ it​ ​ to​ ​ learn​ ​ about​ ​ the​ ​ field​ ​ of​ ​ embedded
​ * ​ ​ software.​ ​ Snehal Sanghvi​ ​ and​ ​ the​ ​ University​ ​ of​ ​ Colorado​ ​ are​ ​ not​ ​ liable​ ​ for​ ​ any​ ​ misuse​ ​ of​ ​ this​ ​ material.
​ *
*****************************************************************************/
/**
​ * ​ ​ @file​ ​ main.cpp
​ * ​ ​ @brief​ ​ Source file having the implementation of the real-time text detection project.
​​ * ​ ​ @author​ ​ Snehal Sanghvi
​ * ​ ​ @date​ ​ May ​ 12 ​ 2017
​ * ​ ​ @version​ ​ 1.0
​ *   @compiler used to process code: GCC compiler
 *	 @functionality implemented: 
 	 Created a real-time system based on NVIDIA Jetson TX1 board making use of EMmbedded Linux OS.
 		 1> Thread1 : capturing image frames
 		 2> Thread2 : extracting a small area around the fingertip performing image processing manouevres
 		              to extract area around fingertip 
 		 3> Thread3 : running a OCR engine on the segmented out image frame to extract words from it
 		 4> Thread4 : passing the extracted words to a speech sysnthesizer to give an audio output of the pointed word		 
 	 These threads were managed by a real-time Rate MOnotonic scheduler and also makes use of semaphores
 	 for better scheduling.
​ */

#include <iostream>
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <malloc.h>
#include "speak_lib.h"
#include <stdlib.h>
#include <string>
#include <fstream>
#include <sched.h>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <time.h>
#include <syslog.h>
#include <math.h>
#include <sys/param.h>

#define SYSLOG

using namespace std;
using namespace cv;

/********************************************************************
Variable Declarations
********************************************************************/
pthread_t image_capture_thread;
pthread_t fingertip_detect_thread;
pthread_t text_extract_thread;
pthread_t speech_synthesis_thread;
pthread_attr_t image_capture_sched_attr;
pthread_attr_t fingertip_detect_sched_attr;
pthread_attr_t text_extract_sched_attr;
pthread_attr_t speech_synthesis_sched_attr;
pthread_attr_t main_sched_attr;
sem_t image_capture_sem,fingertip_detect_sem,text_extract_sem, speech_synthesis_sem;
int rt_max_prio, rt_min_prio, min;
struct sched_param image_capture_param;
struct sched_param fingertip_detect_param;
struct sched_param text_extract_param;
struct sched_param speech_synthesis_param;
struct sched_param nrt_param;
struct sched_param main_param;
pid_t mainpid;


#define NSEC_PER_SEC (1000000000)
#define DELAY_TICKS (1)
#define ERROR (-1)
#define OK (0)
#define NUM_THREADS (4)
#define NUM_CPUS (4)

VideoCapture cap;
int mark = 0; //indicates if the correct frame has been captured
int flag = 0;

espeak_POSITION_TYPE position_type;
espeak_AUDIO_OUTPUT output;
char *path=NULL;
int Buflength = 500, Options=0;
void* user_data;
t_espeak_callback *SynthCallback;
espeak_PARAMETER Parm;
char Voice[] = {"default"};
string line1;
char text[10000];
unsigned int Size1, position=0, end_position=0, flags=espeakCHARS_AUTO, *unique_identifier;

static struct timespec rtclk_dt_1 = {0, 0};
static struct timespec rtclk_start_time_1 = {0, 0};
static struct timespec rtclk_stop_time_1 = {0, 0};

static struct timespec rtclk_dt_2 = {0, 0};
static struct timespec rtclk_start_time_2 = {0, 0};
static struct timespec rtclk_stop_time_2 = {0, 0};

static struct timespec rtclk_dt_3 = {0, 0};
static struct timespec rtclk_start_time_3 = {0, 0};
static struct timespec rtclk_stop_time_3 = {0, 0};

static struct timespec rtclk_dt_4 = {0, 0};
static struct timespec rtclk_start_time_4 = {0, 0};
static struct timespec rtclk_stop_time_4 = {0, 0};


//int minH = 140, maxH = 165, minS = 70, maxS = 90, minV = 240, maxV = 256; //pink - aagam - morning
//int minH = 150, maxH = 175, minS = 100, maxS = 170, minV = 240, maxV = 256; //pink - works
int minH = 30, maxH = 50, minS = 50, maxS = 200, minV = 100, maxV = 250; //green

cv::Mat frame;
cv::Mat temp;
cv::Mat hsv;
cv::Mat roi; 
cv::Rect small_box;
int counter = 0;
//int count = 0;
int count_save = 0;
int cnt = 0;

char buf[10];
string file;
string filename;
Pix *image;
tesseract::TessBaseAPI *api;  
char *outText;
FILE *fp;
FILE *fp2;
int c;

const char* windowName = "Fingertip detection";

void CallbackFunc(int event, int x, int y, int flags, void* userdata)
{
  cv::Mat RGB = frame(cv::Rect(x, y, 1, 1));
  cv::Mat HSV;
int count = 0;

  cv::cvtColor(RGB, HSV, CV_BGR2HSV);
  cv::Vec3b pixel = HSV.at<cv::Vec3b>(0, 0);
  if (event == cv::EVENT_LBUTTONDBLCLK) // on double left clcik
  {
      std::cout << "Click" << std::endl;
      int h = pixel.val[0];
      int s = pixel.val[1];
      int v = pixel.val[2];
      if (count == 0)
      {
          minH = h;
          maxH = h;
          minS = s;
          maxS = s;
          minV = v;
          maxV = v;
      }
      else
      {
          if (h < minH)
          {
              minH = h;
          }
          else if (h > maxH)
          {
              maxH = h;
          }
          if (s < minS)
          {
              minS = s;
          }
          else if (s > maxS)
          {
              maxS = s;
          }
          if (v < minV)
          {
              minV = v;
          }
          else if (v > maxV)
          {
              maxV = v;
          }
          
      }
      count++;
  }
  std::cout << pixel << std::endl;
}



// Simply prints the current POSIX scheduling policy in effect.
void print_scheduler (void)
{
int schedType;
schedType = sched_getscheduler(getpid());
switch(schedType)
{
case SCHED_FIFO:
printf("Pthread Policy is SCHED_FIFO\n");
break;
case SCHED_OTHER:
printf("Pthread Policy is SCHED_OTHER\n");
break;
case SCHED_RR:
printf("Pthread Policy is SCHED_OTHER\n");
break;
default:
printf("Pthread Policy is UNKNOWN\n");
}
}


//function to compute difference between start and stop times
int delta_t(struct timespec *stop, struct timespec *start, struct timespec *delta_t)
{
  int dt_sec=stop->tv_sec - start->tv_sec;
  int dt_nsec=stop->tv_nsec - start->tv_nsec;

  if(dt_sec >= 0)
  {
    if(dt_nsec >= 0)
    {
      delta_t->tv_sec=dt_sec;
      delta_t->tv_nsec=dt_nsec;
    }
    else
    {
      delta_t->tv_sec=dt_sec-1;
      delta_t->tv_nsec=NSEC_PER_SEC+dt_nsec;
    }
  }
  
  else
  {
    if(dt_nsec >= 0)
    {
      delta_t->tv_sec=dt_sec;
      delta_t->tv_nsec=dt_nsec;
    }
    else
    {
      delta_t->tv_sec=dt_sec-1;
      delta_t->tv_nsec=NSEC_PER_SEC+dt_nsec;
    }
  }

  return(OK);
}



/*
Integer to ASCII (Null terminated string)
*/
char * itoa(char *str, int32_t data){

    // If the number is '0'
    if (data == 0){
        *str++ = '0';
        *str-- = '\0';   //Adding null for end of string and resetting str to initial value
        return str;
    }


    int8_t neg = 0;    // To check if the number is negative
    int rem = 0;     //variable to store remainder
    uint8_t length = 0;         //Calculating length of the string

    //If data is negative
    if (data < 0){
        neg = -1;
        data = -data;
    }

    //Dividing with the base to get the value of data in that base and storing it in the string
    while (data != 0){
        rem = data % 10;
        *str++ = (rem > 9)? (rem-10) + 'A' : rem + '0';             //Ternary for base values greater than 10.
        length++;
        data = data/10;
    }

    // If data is negative adding minus sign
    if(neg == -1){
        *str++ = '-';
        length++;
    }

    *str = '\0'; // Append null character for end of string

    // Reverse the string for final output as the loop gives us the last value first

    uint8_t j=0;         //Initializing counter for the loop
    int8_t temp;
    str = str - length;        //Resetting str to initial value
    for(j=0;j<length/2;j++){ //loop to reverse string
        temp=*(str+j);
        *(str+j) = *(str+length-j-1);
        *(str+length-j-1)=temp;
    }
    return str;
}



float innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1)
{

 float dist1 = std::sqrt(  (px1-cx1)*(px1-cx1) + (py1-cy1)*(py1-cy1) );
 float dist2 = std::sqrt(  (px2-cx1)*(px2-cx1) + (py2-cy1)*(py2-cy1) );

 float Ax, Ay;
 float Bx, By;
 float Cx, Cy;

 //find closest point to C  
 //printf("dist = %lf %lf\n", dist1, dist2);  

 Cx = cx1;
 Cy = cy1;
 if(dist1 < dist2)
 {
  Bx = px1;
  By = py1;
  Ax = px2;
  Ay = py2;


 }else{
  Bx = px2;
  By = py2;
  Ax = px1;
  Ay = py1;
 }


 float Q1 = Cx - Ax;
 float Q2 = Cy - Ay;
 float P1 = Bx - Ax;
 float P2 = By - Ay;


 float A = std::acos( (P1*Q1 + P2*Q2) / ( std::sqrt(P1*P1+P2*P2) * std::sqrt(Q1*Q1+Q2*Q2) ) );

 A = A*180/CV_PI;

 return A;
}




/********************************************************************
Threads
********************************************************************/

void *capture_image(void *threadid)
{  int j; 
    while(1){
	//Block on the image capture semaphore
	sem_wait(&image_capture_sem);	

  	clock_gettime(CLOCK_REALTIME, &rtclk_start_time_1);

	for(j = 0; j < 20; j++){
	    count_save++;
	     if(count_save!=14){
		mark = 0;
		cap >> temp;
                imshow(windowName, temp);
	     }
	     else{
		mark = 1;
		cap >> frame;     
                imshow(windowName, frame);
	     }

	    if(j==19 || j==20){
		count_save = 0;
	    }
	      
	 char q = cvWaitKey(33);  
	 }	 

	clock_gettime(CLOCK_REALTIME, &rtclk_stop_time_1);
	delta_t(&rtclk_stop_time_1, &rtclk_start_time_1, &rtclk_dt_1);
	cout << "Execution time of image capture thread is " << (double)(rtclk_dt_1.tv_nsec/1000000.0) << "ms"<< endl;;	
	
	sem_post(&fingertip_detect_sem);
    }
}


void *fingertip_detect(void *threadid)
{ 
    while(1){
	//Block on the fingertip detect semaphore
	sem_wait(&fingertip_detect_sem);	

	clock_gettime(CLOCK_REALTIME, &rtclk_start_time_2);

	flag = 0;
	cv::cvtColor(frame, hsv, CV_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(minH, minS, minV), cv::Scalar(maxH, maxS, maxV), hsv);
	// Pre processing
        int blurSize = 5;
        int elementSize = 5;

        cv::medianBlur(hsv, hsv, blurSize);
        cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * elementSize + 1, 2 * elementSize + 1), cv::Point(elementSize, elementSize));
        cv::dilate(hsv, hsv, element);
        // Contour detection
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(hsv, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

        size_t largestContour = 0;

        for (size_t i = 1; i < contours.size(); i++)
        {
           if (cv::contourArea(contours[i]) > cv::contourArea(contours[largestContour]))
               largestContour = i;
        }
        //cv::drawContours(frame, contours, largestContour, cv::Scalar(0, 0, 255), 1);
      // Convex hull
      if (!contours.empty())
      {
          std::vector<std::vector<cv::Point> > hull(1);
          cv::convexHull(cv::Mat(contours[largestContour]), hull[0], false);
          //cv::drawContours(frame, hull, 0, cv::Scalar(0, 255, 0), 3);
          if (hull[0].size() > 2) 
          {
              std::vector<int> hullIndexes;
              cv::convexHull(cv::Mat(contours[largestContour]), hullIndexes, true);
              std::vector<cv::Vec4i> convexityDefects;
              cv::convexityDefects(cv::Mat(contours[largestContour]), hullIndexes, convexityDefects);
              cv::Rect boundingBox = cv::boundingRect(hull[0]);

	      int big_box_left_x = boundingBox.x - 75;
	      int big_box_width = boundingBox.width + 100;
	      int big_box_left_y = boundingBox.y - 80;
	      //int big_box_left_y = boundingBox.y - 100;
	      //int big_box_height = boundingBox.height + 70;
	      int big_box_height = boundingBox.height + 25;
	      //cv::rectangle(frame, cv::Point( boundingBox.x - 65, boundingBox.y - 110 ), cv::Point(boundingBox.x + boundingBox.width + 5, boundingBox.y + boundingBox.height - 70 ),  cv::Scalar(0,0,255), 2);
       
	     if((big_box_left_x > 0) && (big_box_left_x + big_box_width < 639) && (big_box_left_y > 0) && (big_box_left_y + big_box_height < 479))        {  
             small_box = Rect(big_box_left_x, big_box_left_y, big_box_width, big_box_height);    	      
	     roi = frame(small_box);
	     flag = 1;
	     imshow("small window", roi);

             file = "/home/ubuntu/rtes_project/pics/img_";
             itoa(buf, cnt);
             filename = file + buf + ".png";
             imwrite(filename, roi );
	     cnt++;
             counter++;
             }	   
         }	 
      }

        imshow(windowName, frame);
	
	clock_gettime(CLOCK_REALTIME, &rtclk_stop_time_2);
	delta_t(&rtclk_stop_time_2, &rtclk_start_time_2, &rtclk_dt_2);
	cout << "Execution time of fingertip detect thread is " << (double)(rtclk_dt_2.tv_nsec/1000000.0) << "ms"<< endl;

        //char q = cvWaitKey(33);   
	sem_post(&text_extract_sem);	
    }
}


void *text_extract(void *threadid)
{
    while(1){
	//Block on the text extract semaphore
	sem_wait(&text_extract_sem);	

	clock_gettime(CLOCK_REALTIME, &rtclk_start_time_3);

	if(flag == 1){
            api = new tesseract::TessBaseAPI();
    
  	    // Initialize tesseract-ocr with English, without specifying tessdata path
  	    if (api->Init(NULL, "eng")) {
  	      fprintf(stderr, "Could not initialize tesseract.\n");
  	      exit(1);
  	    }
    	    image = pixRead(filename.c_str());
            api->SetImage(image);
    	    // Get OCR result
    	    outText = api->GetUTF8Text();
	    cout << outText;
        }	
	
	clock_gettime(CLOCK_REALTIME, &rtclk_stop_time_3);
	delta_t(&rtclk_stop_time_3, &rtclk_start_time_3, &rtclk_dt_3);
	cout << "Execution time of text extract thread is " << (double)(rtclk_dt_3.tv_nsec/1000000.0) << "ms"<< endl;	

        //char q = cvWaitKey(33);   
	sem_post(&speech_synthesis_sem);	
    } 
}



void *speech_synthesis(void *threadid){
    while(1){
	sem_wait(&speech_synthesis_sem);

	clock_gettime(CLOCK_REALTIME, &rtclk_start_time_4);

	if(flag == 1){
   	   //cout << "Saying " << outText;
  	   if(strlen(outText) > 2){
   	   	espeak_Synth( outText, strlen(outText), position, position_type, end_position, flags, unique_identifier, user_data );
    	 	espeak_Synchronize( );
       	   }

            api->End();
    	    delete [] outText;
    	    pixDestroy(&image);
	}

	clock_gettime(CLOCK_REALTIME, &rtclk_stop_time_4);
	delta_t(&rtclk_stop_time_4, &rtclk_start_time_4, &rtclk_dt_4);
	cout << "Execution time of speech synthesis thread is " << (double)(rtclk_dt_4.tv_nsec/1000000.0) << "ms"<< endl << endl;	

	sem_post(&image_capture_sem);
    }
}



int main()
{
  int rc , scope , i ;
  sem_init (& image_capture_sem , 0 , 1 );
  sem_init (& fingertip_detect_sem , 0 , 0 );
  sem_init (& text_extract_sem , 0 , 0 );
  sem_init (& speech_synthesis_sem , 0 , 0 );
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);     

  //espeak initializations
  output = AUDIO_OUTPUT_PLAYBACK;
  int I, Run = 1, L;    
  espeak_Initialize(output, 0, path, 1<<15 ); 
  espeak_SetVoiceByName(Voice);

  cv::namedWindow(windowName);
  cap = VideoCapture(0);


  cv::setMouseCallback(windowName, CallbackFunc, NULL);
  int inAngleMin = 200, inAngleMax = 300, angleMin = 180, angleMax = 359, lengthMin = 10, lengthMax = 80;
  
  //setting the CPU cores of all cores to 0.
  for(int i=0; i < NUM_CPUS; i++)
      CPU_SET(i, &cpuset);
  
  double stop_1 = 0;
  printf("Before adjustments to scheduling policy:\n");
  print_scheduler();
  
  //setting pthread attributes
  pthread_attr_init (&image_capture_sched_attr);
  pthread_attr_init (&fingertip_detect_sched_attr);
  pthread_attr_init (&text_extract_sched_attr);
  pthread_attr_init (&main_sched_attr);
  pthread_attr_init (&speech_synthesis_sched_attr);
  pthread_attr_setinheritsched (&image_capture_sched_attr, PTHREAD_EXPLICIT_SCHED);
  pthread_attr_setinheritsched (&fingertip_detect_sched_attr, PTHREAD_EXPLICIT_SCHED);
  pthread_attr_setinheritsched (&text_extract_sched_attr, PTHREAD_EXPLICIT_SCHED);
  pthread_attr_setinheritsched (&speech_synthesis_sched_attr, PTHREAD_EXPLICIT_SCHED);
  pthread_attr_setinheritsched (&main_sched_attr, PTHREAD_EXPLICIT_SCHED);
  pthread_attr_setschedpolicy (&image_capture_sched_attr, SCHED_FIFO);
  pthread_attr_setschedpolicy (&fingertip_detect_sched_attr, SCHED_FIFO);
  pthread_attr_setschedpolicy (&text_extract_sched_attr, SCHED_FIFO);
  pthread_attr_setschedpolicy (&speech_synthesis_sched_attr, SCHED_FIFO);
  pthread_attr_setschedpolicy (&main_sched_attr, SCHED_FIFO);

  mainpid=getpid();
  rt_max_prio = sched_get_priority_max (SCHED_FIFO);
  rt_min_prio = sched_get_priority_min (SCHED_FIFO);

  rc=sched_getparam (mainpid, &nrt_param);

  //setting priorities of each thread
  main_param.sched_priority = rt_max_prio;
  image_capture_param.sched_priority = rt_max_prio-1;
  fingertip_detect_param.sched_priority = rt_max_prio-2;
  text_extract_param.sched_priority = rt_max_prio-3;
  speech_synthesis_param.sched_priority = rt_max_prio-3;
  rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param);
  if (rc)
  {
    printf("ERROR; sched_setscheduler rc is %d\n", rc); 
    perror(NULL); 
    exit(-1);
  }

  printf("After adjustments to scheduling policy:\n");
  print_scheduler();
  printf("min prio = %d, max prio = %d\n", rt_min_prio, rt_max_prio);
  pthread_attr_getscope (&fingertip_detect_sched_attr, &scope);

 // Check the scope of the POSIX scheduling mechanism
 if(scope == PTHREAD_SCOPE_SYSTEM)
     printf("PTHREAD SCOPE SYSTEM\n");
 else if (scope == PTHREAD_SCOPE_PROCESS)
     printf("PTHREAD SCOPE PROCESS\n");
 else
     printf("PTHREAD SCOPE UNKNOWN\n");


  //set the scheduling parameters of the pthreads.
  pthread_attr_setschedparam (&image_capture_sched_attr, &fingertip_detect_param);
  pthread_attr_setschedparam (&fingertip_detect_sched_attr, &fingertip_detect_param);
  pthread_attr_setschedparam (&text_extract_sched_attr, &text_extract_param);
  pthread_attr_setschedparam (&speech_synthesis_sched_attr, &text_extract_param);
  pthread_attr_setschedparam (&main_sched_attr, &main_param);

  //set the affinity of the CPU cores to zero
  rc=pthread_attr_setaffinity_np(&image_capture_sched_attr, sizeof(cpu_set_t), &cpuset);
  rc=pthread_attr_setaffinity_np(&fingertip_detect_sched_attr, sizeof(cpu_set_t), &cpuset);
  rc=pthread_attr_setaffinity_np(&text_extract_sched_attr, sizeof(cpu_set_t), &cpuset);
  rc=pthread_attr_setaffinity_np(&speech_synthesis_sched_attr, sizeof(cpu_set_t), &cpuset);

  //Start profiling
  //clock_gettime(CLOCK_REALTIME, &rtclk_start_time);

  //clock_gettime(CLOCK_REALTIME, &rtclk_start_time);
  rc = pthread_create (& image_capture_thread , & image_capture_sched_attr , capture_image , (void *) 0 );
  if (rc)
  {
     printf("ERROR -> pthread_create() rc is %d\n", rc); 
     perror(NULL); 
     exit(-1);
  }

  rc = pthread_create (&fingertip_detect_thread , &fingertip_detect_sched_attr , fingertip_detect , (void *) 0 );
  if (rc)
  {
     printf("ERROR -> pthread_create() rc is %d\n", rc); 
     perror(NULL); 
     exit(-1);
  }

  rc = pthread_create (&text_extract_thread , &text_extract_sched_attr , text_extract , (void *) 0 );
  if (rc)
  {
     printf("ERROR -> pthread_create() rc is %d\n", rc); 
     perror(NULL); 
     exit(-1);
  }
  
  rc = pthread_create (&speech_synthesis_thread , &speech_synthesis_sched_attr , speech_synthesis , (void *) 0 );
  if (rc)
  {
     printf("ERROR -> pthread_create() rc is %d\n", rc); 
     perror(NULL); 
     exit(-1);
  }

  //Joining the threads
  pthread_join(image_capture_thread, NULL);                
  pthread_join(fingertip_detect_thread, NULL);             
  pthread_join(text_extract_thread, NULL);             
  pthread_join(speech_synthesis_thread, NULL);  
           
  pthread_exit(NULL);
  return 0;
}
