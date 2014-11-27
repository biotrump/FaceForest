/*
 * demo.cpp
 *
 *  Created on: Aug 11, 2012
 *      Author: Matthias Dantone
 */
#include "forest.hpp"
#include "multi_part_sample.hpp"
#include "head_pose_sample.hpp"
#include "face_utils.hpp"
#include <iostream>
#include "optionparser.h"
#include <opencv2/opencv.hpp>
#include <boost/progress.hpp>
#include "face_forest.hpp"
#include "feature_channel_factory.hpp"
#include "timing.hpp"

using namespace std;
using namespace cv;

void train_forest( ForestParam param,
		vector<FaceAnnotation>& annotations ){
	int off_set = 0;
	//init random generator
	boost::mt19937 rng;
	rng.seed( off_set + 1 );
	srand( off_set + 1);

	//shuffle annotations
	std::random_shuffle( annotations.begin(), annotations.end());


	//allocate memory
	std::vector<ImageSample> samples;
	samples.resize( param.nSamplesPerTree );
	std::vector<MPSample*> mp_samples;
	int num_samples = param.nSamplesPerTree*param.nPatchesPerSample;
	mp_samples.reserve( num_samples );

	int patch_size = param.faceSize*param.patchSizeRatio;

	boost::progress_display show_progress( param.nSamplesPerTree );
	for( int i=0; i < static_cast<int>(annotations.size()) and
		static_cast<int>(mp_samples.size()) < num_samples; i++, ++show_progress) {
		// load image
		const cv::Mat image = cv::imread(annotations[i].url,1);
		if (image.data == NULL){
			std::cerr << "could not load " << annotations[i].url << std::endl;
			continue;
		}

		//convert image to grayscale
		Mat img_gray;
		cvtColor( image, img_gray, CV_BGR2GRAY );

		//rescale image to a common size
		cv::Mat img_rescaled;
		float scale =  static_cast<float>(param.faceSize)/annotations[i].bbox.width ;
		rescale_img( img_gray, img_rescaled, scale, annotations[i]);

		//enlarge the bounding box.
		int offset = annotations[i].bbox.width * .1;
		cv::Mat face;
		extract_face( img_rescaled, annotations[i],face, 0,  offset );

		//normalize imgae
		equalizeHist( face, face );

		//create image sample
		samples[i] = ImageSample(face,param.features,true);

		//randomly sample patches within the face
		boost::uniform_int<> dist_x( 1 , face.cols-patch_size-2);
		boost::uniform_int<> dist_y( 1 , face.rows-patch_size-2);
		boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_x(rng, dist_x);
		boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_y(rng, dist_y);
		for( int j = 0; j < param.nPatchesPerSample; j++ ) {
			cv::Rect bbox = cv::Rect( rand_x(), rand_y(), patch_size,patch_size);
			MPSample* s = new MPSample( &samples[i], bbox, Rect(0,0,face.cols, face.rows), annotations[i].parts, param.faceSize, 1 );
			mp_samples.push_back( s );

			//show patch
//			s->show();
		}
	}

	//start the training
	Timing jobTimer;
	jobTimer.start();
    char savePath[200];
	sprintf(savePath,"%s%03d.txt",param.treePath.c_str(),off_set);
	Tree<MPSample> tree( mp_samples, param, &rng, savePath, jobTimer);
}

void eval_forest( FaceForestOptions option,
		int index=0 ){
	VideoCapture vc(index);
	if(!vc.isOpened())
		return;
	//init face forest
	FaceForest ff(option);

	while(1){
		// load image
		Mat image;
		vc >> image; 	//get one frame
		if (image.empty()){
			std::cerr << "empty frame "<< std::endl;
			continue;
		}

		// convert to grayscale
		Mat img_gray;
		cvtColor( image, img_gray, CV_BGR2GRAY );

		vector<Face> faces;
			ff.analize_image( img_gray, faces );

		//cout << "ffd estimated" << endl;
		//draw results
		char key = FaceForest::show_results( image, faces, 5 );
		if( (key == 'q' ) || ( key == 27 )){
			cout << "key=" << key << std::endl;
			break;
		}
	}
}

void eval_forest( FaceForestOptions option,
		vector<FaceAnnotation>& annotations ){
	//init face forest
	FaceForest ff(option);

	for( int i=0; i < static_cast<int>(annotations.size()); ++i){
    cout << annotations[i].url << endl;

		// load image
		Mat image;
		image = cv::imread(annotations[i].url,1);

		if (image.data == NULL){
			std::cerr << "could not load " << annotations[i].url << std::endl;
			continue;
		}

		// convert to grayscale
		Mat img_gray;
		cvtColor( image, img_gray, CV_BGR2GRAY );

		bool use_predefined_bbox = false;//true;
		vector<Face> faces;
		if( use_predefined_bbox ){
			Face face;
			ff.analize_face( img_gray, annotations[i].bbox, face );
			faces.push_back(face);
		}else{
			ff.analize_image( img_gray, faces );
		}

		cout << "ffd estimated" << endl;
		//draw results
		FaceForest::show_results( image, faces );
	}
}

//http://optionparser.sourceforge.net/
//The Lean Mean C++ Option Parser
enum  optionIndex { UNKNOWN, ANNOTATION, CONFIG_FDD, FACE_CASCADE, HEADPOSE,
		HELP, MODE, VIDEO_I};
const option::Descriptor usage[] =
{
	{UNKNOWN, 0,"" , ""    ,option::Arg::None, "USAGE: FaceForest [options]\n"
												"Options:" },
	{ANNOTATION,    0,"a" , "annotation",option::Arg::None,
		" -a --annotation  \tuse annotation files." },
	{CONFIG_FDD,    0,"c" , "config",option::Arg::Optional,
		" -c --config  \tconfig file." },
	{FACE_CASCADE,  0,"f" , "facecascade",option::Arg::Optional,
		" -f --facecascade  \topenCV face cascade xml file" },
	{HEADPOSE,    	0,"p" , "pose",option::Arg::Optional,
		" -p --pose  \thead pose config file." },
	{HELP,    		0,"h" , "help",option::Arg::None,
		" -h --help \tPrint usage and exit." },
	{MODE,    		0,"m" , "mode",option::Arg::Optional,
		" -m[0,1] --mode[0,1],   \tmode 0 is training, mode 1 is runtime." },
	{VIDEO_I,    	0,"i" , "video",option::Arg::Optional,
		" -i[0,1,2] --video[0,1,2] \tvideo port index." },
	{0,0,0,0,0,0}
};

int main(int argc, char** argv)
{
	int vi=0;
	int mode = 1;
	bool fannonation=false;
	std::string ffd_config_file = "data/config_ffd.txt";
	std::string headpose_config_file = "data/config_headpose.txt";
	std::string face_cascade = "data/haarcascade_frontalface_alt.xml";
	argc-=(argc>0); argv+=(argc>0); // skip program name argv[0] if present
	option::Stats  stats(usage, argc, argv);
	option::Option options[stats.options_max], buffer[stats.buffer_max];
	option::Parser parse(usage, argc, argv, options, buffer);

	if (parse.error()){
		cout << "parsing error" << endl;
		return 1;
	}
	cout << "optionsCount:" << parse.optionsCount() << endl;
	for (int i = 0; i < parse.optionsCount(); ++i) {
		option::Option& opt = buffer[i];
		//cout << "[" << i << "]:" <<  opt.index() << endl;
		switch(opt.index()) {
		case ANNOTATION:
			cout << "ANNOTATION" << endl;
			fannonation=true;
			break;
		case CONFIG_FDD:
			cout << "CONFIG_FDD" << CONFIG_FDD <<endl;
			if(opt.arg){
				ffd_config_file = opt.arg;
				cout << "," << ffd_config_file << endl;
			}
			break;
		case FACE_CASCADE:
			cout << "FACE_CASCADE:" << FACE_CASCADE;
			if(opt.arg){
				face_cascade = opt.arg;
				cout << "," <<face_cascade << endl;
			}
			break;
		case HEADPOSE:
			cout << "HEADPOSE" << HEADPOSE;
			if(opt.arg){
				headpose_config_file = opt.arg;
				cout << "," << headpose_config_file << endl;
			}
			break;
		case HELP:
			cout << "HELP" << HELP<<endl;
			option::printUsage(std::cout, usage);
			exit(1);
			break;
		case MODE:
			cout << "MODE:" << MODE <<endl;
			if(opt.arg)
				mode = atoi(opt.arg);
			cout << "mode:" << mode << endl;
			break;
		case VIDEO_I:
			cout << "VIDEO_I:" << VIDEO_I << endl;
			if(opt.arg){
				vi = atoi(opt.arg);
			}
			cout << "vi:" << vi << endl;
			break;
		case UNKNOWN:
		default:
			cout << "unknown" << opt.index() << endl;
			break;
		}
	}

	// parse config file
	ForestParam mp_param;
	assert(loadConfigFile(ffd_config_file, mp_param));

	if( mode == 0){
		// loading GT
		vector<FaceAnnotation> annotations;
		load_annotations( annotations, mp_param.imgPath);

		train_forest( mp_param, annotations );
	}else if( mode == 1 ){
		FaceForestOptions option;
		option.face_detection_option.path_face_cascade = face_cascade;

		ForestParam head_param;
		assert(loadConfigFile(headpose_config_file, head_param));
		option.head_pose_forest_param = head_param;
		option.mp_forest_param = mp_param;
		if(fannonation){
			vector<FaceAnnotation> annotations;
			load_annotations( annotations, mp_param.imgPath);

			eval_forest(option, annotations);
		}else{
			eval_forest(option, vi);
		}
	}

	return 0;
}
