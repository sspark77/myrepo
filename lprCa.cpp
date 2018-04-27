#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>
#include<opencv2/ml/ml.hpp>
#include <locale>
#include <time.h>
#include <locale>
using namespace cv;
using namespace std;


int size_width = (int)1920 * 0.5;
float MIN_CONTOUR_AREA1 = 0.0546875 * size_width;
float MAX_CONTOUR_AREA1 = 0.5 * size_width;
float MIN_BOX_AREA1 = 0.015625 * size_width;
float MAX_BOX_AREA1 = 0.5 * size_width;
float MAX_RATIO_WH = 0.01640625 * size_width;
float MIN_RATIO_WH = 0.000046875 * size_width;
float MAX_WIDTH = 0.046875* size_width;
float MAX_HEIGHT = 0.234375* size_width;
float MIN_WIDTH = 0.0078125* size_width;
float MIN_HEIGHT = 0.01875* size_width;


bool finish = false;
bool obChacnge = false;

wstring LPRResultStr = L"";
wstring newResultStr = L"";
wstring LPRResultType = L"";
wstring LPRResultCount = L"";

ofstream outFile1("outputDP.txt");
ofstream outFile2("outputTC.txt");
ofstream outFile3("outputSR.txt");

int resultCount = 0;
//Type Classification *****
int MIN_CONTOUR_AREA = 10;
int MAX_CONTOUR_AREA = 10;
float min_width = 10;
float min_height = 10;
float max_width = 10;
float max_height = 10;
bool doubleLine = false;
float combinedWidth = 500 * 0.14;

int frameCount = 0;
int frameCount2 = 0;

//Set ROI
int roiX = 200;
int roiY = 150;
int roiWidth = 500;
int roiHeight = 300;

cv::Rect roiRect = { roiX, roiY, roiWidth, roiHeight };

wstring regionVaildChars = L"서울부산대구인천광주대전울산경기강원충북충남전북전남경북경남제주세종";
wstring useVaildChars = L"가나마다라마바사아자거너더러머버서어저고노도로모보소오조구누두루무부수우주허하호배";


//Detect Possible Plates
int GROUP_DIST = 20;
int GROUP_THRE = 2;
int LPR_THRE = 5;

int savedCount = 0;
cv::Mat resultCombinedRes;


struct LPRInfo{
	int frameIdx;
	wstring result;
	cv::Rect plates;
	cv::Point centroid;
	std::vector<int> votingIdx;
	int votingCount;
};

std::vector<LPRInfo> LPRInfos;

bool simStart = false;
bool remStart = false;


int intSizeInStr(std::wstring inputstr){
	int numDigit = 0;
	for (int i = 0; i < inputstr.length(); i++){
		if (isdigit(inputstr[i])){
			//std::wcout << checkStr[i] << endl;
			numDigit++;
		}
	}
	return numDigit;
}
int intNumOfKor(std::wstring inputstr){
	int numDigit = 0;
	int idx = -1;
	for (int i = 0; i < inputstr.length(); i++){
		if (isdigit(inputstr[i])){
			//std::wcout << checkStr[i] << endl;
			numDigit++;
		}
		else{
			idx = i;
		}
	}
	return idx;
}

//Pre-Processing
class ContourWithData {
public:

	std::vector<cv::Point> ptContour;
	cv::Rect boundingRect;
	cv::Point centroid;
	int idx;
	float fltArea;
	float ratioWH;
	int boxArea;

	bool checkIfContourIsValidTypes() {
		if (fltArea < MIN_CONTOUR_AREA || fltArea > MAX_CONTOUR_AREA || boundingRect.width>max_width || boundingRect.width<min_width || boundingRect.height<min_height || boundingRect.height>max_height) return false;
		return true;
	}

	bool checkIfContourIsValid() {
		if (fltArea>MAX_CONTOUR_AREA1 || fltArea < MIN_CONTOUR_AREA1 || ratioWH>MAX_RATIO_WH || ratioWH<MIN_RATIO_WH || boundingRect.width>MAX_WIDTH
			|| boundingRect.height>MAX_HEIGHT || boundingRect.width<MIN_WIDTH || boundingRect.height<MIN_HEIGHT || boxArea>MAX_BOX_AREA1 || boxArea<MIN_BOX_AREA1) return false;
		return true;
	}

	static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {
		return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);
	}
	static bool sortByBoundingRectXPositionReverse(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {
		return(cwdLeft.boundingRect.x > cwdRight.boundingRect.x);
	}

};
void preprocessing(cv::Mat& inputSrc, cv::Mat& grayOut, cv::Mat& blutOut, cv::Mat& threOut, bool fliped, int bulrs, int bulrn, int thres, int thren){

	cv::cvtColor(inputSrc, grayOut, CV_BGR2GRAY);
	if (fliped)
		grayOut = ~grayOut;
	cv::GaussianBlur(grayOut,
		blutOut,
		cv::Size(bulrs, bulrs),
		bulrn);

	cv::adaptiveThreshold(blutOut,
		threOut,
		255,
		cv::ADAPTIVE_THRESH_GAUSSIAN_C,
		cv::THRESH_BINARY_INV,
		thres,
		thren);
}
void validBoxesReturn(cv::Mat& inputThre, std::vector<cv::Rect>& returnBoxes, std::vector<std::vector<cv::Point>>& retrunContours){

	std::vector<std::vector<cv::Point> > ptContours;
	std::vector<cv::Vec4i> v4iHierarchy;
	v4iHierarchy.clear();
	ptContours.clear();
	if (!(inputThre.empty())){
		cv::findContours(inputThre,
			ptContours,
			v4iHierarchy,
			cv::RETR_CCOMP,
			cv::CHAIN_APPROX_SIMPLE, Point(1, 1));

		std::vector<ContourWithData> allContoursWithData;
		std::vector<ContourWithData> validContoursWithData;
		std::vector<cv::Rect> validBox;
		std::vector<std::vector<cv::Point> > vaildConts;

		if (ptContours.size() != 0){
			for (int i = 0; i < ptContours.size(); i++) {
				ContourWithData contourWithData;
				contourWithData.ptContour = ptContours[i];
				contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);
				contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);
				contourWithData.ratioWH = (float)contourWithData.boundingRect.width / contourWithData.boundingRect.height;
				contourWithData.boxArea = contourWithData.boundingRect.width*contourWithData.boundingRect.height;
				allContoursWithData.push_back(contourWithData);
			}

			validBox.clear();

			for (int i = 0; i < allContoursWithData.size(); i++) {
				if (allContoursWithData[i].checkIfContourIsValid()) {
					validContoursWithData.push_back(allContoursWithData[i]);
					validBox.push_back(allContoursWithData[i].boundingRect);
					vaildConts.push_back(allContoursWithData[i].ptContour);
				}
			}

			returnBoxes = validBox;
			retrunContours = vaildConts;
		}
	}


}

//Detect Possible Platses
class DbScan
{
public:
	std::map<int, int> labels;
	vector<Rect>& data;
	int C;
	double eps;
	int mnpts;
	double* dp;
	//memoization table in case of complex dist functions
#define DP(i,j) dp[(data.size()*i)+j]
	DbScan(vector<Rect>& _data, double _eps, int _mnpts) :data(_data)
	{
		C = -1;
		for (int i = 0; i<data.size(); i++)
		{
			labels[i] = -99;
		}
		eps = _eps;
		mnpts = _mnpts;
	}
	void run()
	{
		dp = new double[data.size()*data.size()];
		for (int i = 0; i<data.size(); i++)
		{
			for (int j = 0; j<data.size(); j++)
			{
				if (i == j)
					DP(i, j) = 0;
				else
					DP(i, j) = -1;
			}
		}
		for (int i = 0; i<data.size(); i++)
		{
			if (!isVisited(i))
			{
				vector<int> neighbours = regionQuery(i);
				if (neighbours.size()<mnpts)
				{
					labels[i] = -1;//noise
				}
				else
				{
					C++;
					expandCluster(i, neighbours);
				}
			}
		}
		delete[] dp;
	}
	void expandCluster(int p, vector<int> neighbours)
	{
		labels[p] = C;
		for (int i = 0; i<neighbours.size(); i++)
		{
			if (!isVisited(neighbours[i]))
			{
				labels[neighbours[i]] = C;
				vector<int> neighbours_p = regionQuery(neighbours[i]);
				if (neighbours_p.size() >= mnpts)
				{
					expandCluster(neighbours[i], neighbours_p);
				}
			}
		}
	}

	bool isVisited(int i)
	{
		return labels[i] != -99;
	}

	vector<int> regionQuery(int p)
	{
		vector<int> res;
		for (int i = 0; i<data.size(); i++)
		{
			if (distanceFunc(p, i) <= eps)
			{
				res.push_back(i);
			}
		}
		return res;
	}

	double dist2d(Point2d a, Point2d b)
	{
		return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
	}

	double distanceFunc(int ai, int bi)
	{
		if (DP(ai, bi) != -1)
			return DP(ai, bi);
		Rect a = data[ai];
		Rect b = data[bi];
		/*
		Point2d cena= Point2d(a.x+a.width/2,
		a.y+a.height/2);
		Point2d cenb = Point2d(b.x+b.width/2,
		b.y+b.height/2);
		double dist = sqrt(pow(cena.x-cenb.x,2) + pow(cena.y-cenb.y,2));
		DP(ai,bi)=dist;
		DP(bi,ai)=dist;*/
		Point2d tla = Point2d(a.x, a.y);
		Point2d tra = Point2d(a.x + a.width, a.y);
		Point2d bla = Point2d(a.x, a.y + a.height);
		Point2d bra = Point2d(a.x + a.width, a.y + a.height);

		Point2d tlb = Point2d(b.x, b.y);
		Point2d trb = Point2d(b.x + b.width, b.y);
		Point2d blb = Point2d(b.x, b.y + b.height);
		Point2d brb = Point2d(b.x + b.width, b.y + b.height);

		double minDist = 9999999;

		minDist = min(minDist, dist2d(tla, tlb));
		minDist = min(minDist, dist2d(tla, trb));
		minDist = min(minDist, dist2d(tla, blb));
		minDist = min(minDist, dist2d(tla, brb));

		minDist = min(minDist, dist2d(tra, tlb));
		minDist = min(minDist, dist2d(tra, trb));
		minDist = min(minDist, dist2d(tra, blb));
		minDist = min(minDist, dist2d(tra, brb));

		minDist = min(minDist, dist2d(bla, tlb));
		minDist = min(minDist, dist2d(bla, trb));
		minDist = min(minDist, dist2d(bla, blb));
		minDist = min(minDist, dist2d(bla, brb));

		minDist = min(minDist, dist2d(bra, tlb));
		minDist = min(minDist, dist2d(bra, trb));
		minDist = min(minDist, dist2d(bra, blb));
		minDist = min(minDist, dist2d(bra, brb));
		DP(ai, bi) = minDist;
		DP(bi, ai) = minDist;
		return DP(ai, bi);
	}

	vector<vector<Rect> > getGroups()
	{
		vector<vector<Rect> > ret;
		for (int i = 0; i <= C; i++)
		{
			ret.push_back(vector<Rect>());
			for (int j = 0; j<data.size(); j++)
			{
				if (labels[j] == i)
				{
					ret[ret.size() - 1].push_back(data[j]);
				}
			}
		}
		return ret;
	}
};
cv::Scalar HSVtoRGBcvScalar(int H, int S, int V) {

	int bH = H; // H component
	int bS = S; // S component
	int bV = V; // V component
	double fH, fS, fV;
	double fR, fG, fB;
	const double double_TO_BYTE = 255.0f;
	const double BYTE_TO_double = 1.0f / double_TO_BYTE;

	// Convert from 8-bit integers to doubles
	fH = (double)bH * BYTE_TO_double;
	fS = (double)bS * BYTE_TO_double;
	fV = (double)bV * BYTE_TO_double;

	// Convert from HSV to RGB, using double ranges 0.0 to 1.0
	int iI;
	double fI, fF, p, q, t;

	if (bS == 0) {
		// achromatic (grey)
		fR = fG = fB = fV;
	}
	else {
		// If Hue == 1.0, then wrap it around the circle to 0.0
		if (fH >= 1.0f)
			fH = 0.0f;

		fH *= 6.0; // sector 0 to 5
		fI = floor(fH); // integer part of h (0,1,2,3,4,5 or 6)
		iI = (int)fH; // " " " "
		fF = fH - fI; // factorial part of h (0 to 1)

		p = fV * (1.0f - fS);
		q = fV * (1.0f - fS * fF);
		t = fV * (1.0f - fS * (1.0f - fF));

		switch (iI) {
		case 0:
			fR = fV;
			fG = t;
			fB = p;
			break;
		case 1:
			fR = q;
			fG = fV;
			fB = p;
			break;
		case 2:
			fR = p;
			fG = fV;
			fB = t;
			break;
		case 3:
			fR = p;
			fG = q;
			fB = fV;
			break;
		case 4:
			fR = t;
			fG = p;
			fB = fV;
			break;
		default: // case 5 (or 6):
			fR = fV;
			fG = p;
			fB = q;
			break;
		}
	}

	// Convert from doubles to 8-bit integers
	int bR = (int)(fR * double_TO_BYTE);
	int bG = (int)(fG * double_TO_BYTE);
	int bB = (int)(fB * double_TO_BYTE);

	// Clip the values to make sure it fits within the 8bits.
	if (bR > 255)
		bR = 255;
	if (bR < 0)
		bR = 0;
	if (bG >255)
		bG = 255;
	if (bG < 0)
		bG = 0;
	if (bB > 255)
		bB = 255;
	if (bB < 0)
		bB = 0;

	// Set the RGB cvScalar with G B R, you can use this values as you want too..
	return cv::Scalar(bB, bG, bR); // R component
}
float euclideanDist(Point2f& p, Point2f& q) {
	Point diff = p - q;
	return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}
std::vector<cv::Rect> possiblePlates(cv::Mat& inputSrc){

	std::vector<cv::Rect> returnPlates;
	std::vector<cv::Rect> plates;
	std::vector<std::vector<cv::Point> > plateContours;
	std::vector<cv::Rect> plates1;
	std::vector<std::vector<cv::Point> > plateContours1;

	cv::Mat matGrayscale;
	cv::Mat matBlurred;
	cv::Mat matThresh;
	cv::Mat matThresh1;
	cv::Mat matThreshCopy;

	preprocessing(inputSrc, matGrayscale, matBlurred, matThresh, false, 5, 1, 11, 10);
	matThreshCopy = matThresh.clone();
	validBoxesReturn(matThreshCopy, plates, plateContours);

	preprocessing(inputSrc, matGrayscale, matBlurred, matThresh1, true, 5, 1, 9, 10);
	validBoxesReturn(matThresh1, plates1, plateContours1);

	plates.insert(plates.end(), plates1.begin(), plates1.end());
	plateContours.insert(plateContours.end(), plateContours1.begin(), plateContours1.end());

	DbScan dbscan(plates, GROUP_DIST, GROUP_THRE);
	dbscan.run();
	//done, perform display

	Mat grouped = Mat::zeros(matThreshCopy.size(), CV_8UC3);
	vector<Scalar> colors;
	RNG rng(100);


	for (int i = 0; i <= dbscan.C; i++)
	{
		colors.push_back(HSVtoRGBcvScalar(rng(255), 255, 255));
	}

	for (int i = 0; i<dbscan.data.size(); i++)
	{
		Scalar color;
		if (dbscan.labels[i] == -1)
		{
			color = Scalar(128, 128, 128);
		}
		else
		{
			int label = dbscan.labels[i];
			color = colors[label];
		}
		putText(grouped, to_string(dbscan.labels[i]), dbscan.data[i].tl(), FONT_HERSHEY_COMPLEX, .5, color, 1);
		drawContours(grouped, plateContours, i, color, -1);
	}
	imshow("grouped", grouped);




	//////////////////////////////////////////////////////// Group of size
	int labelCount = dbscan.labels[0];

	for (int i = 0; i < dbscan.labels.size(); i++){
		if (labelCount< dbscan.labels[i])
			labelCount = dbscan.labels[i];
	}
	std::vector<std::vector<std::vector<cv::Point>>> groupOfContours;
	std::vector<std::vector<cv::Point>> cContours;

	for (int j = 0; j < labelCount + 1; j++){
		int currentLabel = j;
		cContours.clear();


		for (int i = 0; i < dbscan.data.size(); i++){
			if (dbscan.labels[i] == -1) continue;
			if (currentLabel == dbscan.labels[i]){
				cContours.push_back(plateContours[i]);

			}
		}
		groupOfContours.push_back(cContours);
	}
	for (int i = 0; i<groupOfContours.size(); i++){

		if (groupOfContours[i].size() < 10 /*|| groupOfContours[i].size() > 20*/) continue;
		cv::Rect cRect = cv::boundingRect(groupOfContours[i][0]);
		int minX = cRect.x;
		int minY = cRect.y;
		int maxX = cRect.x + cRect.width;
		int maxY = cRect.y + cRect.height;

		for (int j = 0; j<groupOfContours[i].size(); j++){
			if (minX>cv::boundingRect(groupOfContours[i][j]).x)
				minX = cv::boundingRect(groupOfContours[i][j]).x;
			if (minY > cv::boundingRect(groupOfContours[i][j]).y)
				minY = cv::boundingRect(groupOfContours[i][j]).y;
			if (maxX < cv::boundingRect(groupOfContours[i][j]).x + cv::boundingRect(groupOfContours[i][j]).width)
				maxX = cv::boundingRect(groupOfContours[i][j]).x + cv::boundingRect(groupOfContours[i][j]).width;
			if (maxY < cv::boundingRect(groupOfContours[i][j]).y + cv::boundingRect(groupOfContours[i][j]).height)
				maxY = cv::boundingRect(groupOfContours[i][j]).y + cv::boundingRect(groupOfContours[i][j]).height;
		}
		cv::Rect bRect;
		bRect.x = minX;
		bRect.y = minY;
		bRect.width = maxX - minX;
		bRect.height = maxY - minY;

		double ratio = (double)bRect.width / bRect.height;

		if (1.0< ratio&&ratio < 7.0){
			//if (bRect.width>150 || bRect.height>80) continue;
			//if (bRect.width<60 || bRect.height<20) continue;
			//if ((int)(bRect.width * bRect.height) <2500) continue;
			//if ((int)(bRect.width * bRect.height) >8500) continue;
			if (bRect.width>130 || bRect.height>60) continue;
			if (bRect.width < 60 || bRect.height < 15) continue;
			//if ((int)(bRect.width * bRect.height) <2500) continue;
			//if ((int)(bRect.width * bRect.height) >5500) continue;
			bRect.x = minX - LPR_THRE;
			bRect.y = minY - LPR_THRE;
			bRect.width = maxX - minX + LPR_THRE * 2;
			bRect.height = maxY - minY + LPR_THRE * 2;
			returnPlates.push_back(bRect);
		}

	}
	return returnPlates;
}


//Training Data Load******************
cv::Ptr<cv::ml::KNearest>  kNearest_all(cv::ml::KNearest::create());
int traningDataLoad(){

	cv::Mat matClassificationInts4;
	cv::FileStorage fsClassifications4("classifications_new1.xml", cv::FileStorage::READ);
	if (fsClassifications4.isOpened() == false) {
		std::cout << "error, unable to open training classifications file, exiting program\n\n";
		return(0);
	}
	fsClassifications4["classifications"] >> matClassificationInts4;
	fsClassifications4.release();
	cv::Mat matTrainingImagesAsFlattenedFloats4;
	cv::FileStorage fsTrainingImages4("images_new1.xml", cv::FileStorage::READ);
	if (fsTrainingImages4.isOpened() == false) {
		std::cout << "error, unable to open training images file, exiting program\n\n";
		return(0);
	}
	fsTrainingImages4["images"] >> matTrainingImagesAsFlattenedFloats4;
	fsTrainingImages4.release();
	kNearest_all->train(matTrainingImagesAsFlattenedFloats4, cv::ml::ROW_SAMPLE, matClassificationInts4);

	return 1;
}
std::wstring regionReturn(int input){
	std::wstring region;
	switch (input)
	{
	case 44036:
		region = L"강원";
		break;
	case 52645:
		region = L"충북";
		break;
	case 51216:
		region = L"전남";
		break;
	case 44217:
		region = L"경북";
		break;
	case 44204:
		region = L"경남";
		break;
	case 51250:
		region = L"제주";
		break;
	case 52632:
		region = L"충남";
		break;
	case 51217:
		region = L"전북";
		break;
	case 44306:
		region = L"광주";
		break;
	case 45846:
		region = L"대전";
		break;
	case 50876:
		region = L"울산";
		break;
	case 44201:
		region = L"경기";
		break;
	case 49457:
		region = L"서울";
		break;
	case 48531:
		region = L"부산";
		break;
	case 45825:
		region = L"대구";
		break;
	case 51083:
		region = L"인천";
		break;
	}
	return region;
}
bool checkRegion(int input){
	std::wstring region;
	switch (input)
	{
	case 44036:
		return true;
		break;
	case 52645:
		return true;
		break;
	case 51216:
		return true;
		break;
	case 44217:
		return true;
		break;
	case 44204:
		return true;
		break;
	case 51250:
		return true;
		break;
	case 52632:
		return true;
		break;
	case 51217:
		return true;
		break;
	case 44306:
		return true;
		break;
	case 45876:
		return true;
		break;
	case 50876:
		return true;
		break;
	case 44201:
		return true;
		break;
	case 49457:
		return true;
		break;
	case 48531:
		return true;
		break;
	case 45825:
		return true;
		break;
	case 51083:
		return true;
	default:
		return false;
	}
}

// Type Classification Data ********** 
std::vector<cv::Rect> plateTypeRetrun(cv::Mat src, bool reverse){

	std::vector<cv::Rect> trminArea;
	int srcWidth = src.size().width;
	int srcHeight = src.size().height;

	MIN_CONTOUR_AREA = srcHeight*0.15;
	MAX_CONTOUR_AREA = 9500;
	max_width = 150;
	max_height = 150;
	min_width = 10;
	min_height = 15;

	std::vector<ContourWithData> allTypeData;
	std::vector<ContourWithData> validTypeData;

	cv::Mat matGrayscale;
	cv::Mat matBlurred;
	cv::Mat matThresh;
	cv::Mat matThreshCopy;

	preprocessing(src, matGrayscale, matBlurred, matThresh, reverse, 5, 1, 55, 10);
	matThreshCopy = matThresh.clone();
	std::vector<std::vector<cv::Point> > ptContours;
	std::vector<cv::Vec4i> v4iHierarchy;

	//Contour 
	cv::findContours(matThreshCopy,
		ptContours,
		v4iHierarchy,
		cv::RETR_LIST,
		cv::CHAIN_APPROX_SIMPLE);


	cv::Mat imgContours2(matThreshCopy.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));
	cv::drawContours(imgContours2, ptContours, -1, cv::Scalar(255, 255, 255, 0));

	//cv::imshow("imgContours2", imgContours2);
	std::vector<std::vector<cv::Point> > validCoutours;


	/******************************************************************************************************************************** AllDataType ***/
	for (int i = 0; i < ptContours.size(); i++) {
		ContourWithData typeData;
		typeData.ptContour = ptContours[i];
		typeData.boundingRect = cv::boundingRect(typeData.ptContour);
		typeData.fltArea = cv::contourArea(typeData.ptContour);
		typeData.ratioWH = (float)typeData.boundingRect.width / typeData.boundingRect.height;
		typeData.boxArea = (float)typeData.boundingRect.width * typeData.boundingRect.height;
		typeData.centroid = cv::Point((typeData.boundingRect.x + (typeData.boundingRect.width / 2)), (typeData.boundingRect.y + (typeData.boundingRect.height / 2)));
		allTypeData.push_back(typeData);
	}


	/******************************************************************************************************************************** Valid DataType ***/
	for (int i = 0; i < allTypeData.size(); i++) {
		if ((allTypeData[i].checkIfContourIsValidTypes()) && allTypeData[i].boundingRect.x + allTypeData[i].boundingRect.width<src.size().width - 30) {
			validTypeData.push_back(allTypeData[i]);
			validCoutours.push_back(allTypeData[i].ptContour);
		}
	}


	cv::Mat validContourImg(matThreshCopy.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));
	cv::drawContours(validContourImg, validCoutours, -1, cv::Scalar(255, 255, 255, 0));
	//cv::imshow("validContourImg", validContourImg);

	cv::Mat sortIMG = src.clone();

	/******************************************************************************************************************************** acrtangent Pairing DataType ***/
	std::vector<ContourWithData> checkTypeData = validTypeData;
	std::vector<std::vector<int>>groupsIdx;
	std::vector<int> indexGrouping;

	std::sort(checkTypeData.begin(), checkTypeData.end(), ContourWithData::sortByBoundingRectXPositionReverse);
	for (int i = 0; i < checkTypeData.size(); i++){
		checkTypeData[i].idx = i;
	}

	for (auto it1 = checkTypeData.begin(); it1 != checkTypeData.end(); ++it1)
	{
		auto it2 = it1 + 1;
		double mfDegree = 0;
		indexGrouping.clear();
		indexGrouping.push_back(it1->idx);
		while (it2 != checkTypeData.end())
		{
			mfDegree = atan2f((float)(*it1).centroid.y - (*it2).centroid.y, (float)(*it1).centroid.x - (*it2).centroid.x) * 180 / 3.1415f;
			mfDegree = abs(mfDegree);
			if (mfDegree < 5){
				//cv::circle(sortIMG, it1->centroid, 3, cv::Scalar(255, 0, 255), 2);
				//cv::circle(sortIMG, it2->centroid, 3, cv::Scalar(255, 0, 255), 2);
				//cv::line(sortIMG, it1->centroid, it2->centroid, cv::Scalar(255, 255, 0));
				////std::cout << it1->idx << "    ,    " << it2->idx << std::endl;
				indexGrouping.push_back(it2->idx);
				it2 = checkTypeData.erase(it2);

			}
			else
				++it2;
		}
		if (indexGrouping.size() != 1)
			groupsIdx.push_back(indexGrouping);
	}


	std::vector<ContourWithData> newData = validTypeData;
	std::sort(newData.begin(), newData.end(), ContourWithData::sortByBoundingRectXPositionReverse);

	int curSize = groupsIdx.size();
	std::vector<int> avgYGroups;
	int totalCount = 0;

	for (auto it1 = groupsIdx.begin(); it1 != groupsIdx.end(); ++it1)
	{
		auto it2 = it1 + 1;

		while (it2 != groupsIdx.end())
		{
			int yDist = abs(newData[(*it1)[0]].centroid.y - newData[(*it2)[0]].centroid.y);
			if (yDist < 50){
				(*it1).insert((*it1).end(), (*it2).begin(), (*it2).end());
				it2 = groupsIdx.erase(it2);
			}
			else
				++it2;
		}
	}

	for (auto it1 = groupsIdx.begin(); it1 != groupsIdx.end();)
	{
		float perC = (float)(*it1).size() / totalCount;
		if ((*it1).size() == 2){
			it1 = groupsIdx.erase(it1);
		}
		else{
			++it1;
		}
	}

	for (int j = 0; j < groupsIdx.size(); j++){
		totalCount += groupsIdx[j].size();
	}
	for (auto it1 = groupsIdx.begin(); it1 != groupsIdx.end();)
	{
		float perC = (float)(*it1).size() / totalCount;
		//std::cout << perC << std::endl;
		if (perC < 0.26){
			it1 = groupsIdx.erase(it1);
		}
		else{
			++it1;
		}
	}


	for (int j = 0; j < groupsIdx.size(); j++){
		//std::cout << j + 1 << "      =  " << groupsIdx[j].size() << std::endl;
		int minX = newData[groupsIdx[j][0]].centroid.x;
		int maxX = newData[groupsIdx[j][0]].centroid.x;
		int avgY = newData[groupsIdx[j][0]].centroid.y;
		for (int i = 0; i < groupsIdx[j].size(); i++){
			if (i == j) continue;
			if (minX>newData[groupsIdx[j][i]].centroid.x)
				minX = newData[groupsIdx[j][i]].centroid.x;

			if (maxX < newData[groupsIdx[j][i]].centroid.x)
				maxX = newData[groupsIdx[j][i]].centroid.x;

			avgY += newData[groupsIdx[j][i]].centroid.y;
		}

		avgY /= groupsIdx[j].size();
		avgYGroups.push_back(avgY);
		cv::Point min = { minX, avgY };
		cv::Point max = { maxX, avgY };
		cv::line(sortIMG, min, max, cv::Scalar(0, 0, 255), 3);

	}

	cv::Rect returnRect;
	for (int j = 0; j < groupsIdx.size(); j++){
		int minX = newData[groupsIdx[j][0]].boundingRect.x;
		int maxX = newData[groupsIdx[j][0]].boundingRect.x + newData[groupsIdx[j][0]].boundingRect.width;
		int avgY = newData[groupsIdx[j][0]].centroid.y;
		int avgHeight = newData[groupsIdx[j][0]].boundingRect.height;
		int minY = newData[groupsIdx[j][0]].boundingRect.y;
		int maxY = newData[groupsIdx[j][0]].boundingRect.y + newData[groupsIdx[j][0]].boundingRect.height;



		for (int i = 0; i < groupsIdx[j].size(); i++){
			if (i == j) continue;
			if (minX>newData[groupsIdx[j][i]].boundingRect.x)
				minX = newData[groupsIdx[j][i]].boundingRect.x;

			if (maxX < newData[groupsIdx[j][i]].boundingRect.x + newData[groupsIdx[j][i]].boundingRect.width)
				maxX = newData[groupsIdx[j][i]].boundingRect.x + newData[groupsIdx[j][i]].boundingRect.width;

			if (minY>newData[groupsIdx[j][i]].boundingRect.y)
				minY = newData[groupsIdx[j][i]].boundingRect.y;

			if (maxY < newData[groupsIdx[j][i]].boundingRect.y + newData[groupsIdx[j][i]].boundingRect.height)
				maxY = newData[groupsIdx[j][i]].boundingRect.y + newData[groupsIdx[j][i]].boundingRect.height;

			avgY += newData[groupsIdx[j][i]].centroid.y;
			avgHeight += newData[groupsIdx[j][i]].boundingRect.height;
		}
		returnRect = { minX, minY, maxX - minX, maxY - minY };
		if (avgHeight<35) continue;
		trminArea.push_back(returnRect);
		cv::rectangle(sortIMG, returnRect, cv::Scalar(0, 255, 255), 3);
	}

	//imshow("sortIMG", sortIMG);
	return trminArea;



}
//************************************

// Combine & Remove Function **********
float euclideanDist(cv::Point& p, cv::Point& q) {
	cv::Point diff = p - q;
	return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}
static bool sortByBoundingRectXPosition(const cv::Rect& cwdLeft, const cv::Rect& cwdRight) {
	return(cwdLeft.x < cwdRight.x);
}
static bool sortByBoundingRectYPosition(const cv::Rect& cwdLeft, const cv::Rect& cwdRight) {
	return(cwdLeft.y < cwdRight.y);
}
cv::Rect trimRects(cv::Rect rec1, cv::Rect rec2){
	cv::Rect result;
	int minX = rec1.x;
	int minY = rec1.y;
	int maxX = rec1.x + rec1.width;
	int maxY = rec1.y + rec1.height;

	if (minX > rec2.x){
		minX = rec2.x;
	}
	if (minY > rec2.y)
		minY = rec2.y;

	if (maxX < (rec2.x + rec2.width))
		maxX = (rec2.x + rec2.width);
	if (maxY < (rec2.y + rec2.height))
		maxY = (rec2.y + rec2.height);

	result = { minX, minY, maxX - minX, maxY - minY };

	return result;
}
std::vector<cv::Rect> removeOverlapping(std::vector<cv::Rect> input, float X){

	std::sort(input.begin(), input.end(), sortByBoundingRectXPosition);
	for (auto it1 = input.begin(); it1 != input.end(); ++it1)
	{
		auto it2 = it1 + 1;

		while (it2 != input.end())
		{
			bool intersects = (((*it1)&(*it2)).area() > 0);

			if (intersects){
				float intersectsArea = ((*it1)&(*it2)).area();
				if (intersectsArea == it2->area())
					it2 = input.erase(it2);
				else{
					++it2;
				}
			}
			else
				++it2;
		}
	}

	return input;
}
std::vector<cv::Rect> combinedNearData(std::vector<cv::Rect> input, float distTH){

	//X Position
	std::sort(input.begin(), input.end(), sortByBoundingRectXPosition);
	for (auto it1 = input.begin(); it1 != input.end(); ++it1)
	{
		auto it2 = it1 + 1;

		while (it2 != input.end())
		{
			cv::Point p1 = { it1->x + it1->width, it1->y };
			cv::Point p2 = { it2->x, it2->y };

			double dist = abs(euclideanDist(p1, p2));
			if (dist< distTH){
				if (it1->width + it2->width < combinedWidth){
					(*it1) = trimRects(*it1, *it2);
					it2 = input.erase(it2);
				}
				else{
					++it2;
				}

			}
			else
				++it2;
		}
	}

	//Y Position
	std::sort(input.begin(), input.end(), sortByBoundingRectYPosition);
	for (auto it1 = input.begin(); it1 != input.end(); ++it1)
	{
		auto it2 = it1 + 1;

		while (it2 != input.end())
		{
			cv::Point p1 = { it1->x, it1->y + it1->height };
			cv::Point p2 = { it2->x, it2->y };

			double dist = abs(euclideanDist(p1, p2));
			if (dist< distTH){
				(*it1) = trimRects(*it1, *it2);
				it2 = input.erase(it2);

			}
			else
				++it2;
		}
	}

	return input;

}
std::vector<cv::Rect> removeSmallData(std::vector<cv::Rect>input, int minWidth, int minHeight){

	for (auto it1 = input.begin(); it1 != input.end();)
	{
		if (it1->width<minWidth)
			it1 = input.erase(it1);
		else if (it1->height < minHeight)
			it1 = input.erase(it1);
		else
			++it1;
	}

	return input;
}
//*************************************


//SinglePlate*************************
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;
std::wstring knnReturnChars(std::vector<cv::Rect> input, cv::Mat threSrc, cv::Ptr<cv::ml::KNearest>  knn){
	int result = 0;
	std::wstring resultStr = L"";

	if (input.size()>0){
		for (int i = 0; i < input.size(); i++) {

			if (0 <= input[i].x
				&& 0 <= input[i].width
				&& input[i].x + input[i].width <= threSrc.cols
				&& 0 <= input[i].y
				&& 0 <= input[i].height
				&& input[i].y + input[i].height <= threSrc.rows){

				cv::Mat matROI = threSrc(input[i]);
				cv::Mat matROIResized;
				cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
				cv::Mat matROIFloat;
				matROIResized.convertTo(matROIFloat, CV_32FC1);
				cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);
				cv::Mat matCurrentChar(0, 0, CV_32F);
				knn->findNearest(matROIFlattenedFloat, 1, matCurrentChar);
				float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

				if (checkRegion(int(fltCurrentChar))){
					resultStr = resultStr + regionReturn(int(fltCurrentChar));
				}
				else{
					resultStr = resultStr + wchar_t(int(fltCurrentChar));
				}
			}

		}
	}

	return resultStr;
}
std::wstring recognitionChars(cv::Mat &imgSrc, int code, float minW, float maxW, float minH, float maxH, int MIN_CONTOUR_AREA, bool flip = false){

	std::vector<ContourWithData> allContoursWithData;
	std::vector<cv::Rect> vaildRects;

	cv::Mat src22 = imgSrc.clone();

	cv::Mat matGrayscale;
	cv::Mat matBlurred;
	cv::Mat matThresh;
	cv::Mat matThreshCopy;

	std::wstring resultChars = L"";

	cv::cvtColor(src22, matGrayscale, CV_BGR2GRAY);

	cv::GaussianBlur(matGrayscale,
		matBlurred,
		cv::Size(7, 7),
		2);

	//cv::imshow("matGrayscale", matGrayscale);

	cv::adaptiveThreshold(matBlurred,
		matThresh,
		255,
		cv::ADAPTIVE_THRESH_MEAN_C,
		cv::THRESH_BINARY_INV,
		21,
		7);

	//cv::imshow("matThresh", matThresh);
	cv::Mat thCopy = matThresh.clone();
	matThreshCopy = matThresh.clone();


	std::vector<std::vector<cv::Point> > ptContours;
	std::vector<cv::Vec4i> v4iHierarchy;

	//Contour 
	cv::findContours(matThreshCopy,
		ptContours,
		v4iHierarchy,
		cv::RETR_LIST,
		cv::CHAIN_APPROX_SIMPLE);


	if (ptContours.size()>5){
		for (int i = 0; i < ptContours.size(); i++) {
			ContourWithData contourWithData;
			contourWithData.ptContour = ptContours[i];
			contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);
			contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);
			contourWithData.ratioWH = (float)contourWithData.boundingRect.width / contourWithData.boundingRect.height;
			allContoursWithData.push_back(contourWithData);

		}
		std::vector<std::vector<cv::Point> > vailContours;
		cv::Rect checkVaild;



		for (int i = 0; i < allContoursWithData.size(); i++) {
			if (allContoursWithData[i].checkIfContourIsValid()) {
				vaildRects.push_back(allContoursWithData[i].boundingRect);
				vailContours.push_back(allContoursWithData[i].ptContour);
			}
		}

		cv::Mat imgContours2(src22.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));
		cv::drawContours(imgContours2, vailContours, -1, cv::Scalar(255, 255, 255, 0));
		//cv::imshow("imgContours2", imgContours2);

		for (int i = 0; i < vaildRects.size(); i++){
			cv::rectangle(src22,
				vaildRects[i],
				cv::Scalar(255, 255, 0),
				2);
			//std::cout << vaildRects[i].size() << std::endl;
		}

		vaildRects = removeOverlapping(vaildRects, 1.1);
		vaildRects = combinedNearData(vaildRects, 10);
		vaildRects = removeOverlapping(vaildRects, 1.1);
		vaildRects = removeSmallData(vaildRects, 25, 25);
		for (int i = 0; i < vaildRects.size(); i++){
			cv::rectangle(src22,
				vaildRects[i],
				cv::Scalar(0, 255, 0),
				2);

		}

		std::sort(vaildRects.begin(), vaildRects.end(), sortByBoundingRectXPosition);

		if (vaildRects.size()>0){
			resultChars = knnReturnChars(vaildRects, thCopy, kNearest_all);
		}

	}


	//cv::imshow("imageSrc", src22);
	return resultChars;
}

//Double plate
cv::Mat trimAreaDouble(cv::Mat inputSrc, bool flip){

	cv::Rect returnResult;
	std::vector<cv::Rect> objects;
	objects.clear();
	std::vector<std::vector<cv::Point> > objectContours;


	cv::Mat matGrayscale;
	cv::Mat matBlurred;
	cv::Mat matThresh;
	cv::Mat matThreshCopy;

	cv::cvtColor(inputSrc, matGrayscale, CV_BGR2GRAY);
	if (!flip)
		matGrayscale = ~matGrayscale;

	cv::GaussianBlur(matGrayscale,
		matBlurred,
		cv::Size(7, 7),
		1);

	cv::adaptiveThreshold(matBlurred,
		matThresh,
		255,
		cv::ADAPTIVE_THRESH_MEAN_C,
		cv::THRESH_BINARY_INV,
		45,
		8);
	validBoxesReturn(matThresh, objects, objectContours);
	//for (int i = 0; i < objects.size(); i++){
	//	cv::rectangle(inputSrc,
	//		objects[i],
	//		cv::Scalar(255, 255, 0),
	//		2);
	//}

	int minX = objects[0].x;
	int minY = objects[0].y;
	int maxX = 0;
	int maxY = 0;


	for (int i = 0; i < objects.size(); i++){
		if (minX>objects[i].x)
			minX = objects[i].x;

		if (maxX <objects[i].x + objects[i].width)
			maxX = objects[i].x + objects[i].width;

		if (minY>objects[i].y)
			minY = objects[i].y;

		if (maxY < objects[i].y + objects[i].height)
			maxY = objects[i].y + objects[i].height;
	}

	returnResult = { minX, minY, maxX - minX, maxY - minY };
	//cv::rectangle(inputSrc,
	//	returnResult,
	//	cv::Scalar(255, 0, 255),
	//	2);


	std::vector<cv::Point> totalPoint;
	totalPoint.clear();
	for (int i = 0; i < objectContours.size(); i++){
		totalPoint.insert(totalPoint.end(), objectContours[i].begin(), objectContours[i].end());
	}
	cv::RotatedRect roRect = minAreaRect(totalPoint);
	Mat M, rotated, cropped;
	// get angle and size from the bounding box
	float angle = roRect.angle;
	Size rect_size = roRect.size;
	// thanks to http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
	if (roRect.angle < -45.) {
		angle += 90.0;
		swap(rect_size.width, rect_size.height);
	}
	// get the rotation matrix
	M = getRotationMatrix2D(roRect.center, angle, 1.0);
	// perform the affine transformation
	warpAffine(inputSrc, rotated, M, inputSrc.size(), INTER_CUBIC);
	// crop the resulting image
	getRectSubPix(rotated, rect_size, roRect.center, cropped);
	//imshow("cropped", cropped);
	//imshow("rotated", rotated);



	//imshow("inputSrc", inputSrc);
	//imshow("matThresh111", matThresh);

	return cropped;
}
std::vector<cv::Rect> combinedNearDataDouble(std::vector<cv::Rect> input, int distTH, int maxW, int maxH){




	std::sort(input.begin(), input.end(), sortByBoundingRectXPosition);
	for (auto it1 = input.begin(); it1 != input.end(); ++it1)
	{
		auto it2 = it1 + 1;

		while (it2 != input.end())
		{
			bool intersects = (((*it1)&(*it2)).area() > 0);

			if (intersects){
				cv::Rect result1 = trimRects((*it1), (*it2));
				if (result1.width < maxW){
					(*it1) = result1;
					it2 = input.erase(it2);
				}
				else{
					++it2;
				}
			}
			else
				++it2;
		}
	}







	//X Position
	std::sort(input.begin(), input.end(), sortByBoundingRectXPosition);
	for (auto it1 = input.begin(); it1 != input.end(); ++it1)
	{
		auto it2 = it1 + 1;

		while (it2 != input.end())
		{
			cv::Point p1 = { it1->x + it1->width, it1->y };
			cv::Point p2 = { it2->x, it2->y };

			double dist = abs(euclideanDist(p1, p2));

			if (dist< distTH){
				//std::cout << dist << std::endl;
				cv::Rect result1 = trimRects((*it1), (*it2));
				if (result1.width < maxW){
					(*it1) = result1;
					it2 = input.erase(it2);
				}
				else{
					++it2;
				}

			}
			else
				++it2;
		}
	}

	//Y Position
	std::sort(input.begin(), input.end(), sortByBoundingRectYPosition);
	for (auto it1 = input.begin(); it1 != input.end(); ++it1)
	{
		auto it2 = it1 + 1;

		while (it2 != input.end())
		{
			cv::Point p1 = { it1->x, it1->y + it1->height };
			cv::Point p2 = { it2->x, it2->y };

			double dist = abs(euclideanDist(p1, p2));
			if (dist< distTH){
				cv::Rect result1 = trimRects((*it1), (*it2));
				if (result1.width < maxW){
					(*it1) = result1;
					it2 = input.erase(it2);
				}
				else{
					++it2;
				}

			}
			else
				++it2;
		}
	}

	for (auto it1 = input.begin(); it1 != input.end(); ++it1)
	{
		auto it2 = it1 + 1;

		while (it2 != input.end())
		{
			bool intersects = (((*it1)&(*it2)).area() > 0);
			if (intersects){
				cv::Rect result1 = trimRects((*it1), (*it2));
				if (result1.width < maxW){
					(*it1) = result1;
					it2 = input.erase(it2);
				}
				else{
					++it2;
				}

			}
			else
				++it2;
		}

	}

	std::sort(input.begin(), input.end(), sortByBoundingRectXPosition);
	for (auto it1 = input.begin(); it1 != input.end(); ++it1)
	{
		auto it2 = it1 + 1;

		while (it2 != input.end())
		{
			bool intersects = (((*it1)&(*it2)).area() > 0);
			if (intersects){
				cv::Rect result1 = trimRects((*it1), (*it2));
				if (result1.width < maxW){
					(*it1) = result1;
					it2 = input.erase(it2);
				}
				else{
					++it2;
				}

			}
			else
				++it2;
		}
	}

	return input;

}
std::wstring recognitionCharsDouble(cv::Mat inputSrc, cv::Point trimPoint, cv::Rect trimRectArea, bool flip){

	std::wstring result = L"";
	if (inputSrc.size().width > 0 && inputSrc.size().height > 0){

		cv::Mat src = inputSrc.clone();
		cv::Point resultPoints = trimPoint;
		cv::Rect area1 = { 0, 0, src.size().width, (resultPoints.y) };
		cv::Rect area2 = { 0, resultPoints.y, src.size().width, (src.size().height - resultPoints.y) };


		cv::Mat matGrayscale;
		cv::Mat matBlurred;
		cv::Mat matThresh;
		cv::Mat matThreshCopy;

		cv::cvtColor(src, matGrayscale, CV_BGR2GRAY);
		if (!flip)
			matGrayscale = ~matGrayscale;

		cv::GaussianBlur(matGrayscale,
			matBlurred,
			cv::Size(7, 7),
			1);

		if (!flip)
			cv::adaptiveThreshold(matBlurred,
			matThresh,
			255,
			cv::ADAPTIVE_THRESH_MEAN_C,
			cv::THRESH_BINARY_INV,
			45,
			8);

		cv::adaptiveThreshold(matBlurred,
			matThresh,
			255,
			cv::ADAPTIVE_THRESH_MEAN_C,
			cv::THRESH_BINARY_INV,
			37,
			5);

		//cv::imshow("matThresh", matThresh);
		matThreshCopy = matThresh.clone();
		cv::Mat thCopy = matThresh.clone();

		std::vector<std::vector<cv::Point> > ptContours;
		std::vector<cv::Vec4i> v4iHierarchy;


		cv::findContours(matThreshCopy,
			ptContours,
			v4iHierarchy,
			cv::RETR_CCOMP,
			cv::CHAIN_APPROX_SIMPLE, Point(1, 1));


		cv::Mat imgContours2(src.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));
		cv::drawContours(imgContours2, ptContours, -1, cv::Scalar(255, 255, 255, 0));
		//cv::imshow("imgContours2", imgContours2);



		std::vector<ContourWithData> allContoursWithData;
		std::vector<std::vector<cv::Point>> validContours;
		std::vector<cv::Rect> validBox;


		for (int i = 0; i < ptContours.size(); i++) {
			ContourWithData contourWithData;
			contourWithData.ptContour = ptContours[i];
			contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);
			contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);
			contourWithData.ratioWH = (float)contourWithData.boundingRect.width / contourWithData.boundingRect.height;
			contourWithData.boxArea = contourWithData.boundingRect.width*contourWithData.boundingRect.height;
			allContoursWithData.push_back(contourWithData);
		}

		validBox.clear();
		for (int i = 0; i < allContoursWithData.size(); i++) {
			if (allContoursWithData[i].checkIfContourIsValid()) {
				validContours.push_back(allContoursWithData[i].ptContour);
				validBox.push_back(allContoursWithData[i].boundingRect);
			}
		}

		cv::Mat validContours2(src.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));
		cv::drawContours(validContours2, validContours, -1, cv::Scalar(255, 255, 255, 0));
		//cv::imshow("validContours2", validContours2);


		if (validBox.size() > 0){


			validBox = removeOverlapping(validBox, 1.1);
			//validBox = combinedNearData(validBox, 10);

			std::vector<cv::Rect> area1Rects;
			std::vector<cv::Rect> area2Rects;


			for (int i = 0; i < validBox.size(); i++){
				cv::rectangle(src,
					validBox[i],
					cv::Scalar(255, 255, 0),
					2);
			}


			for (int i = 0; i < validBox.size(); i++){

				if (area1.y <= validBox[i].y
					&&validBox[i].y < area1.y + area1.height
					&&area1.x <= validBox[i].x
					&&validBox[i].x < area1.x + area1.width
					&&validBox[i].x < trimPoint.x
					&&validBox[i].y < trimPoint.y
					&& validBox[i].x > trimRectArea.x - 5
					&& validBox[i].height < 86
					&& validBox[i].x < 450){
					area1Rects.push_back(validBox[i]);
				}



				if (area2.y <= validBox[i].y
					&&validBox[i].y < area2.y + area2.height
					&&area2.x <= validBox[i].x
					&&validBox[i].x < area2.x + area2.width
					&& (validBox[i].height > 65 || validBox[i].width > 48)
					&& validBox[i].width > 16
					/*&& validBox[i].x > trimRectArea.x*/){
					area2Rects.push_back(validBox[i]);
				}
			}

			int maxWidth = 0;
			int maxHeight = 0;

			std::wstring result1 = L"";
			std::wstring result2 = L"";

			if (area2Rects.size() > 0){

				area2Rects = combinedNearDataDouble(area2Rects, 20, 100, 55);
				area2Rects = removeSmallData(area2Rects, 15, 70);

				std::sort(area2Rects.begin(), area2Rects.end(), sortByBoundingRectXPosition);

				if (area2Rects.size() > 5){
					area2Rects.erase(area2Rects.begin() + 5, area2Rects.end());
				}

				result2 = knnReturnChars(area2Rects, thCopy, kNearest_all);

				for (int i = 0; i < area2Rects.size(); i++){
					cv::rectangle(src,
						area2Rects[i],
						cv::Scalar(0, 255, 0),
						2);
				}
				wcout << area2Rects.size() << "    result2 =    " << result2 << endl;
			}


			if (area1Rects.size() > 0){


				std::sort(area1Rects.begin(), area1Rects.end(), sortByBoundingRectXPosition);
				if (!flip){
					if (area2Rects.size() == 5)
						area1Rects = combinedNearDataDouble(area1Rects, 15, 63, 85);
					else
						area1Rects = combinedNearDataDouble(area1Rects, 15, 90, 85);
				}
				else{
					area1Rects = combinedNearDataDouble(area1Rects, 15, 70, 87);
				}
				area1Rects = removeSmallData(area1Rects, 10, 40);

				for (int i = 0; i < area1Rects.size(); i++){
					//testChars
					//cv::Mat thCopy11 = thCopy.clone();
					//thCopy11 = thCopy11(area1Rects[i]);
					//if (!thCopy11.empty()){
					//	cv::String savedName = "testChars/";
					//	savedName += std::to_string(savedCount);
					//	savedName += "_0000";
					//	savedName += ".jpg";
					//	imwrite(savedName, thCopy11);
					//	savedCount++;
					//}

					cv::rectangle(src,
						area1Rects[i],
						cv::Scalar(0, 255, 255),
						2);
				}

				std::sort(area1Rects.begin(), area1Rects.end(), sortByBoundingRectXPosition);

				result1 = knnReturnChars(area1Rects, thCopy, kNearest_all);
				wcout << "result1 =    " << result1 << endl;

			}

			result = result1 + result2;
		}

		cv::rectangle(src,
			area1,
			cv::Scalar(0, 255, 0),
			2);

		cv::rectangle(src,
			area2,
			cv::Scalar(0, 255, 0),
			2);
		//imshow("src111", src);
	}

	return result;
}
std::wstring checkStringVaild(std::wstring inStr){
	std::wstring resultStr = inStr;

	int idx = intNumOfKor(inStr);
	int numChars = intSizeInStr(inStr);
	int strLength = inStr.length();
	//regionVaildChars
	//useVaildChars

	if (strLength == 7 && idx == 2){
		if (inStr[2] == 48148 || inStr[2] == 49324 || inStr[2] == 50500 || inStr[2] == 51088) resultStr = L"";
		for (int i = 0; i < inStr.size(); i++){
			if (i == 2) {
				bool inRange = false;
				for (int u = 0; u < useVaildChars.length(); u++){
					if (inStr[i] == useVaildChars[u]){
						inRange = true;
					}
				}
				if (!inRange)
					resultStr = L"";
			}
			else{
				if (inStr[i]<48 || inStr[i]>57)resultStr = L"";
				if (inStr[0] == 48 && inStr[1] == 48) resultStr = L"";
				if (inStr[3] == 48 && inStr[4] == 48 && inStr[5] == 48 && inStr[6] == 48) resultStr = L"";
			}
		}
	}
	else if (strLength == 9 && idx == 4){
		for (int i = 0; i < inStr.size(); i++){
			if (i == 4) {
				bool inRange = false;
				for (int u = 0; u < useVaildChars.length(); u++){
					if (inStr[i] == useVaildChars[u]){
						inRange = true;
						break;
					}
				}
				if (!inRange)
					resultStr = L"";
			}
			else if (i == 0 || i == 1){
				bool inRange = false;
				for (int u = 0; u < regionVaildChars.length(); u++){
					if (inStr[i] == regionVaildChars[u]){
						inRange = true;
						break;
					}
				}
				if (!inRange)
					resultStr = L"";
			}
			else{
				if (inStr[i]<48 || inStr[i]>57){
					resultStr = L"";
				}
			}

		}
	}
	else{
		if (strLength == 10){
			wstring checkEceptionStr = L"";
			if (inStr[0] < 58){
				for (int i = 0; i < inStr.size(); i++){
					if (i == 0)continue;
					checkEceptionStr += inStr[i];
				}
				checkEceptionStr = checkStringVaild(checkEceptionStr);
				if (checkEceptionStr.size()>0)
					return checkEceptionStr;
			}

		}
		resultStr = L"";
	}
	return resultStr;
}
std::wstring allProcessLPR2(cv::Mat plates, cv::Point& retrunCentroid){
	cv::Point centroid = { -1, -1 };
	std::wstring allLPR2result = L"";
	if (!plates.empty()){

		cv::Mat src;
		float scaleRatio = (float)500 / plates.size().width;
		cv::resize(plates, src, cv::Size(), scaleRatio, scaleRatio);


		MIN_CONTOUR_AREA1 = 10;
		MAX_CONTOUR_AREA1 = 50000;
		MIN_BOX_AREA1 = 2;
		MAX_BOX_AREA1 = 500000;
		MAX_RATIO_WH = 6;
		MIN_RATIO_WH = 0;
		MAX_WIDTH = 150;
		MAX_HEIGHT = 150;
		MIN_WIDTH = 10;
		MIN_HEIGHT = 10;


		clock_t start_t, end_t;
		start_t = clock();
		std::vector<cv::Rect> trimArea = plateTypeRetrun(src, true);
		end_t = clock();
		double tcTime = (double)(end_t - start_t) / CLOCKS_PER_SEC;
		outFile2 << tcTime << endl;



		if (trimArea.size() != 0){
			if (trimArea.size() == 1){
				//std::cout << "single plate" << endl;
				LPRResultType = L"single plate";
				std::vector<cv::Rect> trimAreaSingle = plateTypeRetrun(src, false);
				cv::Mat srcCopy = src.clone();

				if (trimAreaSingle.size() == 1){

					if (trimAreaSingle[0].width > 0 && trimAreaSingle[0].height > 0){

						cv::Mat srcClone(srcCopy, trimAreaSingle[0]);
						Point center = Point((trimAreaSingle[0].x + trimAreaSingle[0].width) / 2, (trimAreaSingle[0].y + trimAreaSingle[0].height) / 2);
						center.x /= scaleRatio;
						center.y /= scaleRatio;
						centroid = center;

						if (!srcClone.empty()){
							float scaleRatio = (float)400 / srcClone.size().width;
							cv::resize(srcClone, srcClone, cv::Size(), scaleRatio, scaleRatio);
							//cv::imshow("srcClone", srcClone);

							MIN_CONTOUR_AREA1 = srcClone.size().width*0.01;
							MAX_CONTOUR_AREA1 = srcClone.size().width * 100;
							MIN_BOX_AREA1 = 0;
							MAX_BOX_AREA1 = srcClone.size().width * 100;
							MAX_RATIO_WH = srcClone.size().width * 0.1;
							MIN_RATIO_WH = 0;
							MAX_WIDTH = srcClone.size().width * 0.12;
							MAX_HEIGHT = srcClone.size().width * 0.35;
							MIN_WIDTH = srcClone.size().width * 0.03;
							MIN_HEIGHT = srcClone.size().width * 0.035;
							combinedWidth = srcClone.size().width * 0.15;


							start_t = clock();
							allLPR2result = recognitionChars(srcClone, 1, min_width, max_width, min_height, max_height, MIN_CONTOUR_AREA);
							allLPR2result = checkStringVaild(allLPR2result);
							end_t = clock();
							double srTime = (double)(end_t - start_t) / CLOCKS_PER_SEC;
							outFile3 << srTime << endl;

							if (allLPR2result.size() > 6)return allLPR2result;

						}
					}
				}
			}
			else{
				std::cout << "double plate" << endl;
				LPRResultType = L"double plate";
				cv::Mat srcCopy = src.clone();

				std::sort(trimArea.begin(), trimArea.end(), sortByBoundingRectYPosition);
				cv::Point trimPoints = { trimArea[0].x + trimArea[0].width, trimArea[0].y + trimArea[0].height };
				int minX = trimArea[0].x;
				int minY = trimArea[0].y;
				int maxX = 0;
				int maxY = 0;

				for (int i = 0; i < trimArea.size(); i++){
					if (minX>trimArea[i].x)
						minX = trimArea[i].x;

					if (maxX <trimArea[i].x + trimArea[i].width)
						maxX = trimArea[i].x + trimArea[i].width;

					if (minY>trimArea[i].y)
						minY = trimArea[i].y;

					if (maxY < trimArea[i].y + trimArea[i].height)
						maxY = trimArea[i].y + trimArea[i].height;
				}

				cv::Rect trimRectDouble = { minX, minY, maxX - minX, maxY - minY };
				Point center = Point((trimRectDouble.x + trimRectDouble.width) / 2, (trimRectDouble.y + trimRectDouble.height) / 2);
				center.x /= scaleRatio;
				center.y /= scaleRatio;
				centroid = center;
				cv::Mat srcClone = srcCopy.clone();
				//cv::rectangle(srcCopy,
				//	trimRectDouble,
				//	cv::Scalar(0, 255, 0),
				//	2);
				//MIN_CONTOUR_AREA1 = srcClone.size().width * 0.02;
				//MAX_CONTOUR_AREA1 = srcClone.size().width * 20;
				//MIN_BOX_AREA1 = srcClone.size().width * 0.004;
				//MAX_BOX_AREA1 = srcClone.size().width * 20;
				//MAX_RATIO_WH = srcClone.size().width * 0.01;
				//MIN_RATIO_WH = 0;
				//MAX_WIDTH = srcClone.size().width * 0.5;
				//MAX_HEIGHT = srcClone.size().width * 0.5;
				//MIN_WIDTH = srcClone.size().width * 0.02;
				//MIN_HEIGHT = srcClone.size().width * 0.02;
				//combinedWidth = srcClone.size().width * 0.16;

				MIN_CONTOUR_AREA1 = 10;
				MAX_CONTOUR_AREA1 = 50000;
				MIN_BOX_AREA1 = 2;
				MAX_BOX_AREA1 = 500000;
				MAX_RATIO_WH = 6;
				MIN_RATIO_WH = 0;
				MAX_WIDTH = 150;
				MAX_HEIGHT = 150;
				MIN_WIDTH = 12;
				MIN_HEIGHT = 10;

				start_t = clock();
				//std::wcout << recognitionCharsDouble(srcClone, trimPoints, false) << std::endl;
				allLPR2result = recognitionCharsDouble(srcClone, trimPoints, trimRectDouble, false);
				allLPR2result = checkStringVaild(allLPR2result);

				////검은글자의 경우 생각..
				if (allLPR2result.size() < 6){
					allLPR2result = recognitionCharsDouble(srcClone, trimPoints, trimRectDouble, true);
					allLPR2result = checkStringVaild(allLPR2result);
				}

				end_t = clock();
				double srTime = (double)(end_t - start_t) / CLOCKS_PER_SEC;
				outFile3 << srTime << endl;

				if (allLPR2result.size() > 6)return allLPR2result;
			}
		}
	}
	retrunCentroid = centroid;
	return allLPR2result;
}
//Check result validation
unsigned int edit_distance(const std::wstring& s1, const std::wstring& s2)
{
	const std::size_t len1 = s1.size(), len2 = s2.size();
	std::vector<std::vector<unsigned int>> d(len1 + 1, std::vector<unsigned int>(len2 + 1));

	d[0][0] = 0;
	for (unsigned int i = 1; i <= len1; ++i) d[i][0] = i;
	for (unsigned int i = 1; i <= len2; ++i) d[0][i] = i;

	for (unsigned int i = 1; i <= len1; ++i)
	for (unsigned int j = 1; j <= len2; ++j)
		d[i][j] = std::min({ d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1) });
	return d[len1][len2];
}
void votingCheck(LPRInfo inputs){
	int differ = -1;

	if (inputs.result.size() > 5){
		if (LPRInfos.size() == 0){
			LPRInfos.push_back(inputs);
		}
		else{
			std::wstring a = LPRInfos[LPRInfos.size() - 1].result;
			std::wstring b = inputs.result;

			differ = edit_distance(a, b);
			int startLoop = -1;
			int differ2 = -1;
			if (LPRInfos.size() > 10) startLoop = LPRInfos.size() - 5;
			else startLoop = 0;


			if (differ > 4){
				LPRInfos.push_back(inputs);
			}
			else{
				bool found = false;
				bool found4 = false;
				int idx = -1;
				for (int j = startLoop; j < LPRInfos.size(); j++){
					std::wstring c = LPRInfos[j].result;
					differ2 = edit_distance(b, c);
					if (differ2 == 0){
						found = true;
						if (/*inputs.plates.x + inputs.centroid.x>LPRInfos[j].plates.x + LPRInfos[j].centroid.x
							&&*/inputs.plates.y + inputs.centroid.y > LPRInfos[j].plates.y + LPRInfos[j].centroid.y
							&&inputs.frameIdx < LPRInfos[j].frameIdx + 2000){
							LPRInfos[j].votingCount++;
							LPRInfos[j].frameIdx = inputs.frameIdx;
							LPRInfos[j].plates = inputs.plates;
							LPRInfos[j].centroid = inputs.centroid;
							break;
						}
					}
					if (differ2 < 4){
						found4 = true;
						idx = j;
					}
				}
				if (found4&&idx>0 && !found){
					inputs.votingIdx.push_back(idx);
					LPRInfos[idx].votingIdx.push_back(LPRInfos.size() - 1);
					LPRInfos.push_back(inputs);
				}
			}
		}

	}


	//for (int i = 0; i < LPRInfos.size(); i++){
	//	wcout << "									" << LPRInfos[i].votingCount << "  ,  " << LPRInfos[i].result << endl;
	//}


}
int main(int argc, char** argv)
{
	setlocale(LC_ALL, "");
	cout.imbue(locale(""));
	cv::Point trimCent = { -1, -1 };
	std::string filename = "video_night_00.avi";
	VideoCapture capture(0);
	Mat frame;
	if (!capture.isOpened())
		throw "Error when reading steam_avi";

	//namedWindow("w", 1);
	//roiRect = { roiX, roiY, 500, 300 };


	roiX = 300;
	roiY = 120;
	roiRect = { roiX, roiY, 500, 300 };
	traningDataLoad();
	time_t timeBegin = time(0);

	wofstream outFile4("outputResult.txt");
	outFile4.imbue(locale(""));
	for (;;)
	{

		frameCount++;
		frameCount2++;
		/*if (!(frameCount % interval == 0)) continue;*/

		capture >> frame;
		if (frame.empty())
			break;

		//flip(frame, frame, -1);
		Mat oframe = frame.clone();
		//cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
		Mat srcCopy = frame.clone();
		//srcCopy = srcCopy(roiRect);



		MIN_CONTOUR_AREA1 = 5;
		MAX_CONTOUR_AREA1 = 130;
		MIN_BOX_AREA1 = 5;
		MAX_BOX_AREA1 = 3000;
		MAX_RATIO_WH = 3.8;
		MIN_RATIO_WH = 0.1;
		MAX_WIDTH = 20;
		MAX_HEIGHT = 25;
		MIN_WIDTH = 3;
		MIN_HEIGHT = 5;

		clock_t start_t, end_t;
		start_t = clock();
		std::vector<cv::Rect> possibPlates = possiblePlates(srcCopy);
		end_t = clock();
		//cout << "poosiblePlates							:      " << (float)(end_t - start_t) / 1000 << "   s" << endl;
		double cTime = (double)(end_t - start_t) / CLOCKS_PER_SEC;
		cv::String ctimeTxt = std::to_string(cTime);
		//putText(srcCopy, ctimeTxt, cv::Point{ 10, 40 }, FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
		cv::String frameText = std::to_string((int)frameCount);
		//putText(srcCopy, frameText, cv::Point{ 10, 20 }, FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
		outFile1 << cTime << endl;


		if (!(possibPlates.size() == 0)){

			for (int i = 0; i < possibPlates.size(); i++){

				if (0 <= possibPlates[i].x
					&& 0 <= possibPlates[i].width
					&& possibPlates[i].x + possibPlates[i].width <= srcCopy.cols
					&& 0 <= possibPlates[i].y
					&& 0 <= possibPlates[i].height
					&& possibPlates[i].y + possibPlates[i].height <= srcCopy.rows){

					cv::Rect oClip = possibPlates[i];
					//oClip.x += roiX;
					//oClip.y += roiY;
					//oClip.x *= 2;
					//oClip.y *= 2;
					//oClip.width *= 2;
					//oClip.height *= 2;

					cv::Mat oClipSrc = oframe.clone();
					oClipSrc = oClipSrc(oClip);

					newResultStr = allProcessLPR2(oClipSrc, trimCent);
					if (trimCent.x > 0 && trimCent.y > 0 && newResultStr.size() > 5){
						std::vector<int> idx = {};
						LPRInfo newLprs = { frameCount, newResultStr, oClip, trimCent, idx, 1 };
						votingCheck(newLprs);
						wcout << "										" << newResultStr << endl;
						outFile4 << newResultStr << endl;
					}
					wcout << "										" << newResultStr << endl;
					//imshow("oClipSrc", oClipSrc);



				}

				cv::rectangle(srcCopy,
					possibPlates[i],
					cv::Scalar(0, 255, 0),
					2);
			}
			end_t = clock();
			double rcTime = (double)(end_t - start_t) / CLOCKS_PER_SEC;
			cv::String rctimeTxt = std::to_string(rcTime);
			//putText(srcCopy, rctimeTxt, cv::Point{ 10, 60 }, FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
		}


		cv::imshow("srcCopy", srcCopy);
		cv::waitKey(10); // waits to display frame


		//if (frameCount == 100){
		//	break;
		//}

	}//FOREND


	return 0;
}
