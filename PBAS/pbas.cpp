#include "PBAS.h"  
PBAS::PBAS(void) : N(20), R_lower(18), Raute_min(2), T_lower(2), T_upper(200), R_scale(5), R_incdec(0.05), T_dec(0.05), T_inc(1.0)
{
	std::cout << "PBAS()" << std::endl;
	//feature vector
	alpha = 7.0;
	beta = 1.0;
	formerMeanNorm = 0;
	width = 0;
	//result image
	foregroundValue = 255;
	backgroundValue = 0;
	//length of random array
	countOfRandomNumb = 1000;
	//the T(x_i) value needs initiation 
	T_init = R_lower;
	//check if something is moving in the picture
	isMove = false;
	//for init, count number of runs
	runs = 0;
	newInitialization();
}
void PBAS::newInitialization()
{
	//清空存储各项随机数vector
	if (!randomN.empty())
		randomN.clear();
	if (!randomX.empty())
		randomX.clear();
	if (!randomY.empty())
		randomY.clear();
	if (!randomMinDist.empty())
		randomMinDist.clear();
	if (!randomT.empty())
		randomT.clear();
	if (!randomTN.empty())
		randomTN.clear();
	//先分别产生countOfRandomNumb个随机数，提高算法速度
	for (int l = 0; l < countOfRandomNumb; l++)
	{
		randomN.push_back((int)randomGenerator.uniform((int)0, (int)N));
		randomX.push_back((int)randomGenerator.uniform(-1, +2));
		randomY.push_back((int)randomGenerator.uniform(-1, +2));
		randomMinDist.push_back((int)randomGenerator.uniform((int)0, (int)N));
		randomT.push_back((int)randomGenerator.uniform((int)0, (int)T_upper));
		randomTN.push_back((int)randomGenerator.uniform((int)0, (int)T_upper));
	}
}
PBAS::~PBAS(void)
{
	std::cout << "~PBAS()" << std::endl;

	//以下都可以删除......不明白作者为什么写上去
	randomN.clear();
	randomX.clear();
	randomY.clear();
	randomMinDist.clear();
	randomT.clear();
	randomTN.clear();
	for (int k = 0; k < backgroundModel.size(); ++k)
	{
		if (chans == 1)
		{
			backgroundModel.at(k).at(0).release();
			backgroundModel.at(k).at(1).release();
		}
		else
		{
			backgroundModel.at(k).at(0).release();
			backgroundModel.at(k).at(1).release();
			backgroundModel.at(k).at(2).release();
			backgroundModel.at(k).at(3).release();
			backgroundModel.at(k).at(4).release();
			backgroundModel.at(k).at(5).release();
		}
	}
	backgroundModel.clear();
	meanMinDist.release();
	actualR.release();
	actualT.release();
	sobelX.release();
	sobelY.release();
}
bool PBAS::process(cv::Mat* input, cv::Mat* output)
{
	//第一帧先存储图像rows，cols，chans
	if (width != input->cols)
	{
		width = input->cols;
		chans = input->channels();
		height = input->rows;
		if (input->rows < 1 || input->cols < 1)
		{
			std::cout << "Error: Occurrence of to small (or empty?) image size in PBAS. STOPPING " << std::endl;
			return false;
		}
	}
	//iniate the background model，使用前N帧建立背景模型，次函数调用N次
	init(input);
	//分割mask
	resultMap = new cv::Mat(input->rows, input->cols, CV_8UC1);
	//calculate features
	//计算当前图像Features，（梯度，灰度）
	calculateFeatures(&currentFeatures, input);
	//set sumMagnitude to zero at beginning and then sum up in the loop
	//计算公式5用到的Im
	sumMagnitude = 0;
	long glCounterFore = 0;
	isMove = false;
	//Here starts the whole processing of each pixel of the image
	// for each pixel
	for (int j = 0; j < resultMap->rows; ++j)
	{
		//分割mask行指针
		resultMap_Pt = resultMap->ptr<uchar>(j);

		//对当前特征图(彩色6个Mat，灰度2个Mat)
		currentFeaturesM_Pt.clear();
		currentFeaturesC_Pt.clear();
		std::vector<float*> fT;
		std::vector<uchar*> uT;
		B_Mag_Pts.clear();
		B_Col_Pts.clear();
		for (int z = 0; z < chans; ++z)
		{
			//features vector中的梯度图的行指针
			currentFeaturesM_Pt.push_back(currentFeatures.at(z).ptr<float>(j));
			//feature vector中的灰度图的行指针
			currentFeaturesC_Pt.push_back(currentFeatures.at(z + chans).ptr<uchar>(j));
			//背景模型
			B_Mag_Pts.push_back(fT);
			B_Col_Pts.push_back(uT);
		}
		//同样取各个行指针
		meanMinDist_Pt = meanMinDist.ptr<float>(j);
		actualR_Pt = actualR.ptr<float>(j);
		actualT_Pt = actualT.ptr<float>(j);
		//如果是彩色图像，B_Mag_Pts，B_Col_Pts就分别有3个vector<float*>, 3个vector<uchar*>
		//其中每个对应一个通道
		//而一个vector<float*>存储runs帧图像的行指针..比如runs=N，则有N个float*，分别对应每个背景特征图
		//中梯度图的行地址，同理B_Col_Pts则是灰度图
		for (int k = 0; k < runs; ++k)
		{
			for (int z = 0; z < chans; ++z)
			{
				B_Mag_Pts.at(z).push_back(backgroundModel.at(k).at(z).ptr<float>(j));
				B_Col_Pts.at(z).push_back(backgroundModel.at(k).at(z + chans).ptr<uchar>(j));
			}
		}
		//对每个像素依次操作如下
		for (int i = 0; i < resultMap->cols; ++i)
		{
			//Compare each pixel to in the worst runtime-case each background model
			int count = 0;
			int index = 0;
			double norm = 0.0;
			double dist = 0.0;
			double minDist = 1000.0;
			//在countOfRandomNumb中随机取一个数
			int entry = randomGenerator.uniform(3, countOfRandomNumb - 4);
			do
			{
				//如果是彩色图像，
				if (chans == 3)
				{
					//R,G,B通道分别： 第index+1个背景图中的第i+1个位置依次与  当前图像的当前位置，比较梯度
					//sqrt(Mr^2 + Mg^2 + Mb^2)   其中Mi表示对应通道的梯度差值
					norm = sqrt(
						(((double)B_Mag_Pts.at(0).at(index)[i] - ((double)*currentFeaturesM_Pt.at(0)))  *   ((double)B_Mag_Pts.at(0).at(index)[i] - ((double)*currentFeaturesM_Pt.at(0)))) +
						(((double)B_Mag_Pts.at(1).at(index)[i] - ((double)*currentFeaturesM_Pt.at(1)))  *   ((double)B_Mag_Pts.at(1).at(index)[i] - ((double)*currentFeaturesM_Pt.at(1)))) +
						(((double)B_Mag_Pts.at(2).at(index)[i] - ((double)*currentFeaturesM_Pt.at(2)))  *   ((double)B_Mag_Pts.at(2).at(index)[i] - ((double)*currentFeaturesM_Pt.at(2))))
						);
					//同上，只不过此处是灰度值的比较
					dist = sqrt(
						(((double)B_Col_Pts.at(0).at(index)[i] - ((double)*currentFeaturesC_Pt.at(0)))  *   ((double)B_Col_Pts.at(0).at(index)[i] - ((double)*currentFeaturesC_Pt.at(0)))) +
						(((double)B_Col_Pts.at(1).at(index)[i] - ((double)*currentFeaturesC_Pt.at(1)))  *   ((double)B_Col_Pts.at(1).at(index)[i] - ((double)*currentFeaturesC_Pt.at(1)))) +
						(((double)B_Col_Pts.at(2).at(index)[i] - ((double)*currentFeaturesC_Pt.at(2)))  *   ((double)B_Col_Pts.at(2).at(index)[i] - ((double)*currentFeaturesC_Pt.at(2))))
						);
				}
				else  //灰度图像
				{
					//比上面更简单  abs(Mb - Mc)^2
					norm = abs(
						(((double)B_Mag_Pts.at(0).at(index)[i] - ((double)*currentFeaturesM_Pt.at(0)))  *   ((double)B_Mag_Pts.at(0).at(index)[i] - ((double)*currentFeaturesM_Pt.at(0))))
						);
					dist = abs(
						(((double)B_Col_Pts.at(0).at(index)[i] - ((double)*currentFeaturesC_Pt.at(0)))  *   ((double)B_Col_Pts.at(0).at(index)[i] - ((double)*currentFeaturesC_Pt.at(0))))
						);
				}
				//根据公式5计算距离
				dist = ((double)alpha*(norm / formerMeanMag) + beta*dist);
				//如果小于距离阈值，记录，同时匹配数+1
				if ((dist < *actualR_Pt))
				{
					++count;
					if (minDist > dist)
						minDist = dist;
				}
				//否则，记下当前位置的梯度值，【未解决】
				//此处有点奇怪，公式5是说Im表示前一帧图像的梯度平均，为什么这里只计算不匹配背景的位置梯度？？？）
				//我试过通过累计所有位置的梯度，然后除以像素个数，好像效果没多大区别...alpha值我没动过，可能调下alpha
				else
				{
					sumMagnitude += (double)(norm);
					++glCounterFore;
				}
				++index;
			} while ((count < Raute_min) && (index < runs));  //当前匹配数小于2或者背景没有遍历完，循环以上过程

			//#############################################
			//update backgroundmodel
			// is BACKGROUND    如果是背景       
			if (count >= Raute_min)
			{
				*resultMap_Pt = 0;
				//使用这种方式替代随机数产生器，提高算法速度
				//比如T_upper=200， actualT=2；则ratio=100， 那么ratio小于 randomT的概率多大呢，1/2，对吧。  和咱们的
				//更新率1/T = 1/2  一样...其他数值同理
				double ratio = std::ceil((double)T_upper / (double)(*actualT_Pt));
				//in the first run every distance is zero, because there is no background model
				//in the secont run, we have already one image as background model, hence a 
				// reasonable minDist could be found -> because of the partly 1/run changing in the running average, we set in the first try meanMinDist to the actual minDist value
				//存储当前minDist到meanMinDist矩阵中
				//若有新的minDist更新，重新计算meanMinDist
				if (runs < N && runs > 2)
				{
					*meanMinDist_Pt = ((((float)(runs - 1)) * (*meanMinDist_Pt)) + (float)minDist) / ((float)runs);
				}
				else if (runs < N && runs == 2)
				{
					*meanMinDist_Pt = (float)minDist;
				}
				//1. update model
				if (runs == N)
				{
					//Update current pixel
					//check if random numer is smaller than ratio
					//更新当前背景模型
					if (randomT.at(entry) < ratio)
					{
						// replace randomly chosen sample
						int rand = randomN.at(entry + 1); //randomGenerator.uniform((int)0,(int)N-1);
						//替换梯度，灰度值
						for (int z = 0; z < chans; ++z)
						{
							B_Mag_Pts.at(z).at(rand)[i] = (float)*currentFeaturesM_Pt.at(z);
							B_Col_Pts.at(z).at(rand)[i] = (uchar)*currentFeaturesC_Pt.at(z);
						}
						//更新meanMinDist
						*meanMinDist_Pt = ((((float)(N - 1)) * (*meanMinDist_Pt)) + (float)minDist) / ((float)N);
					}
					//邻域更新
					//Update neighboring pixel model
					if (randomTN.at(entry) < ratio)
					{
						//choose neighboring pixel randomly
						int xNeigh = randomX.at(entry) + i;
						int yNeigh = randomY.at(entry) + j;
						//坐标测试
						checkValid(&xNeigh, &yNeigh);
						// replace randomly chosen sample
						int rand = randomN.at(entry - 1);
						//邻域更新
						for (int z = 0; z < chans; ++z)
						{
							(backgroundModel.at(rand)).at(z).at<float>(yNeigh, xNeigh) = currentFeatures.at(z).at<float>(yNeigh, xNeigh);
							(backgroundModel.at(rand)).at(z + chans).at<uchar>(yNeigh, xNeigh) = currentFeatures.at(z + chans).at<uchar>(yNeigh, xNeigh);
						}
					}
				}
			}
			else  //前景
			{
				// store pixel as foreground
				*resultMap_Pt = 255;
				//there is some movement
				isMove = true;
			}
			//#######################//#######################//#######################//#######################
			//control loops
			//#######################//#######################//#######################//#######################
			//update R      
			//更新R， 另一个版本的是在背景中更新R....【未解决】
			decisionThresholdRegulator(actualR_Pt, meanMinDist_Pt);
			//更新T
			//update T
			learningRateRegulator(actualT_Pt, meanMinDist_Pt, resultMap_Pt);
			//#######################//#######################//#######################//#######################
			//#######################//#######################//#######################//#######################           
			//jump to next pixel
			++resultMap_Pt;
			for (int z = 0; z < chans; ++z)
			{
				++currentFeaturesM_Pt.at(z);
				++currentFeaturesC_Pt.at(z);
			}
			++meanMinDist_Pt;
			++actualR_Pt;
			++actualT_Pt;
		}
	}
	resultMap->copyTo(*output);
	//if there is no foreground -> no magnitudes fount
	//-> initiate some low value to prevent diving through zero
	//计算meanMag既公式5中的Im
	double meanMag = sumMagnitude / (double)(glCounterFore + 1); //height*width);
	if (meanMag > 20)
		formerMeanMag = meanMag;
	else
		formerMeanMag = 20;
	delete resultMap;
	for (int z = 0; z < chans; ++z)
	{
		currentFeatures.at(z + chans).release();
		currentFeatures.at(z).release();
	}

	return true;
}

//公式（3），更新Decision Threshold图(Mat)
//Pt指向当前R(xi), meanDist既Dmin均值
void PBAS::decisionThresholdRegulator(float* pt, float* meanDist)
{
	//update R
	double tempR = *pt;
	double newThresh = (*meanDist)*R_scale;
	//公式（3）
	if (tempR < newThresh)
	{
		tempR += tempR * R_incdec;
	}
	else
	{
		tempR -= tempR * R_incdec;
	}

	//边界保护所有灰度阈值必须大于R_lower
	if (tempR >= R_lower)
		*pt = (float)tempR;
	else
		*pt = (float)R_lower;
}

//公式4，pt存储是当前learning rate 图(Mat),  meanDist既Dmin均值，
void PBAS::learningRateRegulator(float* pt, float* meanDist, uchar* isFore)
{
	//time update
	double tempT = *pt;

	//根据公式（4）更新
	//若是背景
	if ((int)*isFore < 128)
	{
		tempT -= T_inc / (*meanDist + 1.0);
	}
	else  //若是前景
	{
		tempT += T_dec / (*meanDist + 1.0);
	}
	//最后更新的T(xi)必须在T_lower 与 T_upper之间
	if (tempT > T_lower && tempT < T_upper)
		*pt = (float)tempT;
}

//边界保护。
//因为模型Update的时候有一定概率更新邻域，而邻域位置是随机的，所以必须保证邻域位置有效
//既(x,y)坐标必须是有效数值
void PBAS::checkValid(int *x, int *y)
{
	if (*x < 0)
	{
		*x = 0;
	}
	else if (*x >= width)
	{
		*x = width - 1;
	}
	if (*y < 0)
	{
		*y = 0;
	}
	else if (*y >= height)
	{
		*y = height - 1;
	}
}

//初始化建立背景模型！！！！！！！！！！！！！
void PBAS::init(cv::Mat* input)
{
	//对前N帧图像，依次计算每帧图像的Features，存入backgroundModel中
	//最后runs = N，不再变化
	if (runs < N)
	{
		std::vector<cv::Mat> init;
		calculateFeatures(&init, input);
		backgroundModel.push_back(init);
		//以下可删
		if (chans == 1)
		{
			init.at(0).release();
			init.at(1).release();
		}
		else
		{
			init.at(0).release();
			init.at(1).release();
			init.at(2).release();
			init.at(3).release();
			init.at(4).release();
			init.at(5).release();
		}
		init.clear();

		//若是第一次，需要分配内存
		if (runs == 0)
		{
			//存储公式3用到的Dmin均值，开始填充0
			meanMinDist.create(input->size(), CV_32FC1);
			meanMinDist.zeros(input->rows, input->cols, CV_32FC1);
			//当前Decision Threshold矩阵，每个“像素”点赋值为R_lower,一般为18
			actualR.create(input->rows, input->cols, CV_32FC1);
			//当前Update Learning Rate矩阵，每个“像素”点赋值为T_init,一般为18
			actualT.create(input->rows, input->cols, CV_32FC1);
			float* ptRs, *ptTs; //, *ptM;
			for (int rows = 0; rows < actualR.rows; ++rows)
			{
				ptRs = actualR.ptr<float>(rows);
				ptTs = actualT.ptr<float>(rows);
				for (int cols = 0; cols < actualR.cols; ++cols)
				{
					//依次赋值
					ptRs[cols] = (float)R_lower;
					ptTs[cols] = (float)T_init;
				}
			}
		}
		++runs;
	}
}
//计算当前帧的Feature图
void PBAS::calculateFeatures(std::vector<cv::Mat>* feature, cv::Mat* inputImage)
{
	//清空feature内容，因为建模完成后，每次计算的feature都存在currentFeature中，所以需要清空
	if (!feature->empty())
		feature->clear();
	cv::Mat mag[3], dir;
	//如果是彩色图像
	if (inputImage->channels() == 3)
	{
		//拆分为R G B通道
		std::vector<cv::Mat> rgbChannels(3);
		cv::split(*inputImage, rgbChannels);
		//依次对每个通道计算梯度，然后push到Feature中，此时feature存入了3个Mat，分别是
		//R G B通道的梯度图像
		for (int l = 0; l < 3; ++l)
		{
			cv::Sobel(rgbChannels.at(l), sobelX, CV_32F, 1, 0, 3, 1, 0.0);
			cv::Sobel(rgbChannels.at(l), sobelY, CV_32F, 0, 1, 3, 1, 0.0);
			// Compute the L2 norm and direction of the gradient
			cv::cartToPolar(sobelX, sobelY, mag[l], dir, true);
			feature->push_back(mag[l]);
			sobelX.release();
			sobelY.release();
		}
		//然后再把R G B通道的灰度值push到feature 中...
		feature->push_back(rgbChannels.at(0));
		feature->push_back(rgbChannels.at(1));
		feature->push_back(rgbChannels.at(2));
		rgbChannels.at(0).release();
		rgbChannels.at(1).release();
		rgbChannels.at(2).release();
		//总结就是：
		//feature这个vector<Mat> 最后存了6个Mat，分别是梯度，梯度，梯度，灰度，灰度，灰度

	}
	else
	{
		//灰度图像同理，
		//只不过最后feature只存入2个Mat， 分别对应梯度图，灰度值
		cv::Sobel(*inputImage, sobelX, CV_32F, 1, 0, 3, 1, 0.0);
		cv::Sobel(*inputImage, sobelY, CV_32F, 0, 1, 3, 1, 0.0);
		// Compute the L2 norm and direction of the gradient
		cv::cartToPolar(sobelX, sobelY, mag[0], dir, true);
		feature->push_back(mag[0]);
		cv::Mat temp;
		inputImage->copyTo(temp);
		feature->push_back(temp);
		temp.release();
	}
	mag[0].release();
	mag[1].release();
	mag[2].release();
	dir.release();
}

//以下都是设置参数函数，不说了................
void PBAS::setN(int temp)
{
	N = temp;
	newInitialization();     //因为N关系到随机数产生器
}
void PBAS::setRaute_min(int temp)
{
	Raute_min = temp;
}
void PBAS::setR_lower(double temp)
{
	R_lower = temp;
}
void PBAS::setR_incdec(double temp)
{
	R_incdec = temp;
}
void PBAS::setR_scale(double temp)
{
	R_scale = temp;
}
void PBAS::setT_init(double temp)
{
	T_init = temp;
}
void PBAS::setT_lower(double temp)
{
	T_lower = temp;
}
void PBAS::setT_upper(double temp)
{
	T_upper = temp;
	newInitialization();   //因为 T_upper关系到随机数产生器
}
void PBAS::setT_dec(double temp)
{
	T_dec = temp;
}
void PBAS::setT_inc(double temp)
{
	T_inc = temp;
}
void PBAS::setAlpha(double temp)
{
	alpha = temp;
}
void PBAS::setBeta(double temp)
{
	beta = temp;
}
bool PBAS::isMovement()
{
	return isMove;
}
