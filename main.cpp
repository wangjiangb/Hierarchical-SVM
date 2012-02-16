#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "svm_tree.h"

using namespace std;

const int testCnt = 2;
const int randSamplingCnt = 10;
const int entropySamplingCnt = 10;

void RandSampling()
{
	SvmTree tree;
	SvmNode *T;
	tree.Create_Tree(T,"list/treeFile.txt");

	tree.LoadSeedImg(T,"list/SeedImages.txt","2");
	tree.PrintNodesImgs(T);
	tree.GetNodesTrain(T);
	
	RandomInstanceSelector rm;
	rm.SetImagePool("list/TrainImages.txt");
	string label;
	string img;
	string node;
	rm.SetTree(T,&tree);

	for(int cnt = 0; cnt<testCnt; cnt++)
	{
		int rcnt = 0;
		bool doneLab;
		while(rcnt < randSamplingCnt)
		{
			rm.SelectImageLabel(img,label,node);
			//cout<<label<<endl;
			//cout<<img<<endl;
			//cout<<node<<endl;
			doneLab = tree.AddTrainImg(T,node,img,label);
			
			if(doneLab)
				rcnt++;
		}

		//tree.PrintNodesImgs(T);
		tree.GetNodesTrain(T);
		tree.ClassifyImages(T,"list/TestImages.txt");
	}
	
	//string res = tree.ClassifyImg(T,"datas/6.txt");
	//cout<<res<<endl;
	//tree.ClassifyImages(T,"list/TestImages.txt");
}

void EntropySampling()
{
	SvmTree tree;
	SvmNode *T;
	tree.Create_Tree(T,"list/treeFile.txt");

	tree.LoadSeedImg(T,"list/SeedImages.txt","2");
	//tree.PrintNodesImgs(T);
	tree.GetNodesTrain(T);
	
	EntropyInstanceSelector en;
	en.SetImagePool("list/TrainImages.txt");
	string label;
	string img;
	string node;
	en.SetTree(T,&tree);
	en.SelectImageLabel(img,label,node);
	cout<<img<<endl;
	cout<<node<<endl;
	cout<<label<<endl;

	/*for(int cnt = 0; cnt<testCnt; cnt++)
	{
		int rcnt = 0;
		bool doneLab;
		while(rcnt < randSamplingCnt)
		{
			rm.SelectImageLabel(img,label,node);
			//cout<<label<<endl;
			//cout<<img<<endl;
			//cout<<node<<endl;
			doneLab = tree.AddTrainImg(T,node,img,label);
			
			if(doneLab)
				rcnt++;
		}

		//tree.PrintNodesImgs(T);
		tree.GetNodesTrain(T);
		tree.ClassifyImages(T,"list/TestImages.txt");
	}*/

}

int main()
{
	EntropySampling();
	//RandSampling();
	return 0;
}
