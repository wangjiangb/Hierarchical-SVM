#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/function.hpp>
#include <stdio.h>
#include "svm_tree.h"

using namespace std;
typedef boost::function<float ( LabelProbVector& labels, Tree<ClassifierNode>& tree )> UncertaintyFunction;
//this test the synthetic data 

double getTreeLoss(Tree<ClassifierNode>& treeClassifier, const vector<string> expected_labels, const string& real_label)
{
   double treeLoss = 0;
   for (int i = 0; i < expected_labels.size(); ++i)
   {
      if (expected_labels[i] != real_label)
         treeLoss += treeClassifier.Distance(expected_labels[i], real_label);
   }
   return treeLoss;  


}

//set the first num_seed_image as the seed iamge, and mark the rest of them as the non-seed imagex
void set_seed_images(ClassifierNode * node, int num_seed_imgs)
{
  for (int i=0;i<node->m_num_features;++i)
  {
    if (i<num_seed_imgs)
      node->m_is_used[i] = true;
    else
      node->m_is_used[i] = false;    
  }  
}



double test_classifier(Tree<ClassifierNode>& treeClassifier, const string& tree_file_name, const string test_file_name)
{
  Tree<ClassifierNode> treeClassifier_test;
  treeClassifier_test.CreateTree(tree_file_name);
  LoadSeedImg(treeClassifier_test, test_file_name,"1");
  vector<TreeNode<ClassifierNode> *> node_vec_leave;  
  treeClassifier_test.AllLeaves(treeClassifier_test.Root(), node_vec_leave);
  int total_right_count = 0;
  int total_count = 0;  
  for (int i = 0; i < node_vec_leave.size(); ++i)
  {
    ClassifierNode* node  = (node_vec_leave[i]->m_data);
    string test_file_name = node->imgPairsVec[0];
    string expected_label = node_vec_leave[i]->m_name;    
    //cout<<test_file_name<<endl;
    vector<string> labels;    
    ClassifyImages(treeClassifier, test_file_name,labels);
    int right_count = getRightCount(labels, expected_label);    
    total_right_count += right_count;
    total_count += labels.size();    
  }
  return double(total_right_count)/total_count;
  treeClassifier.ReleaseTree();  
}

void set_seed_images_classifier(Tree<ClassifierNode>& treeClassifier, int num_seed_imgs)
{
  vector<TreeNode<ClassifierNode> *> node_vec_leave;
  treeClassifier.AllLeaves(treeClassifier.Root(), node_vec_leave);
  for (int i = 0; i < node_vec_leave.size(); ++i)
  {
    ClassifierNode* node  = (node_vec_leave[i]->m_data);
    set_seed_images((node_vec_leave[i]->m_data), num_seed_imgs);    
  }
  
}

void ActiveLearning( vector<double> &accuracy_array, Tree<ClassifierNode> &treeClassifier,const UncertaintyFunction& func , const string& tree_filename, const string& seed_filename, const string& test_filename) 
{

   accuracy_array.clear();
   svm_problem prob;
   vector<DataPointer> isUsedPointers;
   vector<LabelProbVector>  all_label_probs;
   treeClassifier.CreateTree(tree_filename);
   LoadSeedImg(treeClassifier, seed_filename,"1");  
   GetNodesTrain(treeClassifier);
   //accuracy_array.push_back(test_classifier(treeClassifier, "synthetic_data/TreeFile.txt", "synthetic_data/test_images.txt"));
   // //load another tree just for tesing purpose
   // cout<<test_classifier(treeClassifier, "synthetic_data/TreeFile.txt", "synthetic_data/test_images.txt")<<endl;
   int  num_seed_imgs_array[] = {1,5, 10, 20, 40, 80, 160, 320, 640, 1000};  
   int size_seed_array = 3;
   int num_labeled_data = 400;
   set_seed_images_classifier(treeClassifier, num_seed_imgs_array[0]);
   for (int i = 0; i < num_labeled_data; ++i)
   {    
      GetNodesTrain(treeClassifier);
      double accuracy = test_classifier(treeClassifier, tree_filename, test_filename);
      cout<<"accuracy:"<<accuracy<<endl;
      accuracy_array.push_back(accuracy);
      ClassifierNode* node = treeClassifier.Root()->m_data;    
      node->getDataVector(treeClassifier,*treeClassifier.Root(),false,prob, isUsedPointers);    
      ClassifyImagesProb(treeClassifier,prob.x,prob.l,all_label_probs);
      double max_ent = 0;
      int selected_index = -1;    
      for (int j=0;j< all_label_probs.size(); ++j)
      {
         double entropy = func(all_label_probs[j],treeClassifier);
         if (entropy> max_ent)
         {          
            max_ent = entropy;
            selected_index = j;
         }
      }
      TreeNode<ClassifierNode>  * node_selected = isUsedPointers[selected_index].first;      
      int index = isUsedPointers[selected_index].second;
      for_each( all_label_probs[selected_index].begin(), all_label_probs[selected_index].end(),[](LabelPair& p){
          cout<<p.first<<":"<<p.second<<" ";
        });
      cout<<endl;      
      cout<<"selected index "<<selected_index<<" "<< node_selected->m_name<<":"<<index<<endl;
      node_selected->m_data->m_is_used[index] = true;
      delete[] prob.x;
      delete[] prob.y;
   }
}

void UniformSamplingActiveLearning( vector<double> &accuracy_array, Tree<ClassifierNode> &treeClassifier ) 
{  
   accuracy_array.clear();
   svm_problem prob;
   vector<DataPointer> isUsedPointers;
   vector<LabelProbVector>  all_label_probs;
   treeClassifier.CreateTree("synthetic_data/TreeFile_new.txt");
   LoadSeedImg(treeClassifier, "synthetic_data/SeedImages_new.txt","1");  
   GetNodesTrain(treeClassifier);
   int  num_seed_imgs_array[] = {1,5, 10, 20, 40, 80, 160, 320, 640, 1000};  
   int size_seed_array = 6;
   for (int i=0;i<size_seed_array;++i)
   {
      set_seed_images_classifier(treeClassifier, num_seed_imgs_array[i]);
      GetNodesTrain(treeClassifier);
      double accuracy = test_classifier(treeClassifier, "synthetic_data/TreeFile_new.txt", "synthetic_data/test_images_new.txt");
      cout<<"accuracy:"<<accuracy<<endl;
      accuracy_array.push_back(accuracy);
   }

}

void printUsage()
{
  cout<<"usage activeLearning TreeFile seedFile testFile outputFile mode"<<endl;  
}


int main(int argc, char** argv)
{
  Tree<ClassifierNode> treeClassifier;
  TreeNode<ClassifierNode> *node1;
  vector<double> accuracy_array;
  if (argc<5)
  {
    printUsage();
    return -1;    
  }
  
  string  tree_filename = argv[1];
  string  seed_filename = argv[2];
  string  test_filename = argv[3];
  string  output_filename = argv[4];
  int mode = atoi(argv[5]);
  if (mode==0)
    ActiveLearning(accuracy_array, treeClassifier,&computeEntropy, tree_filename, seed_filename, test_filename);
  else if (mode==1)
    ActiveLearning(accuracy_array, treeClassifier,&computeVariance, tree_filename, seed_filename, test_filename);
  else
  {
    printUsage();
    return -1;
  }
  ofstream outputFile(output_filename);
  for (int i = 0; i < accuracy_array.size(); ++i)
  {
     outputFile<<accuracy_array[i]<<endl;    
  }
  
  //UniformSamplingActiveLearning(accuracy_array, treeClassifier);

  

//   accuracy_array.clear();
//   set_seed_images_classifier(treeClassifier, num_seed_imgs_array[3]);
//   for (int i = 0; i < num_labeled_data; ++i)
//   {    
//     GetNodesTrain(treeClassifier);
//     double accuracy = test_classifier(treeClassifier, "synthetic_data/TreeFile.txt", "synthetic_data/test_images.txt");
//     cout<<"accuracy:"<<accuracy<<endl;
//     accuracy_array.push_back(accuracy);
//     ClassifierNode* node = treeClassifier.Root()->m_data;    
//     node->getDataVector(treeClassifier,*treeClassifier.Root(),false,prob, isUsedPointers);    
//     ClassifyImagesProb(treeClassifier,prob.x,prob.l,all_label_probs);
//     double max_ent = 0;
//     int selected_index = -1;    
//     for (int j=0;j< all_label_probs.size(); ++j)
//     {
//       double entropy = computeVariance(all_label_probs[j], treeClassifier);
//       if (entropy> max_ent)
//       {          
//         max_ent = entropy;
//         selected_index = j;
//       }
//     }
//     TreeNode<ClassifierNode>  * node_selected = isUsedPointers[selected_index].first;
//     int index = isUsedPointers[selected_index].second;    
//     cout<<"selected index "<<selected_index<<" "<< node_selected->m_name<<":"<<index<<endl;
//     node_selected->m_data->m_is_used[index] = true;
//     delete[] prob.x;
//     delete[] prob.y;
//     node->getDataVector(treeClassifier,*treeClassifier.Root(),true,prob, isUsedPointers);
//     cout<<"num of training data "<<prob.l<<endl;
//     delete[] prob.x;
//     delete[] prob.y;
// 
//   }
//   ofstream outputFile2("acc_variance.txt");
//   for (int i = 0; i < accuracy_array.size(); ++i)
//   {
//     outputFile2<<accuracy_array[i]<<endl;    
//   }

  return 0;
}
