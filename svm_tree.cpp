#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <algorithm>
#include <string>
#include <time.h>
#include <boost/tokenizer.hpp>
#include "svm_tree.h"
#include "svm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
using namespace std;
using namespace boost;
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
typedef TreeNode<ClassifierNode>* PTreeNode;

 











void ClassifierNode::Pushdata( string name)
{
  imgPairsVec.push_back(name);
}

void ClassifierNode::PrintImages()
{
  for(string::size_type i=0; i<imgPairsVec.size(); i++)
  {
    //if(imgPairsVec[i].label == 1)
    cout<<imgPairsVec[i]+"\t";
  }
}

bool ClassifierNode::IsExist( string imgName )
{
  bool lab = false;
  for(string::size_type i=0; i<imgPairsVec.size(); i++)
  {
    if(imgName == imgPairsVec[i])
    {
      lab = true;
      break;
    }	
  }
  return lab;
}
static char *line = NULL;
static int max_line_len;
static char* readline(FILE *input)
{
  int len;
	
  if(fgets(line,max_line_len,input) == NULL)
    return NULL;

  while(strrchr(line,'\n') == NULL)
  {
    max_line_len *= 2;
    line = (char *) realloc(line,max_line_len);
    len = (int) strlen(line);
    if(fgets(line+len,max_line_len-len,input) == NULL)
      break;
  }
  return line;
}

void exit_input_error(int line_num)
{
  fprintf(stderr,"Wrong input format at line %d\n", line_num);
  exit(1);
}

void LoadSeedImg( Tree<ClassifierNode>& tree, string seedFile, string strategy )
{  
  ifstream TreeNodes(seedFile.c_str());   
  char_separator<char> sep(" ");
  typedef tokenizer<char_separator<char> > char_tokenizer;
  while(!TreeNodes.eof())
  {
    string org;
    int cnt = 0;
    getline(TreeNodes,org);
    char_tokenizer tokens(org, sep);
    string label, filename;
    int count = 0;
    for (char_tokenizer::iterator tok_iter = tokens.begin(); tok_iter!=tokens.end();++tok_iter)
    {
      if (count==0)
        label = *tok_iter;
      if (count==1)
        filename = *tok_iter;
      count++;          
    }
    if (count<2)
      continue;
    PTreeNode pnode;
    pnode = tree.FindByName(label);
    if (pnode!=NULL)
      pnode->m_data->Pushdata(filename);      
  }
    
}

void GetNodesTrain( Tree<ClassifierNode>& T )
{
  Tree<ClassifierNode>::PTreeNode pnode = T.Begin();
  for (;pnode!=T.End();pnode=T.Next(pnode))
  {
    if (!T.IsLeave(pnode))
      pnode->m_data->SvmTrain(T, *pnode);    
  }  
}

//the file is assumed to have  target label, but we do not use it 
void ClassifyImages(Tree<ClassifierNode>& T, const string& testFile, vector<string>& labels)
{ 
  struct svm_node ** data=NULL;
  vector<double> realLabels;    
  int num_data;
  ReadSVMDataFile(testFile,data, realLabels,num_data);
  for (int i =0; i<num_data;++i)
  {
    string predict_label_str = ClassifyImg(T, data[i]);
    //fprintf(output,"%s\n",predict_label_str);
    labels.push_back(predict_label_str);       
  }	
  releaseSvmData(data);
}

std::string ClassifyImg( Tree<ClassifierNode>& T, svm_node* x )
{
  
  PTreeNode pnode;
  vector<PTreeNode> children;  
  
  pnode = T.Root();
  while (!T.IsLeave(pnode))
  {
    int label = pnode->m_data->SvmPredict(x);
    T.AllChildrens(pnode, children);
    pnode = children[label];    
  }
  return pnode->GetNodeName();  
}


void ClassifyImagesProb(Tree<ClassifierNode>& T, const string& testFile, vector< vector<LabelPair> > & all_label_probs)
{  
  struct svm_node ** data=NULL;
  vector<double> realLabels;    
  int num_data;
  ReadSVMDataFile(testFile,data, realLabels,num_data);
  ClassifyImagesProb(T, data, num_data,all_label_probs);
}

void ClassifyImagesProb( Tree<ClassifierNode>& T, svm_node** data, int num_data, vector< LabelProbVector> & all_label_probs )
{
  all_label_probs.clear();
  for (int i =0; i<num_data;++i)
  {
    vector<LabelPair> prob_labels;
    ClassifyImgProb(T, data[i], prob_labels);
    string predict_label_str = ClassifyImg(T, data[i]);
    all_label_probs.push_back(prob_labels);
    //fprintf(output,"%s\n",predict_label_str);		
  }
}


void  ClassifyImgProb(Tree<ClassifierNode>& T, svm_node* x_in, vector<LabelPair>& prob_labels)
{
  int num_features=0;  
  while (true)
  {
    if (x_in[num_features].index==-1)
      break;
    num_features++;
  }
  svm_node* x = new svm_node[num_features];
  for (int i = 0; i < num_features; ++i)
  {
    x[i] = x_in[i];
  }
  PTreeNode pnode,  current_node;
  double current_prob = 1;
  vector<PTreeNode> children;  
  prob_labels.clear();
  pnode = T.Root();
  prob_labels.push_back(make_pair(pnode->m_name, 1));
  int index = 0;  
  while (!T.IsLeave(pnode))
  {
    T.AllChildrens(pnode, children);
    int label = pnode->m_data->SvmPredictProb(x,children.size());   
    for (int i = 0; i < children.size(); ++i)
    {
      current_node = children[i];
      prob_labels.push_back(make_pair(current_node->GetNodeName(), current_prob*pnode->m_data->m_prob[i]));
    }
    vector<LabelPair>::iterator iter (&prob_labels[index]);    
    prob_labels.erase(iter, iter+1);    
    index  = 0;    
    pnode = T.FindByName(prob_labels[index].first);
    current_prob = prob_labels[index].second;
    while (T.IsLeave(pnode)&&index<prob_labels.size())
    {
      ++index;
      pnode = T.FindByName(prob_labels[index].first);
      current_prob = prob_labels[index].second;
    }
    if (index>=prob_labels.size())
      break;    
  }
  delete[] x;
}



double computeEntropy( LabelProbVector& prob_vector,  Tree<ClassifierNode>& tree )
{
  double entropy = 0;  
  for (int i = 0; i < prob_vector.size(); ++i)
  {
    if (abs(prob_vector[i].second)>1e-8)
      entropy += prob_vector[i].second*log(prob_vector[i].second);
  }
  return -entropy;  
}


double computeVariance(LabelProbVector& prob_vector, Tree<ClassifierNode>& tree)
{
  double variance = 0 ;
  double distance;
  for (int i = 0; i < prob_vector.size(); ++i)
  {
    for (int j=i+1;j<prob_vector.size(); ++j)
    {
      distance = tree.Distance(prob_vector[i].first, prob_vector[j].first);
      variance += exp(distance/1.5)*prob_vector[i].second* prob_vector[j].second;
    }    
  }
  return variance;
}








void ReadSVMDataFile(const string& filename, svm_node**& data, vector<double>& labels, int& data_size)
{
  if (data!=NULL)
  {
    delete[] data;
    data = NULL;    
  }
	

  int elements, max_index, inst_max_index, i, j;	
  FILE *fp = fopen(filename.c_str(),"r");  
  char *endptr;
  char *idx, *val, *label;
  
  if(fp == NULL) 
  {
    fprintf(stderr,"can't open input file %s\n",filename.c_str());
    return;
  }
  max_line_len = 1024;
  line = Malloc(char,max_line_len);
  data_size = 0;
  elements= 0;
  // the following code are used to check the number of the features
  while(readline(fp)!=NULL)
  {
    char *p = strtok(line," \t"); // label

    // features
    while(1)
    {
      p = strtok(NULL," \t");
      if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
        break;
      ++elements;
    }
    ++elements;
    ++data_size;    
  }
  rewind(fp);
  data = new svm_node*[data_size];
  labels.resize(data_size);
	
  svm_node* x_space= new svm_node[elements];
  //the trick of this code, read the data into one large array, while the pointer point to differnt part of this array
  max_index = 0;
  j=0;
  for(i=0;i<data_size;i++)
  {		
    inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
    readline(fp);
    data[i] = &x_space[j];
    label = strtok(line," \t\n");
    if(label == NULL) // empty line
      exit_input_error(i+1);

    labels[i] = strtod(label,&endptr);
    if(endptr == label || *endptr != '\0')
      exit_input_error(i+1);

    while(1)
    {
      idx = strtok(NULL,":");
      val = strtok(NULL," \t");

      if(val == NULL)
        break;

      errno = 0;
      x_space[j].index = (int) strtol(idx,&endptr,10);
      if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
        exit_input_error(i+1);
      else
        inst_max_index = x_space[j].index;

      errno = 0;
      x_space[j].value = strtod(val,&endptr);
      if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
        exit_input_error(i+1);

      ++j;
    }

    if(inst_max_index > max_index)
      max_index = inst_max_index;
    x_space[j++].index = -1;
  }
  fclose(fp);  
}

void releaseSvmData( svm_node **& node )
{
  if (node!=NULL)
  {
    delete[] node[0];
    delete[] node;		
  }
}

void ClassifierNode::ReadDataFile()  
{
  ReadSVMDataFile(imgPairsVec[0],m_data, m_label, m_num_features);
  m_is_used.resize(m_num_features);
  for (int i =0;i<m_num_features;++i)
  {
    m_is_used[i] = true;
  }
   
}

void ClassifierNode::SvmTrain(Tree<ClassifierNode>& tree, TreeNode<ClassifierNode>& node)
{
  svm_problem prob;
  vector<DataPointer> isUsedPointers;
  getDataVector(tree,node,true,prob,isUsedPointers);
  cout<<": training node "<<node.GetNodeName()<<endl;
  
  //cout <<"number of training data "<<prob.l<<endl;
  svm_parameter param;
  //svm setting;
  param.svm_type = 0;
  param.kernel_type = 0;//linear kernel
  param.degree = 3;
  param.gamma = 0;
  param.coef0 = 0;
  param.nu = 0.5;
  param.cache_size = 100;
  param.C = 1;
  param.eps = 1e-3;
  param.p = 0.1;
  param.shrinking = 1;
  param.probability = 1;
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;

  if(param.gamma == 0) param.gamma = 0.5;

  model = svm_train(&prob,&param);
  //cout<<": training finished"<<endl;
  delete [] prob.x;
  delete [] prob.y;  
}

double ClassifierNode::SvmPredict( svm_node* x )
{
  return svm_predict(model,x);
}

double ClassifierNode::SvmPredictProb( svm_node* x, int numChildren )
{
  m_prob.resize(numChildren);
  double label = svm_predict_probability(model,x,&m_prob[0]);  
}

void ClassifierNode::getDataVector( Tree<ClassifierNode>& tree, TreeNode<ClassifierNode>& node, bool isUsed,svm_problem& prob, vector<DataPointer>& isUsedPointers )
{
  typedef TreeNode<ClassifierNode>* PSvmTreeNode;
  isUsedPointers.clear();    
  if (tree.IsLeave(&node))
    return;

  vector<PSvmTreeNode> children;
  vector<PSvmTreeNode> leaves;  
  ClassifierNode*  childNode;
  tree.AllChildrens(&node, children);
  prob.l = 0;
  // the following for only count the total number of the training data
  for (size_t i =0; i< children.size(); ++i)
  {
    tree.AllLeaves(children[i],leaves);
    for (size_t k =0;k< leaves.size(); ++k)
    {
      if (leaves[k]->m_data->m_data==NULL)
      {
        leaves[k]->m_data->ReadDataFile();      
      }
      for (size_t j=0;j<leaves[k]->m_data->m_num_features;++j)
      {       
        if (leaves[k]->m_data->m_is_used[j]==isUsed)
        {          
          prob.l++;          
        }
      }
    }    
  }
  prob.y = new double[prob.l];
  prob.x = new svm_node*[prob.l];
  int element = 0;
  int num_of_features_i;
  //go over all the children
  for (size_t i =0; i< children.size(); ++i)
  {
    tree.AllLeaves(children[i],leaves);
    num_of_features_i = 0;
    //get the data from the leaf nodes of each child
    for (size_t k =0;k< leaves.size(); ++k)
    {    
      childNode = (children[i]->m_data);
      if (leaves[k]->m_data->m_data==NULL)
      {
        leaves[k]->m_data->ReadDataFile();      
      }
      for (size_t j=0;j<leaves[k]->m_data->m_num_features;++j)
      {
        if (leaves[k]->m_data->m_is_used[j]==isUsed)
        {
          prob.x[element] = leaves[k]->m_data->m_data[j];
          prob.y[element] = i;
          ++num_of_features_i;          
          isUsedPointers.push_back(make_pair(leaves[k], j));
          element++;          
        }                   
      }
	  cout<<"the number of training data for "<<i<<" is:"<<num_of_features_i<<endl;    	
    }   
  }
}
