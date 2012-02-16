#include <iostream>
#include <fstream>
#include <vector>
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
        pnode->m_data.Pushdata(filename);      
    }
    
}

void GetNodesTrain( Tree<ClassifierNode>& T )
{
  Tree<ClassifierNode>::PTreeNode pnode = T.Begin();
  for (;pnode!=T.End();pnode=T.Next(pnode))
  {
    if (!T.IsLeave(pnode))
      pnode->m_data.SvmTrain(T, *pnode);    
  }  
}

//the file is assumed to have  target label, but we do not use it 
void ClassifyImages(Tree<ClassifierNode>& T, const string& testFile, vector<string>& labels)
{  
  	max_line_len = 1024;
    labels.clear();
    int total = 0;    
    FILE* input = fopen(testFile.c_str(), "r");
    int max_nr_attr = 64;
    struct svm_node *x;
	line = (char *)malloc(max_line_len*sizeof(char));
    x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
	while(readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			++i;
		}
		x[i].index = -1;
        
        string predict_label_str = ClassifyImg(T, x);
        //fprintf(output,"%s\n",predict_label_str);
        labels.push_back(predict_label_str);        
		++total;
	}
}

std::string ClassifyImg( Tree<ClassifierNode>& T, svm_node* x )
{
  
  PTreeNode pnode;
  vector<PTreeNode> children;  
  
  pnode = T.Root();
  while (!T.IsLeave(pnode))
  {
    int label = pnode->m_data.SvmPredict(x);
    T.AllChildrens(pnode, children);
    pnode = children[label];    
  }
  return pnode->GetNodeName();  
}






void ClassifierNode::ReadDataFile()  //this currently only reads one line 
{
  if (m_data!=NULL)
  {
    delete[] m_data;
    m_data = NULL;    
  }
  if (m_label!=NULL)
  {
    delete [] m_label;
    m_label = NULL;    
  }
  
  int elements, max_index, inst_max_index, i, j;
  string filename = imgPairsVec[0];
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
  m_num_features = 0;
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
    ++m_num_features;    
  }
  rewind(fp);
  m_data = new svm_node*[m_num_features];
  m_label = new double[m_num_features];
  svm_node* x_space= new svm_node[elements];
  //the trick of this code, read the data into one large array, while the pointer point to differnt part of this array
  max_index = 0;
  j=0;
  for(i=0;i<m_num_features;i++)
  {
    inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
    readline(fp);
    m_data[i] = &x_space[j];
    label = strtok(line," \t\n");
    if(label == NULL) // empty line
      exit_input_error(i+1);

    m_label[i] = strtod(label,&endptr);
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
  
}

void ClassifierNode::SvmTrain(Tree<ClassifierNode>& tree, TreeNode<ClassifierNode>& node)
{
  svm_problem prob;
  typedef TreeNode<ClassifierNode>* PSvmTreeNode;
  if (tree.IsLeave(&node))
    return;
  cout<<": training node "<<node.GetNodeName()<<endl;
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
      if (leaves[k]->m_data.m_data==NULL)
      {
        leaves[k]->m_data.ReadDataFile();      
      }
      prob.l += leaves[k]->m_data.m_num_features;
    }    
  }
  prob.y = new double[prob.l];
  prob.x = new svm_node*[prob.l];
  int element = 0;
  //go over all the children
  for (size_t i =0; i< children.size(); ++i)
  {
    tree.AllLeaves(children[i],leaves);
    //get the data from the leaf nodes of each child
    for (size_t k =0;k< leaves.size(); ++k)
    {    
      childNode = &(children[i]->m_data);
      if (leaves[k]->m_data.m_data==NULL)
      {
        leaves[k]->m_data.ReadDataFile();      
      }
      for (size_t j=0;j<leaves[k]->m_data.m_num_features;++j)
      {
        prob.x[element+j] = leaves[k]->m_data.m_data[j];
        prob.y[element+j] = i;        
      }    
      element +=leaves[k]->m_data.m_num_features;
    }    
  }
  cout <<"number of training data "<<prob.l<<endl;
  model = svm_train(&prob,&param);
  cout<<": training finished"<<endl;
  delete [] prob.x;
  delete [] prob.y;  
}

double ClassifierNode::SvmPredict( svm_node* x )
{
  return svm_predict(model,x);
}

double ClassifierNode::SvmPredictProb( svm_node* x )
{
  if (m_prob!=NULL)
  {
    delete[] m_prob;
    m_prob =  NULL;
  }
  m_prob  = new double[model->nr_class];
  double label = svm_predict_probability(model,x,m_prob);  
}
