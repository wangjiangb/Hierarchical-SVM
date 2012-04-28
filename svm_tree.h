#include <string>
#include <stack>
#include <math.h>
#include <vector>
#include <fstream>
#include <algorithm>
#include <boost/tuple/tuple.hpp>
#include <boost/tokenizer.hpp>
#include "svm.h"
using namespace std;
using namespace boost;

typedef pair<string, double> LabelPair;
typedef  vector<LabelPair> LabelProbVector;
typedef pair<LabelProbVector, bool*> CombinedProbVector;
typedef vector<bool>::iterator BoolIterator;



void releaseSvmData(svm_node **& node);


//This function assumes all the labels in the expected_labels should be the same.
template <typename T>
int getRightCount(const vector<T> expected_labels, const T& real_label)
{
  double right_labels = 0;
  for (int i = 0; i < expected_labels.size(); ++i)
  {
    if (expected_labels[i] == real_label)
      right_labels++;
  }
  return right_labels;  
}


template <typename T>
class Tree;




template <typename T>
class TreeNode
{
public:
	friend class Tree<T>;
   TreeNode():m_firstchild(NULL), m_nextsibling(NULL), m_parent(NULL), m_level(0)
	{
      m_data = new T();      
	}
 TreeNode(const string& name, const T &d): m_name(name),m_firstchild(NULL), m_nextsibling(NULL), m_parent(NULL)
	{
      m_data = new T(d);      
	}
    ~TreeNode()
    {
      if (m_data!=NULL)
        delete m_data;
    }    
    string GetNodeName()
    {
      return m_name;
    }
	void SetNodeName(const string& name)
	{
      m_name = name;
	}
	void PrintNodeName()
	{
      cout<<m_name<<endl;
	}
	TreeNode<T> *m_firstchild, *m_nextsibling, *m_parent;
    int m_level;
	string m_name;
	T* m_data;	
};

template <typename T>
class Tree
{
 public:
  typedef TreeNode<T>* PTreeNode;
	Tree():m_root(NULL)
	{
	}
  	~Tree()
	{
	}
	TreeNode<T> *Root();
	TreeNode<T> *Begin();    
	TreeNode<T> *End(); 
	TreeNode<T> *Next(TreeNode<T> * node);  //simple in-order iterator    
	void CreateTree( const string& treeFile); //read a tree structure from a file
	void ReleaseTree(); //release all the memories
	int AddNode(TreeNode<T> *target_node, TreeNode<T> *new_node);//insert a new node to a given node, the new node can be a tree
    void AllLeaves(PTreeNode node, vector<PTreeNode>& node_vec); // return all the leaves of the node;
    void AllChildrens(PTreeNode node, vector<PTreeNode>& node_vec); // return all the leaves of a node;
    bool IsLeave(PTreeNode node);
    int Distance(const string& name1, const string& name2)
    {
      TreeNode<T>* node1 = FindByName(name1);
      TreeNode<T>* node2 = FindByName(name2);
      TreeNode<T>* temp;
      if (node1->m_level < node2->m_level)
        swap(node1,node2);
      int diff = node1->m_level - node2->m_level;
      for (int i =0;i<diff; ++i)
      {
        node1 = node1->m_parent;
      }
      while (node1!=node2)
      {
        node1 = node1->m_parent;
        node2 = node2->m_parent;
        diff+=2;
      }
      return diff;        
    }
    
	void SetRoot(PTreeNode node)
	{
		m_root = node;
	}
	TreeNode<T>* FindByName(const string& name);
private:
	TreeNode<T> *m_root;
};
class ClassifierNode;
typedef pair<TreeNode<ClassifierNode>*, int> DataPointer;


class ClassifierNode
{
private:	
	svm_model *model;    
public:
    vector<string> imgPairsVec;//training data files of each node
	svm_node** m_data;// the vector for data 
	 vector<double> m_label; //the associated label for the data 
    vector<bool> m_is_used;    
	int m_num_features;  //number of the the training data  
    vector<double> m_prob;    
    ClassifierNode():m_data(NULL), model(NULL)
	{
	}
   ~ClassifierNode()
   {
      releaseSvmData(m_data);      
      if (model!=NULL)
         delete model;
   }
	vector<double> prob;      
	void Pushdata(string name);//push images into the trainging vector
    string GetDataFile()
    {
      return  imgPairsVec[0];
    }
	void PrintImages();
	bool IsExist(string imgName);
	void ReadDataFile(); 
	void SvmTrain(Tree<ClassifierNode>& tree, TreeNode<ClassifierNode>& node);//the training require the knowledge of the tree and its node   
    void getDataVector( Tree<ClassifierNode>& tree, TreeNode<ClassifierNode>& node, bool isUsed,svm_problem& prob, vector<DataPointer>& isUsedPointers); //get all the data that should be used or should not be used 
	double SvmPredict(svm_node* x);//return 0 or 1
   double SvmPredictProb(svm_node* x, int numChildren);//return the probability

};

void ReadSVMDataFile(const string& filename, svm_node**& data, vector<double>& labels, int& data_size);
void LoadSeedImg(Tree<ClassifierNode>& node, string seedFile, string strategy); //load the seed images for a given node
void PrintNodesImgs(Tree<ClassifierNode>& T);
void GetNodesTrain(Tree<ClassifierNode>& T); //training all the nodes in a tree
void ClassifyImages(Tree<ClassifierNode>& T, const string& testFile, vector<string>& labels);
void ClassifyImagesProb(Tree<ClassifierNode>& T, const string& testFile, vector< LabelProbVector> & all_label_probs);
void ClassifyImagesProb(Tree<ClassifierNode>& T, svm_node** nodeArray, int array_size, vector< LabelProbVector> & all_label_probs);
string ClassifyImg(Tree<ClassifierNode>& T, svm_node* x);
void  ClassifyImgProb(Tree<ClassifierNode>& T, svm_node* x, LabelProbVector& prob_labels);
template <typename T>
bool Tree<T>::IsLeave(PTreeNode node)
{
  return (node->m_firstchild==NULL);  
}

template <typename T>
void Tree<T>::AllLeaves(PTreeNode node,vector<PTreeNode>& node_vec)
{
  node_vec.clear();
  if (IsLeave(node))
  {
    node_vec.push_back(node);
    return;    
  }
  
  PTreeNode nodeT = node;
  nodeT = Next(nodeT);
  while (nodeT!=0&&nodeT!=node->m_nextsibling)
  {
   
    if (IsLeave(nodeT))
      node_vec.push_back(nodeT);
     nodeT = Next(nodeT);
  }
}


template <typename T>
void Tree<T>::AllChildrens(PTreeNode node,vector<PTreeNode>& node_vec)
{
  node_vec.clear();
  PTreeNode nodeChild = node->m_firstchild;
  while (nodeChild!=NULL)
  {    
    node_vec.push_back(nodeChild);
    nodeChild = nodeChild->m_nextsibling;
  }
}
template <typename T>
TreeNode<T>* Tree<T>::FindByName( const string& name )
{
  TreeNode<T> *node = Begin();	
  for (;node!=End();node=Next(node))
  {
    if (node->GetNodeName()==name)
      return node;
  }
  return NULL;
}




template <typename T>
int Tree<T>::AddNode( TreeNode<T> *target_node, TreeNode<T> *new_node )
{	
	if (target_node==new_node)
		return -1;
	if (target_node->m_firstchild == NULL) 
	{
		target_node->m_firstchild = new_node;		
	}else
	{	
		PTreeNode node=target_node->m_firstchild;		
		while (node->m_nextsibling != NULL)
		{
			node = node->m_nextsibling;
		}
        node->m_nextsibling = new_node;
	}
    new_node->m_parent = target_node;
	new_node->m_nextsibling  = NULL;
    new_node->m_level += target_node->m_level+1;    
    
	return 1;
}


template <typename T>
void Tree<T>::ReleaseTree()
{
	typedef vector<TreeNode<T> *> NodeVec;
	TreeNode<T> *node = Begin();
	NodeVec node_vec;
	for (;node!=End();node=Next(node))
	{
		node_vec.push_back(node);
	}
    for (int i = 0; i < node_vec.size(); ++i)
    {
      delete node_vec[i];
    }    
}

template <typename T>
void Tree<T>::CreateTree( const string& filename )
{
  	ifstream TreeNodes(filename.c_str());
    if (m_root!=NULL)
      ReleaseTree();    
	m_root =NULL;
    char_separator<char> sep(" ");
    typedef tokenizer<char_separator<char> > char_tokenizer;
    
    while(!TreeNodes.eof())
	{
      string org;
		int cnt = 0;
		getline(TreeNodes,org);
        char_tokenizer tokens(org, sep);
        string parent, child;
        int count = 0;        
        for (char_tokenizer::iterator tok_iter = tokens.begin(); tok_iter!=tokens.end();++tok_iter)
        {
          if (count==0)
            parent = *tok_iter;
          if (count==1)
            child = *tok_iter;
          count++;          
        }
        if (count<2)
          continue;        
		//put into vectors
        PTreeNode node1;
        PTreeNode node2 = new TreeNode<T>();
        node2->SetNodeName(child);
        if (parent =="#")
        {
          m_root = node2;
        }else
        {
          node1 = FindByName(parent);
          AddNode(node1, node2);          
        }        
    }
}

template <typename T>
TreeNode<T>
* Tree<T>::End()
{
	return NULL;
}

template <typename T>
TreeNode<T>
* Tree<T>::Next( TreeNode<T> * node )
{
	if (node->m_firstchild!=NULL)
		return node->m_firstchild;
	if (node->m_nextsibling!=NULL)
		return node->m_nextsibling;
	while (node!=NULL&&node->m_nextsibling==NULL)
		node = node->m_parent;
	if (node!=NULL)
		return node->m_nextsibling;
	else
		return NULL;
}

template <typename T>
TreeNode<T>
* Tree<T>::Begin()
{
	return Root();
}

template <typename T>
TreeNode<T>
* Tree<T>::Root()
{
	return m_root;
}

double computeEntropy(LabelProbVector& prob_vector,  Tree<ClassifierNode>& tree);
double computeVariance(LabelProbVector& prob_vector, Tree<ClassifierNode>& tree);

