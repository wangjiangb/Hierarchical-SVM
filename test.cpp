#include <gtest/gtest.h>
#include "svm_tree.h"
#include <iostream>
#include <tr1/unordered_set>
using namespace std;

TEST(Tree,tree_traverse)
{
	Tree<double> treeDouble;
	TreeNode<double> *node1 = new TreeNode<double>();
	TreeNode<double> *node2 = new TreeNode<double>();
	TreeNode<double> *node3 = new TreeNode<double>();
	TreeNode<double> *node4 = new TreeNode<double>();
	TreeNode<double> *node5 = new TreeNode<double>();
	node1->SetNodeName("A");
	node2->SetNodeName("B");
	node3->SetNodeName("C");
	node4->SetNodeName("D");
	node5->SetNodeName("E");
	treeDouble.SetRoot(node1);
    treeDouble.AddNode(node1, node2);
	treeDouble.AddNode(node1, node3);
	treeDouble.AddNode(node2, node4);
    treeDouble.AddNode(node2, node5);
    node1 = treeDouble.Begin();
    vector<TreeNode<double> *> node_vec;
    int node_count = 0;
    for (;node1!=treeDouble.End();node1=treeDouble.Next(node1))
    {
      node_vec.push_back(node1);
      node_count++;
    }
    ASSERT_EQ(5, node_count);
	ASSERT_EQ("A", node_vec[0]->GetNodeName());
    ASSERT_EQ("B", node_vec[1]->GetNodeName());
    ASSERT_EQ("D", node_vec[2]->GetNodeName());
    ASSERT_EQ("E", node_vec[3]->GetNodeName());
    ASSERT_EQ("C", node_vec[4]->GetNodeName());
    treeDouble.ReleaseTree();
}
TEST(Tree,findName)
{
	Tree<double> treeDouble;
	TreeNode<double> *node1 = new TreeNode<double>();
	TreeNode<double> *node2 = new TreeNode<double>();
	TreeNode<double> *node3 = new TreeNode<double>();
	TreeNode<double> *node4 = new TreeNode<double>();
	TreeNode<double> *node5 = new TreeNode<double>();
	node1->SetNodeName("A");
	node2->SetNodeName("B");
	node3->SetNodeName("C");
	node4->SetNodeName("D");
	node5->SetNodeName("E");
    treeDouble.SetRoot(node1);
    treeDouble.AddNode(node1, node2);
	treeDouble.AddNode(node1, node3);
	treeDouble.AddNode(node2, node4);
    treeDouble.AddNode(node2, node5);
    node1 = treeDouble.FindByName("A");
    ASSERT_EQ("A", node1->GetNodeName());
    node1 = treeDouble.FindByName("B");
    ASSERT_EQ("B", node1->GetNodeName());
    node1 = treeDouble.FindByName("C");
    ASSERT_EQ("C", node1->GetNodeName());
    node1 = treeDouble.FindByName("D");
    ASSERT_EQ("D", node1->GetNodeName());
    node1 = treeDouble.FindByName("E");
    ASSERT_EQ("E", node1->GetNodeName());
}
TEST(ReadTreeStructure, test_correctness)
{
    Tree<double> treeDouble;
    TreeNode<double> *node1;
    treeDouble.CreateTree("treeFile.txt");
    node1 = treeDouble.Begin();
    vector<TreeNode<double> *> node_vec;
    int node_count = 0;
    for (;node1!=treeDouble.End();node1=treeDouble.Next(node1))
    {
      node_vec.push_back(node1);
      node_count++;
    }
    ASSERT_EQ(5, node_count);
	ASSERT_EQ("A", node_vec[0]->GetNodeName());
    ASSERT_EQ("B", node_vec[1]->GetNodeName());
    ASSERT_EQ("D", node_vec[2]->GetNodeName());
    ASSERT_EQ("E", node_vec[3]->GetNodeName());
    ASSERT_EQ("C", node_vec[4]->GetNodeName());
    treeDouble.ReleaseTree();
    
}

TEST(ListAllLeaves, test_correctness)
{
    Tree<double> treeDouble;
    TreeNode<double> *node1;
    treeDouble.CreateTree("treeFile.txt");
    node1 = treeDouble.Begin();
    vector<TreeNode<double> *> node_vec, node_vec_leave;
    tr1::unordered_set<string> node_set;    
    int node_count = 0;
    for (;node1!=treeDouble.End();node1=treeDouble.Next(node1))
    {
      node_vec.push_back(node1);
      node_count++;
    }
    treeDouble.AllLeaves(node_vec[0], node_vec_leave);
    ASSERT_EQ(3, node_vec_leave.size());
    node_set.clear();
    for (int i = 0; i < node_vec_leave.size(); ++i)
    {
      node_set.insert(node_vec_leave[i]->GetNodeName());      
    }
    ASSERT_TRUE(node_set.find("D")!=node_set.end());
    ASSERT_TRUE(node_set.find("E")!=node_set.end());
    
    ASSERT_TRUE(node_set.find("C")!=node_set.end());    
    ASSERT_TRUE(node_set.find("B")==node_set.end());
    treeDouble.AllLeaves(node_vec[1], node_vec_leave);
    ASSERT_EQ(2, node_vec_leave.size());
    node_set.clear();
    for (int i = 0; i < node_vec_leave.size(); ++i)
    {
      node_set.insert(node_vec_leave[i]->GetNodeName());      
    }
    ASSERT_TRUE(node_set.find("D")!=node_set.end());
    ASSERT_TRUE(node_set.find("E")!=node_set.end());
    ASSERT_TRUE(node_set.find("C")==node_set.end());    
    ASSERT_TRUE(node_set.find("B")==node_set.end());
    
}

TEST(TESTLOADING, test_loading)
{
  ClassifierNode node;
  node.Pushdata("1.txt");
  node.ReadDataFile();  
  ASSERT_EQ(2, node.m_num_features);
  ASSERT_EQ(1, node.m_data[0][0].value);
  ASSERT_EQ(2, node.m_data[0][1].value);
  ASSERT_EQ(3, node.m_data[0][2].value);
  ASSERT_EQ(4, node.m_data[0][3].value);
  ASSERT_EQ(-1, node.m_data[0][4].index);
  ASSERT_EQ(2, node.m_data[1][0].value);
  ASSERT_EQ(3, node.m_data[1][1].value);
  ASSERT_EQ(4, node.m_data[1][2].value);
  ASSERT_EQ(5, node.m_data[1][3].value);
  ASSERT_EQ(-1, node.m_data[1][4].index);
  
}

TEST(TestSeedsImages, test_loading)
{
  Tree<ClassifierNode> treeClassifier;
  TreeNode<ClassifierNode> *node1;
  treeClassifier.CreateTree("treeFile.txt");
  LoadSeedImg(treeClassifier, "list/SeedImages.txt","1");
  node1 = treeClassifier.FindByName("C");
  ASSERT_EQ("list/1.txt",node1->m_data.GetDataFile());
  node1 = treeClassifier.FindByName("D");
  ASSERT_EQ("list/2.txt",node1->m_data.GetDataFile());
  node1 = treeClassifier.FindByName("E");
  ASSERT_EQ("list/3.txt",node1->m_data.GetDataFile());  
}


TEST(TestSeedsImages, test_training)
{
  Tree<ClassifierNode> treeClassifier;
  TreeNode<ClassifierNode> *node1;
  treeClassifier.CreateTree("treeFile.txt");
  LoadSeedImg(treeClassifier, "list/SeedImages.txt","1");
  GetNodesTrain(treeClassifier);
  
}


TEST(TestSeedsImages, test_predict)
{
  Tree<ClassifierNode> treeClassifier;
  TreeNode<ClassifierNode> *node1;
  treeClassifier.CreateTree("treeFile.txt");
  LoadSeedImg(treeClassifier, "list/SeedImages.txt","1");
  GetNodesTrain(treeClassifier);
  vector<string> labels;  
  ClassifyImages(treeClassifier, "test.txt",labels);
  ASSERT_EQ(3,labels.size());
  ASSERT_EQ("C",labels[0]);
  ASSERT_EQ("D",labels[1]);
  ASSERT_EQ("E",labels[2]);  
}
