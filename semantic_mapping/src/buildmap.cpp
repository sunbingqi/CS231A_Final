#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"

#include <Eigen/Geometry>
#include <boost/format.hpp>

using namespace std;
using namespace Eigen;

const int ROW = 375;
const int COL = 1242;

/*
  seqNum = the sequence number of the KITTI odometry dataset
  calib_dir = [path to calib.txt]
  data_dir = [directory] to .bin files.
  pose_dir = [path to ground truth poses [seqNum].txt]
  image_dir = [path to prediction rgb images]
  output_dir = [path to output 3d positions and rgb colors txt]
*/
const string seqNum = "00";
const string calib_dir = "../lidar_mapping_data/calib.txt";
const string data_dir = "../lidar_mapping_data/" + seqNum + "/bin/";
const string pose_dir = "../lidar_mapping_data/" + seqNum + "/" + seqNum + ".txt";
const string image_dir = "../lidar_mapping_data/" + seqNum + "/rgb/";
const string output_dir = "../lidar_mapping_data/" + seqNum + "/frames/";

// bool values for different purposes.
const bool fullMap = false;
const int partMap = 100;

MatrixXf readbinfile(const string dir);

template <size_t Rows, size_t Cols>
Eigen::Matrix<float, Rows, Cols> ReadCalibrationVariable(
  const std::string& prefix, std::istream& file);

vector<Matrix4f> ReadPoses(const string filename);


int main(int argc, char const *argv[]){

  // KITTI intrinsic & extrinsic, using left image.
  std::ifstream calib_file(calib_dir);
  MatrixXf K;
  K.resize(3, 4);
  K = ReadCalibrationVariable<3, 4>("P0:", calib_file);
  Matrix4f T = Matrix4f::Identity();
  T.block(0, 0, 3, 4) = ReadCalibrationVariable<3, 4>("Tr:", calib_file);

  cv::Mat map, prediction;
  
  vector<Matrix4f> v = ReadPoses(pose_dir);
  int size;
  if (fullMap){
    cout << "Mapping with whole dataset." << endl;
    size = v.size();
  } else {
    cout << "Mapping with " << partMap << " point clouds." << endl;
    size = partMap;
  }

  for (int k = 0; k < size; k++){
    cout << "Processing " << k + 1 << "-th frame." << endl;
    char buff1[100];
    snprintf(buff1, sizeof(buff1), "%006d.bin", k);
    string file = data_dir + string(buff1);
    char buff2[100];
    snprintf(buff2, sizeof(buff2), "%006d.png", k);
    string pred = image_dir + string(buff2);
    prediction = cv::imread(pred, cv::IMREAD_COLOR);
    
    Matrix4f pose = v[k];
    MatrixXf data = readbinfile(file);
    MatrixXf camPts = K * T * data;

    ofstream output;
    output.open (output_dir + "frame" + to_string(k) + ".txt");
    
    for (int i = 0; i < camPts.cols(); i++){
      Vector4f point(data(0, i), data(1, i), data(2, i), 1);
      Vector4f pt = pose * T * point;
      if (camPts(2, i) > 0){
        float x = camPts(0, i) / camPts(2, i);
        float y = camPts(1, i) / camPts(2, i);
        if ( (x > 0 && x < COL - 0.5) && (y > 0 && y < ROW - 0.5) ){
          cv::Vec3b intensity = prediction.at<cv::Vec3b>(y, x);
          output << to_string(pt[0]) << "\t";
          output << to_string(pt[1]) << "\t";
          output << to_string(pt[2]) << "\t";
          output << to_string(intensity.val[2]) << "\t";
          output << to_string(intensity.val[1]) << "\t";
          output << to_string(intensity.val[0]) << "\n";
        }
      }
    }
    output.close();
  }
  return 0;
}

MatrixXf readbinfile(const string dir){

  ifstream fin(dir.c_str(), ios::binary);
  assert(fin);

  fin.seekg(0, ios::end);
  const size_t num_elements = fin.tellg() / sizeof(float);
  fin.seekg(0, ios::beg);

  vector<float> l_data(num_elements);
  fin.read(reinterpret_cast<char*>(&l_data[0]), num_elements*sizeof(float));

  MatrixXf data = Map<MatrixXf>(l_data.data(), 4, l_data.size()/4);

  return data;
}

template <size_t Rows, size_t Cols>
Eigen::Matrix<float, Rows, Cols> ReadCalibrationVariable(
    const std::string& prefix, std::istream& file) {
  // rewind
  file.seekg(0, std::ios::beg);
  file.clear();

  double buff[20] = {0};

  int itter = 0;

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;

    size_t found = line.find(prefix);
    if (found != std::string::npos) {
      std::cout << prefix << std::endl;
      std::cout << line << std::endl;
      std::stringstream stream(
          line.substr(found + prefix.length(), std::string::npos));
      while (!stream.eof()) {
        stream >> buff[itter++];
      }
      std::cout << std::endl;
      break;
    }
  }

  Eigen::Matrix<float, Rows, Cols> results;
  for (size_t i = 0; i < Rows; i++) {
    for (size_t j = 0; j < Cols; j++) {
      results(i, j) = buff[Cols * i + j];
    }
  }
  return results;
}

vector<Matrix4f> ReadPoses(const string filename){
  vector<Matrix4f> poses;
  ifstream file(filename);
  std::string line;
  while (std::getline(file, line)) {
    double buff[20] = {0};
    stringstream stream(line);
    int itter = 0;
    while (!stream.eof()) {
      stream >> buff[itter++];
    }
    Matrix4f result = Matrix4f::Identity();
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 4; j++) {
        result(i, j) = buff[4 * i + j];
      }
    }
    poses.push_back(result);
  }
  return poses;
}
