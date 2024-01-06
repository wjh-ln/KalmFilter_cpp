#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <random>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

ofstream fout;

int main(void)
{
    fout.open("data.txt");

    int k_steps = 100;     // 循环次数
    double delta_t = 0.05; // 时间间隔

    // 设置随机数生成器和正态分布
    std::random_device w1_rd;
    std::mt19937 w1_gen(w1_rd());                             // 使用随机设备作为种子
    std::normal_distribution<double> w1_dist(0.0, sqrt(0.1)); // 均值为0，标准差为1的正态分布
    std::random_device w2_rd;
    std::mt19937 w2_gen(w2_rd());
    std::normal_distribution<double> w2_dist(0.0, sqrt(0.1));

    std::random_device v1_rd;
    std::mt19937 v1_gen(v1_rd());
    std::normal_distribution<double> v1_dist(0.0, sqrt(1));
    std::random_device v2_rd;
    std::mt19937 v2_gen(v2_rd());
    std::normal_distribution<double> v2_dist(0.0, sqrt(1));

    // 建立预测模型
    MatrixXd A{{1, delta_t},
               {0, 1}};
    VectorXd B{{0.5 * pow(delta_t, 2),
                delta_t}};
    double u = 1;

    // 建立观测矩阵
    MatrixXd H{{1, 0},
               {0, 1}};
    // 定义测量值
    VectorXd Z{{0,
                0}};

    // 假设的过程预测噪声的协方差矩阵
    MatrixXd Q{{0.1, 0},
               {0, 0.1}};
    // 假设的观测噪声协方差矩阵
    MatrixXd R{{0.001, 0},
               {0, 0.001}};

    // 定义实际状态
    VectorXd X{{0,
                1}};
    // 定义先验估计值
    VectorXd X_hat_minus{{0,
                          1}};
    // 定义后验估计值
    VectorXd X_hat{{0,
                    1}};
    // 定义先验估计误差协方差矩阵
    MatrixXd P{{1, 0},
               {0, 1}};

    double Emse[2] = {0.0, 0.0};   // 估计值的均方误差
    double Emse_z[2] = {0.0, 0.0}; // 测量值的均方误差

    for (int i = 0; i < k_steps; i++)
    {
        VectorXd W{{w1_dist(w1_gen), w2_dist(w2_gen)}};
        VectorXd V{{v1_dist(v1_gen), v2_dist(v2_gen)}};
        // 系统状态空间方程，计算实际状态变量
        X = A * X + B * u + W;
        // 实际测量值，来自传感器，包含误差
        Z = H * X + V;

        // 卡尔曼滤波
        // 计算先验状态估计
        X_hat_minus = A * X_hat + B * u;
        // 计算先验状态估计协方差矩阵
        MatrixXd P_minus = A * P * A.transpose() + Q;
        // 计算卡尔曼增益
        MatrixXd K = (P_minus * H.transpose()) * (H * P_minus * H.transpose() + R).inverse();
        // 更新后验估计
        X_hat = X_hat_minus + K * (Z - H * X_hat_minus);
        // 后验估计误差协方差矩阵
        P = P_minus - K * H * P_minus;

        fout << X(0) << " " << X(1) << " "
             << Z(0) << " " << Z(1) << " "
             << X_hat(0) << " " << X_hat(1) << " "
             << X_hat_minus(0) << " " << X_hat_minus(1) << endl;

        Emse[0] = Emse[0] + pow((X(0) - X_hat(0)), 2);
        Emse[1] = Emse[1] + pow((X(1) - X_hat(1)), 2);

        Emse_z[0] = Emse_z[0] + pow((X(0) - Z(0)), 2);
        Emse_z[1] = Emse_z[1] + pow((X(1) - Z(1)), 2);
    }
    Emse[0] = Emse[0] / k_steps; // 计算估计值的均方误差
    Emse[1] = Emse[1] / k_steps;
    Emse_z[0] = Emse_z[0] / k_steps; // 计算测量值的均方误差
    Emse_z[1] = Emse_z[1] / k_steps;

    cout << Emse[0] << " " << Emse[1] << " " << Emse_z[0] << " " << Emse_z[1] << endl;

    return 0;
}