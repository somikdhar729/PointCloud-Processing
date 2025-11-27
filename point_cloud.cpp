#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <random>
#include <Eigen/Geometry>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

// generates a point cloud representing a box shape that is open on one side
// Points are evenly distributed
// Starts from 0 and end at full dimension
// Function to generate a box-shaped point cloud open on one side
PointCloudT::Ptr generateBoxPointCloud(float width, float height, float depth, int total_points)
{
    PointCloudT::Ptr cloud(new PointCloudT);

    // Calculate surface areas
    float area_xy = width * height;
    float area_xz = width * depth;
    float area_yz = height * depth;
    float total_area = 2 * (area_xy + area_xz + area_yz) - area_yz; // Subtract front face

    // Calculate points per face
    int points_xy = static_cast<int>(total_points * area_xy / total_area);
    int points_xz = static_cast<int>(total_points * area_xz / total_area);
    int points_yz = static_cast<int>(total_points * area_yz / total_area);

    // Generate points for each face
    // lambda function: generates a 3D point cloud for a surface based on a custom mathematical function (func)
    auto generate_face = [&](int nx, int ny, auto func) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                float x = static_cast<float>(i) / (nx - 1);
                float y = static_cast<float>(j) / (ny - 1);
                cloud->points.push_back(func(x, y));
            }
        }
        };

    // Bottom and top faces
    generate_face(sqrt(points_xy), sqrt(points_xy), [&](float x, float y) { return PointT(x * width, y * height, 0); });
    generate_face(sqrt(points_xy), sqrt(points_xy), [&](float x, float y) { return PointT(x * width, y * height, depth); });

    // Left and right faces
    generate_face(sqrt(points_yz), sqrt(points_yz), [&](float y, float z) { return PointT(0, y * height, z * depth); });
    generate_face(sqrt(points_yz), sqrt(points_yz), [&](float y, float z) { return PointT(width, y * height, z * depth); });

    // Back face
    generate_face(sqrt(points_xz), sqrt(points_xz), [&](float x, float z) { return PointT(x * width, height, z * depth); });

    // Front face
    //generate_face(sqrt(points_xz), sqrt(points_xz), [&](float x, float z) { return PointT(x * width, 0, z * depth); });
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}


void addGaussianNoise(PointCloudT::Ptr cloud, float mean = 0.0f, float stddev = 0.01f)
{
    std::default_random_engine generator; //  Creates a random number generator
    std::normal_distribution<float> distribution(mean, stddev);
    
    for (auto& point : *cloud)
    {
        point.x += distribution(generator);
        point.y += distribution(generator);
        point.z += distribution(generator);
    }
}

void removeNoise(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_filtered)
{
    // works by analyzing the distribution of point - to - neighbor distances in the cloud.Points with a mean distance larger than the global mean distance plus a standard deviation multiplier are considered outliers and removed.
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(75); // number of nearest neighbors to use for mean distance estimation
    sor.setStddevMulThresh(0.5); // standard deviation multiplier threshold
    sor.filter(*cloud_filtered);
}

void rotatePointCloud(PointCloudT::Ptr cloud, const Eigen::Quaternionf& q)
{
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3,3>(0,0) = q.toRotationMatrix();
    pcl::transformPointCloud(*cloud, *cloud, transform);
}

void translatePointCloud(PointCloudT::Ptr cloud, float x, float y, float z)
{
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3,1>(0,3) = Eigen::Vector3f(x, y, z);
    pcl::transformPointCloud(*cloud, *cloud, transform);
}

pcl::PointCloud<pcl::Normal>::Ptr computeNormals(PointCloudT::Ptr cloud)
{
    // estimates normals by analyzing the local neighborhood of each point within the specified radius
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    ne.setSearchMethod(tree);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch(0.03);
    ne.compute(*cloud_normals);
    return cloud_normals;
}

int main()
{
    // Create box point cloud
    PointCloudT::Ptr cloud = generateBoxPointCloud(1.0, 2.5, 5.0, 5000);

    // Add Gaussian noise
    PointCloudT::Ptr noisy_cloud(new PointCloudT);
    addGaussianNoise(cloud, 0.0, 0.5);
    pcl::copyPointCloud(*cloud, *noisy_cloud);

    // Remove noise
    PointCloudT::Ptr denoise_cloud(new PointCloudT);
    PointCloudT::Ptr cloud_filtered(new PointCloudT);
    removeNoise(cloud, cloud_filtered);
    pcl::copyPointCloud(*cloud_filtered, *denoise_cloud);

    // Create a copy for rotation
    PointCloudT::Ptr cloud_rotated(new PointCloudT); // Will be used for visualization purposes
    pcl::copyPointCloud(*cloud_filtered, *cloud_rotated);

    // Rotate point cloud
    // quaternion = [0.2705, -0.2705, 0.6532, 0.6532]
    Eigen::Quaternionf q(0.6532, 0.2705, -0.2705, 0.6532); //= Eigen::AngleAxisf(M_PI / 4, Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf(M_PI / 4, Eigen::Vector3f::UnitY());
    rotatePointCloud(cloud_rotated, q);

    // Create a copy for translation
    PointCloudT::Ptr cloud_translated(new PointCloudT);
    pcl::copyPointCloud(*cloud_rotated, *cloud_translated);

    // Translate point cloud
    translatePointCloud(cloud_translated, 2.0f, 2.5f, 2.5f);

    // Compute normals for the translated cloud
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = computeNormals(cloud_translated);

    // Visualization
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(1.0);

    // Add original point cloud (red)
    viewer->addPointCloud(cloud_filtered, "original");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "original");

    // Noisy Cloud
     viewer->addPointCloud(noisy_cloud, "noisy");
     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 1, "noisy");

    // Denoise Cloud
     viewer->addPointCloud(denoise_cloud, "denoise");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "denoise");

    // Add rotated point cloud (green)
     viewer->addPointCloud(cloud_rotated, "rotated");
     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "rotated");

    // Add translated point cloud (blue)
     viewer->addPointCloud(cloud_translated, "translated");
     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 1, "translated");

    // Add normals to the translated cloud
    viewer->addPointCloudNormals<PointT, pcl::Normal>(cloud_translated, cloud_normals, 10, 0.5, "normals");
    

    // Adjust camera viewpoint to show all point clouds
    viewer->setCameraPosition(0, 0, -3, 0, -1, 0);

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
    }

    return 0;
}