//
// Created by phanquanghung on 10/11/2023.
//
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include "layer.h"
#include "net.h"
#include "benchmark.h"
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <vector>

// !< add by tianzx 2023.10.20
#include <iostream>
using namespace std;
using namespace cv;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Net yolo;
const int target_size = 412;
const float prob_threshold = 0.4f;
const float nms_threshold = 0.5f;
const int label_iris = 0;
const int label_eyelid = 1;
const float label_mark_split = 789;
const int num_mark = 3;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    cv::Mat mask;
    std::vector<float> mask_feat;
};
struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static void slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Crop");

    // set param
    ncnn::ParamDict pd;

    ncnn::Mat axes = ncnn::Mat(1);
    axes.fill(axis);
    ncnn::Mat ends = ncnn::Mat(1);
    ends.fill(end);
    ncnn::Mat starts = ncnn::Mat(1);
    starts.fill(start);
    pd.set(9, starts);// start
    pd.set(10, ends);// end
    pd.set(11, axes);//axes

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void interp(const ncnn::Mat& in, const float& scale, const int& out_w, const int& out_h, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Interp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 2);// resize_type
    pd.set(1, scale);// height_scale
    pd.set(2, scale);// width_scale
    pd.set(3, out_h);// height
    pd.set(4, out_w);// width

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void reshape(const ncnn::Mat& in, ncnn::Mat& out, int c, int h, int w, int d)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Reshape");

    // set param
    ncnn::ParamDict pd;

    pd.set(0, w);// start
    pd.set(1, h);// end
    if (d > 0)
        pd.set(11, d);//axes
    pd.set(2, c);//axes
    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void sigmoid(ncnn::Mat& bottom)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Sigmoid");

    op->create_pipeline(opt);

    // forward

    op->forward_inplace(bottom, opt);
    op->destroy_pipeline(opt);

    delete op;
}
static void matmul(const std::vector<ncnn::Mat>& bottom_blobs, ncnn::Mat& top_blob)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("MatMul");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 0);// axis

    op->load_param(pd);

    op->create_pipeline(opt);
    std::vector<ncnn::Mat> top_blobs(1);
    op->forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];

    op->destroy_pipeline(opt);

    delete op;
}
static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}
inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}
static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects)
{
    const int num_points = grid_strides.size();
    // const int num_class = 80;
    const int num_class = 2;
    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++)
    {
        const float* scores = pred.row(i) + 4 * reg_max_1;

        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float confidence = scores[k];
            if (confidence > score)
            {
                label = k;
                score = confidence;
            }
        }
        float box_prob = sigmoid(score);
        if (box_prob >= prob_threshold)
        {
            ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));
            {
                ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                ncnn::ParamDict pd;
                pd.set(0, 1); // axis
                pd.set(1, 1);
                softmax->load_param(pd);

                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;

                softmax->create_pipeline(opt);

                softmax->forward_inplace(bbox_pred, opt);

                softmax->destroy_pipeline(opt);

                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++)
            {
                float dis = 0.f;
                const float* dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++)
                {
                    dis += l * dis_after_sm[l];
                }

                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob;
            obj.mask_feat.resize(32);
            std::copy(pred.row(i) + 64 + num_class, pred.row(i) + 64 + num_class + 32, obj.mask_feat.begin());
            objects.push_back(obj);
        }
    }
}
static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}
static void decode_mask(const ncnn::Mat& mask_feat, const int& img_w, const int& img_h,
                        const ncnn::Mat& mask_proto, const ncnn::Mat& in_pad, const int& wpad, const int& hpad,
                        ncnn::Mat& mask_pred_result)
{
    ncnn::Mat masks;
    matmul(std::vector<ncnn::Mat>{mask_feat, mask_proto}, masks);
    fprintf(stdout, "Mask matmul: channel %d width: %d height: %d\n", masks.c, masks.w, masks.h);
    sigmoid(masks);
    fprintf(stdout, "Mask sigmoid: channel %d width: %d height: %d\n", masks.c, masks.w, masks.h);
    reshape(masks, masks, masks.h, in_pad.h / 4, in_pad.w / 4, 0);
    fprintf(stdout, "Mask reshape: channel %d width: %d height: %d\n", masks.c, masks.w, masks.h);

    slice(masks, mask_pred_result, (wpad / 2) / 4, (in_pad.w - wpad / 2) / 4, 2);
    slice(mask_pred_result, mask_pred_result, (hpad / 2) / 4, (in_pad.h - hpad / 2) / 4, 1);
    fprintf(stdout, "mask_pred_result slice: channel %d width: %d height: %d\n", mask_pred_result.c, mask_pred_result.w, mask_pred_result.h);
    interp(mask_pred_result, 4.0, img_w, img_h, mask_pred_result);
    fprintf(stdout, "mask_pred_result interp: channel %d width: %d height: %d\n", mask_pred_result.c, mask_pred_result.w, mask_pred_result.h);

}
static int detect(ncnn::Mat in_pad,
                  vector<Object> & objects,
                  int width,
                  int height,
                  float scale,
                  int wpad,
                  int hpad,
                  float prob_threshold,
                  float nms_threshold){
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);


    ncnn::Extractor ex = yolo.create_extractor();
    ex.input("images", in_pad);

    ncnn::Mat out;
    ex.extract("output0", out);
    // ex.extract("output", out);

    ncnn::Mat mask_proto;
    ex.extract("output1", mask_proto);
    // ex.extract("seg", mask_proto);

    std::vector<int> strides = { 8, 16, 32 };
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);

    std::vector<Object> proposals;
    std::vector<Object> objects8;
    generate_proposals(grid_strides, out, prob_threshold, objects8);

    proposals.insert(proposals.end(), objects8.begin(), objects8.end());

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
    for (int i = 0; i < count; i++) {
        float* mask_feat_ptr = mask_feat.row(i);
        std::memcpy(mask_feat_ptr, proposals[picked[i]].mask_feat.data(), sizeof(float) * proposals[picked[i]].mask_feat.size());
    }

    ncnn::Mat mask_pred_result;
    decode_mask(mask_feat, width, height, mask_proto, in_pad, wpad, hpad, mask_pred_result);

    objects.resize(count);
    fprintf(stdout, " mask_proto channel: %d width: %d height: %d\n", mask_proto.c, mask_proto.w, mask_proto.h);
    for (int i = 0; i < count; i++) {

        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
        __android_log_print(ANDROID_LOG_DEBUG, "yolo_segment_jni", "detect box: x: %f, y: %f, w: %f, h: %f",
                            x0, y0, x1- x0, y1 - y0);

        objects[i].mask = cv::Mat::zeros(height, width, CV_32FC1);
        cv::Mat mask = cv::Mat(height, width, CV_32FC1, (float *) mask_pred_result.channel(i));
        mask(objects[i].rect).copyTo(objects[i].mask(objects[i].rect));
    }
    return 0;
}

static void get_segment_boundary(vector<Object> & objects,
                                 vector<float> & boundaries,
                                 int width,
                                 int height
                                 ){
    __android_log_print(ANDROID_LOG_DEBUG, "yolo_segment_jni", "input size (%d %d)", width, height);
    cv::Mat mask_iris = Mat::zeros(height, width, CV_8U);
    cv::Mat mask_eyelid = Mat::zeros(height, width, CV_8U);
    for(size_t i = 0; i < objects.size(); i ++){
        const Object& obj = objects[i];
        for (int y = 0; y < height; y++) {
            uchar* iris_ptr = mask_iris.ptr(y);
            uchar* eyelid_ptr = mask_eyelid.ptr(y);
            const float* mask_ptr = obj.mask.ptr<float>(y);
            for (int x = 0; x < width; x++) {
                if (mask_ptr[x] >= 0.5)
                {
                    if(obj.label == label_iris){
                        iris_ptr[0] = cv::saturate_cast<uchar>(255);
                    }
                    eyelid_ptr[0] = cv::saturate_cast<uchar>(255);
                }
                iris_ptr += 1;
                eyelid_ptr += 1;
            }
        }
    }

    /// get contours for eyelid /////
    vector<vector<Point> > eyelid_contours;
    vector<Vec4i> eyelid_hierarchy;
    findContours(mask_eyelid, eyelid_contours, eyelid_hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    /// get contours for iris /////
    vector<vector<Point> > iris_contours;
    vector<Vec4i> iris_hierarchy;

    findContours(mask_iris, iris_contours, iris_hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    int num_contour = 0;
    for( size_t i = 0; i< eyelid_contours.size(); i++ ){
        vector<Point> contours = eyelid_contours.at(i);
        if(contours.size() <=  5){
            continue;
        }
        num_contour += 1;
    }
    for( size_t i = 0; i< iris_contours.size(); i++ ){
        vector<Point> contours = iris_contours.at(i);
        if (contours.size() <=  5){
            continue;
        }
        num_contour += 1;
    }
    boundaries.push_back(num_contour);
    for( size_t i = 0; i< eyelid_contours.size(); i++ ){
        vector<Point> contours = eyelid_contours.at(i);
        if (contours.size() <=  5){
            continue;
        }
        boundaries.push_back(label_eyelid);
        boundaries.push_back(contours.size());
    }
    for( size_t i = 0; i< iris_contours.size(); i++ ){
        vector<Point> contours = iris_contours.at(i);
        if (contours.size() <=  5){
            continue;
        }
        boundaries.push_back(label_iris);
        boundaries.push_back(contours.size());
    }

    for( size_t i = 0; i< eyelid_contours.size(); i++ )
    {
        vector<Point> contours = eyelid_contours.at(i);
        if (contours.size() <=  5){
            continue;
        }

        for(size_t ci = 0; ci < contours.size(); ci ++){
            Point p = contours.at(ci);
            boundaries.push_back(p.x);
            boundaries.push_back(p.y);
        }
    }


    for( size_t i = 0; i< iris_contours.size(); i++ )
    {
        vector<Point> contours = iris_contours.at(i);
        if (contours.size() <=  5){
            continue;
        }

        for(size_t ci = 0; ci < contours.size(); ci ++){
            Point p = contours.at(ci);

            boundaries.push_back(p.x);
            boundaries.push_back(p.y);
        }
    }

//    for(int i = 0; i < boundaries.size(); i ++){
//        __android_log_print(ANDROID_LOG_DEBUG, "yolo_segment_jni", "Detect boundary %f", boundaries.at(i));
//    }
}

extern "C"
{

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "yolo_segment_jni", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "yolo_segment_jni", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

JNIEXPORT jboolean JNICALL
//Java_com_tencent_yolov5ncnn_YoloSegment_Init(JNIEnv *env, jobject thiz, jobject assetManager) {
Java_com_songu_beaueye_engine_YoloSegment_Init(JNIEnv *env, jobject thiz, jobject assetManager) {
    // TODO: implement Init()
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_packing_layout = true;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    yolo.opt = opt;

    // init param
    {
        int ret = yolo.load_param(mgr, "best.param");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "yolo_segment_jni", "load_param failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        int ret = yolo.load_model(mgr, "best.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "yolo_segment_jni", "load_model failed");
            return JNI_FALSE;
        }
    }

    return JNI_TRUE;

}

JNIEXPORT jfloatArray JNICALL
//Java_com_tencent_yolov5ncnn_YoloSegment_Detect(JNIEnv *env, jobject thiz, jobject bitmap,
//                                               jboolean use_gpu) {
Java_com_songu_beaueye_engine_YoloSegment_Detect(JNIEnv *env, jobject thiz, jobject bitmap,
                                               jboolean use_gpu) {
    // TODO: implement Detect()
    __android_log_print(ANDROID_LOG_DEBUG, "yolo_segment_jni", "Start Detect");
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
    {
        return NULL;
    }

    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    const int width = info.width;
    const int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    // ncnn from bitmap
    // letterbox pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
//    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);
    std::vector<Object> objects;
    std::vector<float> segment_points;
    detect(in_pad, objects, width, height, scale, wpad, hpad, prob_threshold, nms_threshold);
    get_segment_boundary(objects, segment_points, width, height);
    int size_boundary = segment_points.size();
    jfloat returned_array [size_boundary];
    for (int i = 0; i < size_boundary; i ++){
        returned_array[i] = segment_points.at(i);
    }
    jfloatArray  result = env->NewFloatArray(size_boundary);
    env->SetFloatArrayRegion(result, 0, size_boundary, returned_array);
    return result;
}
}
