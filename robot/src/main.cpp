// g++ -o main main.cpp -I /rahome/sloewe/torch/install/include -L /rahome/sloewe/torch/install/lib -llua
// g++ -o main main.cpp -I /localhome/sloewe/torch/install/include -L /localhome/sloewe/torch/install/lib -llua
// g++ -o main main.cpp -I /data/sloewe/torch/install/include -L /data/sloewe/torch/install/lib -llua
// th -ldisplay.start 8000 0.0.0.0
// localhost:8000 

#include <stdio.h>
#include <iostream>
#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;

extern "C"{
#include "TH/TH.h"
#include "luaT.h"
#include "lualib.h"
#include "lauxlib.h"
//#include "TH/generic/THTensor.h"
#include <TH/THStorage.h>
#include <TH/THTensor.h>
#include <malloc.h>

int luaopen_libpaths(lua_State *L);
int luaopen_libtorch(lua_State *L);
int luaopen_libnn(lua_State *L);
int luaopen_libnnx(lua_State *L);
int luaopen_libimage(lua_State *L);
}

static long getAllocSize(void *ptr) {
  return malloc_usable_size(ptr);
}

static void *OpenCVMalloc(void */*allocatorContext*/, long size) {
    return cv::fastMalloc(size);
}

static void *OpenCVRealloc(void */*allocatorContext*/, void *ptr, long size) {
    // https://github.com/Itseez/opencv/blob/master/modules/core/src/alloc.cpp#L62
    void *oldMem = ((unsigned char**)ptr)[-1];
    void *newMem = cv::fastMalloc(size);
    memcpy(newMem, oldMem, getAllocSize(oldMem));
    return newMem;
}

static void OpenCVFree(void */*allocatorContext*/, void *ptr) {
    cv::fastFree(ptr);
}

static THAllocator OpenCVCompatibleAllocator;

extern "C"
void initAllocator() {
    OpenCVCompatibleAllocator.malloc = OpenCVMalloc;
    OpenCVCompatibleAllocator.realloc = OpenCVRealloc;
    OpenCVCompatibleAllocator.free = OpenCVFree;
}

THByteTensor* matToTHByte(cv::Mat & matArg, THByteTensor *outputPtr) {
    if (matArg.empty()) {
        cerr << "Empty cv Mat" << endl;
    }

    cv::Mat *matPtr;

    if (matArg.depth() == CV_16U) {
        matPtr = new cv::Mat;
        matArg.convertTo(*matPtr, CV_32F, 1.0 / 65535);
    } else {
        matPtr = new cv::Mat(matArg.clone());
    }

    // For convenience
    cv::Mat& mat = *matPtr;

    // Build new storage on top of the Mat
    outputPtr->storage = THByteStorage_newWithDataAndAllocator(
                mat.data,
                mat.step[0] * mat.rows,
            &OpenCVCompatibleAllocator,
            nullptr
            );


    int sizeMultiplier;
    if (mat.channels() == 1) {
        outputPtr->nDimension = mat.dims;
        sizeMultiplier = cv::getElemSize(mat.depth());
    } else {
        outputPtr->nDimension = mat.dims + 1;
        sizeMultiplier = mat.elemSize1();
    }

    outputPtr->size   = static_cast<long *>(THAlloc(sizeof(long) * outputPtr->nDimension));
    outputPtr->stride = static_cast<long *>(THAlloc(sizeof(long) * outputPtr->nDimension));

    if (mat.channels() > 1) {
        outputPtr->size[outputPtr->nDimension - 1] = mat.channels();
        outputPtr->stride[outputPtr->nDimension - 1] = 1;
    }

    for (int i = 0; i < mat.dims; ++i) {
        outputPtr->size[i] = mat.size[i];
        outputPtr->stride[i] = mat.step[i] / sizeMultiplier;
    }

    outputPtr->storageOffset = 0;
    delete matPtr;

    return outputPtr;
}

cv::Mat thByteToMat(const THByteTensor* T) {
    int rows = T->size[0];
    int cols = T->size[1];
    unsigned char* data = T->storage->data;

    cv::Mat mat(rows, cols, CV_8UC1, cv::Scalar());
    for(int i = 0 ; i < rows * cols ; ++i) {
        mat.at<uchar>(i) = data[i];
    }

    return mat;
}


int main()
{
  cv::Mat matImage = cv::imread("/rahome/sloewe/torch/myProjects/fullconv_rgbd/robot/person.bmp");

  lua_State *L = luaL_newstate();
  luaL_openlibs( L );
    
  if (luaL_dofile(L, "classifyImage.lua"))
  {
    printf("%s\n", lua_tostring(L, -1));
  }
  
  luaT_newmetatable(L, "torch.ByteTensor", NULL, NULL, NULL, NULL);

  lua_getglobal(L, "tsImage");
  THByteTensor *thImage;
  if (luaT_isudata(L, -1, "torch.ByteTensor") )
  {
    thImage = static_cast<THByteTensor*>(luaT_toudata(L, -1,"torch.ByteTensor") );
  }

  lua_gc(L,LUA_GCCOLLECT,0);
    
  matToTHByte(matImage, thImage);

  lua_getglobal(L, "classify");  // function to be called
  if(!lua_isfunction(L,-1))
  {
    lua_pop(L,1);
  }
    
  if (lua_pcall(L, 0, LUA_MULTRET, 0) != 0)
    luaL_error(L, "error running function `classify': %s \n", lua_tostring(L, -1));

  void *z = luaT_toudata(L, -1, "torch.ByteTensor");
  THByteTensor *res = THByteTensor_new();
  res = (THByteTensor*) z;
  lua_pop(L, 1);

  cv::Mat resultImage = thByteToMat(res);   
    

  namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
  imshow( "Display window", resultImage );
    waitKey(0);  
  return 0;
}
