#include <opencv2/opencv.hpp>
#include <chrono>

// Размытие
void blurEffect(cv::Mat& frame) {
    cv::GaussianBlur(frame, frame, cv::Size(15, 15), 0);
}

// Инверсия
void invertEffect(cv::Mat& frame) {
    cv::bitwise_not(frame, frame);
}

// Черно-белое
void grayEffect(cv::Mat& frame) {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        return -1;
    }
    
    const char* effectNames[] = {
        "1: ORIGINAL",
        "2: BLUR",
        "3: INVERT",
        "4: GRAYSCALE"
    };

    void (*effects[])(cv::Mat&) = {
        [](cv::Mat& frame){},
        blurEffect,
        invertEffect,
        grayEffect
    };

    int currentEffect = 0;
    cv::Mat frame;

    // Переменные для расчета FPS
    auto startTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    double fps = 0;
    
    while(true) {
        // Захватываем кадр с камеры
        cap >> frame;
        if(frame.empty()) break;

        // Применяем выбранный эффект
        effects[currentEffect](frame);

        // Расчет FPS
        frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count();

        if (elapsedTime >= 1000) {  // Каждую секунду обновляем FPS
            fps = frameCount * 1000.0 / elapsedTime;
            frameCount = 0;
            startTime = currentTime;
        }

        // Добавляем название эффекта и FPS на кадр
        cv::putText(frame, effectNames[currentEffect],
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                   cv::Scalar(0, 255, 0), 2);

        cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(fps)),
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                   cv::Scalar(0, 255, 0), 2);
        
        cv::imshow("Video Effects", frame);
        
        int key = cv::waitKey(1);
        if(key == 27) break; // ESC - выход

        // Переключаем эффекты
        if(key >= '1' && key <= '4') {
            currentEffect = key - '1';
        }
    }

    return 0;
}

