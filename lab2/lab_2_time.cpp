#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>

// Эффект размытия
void blurEffect(cv::Mat& frame) {
    cv::GaussianBlur(frame, frame, cv::Size(15, 15), 0);
}

// Эффект инверсии цветов
void invertEffect(cv::Mat& frame) {
    cv::bitwise_not(frame, frame);
}

// Эффект черно-белого
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
    
    // Переменные для расчета FPS и времени
    auto programStart = std::chrono::high_resolution_clock::now();
    int totalFrames = 0;
    
    // Переменные для измерения времени операций
    double totalCaptureTime = 0;
    double totalProcessTime = 0;
    double totalDisplayTime = 0;

    // Главный цикл обработки видео
    while(true) {
        auto frameStart = std::chrono::high_resolution_clock::now();
        auto captureStart = frameStart;

        cap >> frame;
        if(frame.empty()) break;
        
        auto captureEnd = std::chrono::high_resolution_clock::now();
        auto processStart = captureEnd;

        effects[currentEffect](frame);
        
        auto processEnd = std::chrono::high_resolution_clock::now();
        auto displayStart = processEnd;
        
        cv::putText(frame, effectNames[currentEffect], 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                   cv::Scalar(0, 255, 0), 2);
        
        cv::imshow("Video Effects", frame);
        
        auto displayEnd = std::chrono::high_resolution_clock::now();

        // Измеряем время каждой операции (в миллисекундах)
        double captureTime = std::chrono::duration_cast<std::chrono::microseconds>(captureEnd - captureStart).count() / 1000.0;
        double processTime = std::chrono::duration_cast<std::chrono::microseconds>(processEnd - processStart).count() / 1000.0;
        double displayTime = std::chrono::duration_cast<std::chrono::microseconds>(displayEnd - displayStart).count() / 1000.0;
        
        // Суммируем общее время
        totalCaptureTime += captureTime;
        totalProcessTime += processTime;
        totalDisplayTime += displayTime;
        totalFrames++;

        int key = cv::waitKey(1);
        if(key == 27) break; // ESC - выход
        
        if(key >= '1' && key <= '4') {
            currentEffect = key - '1';
        }
    }

    // Выводим итоговую статистику после завершения программы
    if (totalFrames > 0) {
        auto programEnd = std::chrono::high_resolution_clock::now();
        double totalProgramTime = std::chrono::duration_cast<std::chrono::microseconds>(programEnd - programStart).count() / 1000000.0;
        
        double avgFPS = totalFrames / totalProgramTime;
        double avgFrameTime = totalProgramTime * 1000 / totalFrames;
        
        double avgCaptureTime = totalCaptureTime / totalFrames;
        double avgProcessTime = totalProcessTime / totalFrames;
        double avgDisplayTime = totalDisplayTime / totalFrames;
        
        double capturePercent = (avgCaptureTime / avgFrameTime) * 100;
        double processPercent = (avgProcessTime / avgFrameTime) * 100;
        double displayPercent = (avgDisplayTime / avgFrameTime) * 100;
        
        std::cout << "Total frames processed: " << totalFrames << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(4) << totalProgramTime << " seconds" << std::endl;
        std::cout << "Average FPS: " << std::fixed << std::setprecision(4) << avgFPS << std::endl;
        std::cout << "Average frame time: " << std::fixed << std::setprecision(4) << avgFrameTime << " ms" << std::endl;
        std::cout << std::endl;
        std::cout << "Time distribution per frame:" << std::endl;
        std::cout << "Capture: " << std::fixed << std::setprecision(4) << avgCaptureTime << " ms (" 
                  << std::fixed << std::setprecision(4) << capturePercent << "%)" << std::endl;
        std::cout << "Process: " << std::fixed << std::setprecision(4) << avgProcessTime << " ms (" 
                  << std::fixed << std::setprecision(4) << processPercent << "%)" << std::endl;
        std::cout << "Display: " << std::fixed << std::setprecision(4) << avgDisplayTime << " ms (" 
                  << std::fixed << std::setprecision(4) << displayPercent << "%)" << std::endl;
    }

    return 0;
}
