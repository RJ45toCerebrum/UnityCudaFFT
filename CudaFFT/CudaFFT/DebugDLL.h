#pragma once

#include "IUnityInterface.h"
#include <stdio.h>
#include <string>
#include <stdio.h>
#include <sstream>

//Color Enum
enum class Color { Red = -1, Green, Blue, Black, White, Yellow, Orange };

extern "C"
{
    // Create a callback delegate. Unity will recieve all data in char* format
    typedef void(*UnityLogCallback)(const char* pMessage, int pColor, int pSize);

    static UnityLogCallback callbackInstance = nullptr;

    // So DLL can send logs to unity; Called within unity
    UNITY_INTERFACE_EXPORT void __cdecl registerDebugCallback(UnityLogCallback pF);
}

class DebugDLL
{
public:
    static std::stringstream ss;

public:
    static void log(const char* pLogData, Color pColor = Color::Black);
    static void log(const std::string* pLogData, Color pColor = Color::Black);
    static void log(const int pLogData, Color pColor = Color::Black);
    static void log(const char pLogData, Color pColor = Color::Black);
    static void log(const float pLogData, Color pColor = Color::Black);
    static void log(const double pLogData, Color pColor = Color::Black);
    static void log(const bool pLogData, Color pColor = Color::Black);

    static const char* currentMessage();
    static void clear();

private:
    inline static void sendLog(const Color& pColor);
};