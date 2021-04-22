#include "DebugDLL.h"


std::stringstream DebugDLL::ss = std::stringstream();

// Set static callbackInstance so unity can recieve Log messages from DLL
UNITY_INTERFACE_EXPORT void __cdecl registerDebugCallback(UnityLogCallback pF) {
    callbackInstance = pF;
}


void DebugDLL::log(const char* pLogData, Color pColor) {
    if (callbackInstance != nullptr)
        callbackInstance(pLogData, (int)pColor, (int)strlen(pLogData));
}

void DebugDLL::log(const std::string* pLogData, Color pColor) {
    if (callbackInstance != nullptr) {
        const char* tmsg = pLogData->c_str();
        callbackInstance(tmsg, (int)pColor, (int)strlen(tmsg));
    }
}

void DebugDLL::log(const int pLogData, Color pColor) {
    ss << pLogData;
    sendLog(pColor);
}

void DebugDLL::log(const char pLogData, Color pColor) {
    ss << pLogData;
    sendLog(pColor);
}

void DebugDLL::log(const float pLogData, Color pColor) {
    ss << pLogData;
    sendLog(pColor);
}

void DebugDLL::log(const double pLogData, Color pColor) {
    ss << pLogData;
    sendLog(pColor);
}

void DebugDLL::log(const bool pLogData, Color pColor) {
    if (pLogData)
        ss << "true";
    else
        ss << "false";

    sendLog(pColor);
}

const char* DebugDLL::currentMessage() {
    return DebugDLL::ss.str().c_str();
}

void DebugDLL::clear() {
    DebugDLL::ss.str().clear();
}

void DebugDLL::sendLog(const Color& pColor) {
    if (callbackInstance != nullptr) {
        const std::string tmp = ss.str();
        const char* tmsg = tmp.c_str();
        callbackInstance(tmsg, (int)pColor, (int)strlen(tmsg));
        DebugDLL::clear();
    }
}