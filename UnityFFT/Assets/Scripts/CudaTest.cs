using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class CudaTest : MonoBehaviour
{
    private const string CUDA_DLL = "CudaFFT";

    private delegate void DebugDLL(IntPtr pMessagePtr, int pColor, int pSize);
    
    [DllImport(CUDA_DLL, CallingConvention = CallingConvention.Cdecl)]
    private static extern void registerDebugCallback(DebugDLL pDebugMessageCallback);
    
    [DllImport(CUDA_DLL, CallingConvention = CallingConvention.Cdecl)]
    private static extern int cudaTest(float[] pData, int pSize);

    public AudioClip audioClip;

    private void Start() {
        audioClip.LoadAudioData();
        int N = audioClip.samples * audioClip.channels;
        float[] data = new float[N];
        audioClip.GetData(data, 0);
        Debug.Log($"Before: {data[777777]}");
        
        int exitCode = cudaTest(data, data.Length);
        Debug.Log($"Cuda Exit code = {exitCode}");
    }

    [AOT.MonoPInvokeCallback(typeof(DebugDLL))]
    private static void DebugMessageCallback(IntPtr pMessagePtr, int pColor, int pSize) {
        string message = Marshal.PtrToStringAnsi(pMessagePtr, pSize);
        if (pColor == -1)
            Debug.LogError(message);
        else
            Debug.Log(message);
    }
    
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void RegisterDebugMessages() {
        registerDebugCallback(DebugMessageCallback);
    }
}
