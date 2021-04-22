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

    
    private void Start()
    {
        audioClip.LoadAudioData();

        if (TryGetChunkFromAudioClip(audioClip, 48000, out float[] data))
        {
            Debug.Log($"Before: {data[0]}");
            int exitCode = cudaTest(data, data.Length);
            Debug.Log($"Cuda Exit code = {exitCode}");
            Debug.Log($"After: {data[0]}");

            float largestValue = -1;
            for (uint i = 0; i < data.Length; i++) {
                if (data[i] > largestValue)
                    largestValue = data[i];
            }
            
            Debug.Log($"Largest Value: {largestValue}");
        }
        else {
            Debug.LogError($"({audioClip.name}, {48000})");
        }
    }

    
    private static float[] GetSingleChannelFromAudioClip(AudioClip pAudioClip, uint pChannelID = 0)
    {
        int channelCount = pAudioClip.channels;
        if (pChannelID >= channelCount)
            return null;

        int sampleCount = pAudioClip.samples;

        float singleChannelLength = (float)sampleCount / channelCount;
        if ((singleChannelLength % 1) > 0f)
            return null;

        float[] data = new float[(uint)singleChannelLength];
        if (pAudioClip.GetData(data, channelCount))
            return data;

        return null;
    }
    
    // Highly optimized input sizes can be written in the form:
    // 2^a * 3^b * 5^c * 7^d (aka its prime factorization does not contain a prime > 7)
    // Non-optimal input size example:  48007 => 61^1 * 787^1
    // Optimal input size example: 48000 => 2^7 * 3^1 * 5^3
    private static bool TryGetChunkFromAudioClip(AudioClip pAudioClip, uint pBlockSize, out float[] pData, uint pChannelID = 0)
    {
        pData = null;
        
        int channelCount = pAudioClip.channels;
        if (pChannelID >= channelCount)
            return false;

        int sampleCount = pAudioClip.samples;
        if (sampleCount < pBlockSize)
            return false;

        float[] data = new float[pBlockSize];
        if (pAudioClip.GetData(data, channelCount)) {
            pData = data;
            return true;
        }
        
        return false;
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
