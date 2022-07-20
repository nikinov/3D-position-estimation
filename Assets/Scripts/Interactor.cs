using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class Interactor : MonoBehaviour
{
    [SerializeField] private TMP_InputField inputField;
    [SerializeField] private TextMeshProUGUI positionText;

    public void DisplayCalculations(){
        var detections = PositionCalculation.Detect("/Volumes/SSDexternal/Windows/MLGit/TennisARPriv/Assets/StreamingAssets/AI/testImage.jpg", Application.dataPath + "/StreamingAssets/AI/best.onnx");
        
        for (int i = 0; i < detections.Count; i++)
        {
            var detection = detections[i];
            double[,] outputPossition;
            PositionCalculation.CalculateProjection(detection, out outputPossition);
            positionText.text += outputPossition[0, 0] + " " + outputPossition[1, 0] + " " + outputPossition[2, 0] + " " + "\n";
        }
    }
}
