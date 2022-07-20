using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using OpenCVForUnity.CoreModule;
using UnityEngine;
using OpenCVForUnity.DnnModule;
using OpenCVForUnity.ImgcodecsModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using UnityEngine.UI;
using UnityEngine.UIElements;
using Random = System.Random;
using Rect = OpenCVForUnity.CoreModule.Rect;


public class PositionCalculation
{
    // params
    private const float INPUT_WIDTH = (float) 640.0;
    private const float INPUT_HEIGHT = (float) 640.0;
    private const float SCORE_THRESHOLD = (float) 0.2;
    private const float NMS_THRESHOLD = (float) 0.4;
    private const float CONFIDENCE_THRESHOLD = (float) 0.3;

    // detection result
    public struct Detection
    {
        public int classId;
        public double confidence;
        public Rect2d box2d;
        public int size;
    };
    
    // public detection wrapper function
    public static List<Detection> Detect(string imagePath, string modelPath)
    {
        // load model
        Net net = Dnn.readNet(modelPath);
        if (net == null)
        {
            throw new InvalidDataException("Failed to load model from " + modelPath);
        }

        // load image
        Mat img = Imgcodecs.imread(imagePath);
        if (img.empty())
        {
            throw new InvalidDataException("Failed to load image from " + imagePath);
        }

        // detect
        List<Detection> detections = Detect(img, net);

        // release
        img.Dispose();
        net.Dispose();
        return detections;
    }

    // projection calculation
    public static void CalculateProjection(Detection detection, out Double[,] Possition)
    {
        var cameraMetrix = new Double[,] {{2022.79163, 0, 765.41048}, {0, 2072.11987, 586.02691}, { 0, 0, 1 }};
        var possitionMetrix = new Double[,]
            {{detection.box2d.x - detection.box2d.width / 2}, {detection.box2d.y - detection.box2d.height / 2}, { detection.size}};
        
        Possition = MatrixMultiply(cameraMetrix, possitionMetrix);
    }

    // image formatter
    private static Mat FormatForYolo(Mat source)
    {
        int col = source.cols();
        int row = source.rows();
        int _max = Math.Max(col, row);
        Mat resized = Mat.zeros(_max, _max, CvType.CV_8UC3);
        source.copyTo(resized);
        return resized;
    }

    // object detection
    public static List<Detection> Detect(Mat image, Net net)
    {
        #region Preprocessing
        List<Mat> prediction = new List<Mat>();
        List<Detection> detections = new List<Detection>();
        var input = FormatForYolo(image);
        Mat blob = Dnn.blobFromImage(input, 1/255.0, new Size(INPUT_WIDTH, INPUT_HEIGHT), new Scalar(0, 0, 0), true, false);
        #endregion

        #region Forward pass
        net.setInput(blob);
        net.forward(prediction);
        blob.Dispose();
        #endregion

        #region Postprocessing
        prediction[0] = prediction[0].reshape(1, (int) prediction[0].total() / 6);

        double xFactor = input.cols() / INPUT_WIDTH;
        double yFactor = input.rows() / INPUT_HEIGHT;

        // outputs
        var bboxesRec2d = new List<Rect2d>();
        var bboxesRec = new List<Rect>();
        var confidences = new List<float>();
        var classId = new List<int>();

        #region Confidence filtering
        // filter out confidence scores less than threshold
        for (int i = 0; i < prediction[0].rows(); i++)
        {

            var confidence = prediction[0].get(i, 4)[0];

            if (confidence > CONFIDENCE_THRESHOLD)
            {
                var objPredictions = new List<double>();

                for (int j = 5; j < prediction[0].cols(); j++)
                {
                    objPredictions.Add(prediction[0].get(i, j)[0]);
                }

                var halfWidth = (prediction[0].get(i, 2)[0] / 2)*xFactor;
                var halfHeight = (prediction[0].get(i, 3)[0] / 2)*yFactor;
                var xCenter = (prediction[0].get(i, 0)[0])*xFactor;
                var yCenter = (prediction[0].get(i, 1)[0])*yFactor;
                
                double left = xCenter-halfWidth;
                double top = yCenter-halfHeight;
                double right = xCenter+halfWidth;
                double bottom = yCenter+halfHeight;

                classId.Add(objPredictions.IndexOf(objPredictions.Max()));
                bboxesRec2d.Add(new Rect2d((int) left, (int) top, (int) halfWidth*2, (int) halfHeight*2));
                bboxesRec.Add(new Rect((int) left, (int) top, (int) halfWidth*2, (int) halfHeight*2));
                confidences.Add((float)confidence);
            }
        }
        #endregion

        #region Non-maximum suppression
        var bboxes = new MatOfRect2d(bboxesRec2d.ToArray());
        MatOfInt indices = new MatOfInt();
        MatOfFloat floatConfidences = new MatOfFloat(confidences.ToArray());
        
        Dnn.NMSBoxes(bboxes, floatConfidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
        var indic = indices.toArray();
        #endregion

        #region Image making
        for (int i = 0; i < indic.Length; i++)
        {
            var detection = new Detection();
            detection.box2d = bboxesRec2d[indic[i]];
            detection.classId = classId[indic[i]];
            detection.confidence = confidences[indic[i]];
            var croppedImage = new Mat(input, bboxesRec[indic[i]]);
            Imgproc.cvtColor(croppedImage, croppedImage, Imgproc.COLOR_BGR2GRAY);
            Imgproc.threshold(croppedImage, croppedImage, 128, 255, Imgproc.THRESH_OTSU);
            int size = 0;
            for (int j = 0; j < croppedImage.cols(); j++)
            {
                for (int k = 0; k < croppedImage.rows(); k++)
                {
                    var pixel = croppedImage.get(k, j)[0];
                    if (pixel == 255)
                    {
                        size += 1;
                    }
                }
            }

            detection.size = size;
            detections.Add(detection);
        }
        #endregion
        
        #endregion
        
        return detections;
    }

    // matrix multiplication
    private static Double[,] MatrixMultiply(Double[,] matA, Double[,] matB)
    {
        Double[,] res;
        int rowsA = matA.GetLength(0);
        int columsA = matA.GetLength(1);
        int rowsB = matB.GetLength(0);
        int columsB = matB.GetLength(1);

        if (columsA != rowsB)
        {
            return null;
        }
        else
        {
            res = new double[rowsA, columsB];
            for (var a = 0; a < columsB; a++)
            {
                for (var i = 0; i < rowsA; i++)
                {
                    double sum = 0;
                    for (int j = 0; j < columsA; j++)
                    {
                        sum += matA[i, j] * matB[j, a];
                    }

                    res[i, a] = sum;
                    
                }
            }
        }

        return res;
    }
}
