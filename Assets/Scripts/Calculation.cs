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

public class Calculation : MonoBehaviour
{
    private const float INPUT_WIDTH = (float) 640.0;
    private const float INPUT_HEIGHT = (float) 640.0;
    private const float SCORE_THRESHOLD = (float) 0.2;
    private const float NMS_THRESHOLD = (float) 0.4;
    private const float CONFIDENCE_THRESHOLD = (float) 0.4;
    
    
    private const string model = "/StreamingAssets/AI/yolov3-tiny.weights";

    private const string config = "/StreamingAssets/AI/yolov3-tiny.cfg";

    private const string MY_MODEL = "D:/Windows/Prototype/Ultralitics_test/yolov5/runs/train/exp3/weights/best.onnx";
    //private const string OPTIMIZED_MODEL = ;
    
    public delegate void PerformanceAction(List<Stopwatch> stopwatches);
    public event PerformanceAction OnPerformanceCalculation; 
    
    struct Detection
    {
        public int classId;
        public double confidence;
        public Rect2d box2d;
        public Rect box;
        public int size;
    };
    
    // Start is called before the first frame update
    void Start()
    {
        CalculateMat();
    }

    public void CalculateMat()
    {
        Mat mat = Imgcodecs.imread(
            Application.dataPath + "/StreamingAssets/AI/testImage.jpg");

        List<Detection> detections;
        List<string> classNames;

        Detect(mat, Application.dataPath + "/StreamingAssets/AI/best.onnx",
            out detections, out classNames);
        
    }

    private Mat FormatForYolo(Mat source)
    {
        int col = source.cols();
        int row = source.rows();
        int _max = Math.Max(col, row);
        Mat resized = Mat.zeros(_max, _max, CvType.CV_8UC3);
        source.copyTo(resized);
        return resized;
    }

    private void Detect(Mat image, string onnxNetPath, out List<Detection> detections, out List<string> classNames)
    {
        // performance stopwatches
        Stopwatch stopWatchReadingInput = new Stopwatch();
        Stopwatch stopWatchForwardPass = new Stopwatch();
        Stopwatch stopWatchOutputExtractionForLoop = new Stopwatch();
        Stopwatch stopWatchBBoxesForLoop = new Stopwatch();
        Stopwatch stopWatchMetix = new Stopwatch();
        
        // get input and model
        stopWatchReadingInput.Start();
        List<Mat> predictions = new List<Mat>();
        var input_image = FormatForYolo(image);
        var blob = Dnn.blobFromImage(input_image, 1 / 255.0, new Size(INPUT_WIDTH, INPUT_HEIGHT), new Scalar(0, 0, 0),
            true, false);
        var net = Dnn.readNet(onnxNetPath);
        //var config_filepath = Application.dataPath + config;
        //var model_filepath = Application.dataPath + model;
        //var net = Dnn.readNet(model_filepath, config_filepath);
        stopWatchReadingInput.Stop();
        
        // forward pass
        stopWatchForwardPass.Start();
        net.setInput(blob);
        net.forward(predictions);
        stopWatchForwardPass.Stop();

        detections = new List<Detection>();
        classNames = new List<string>();

        //var outBlobTypes = getOutputsTypes(net);
        
        // Extraction for loop
        stopWatchOutputExtractionForLoop.Start();
        double xFactor = input_image.cols() / INPUT_WIDTH;
        double yFactor = input_image.rows() / INPUT_HEIGHT;

        predictions[0] = predictions[0].reshape(1, (int) predictions[0].total() / 6);

        detections = new List<Detection>();
        classNames = new List<string>();
        
        // outputs
        var bboxesRec2d = new List<Rect2d>();
        var bboxesRec = new List<Rect>();
        var confidences = new List<float>();
        var classId = new List<int>();

        // output structure https://medium.com/mlearning-ai/detecting-objects-with-yolov5-opencv-python-and-c-c7cf13d1483c
        
        for (int i = 0; i < predictions[0].rows(); i++)
        {
            var confidence = predictions[0].get(i, 4)[0];
            
            if (confidence > CONFIDENCE_THRESHOLD)
            {
                var objPredictions = new List<double>();

                for (int j = 5; j < predictions[0].cols(); j++)
                {
                    objPredictions.Add(predictions[0].get(i, j)[0]);
                }

                var halfWidth = (predictions[0].get(i, 2)[0] / 2)*xFactor;
                var halfHeight = (predictions[0].get(i, 3)[0] / 2)*yFactor;
                var xCenter = (predictions[0].get(i, 0)[0])*xFactor;
                var yCenter = (predictions[0].get(i, 1)[0])*yFactor;
                
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
        stopWatchOutputExtractionForLoop.Stop();
        
        // BBoxes extraction
        stopWatchBBoxesForLoop.Start();
        var bboxes = new MatOfRect2d(bboxesRec2d.ToArray());
        MatOfInt indices = new MatOfInt();
        MatOfFloat floatConfidences = new MatOfFloat(confidences.ToArray());
        
        Dnn.NMSBoxes(bboxes, floatConfidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

        var indic = indices.toArray();
        
        for (int i = 0; i < indic.Length; i++)
        {
            var detection = new Detection();
            detection.box2d = bboxesRec2d[indic[i]];
            detection.box = bboxesRec[indic[i]];
            detection.classId = classId[indic[i]];
            detection.confidence = confidences[indic[i]];
            var croppedImage = new Mat(input_image, detection.box);
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
        stopWatchBBoxesForLoop.Stop();
        
        // masking
        stopWatchMetix.Start();

        
        // projection calculation example
        Double[,] possition;
        CalculateProjection(detections[0], out possition);

        stopWatchMetix.Stop();
        
        classNames = new List<string>();
        
        var stopwatches = new List<Stopwatch>();
        stopwatches.Add(stopWatchReadingInput);
        stopwatches.Add(stopWatchForwardPass);
        stopwatches.Add(stopWatchOutputExtractionForLoop);
        stopwatches.Add(stopWatchBBoxesForLoop);
        stopwatches.Add(stopWatchMetix);
        OnPerformanceCalculation?.Invoke(stopwatches);
    }

    private void CalculateProjection(Detection detection, out Double[,] Possition)
    {
        var cameraMetrix = new Double[,] {{2022.79163, 0, 765.41048}, {0, 2072.11987, 586.02691}, { 0, 0, 1 }};
        var possitionMetrix = new Double[,]
            {{detection.box.x - detection.box.width / 2}, {detection.box.y - detection.box.height / 2}, { detection.size}};
        
        Possition = MatrixMultiply(cameraMetrix, possitionMetrix);
    }

    Double[,] MatrixMultiply(Double[,] matA, Double[,] matB)
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
    
    protected virtual List<string> getOutputsTypes(Net net)
    {
        List<string> types = new List<string>();

        MatOfInt outLayers = net.getUnconnectedOutLayers();
        for (int i = 0; i < outLayers.total(); ++i)
        {
            types.Add(net.getLayer(new DictValue((int)outLayers.get(i, 0)[0])).get_type());
        }
        outLayers.Dispose();

        return types;
    }
    
    static double standardDeviation(IEnumerable<double> sequence)
    {
        double result = 0;

        if (sequence.Any())
        {
            double average = sequence.Average();
            double sum = sequence.Sum(d => Math.Pow(d - average, 2));
            result = Math.Sqrt((sum) / sequence.Count());
        }
        return result;
    }

    private void GetImageMask(List<Detection> detections, Mat img, out Mat mask)
    {
        mask = new Mat();
        //Imgproc.bilateralFilter(img, img, 5, 75, 75);
        int i = 0;
        
        //Imgproc.threshold(img, dist, 128, 255, Imgproc.THRESH_OTSU);
        
        foreach (var detection in detections)
        {
            //Imgcodecs.imwrite("D:/Windows/Prototype/Ultralitics_test/tes" + i.ToString() + ".png", croppedImage);
            
            Imgproc.rectangle(img, detection.box, new Scalar(255, 0, 255));
            i++;
        }
        
        Imgcodecs.imwrite("D:/Windows/Prototype/Ultralitics_test/test.png", img);
    }

    private void FocusParam()
    {
        double fx = 0;
        double F = 3;
        double W = 6;
        double w = 1920;

        fx = F * W / w;
    }

    private int GetElapsedMs(Stopwatch stopwatch)
    {
        TimeSpan ts = stopwatch.Elapsed;
        return ts.Milliseconds;
    }

    private void GetIntrinsics()
    {
        
    }
}
