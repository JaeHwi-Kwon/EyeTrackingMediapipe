using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using System.IO;
using Mediapipe;
using Mediapipe.Unity.EyeTrackingSystem;
using System;

public class GazeCalibrationToScreen : MonoBehaviour
{
    [SerializeField] private float width;
    [SerializeField] private float height;
    [SerializeField] private GameObject CircleForCalibration;
    [SerializeField] private GameObject LandMarkParserObj;


    // ������(���� Ȥ�� ������ �� �� ������ ����)
    private Vector2[] eyeDir;
    // �Ӹ� ����(����, ����) ����
    private Vector2[] headPose; // -> -180~180 ������ ���̹Ƿ� Normalize �ʿ�
    // ����� �� �迭�� ���� �ε��� ��
    private int curIdx;

    public int training_epochs = 1000;

    private float learning_rate = 0.01f;
    private int display_step = 50;

    private float weight, bias;

    private NDArray train_X_XAxis, train_Y_XAxis;
    private NDArray train_X_YAxis, train_Y_YAxis;
    int n_samples;

    private bool calibration_Done;

    /*
     �����ִ� ��ǥ�� ��ġ�� �����ϴ� �Լ�
     ������ = ((headPoseHorizontal + eyeX) * a + b, (headPoseVertical + eyeY) * c + d) 
     �� X ��� Y �࿡ ���� ���� ȸ�� �м� ����
     */

    private Vector2[] Coordinates;

    // Start is called before the first frame update
    void Start()
    {
        eyeDir = new Vector2[9];
        headPose = new Vector2[9];
        curIdx = 0;
        calibration_Done = false;
        // Ķ���극�̼ǿ� �ʿ��� ������ �� 9��
        Coordinates = new Vector2[] {   new Vector2(-0.4f,0.4f),
                                        new Vector2(0,0.3f),
                                        new Vector2(0.4f,0.4f),
                                        new Vector2(-0.4f,0),
                                        new Vector2(0,0),
                                        new Vector2(0.4f,0),
                                        new Vector2(-0.4f,-0.4f),
                                        new Vector2(0,-0.4f),
                                        new Vector2(0.4f,-0.4f)};

        transform.localPosition = new Vector2(Coordinates[curIdx].x * width, Coordinates[curIdx].y * height);
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {

            if (curIdx <= 8)
            {
                Debug.Log($"{curIdx + 1} ��° Ķ���극�̼� ����.");
                var landmark = LandMarkParserObj.GetComponent<FaceLandmarksParser>().FaceAndIrisLandmarks;
                eyeDir[curIdx] = LandMarkParserObj.GetComponent<GazeEstimator>().EyeTracker(landmark);
                headPose[curIdx] = LandMarkParserObj.GetComponent<GazeEstimator>().HeadPoseTracker(landmark);
                curIdx++;
                if( curIdx > 8 && !calibration_Done)
                {
                    CircleForCalibration.transform.localScale = Vector3.zero;
                    RunLinearRegression();
                    calibration_Done = true;
                    return;
                }
                transform.localPosition = new Vector2(Coordinates[curIdx].x * width, Coordinates[curIdx].y * height);
            }
        }
    }

    private bool RunLinearRegression()
    {
        tf.compat.v1.disable_eager_execution();

        PrepareData(eyeDir, headPose);

        var X_x = tf.placeholder(tf.float32);
        var Y_x = tf.placeholder(tf.float32);

        var W_x = tf.Variable(0.1f, name: "Xweight");
        var b_x = tf.Variable(0.2f, name: "Xbias");

        var X_y = tf.placeholder(tf.float32);
        var Y_y = tf.placeholder(tf.float32);

        var W_y = tf.Variable(0.1f, name: "Yweight");
        var b_y = tf.Variable(0.2f, name: "Ybias");

        var pred_x = tf.add(tf.multiply(X_x, W_x), b_x);
        var pred_y = tf.add(tf.multiply(X_y, W_y), b_y);

        var cost_x = tf.reduce_sum(tf.pow(pred_x - Y_x, 2.0f)) / (2.0f * n_samples);
        var cost_y = tf.reduce_sum(tf.pow(pred_y - Y_y, 2.0f)) / (2.0f * n_samples);

        var optimizer_x = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_x);
        var optimizer_y = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_y);

        var init = tf.global_variables_initializer();

        using var sess = tf.Session();

        sess.run(init);

        // Fit all training data
        for (int epoch = 0; epoch < training_epochs; epoch++)
        {
            foreach (var (x, y) in zip<float>(train_X_XAxis, train_Y_XAxis))
                sess.run(optimizer_x, (X_x, x), (Y_x, y));
            foreach (var (x, y) in zip<float>(train_X_YAxis, train_Y_YAxis))
                sess.run(optimizer_y, (X_y, x), (Y_y, y));

        }

        Debug.Log($"[X��] W={sess.run(W_x)} b={sess.run(b_x)}");
        Debug.Log($"[Y��] W={sess.run(W_y)} b={sess.run(b_y)}");

        float XWeight, XBias;
        float YWeight, YBias;

        XWeight = sess.run(W_x).numpy();
        YWeight = sess.run(W_y).numpy(); ;
        XBias = sess.run(b_x).numpy(); ;
        YBias = sess.run(b_y).numpy(); ;

        Debug.Log($"{XWeight} {YWeight} {XBias} {YBias}");

        LandMarkParserObj.GetComponent<GazeEstimator>().SaveCalibrationResult(XWeight,XBias,YWeight,YBias);
        return true;
    }

    private void PrepareData(Vector2[] eye, Vector2[] head)
    {
        float[] GazeX = new float[9];
        float[] coordinatesX = new float[9];
        float[] GazeY = new float[9];
        float[] coordinatesY = new float[9];
        for (int i=0;i<9;i++)
        {

            coordinatesX[i] = Coordinates[i].x;
            coordinatesY[i] = Coordinates[i].y;

            //GazeX[i] = eye[i].x;
            //GazeY[i] = eye[i].y;

            GazeX[i] = head[i].x;
            GazeY[i] = head[i].y;

            //GazeX[i] = eye[i].x + head[i].x;
            //GazeY[i] = eye[i].y + head[i].y;




        }


        train_X_XAxis = np.array(GazeX);
        train_Y_XAxis = np.array(coordinatesX);

        train_X_YAxis = np.array(GazeY);
        train_Y_YAxis = np.array(coordinatesY);

        n_samples = (int)train_X_XAxis.shape[0];
    }
}
