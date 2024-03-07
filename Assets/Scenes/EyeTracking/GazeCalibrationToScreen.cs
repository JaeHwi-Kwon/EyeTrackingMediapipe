using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Mediapipe.Unity.EyeTrackingSystem;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using Tensorflow;

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

    private float learning_rate = 0.5f;

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
                                        new Vector2(0,0.4f),
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

    // ����ȸ�� �н�
    private bool RunLinearRegression()
    {
        tf.compat.v1.disable_eager_execution();

        PrepareData(eyeDir, headPose);

        var X1 = tf.placeholder(tf.float32);
        var X2 = tf.placeholder(tf.float32);
        var Y = tf.placeholder(tf.float32);

        var W1 = tf.Variable(0.0f);
        var W2 = tf.Variable(0.0f);
        var b = tf.Variable(0.0f);

        var pred = tf.add(tf.add(tf.multiply(X1, W1), tf.multiply(X2, W2)), b);
        var cost = tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * n_samples);

        var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost);

        var init = tf.global_variables_initializer();

        using var sess = tf.Session();
        sess.run(init);

        for(int epoch = 0;epoch < training_epochs; epoch++)
        {

            for (int i = 0; i < 9; i++)
            {
                var x1 = train_X_XAxis[0, i];
                var x2 = train_X_XAxis[1, i];
                var y = train_Y_XAxis[i];

                sess.run(optimizer, (X1, x1), (X2, x2), (Y, y));
            }
        }

        float XWeight1, XWeight2, XBias;
        XWeight1 = sess.run(W1).numpy();
        XWeight2 = sess.run(W2).numpy();
        XBias = sess.run(b).numpy();

        Debug.Log($"XW1 : {XWeight1}, XW2 : {XWeight2}, xb: {XBias}");

        //===================================================================================================================

        sess.run(init);

        for (int epoch = 0; epoch < training_epochs; epoch++)
        {

            for (int i = 0; i < 9; i++)
            {
                var x1 = train_X_YAxis[0, i];
                var x2 = train_X_YAxis[1, i];
                var y = train_Y_YAxis[i];

                sess.run(optimizer, (X1, x1), (X2, x2), (Y, y));
            }
        }

        float YWeight1, YWeight2, YBias;
        YWeight1 = sess.run(W1).numpy();
        YWeight2 = sess.run(W2).numpy();
        YBias = sess.run(b).numpy();

        Debug.Log($"YW1 : {YWeight1}, YW2 : {YWeight2}, Yb: {YBias}");

        LandMarkParserObj.GetComponent<GazeEstimator>().SaveCalibrationResult(XWeight1, XWeight2, XBias, YWeight1, YWeight2, YBias);

        return true;
    }

    // �н� �����ͼ� �غ�
    private void PrepareData(Vector2[] eye, Vector2[] head)
    {

        Debug.Log($"HeadPoseList : {head}");
        Debug.Log($"EyeDirList : {eye}");
        float[] HeadPoseX = new float[9];
        float[] HeadPoseY = new float[9];

        float[] EyeDirX = new float[9];
        float[] EyeDirY = new float[9];

        float[,] gazeX = new float[2, 9];
        float[,] gazeY = new float[2, 9];

        float[] coordinatesX = new float[9];
        float[] coordinatesY = new float[9];
        for (int i=0;i<9;i++)
        {
            float[] templist = new float[2];
            coordinatesX[i] = Coordinates[i].x;
            coordinatesY[i] = Coordinates[i].y;

            EyeDirX[i] = eye[i].x;
            EyeDirY[i] = eye[i].y;

            HeadPoseX[i] = head[i].x;
            HeadPoseY[i] = head[i].y;

            gazeX[0, i] = head[i].x;
            gazeX[1, i] = eye[i].x;

            gazeY[0, i] = head[i].y;
            gazeY[1, i] = eye[i].y;
        }

        // X �����ν��� X ������
        train_X_XAxis = np.array(gazeX);

        // Y �����ν��� X ������
        train_X_YAxis = np.array(gazeY);

        // X, Y �����ν��� Y �����
        train_Y_XAxis = np.array(coordinatesX);
        train_Y_YAxis = np.array(coordinatesY);

        n_samples = (int)train_X_XAxis.shape[0];
    }
}
